import wandb
import torch
import pickle
import torch.nn as nn

from typing import List

from src.model.base import ODBinaryMetricClassifier
from src.model.clip import VideoAttrExtractor
from src.utility.loss import focal_loss


class BinaryLinearClassifier(nn.Module):
    def __init__(
        self,
        *args,
        **kargs,
    ):
        super().__init__()
        self.encoder = VideoAttrExtractor(
            *args,
            **kargs
        )
        self.encoder.model.requires_grad_(True)
        self.projs = self.make_linear(self.encoder.embed_dim)

    def make_linear(self, embed_dim):
        linear = nn.Linear(
            embed_dim,
            2
        )
        nn.init.normal_(linear.weight, std=0.001)
        nn.init.normal_(linear.bias, std=0.001)
        return linear

    @property
    def transform(self):
        return self.encoder.transform

    @property
    def n_px(self):
        return self.encoder.model.input_resolution

    def forward(self, x, *args, **kargs):
        results = self.encoder(x)
        embeds = results["embeds"]
        logits = self.projs(embeds.mean(1))
        return dict(
            logits=logits,
            ** results
        )


class FullTuneVideoLearner(ODBinaryMetricClassifier):
    def __init__(
        self,
        architecture: str = 'ViT-B/16',
        text_embed: bool = False,
        attn_record: bool = False,
        pretrain: str = None,
        label_weights: List[float] = [1, 1],
        cls_weight: float = 10.0,
        store_attrs: List[str] = [],
    ):
        super().__init__()
        self.save_hyperparameters()
        params = dict(
            architecture=architecture,
            text_embed=text_embed,
            attn_record=attn_record,
            pretrain=pretrain,
            store_attrs=store_attrs,
        )
        self.model = BinaryLinearClassifier(**params)
        self.label_weights = torch.tensor(label_weights)
        self.cls_weight = cls_weight

    @property
    def transform(self):
        return self.model.transform

    @property
    def n_px(self):
        return self.model.n_px

    def shared_step(self, batch, stage):
        x, y, z = batch["xyz"]
        indices = batch["indices"]
        dts_name = batch["dts_name"]
        names = batch["names"]

        output = self(x, **z)
        logits = output["logits"]
        loss = 0
        # classification loss
        if (stage == "train"):
            cls_loss = focal_loss(
                logits,
                y,
                gamma=4,
                weight=(
                    self.label_weights.to(y.device)
                    if stage == "train" else
                    None
                )
            )
            loss += cls_loss.mean() * self.cls_weight
            self.log(
                f"{stage}/{dts_name}/loss",
                cls_loss.mean(),
                batch_size=logits.shape[0]
            )
        else:
            # classification loss
            cls_loss = nn.functional.cross_entropy(
                logits,
                y,
                reduction="none",
                weight=(
                    self.label_weights.to(y.device)
                    if stage == "train" else
                    None
                )
            )
            loss += cls_loss.mean()

        return {
            "logits": logits,
            "labels": y,
            "loss": loss,
            "dts_name": dts_name,
            "indices": indices,
            "output": output
        }


if __name__ == "__main__":
    frames = 5
    model = BinaryLinearClassifier(
        architecture="ViT-L/14",
        attn_record=True,
        text_embed=False
    )
    model.to("cuda")
    result = model(torch.randn(5, frames, 3, 224, 224).to("cuda"))
    logit = result["logits"]
    logit.sum().backward()
    print([m for m, v in model.named_parameters() if v.requires_grad])
    print("done")
