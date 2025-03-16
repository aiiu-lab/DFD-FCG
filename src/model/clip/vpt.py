import wandb
import torch
import pickle
import random

import torch.nn as nn
import torch.nn.functional as F

from operator import or_
from typing import List
from functools import reduce
from enum import IntFlag, auto

from src.model.base import ODBinaryMetricClassifier
from src.model.clip import VideoAttrExtractor
from src.utility.loss import focal_loss


class PromptMode(IntFlag):
    DEEP = auto()  # spatial
    SHALLOW = auto()  # temporal


class PromptedVideoAttrExtractor(VideoAttrExtractor):
    def __init__(
        self,
        # VideoAttrExtractor
        architecture,
        text_embed,
        pretrain=None,
        store_attrs=[],
        attn_record=False,
        # visual prompting
        num_prompts=1,
        prompt_mode=PromptMode.DEEP
    ):
        super(PromptedVideoAttrExtractor, self).__init__(
            architecture=architecture,
            text_embed=text_embed,
            store_attrs=store_attrs,
            attn_record=attn_record,
            pretrain=pretrain
        )
        self.num_prompts = num_prompts
        self.prompt_mode = prompt_mode

        if (prompt_mode == PromptMode.DEEP):
            # shape = batch,frames,layers,prompts,dim
            self.visual_prompts = nn.Parameter(torch.zeros(1, 1, self.n_layers, num_prompts, self.feat_dim), requires_grad=True)
        elif (prompt_mode == PromptMode.SHALLOW):
            # shape = batch,frames,layers,prompts,dim
            self.visual_prompts = nn.Parameter(torch.zeros(1, 1, 1, num_prompts, self.feat_dim), requires_grad=True)
        nn.init.normal_(self.visual_prompts, std=0.001)

    def forward(self, x):

        # first, we prepare the encoder before the transformer layers.
        x = self.model._prepare(x)

        x = torch.cat([x, self.visual_prompts[:, :, 0].repeat(x.shape[0], x.shape[1], 1, 1)], dim=-2)

        # now, we alternate between synoptic and encoder layers
        for i, enc_blk in enumerate(self.model.transformer.resblocks):
            if i > 0 and self.prompt_mode == PromptMode.DEEP:
                x[:, :, :self.num_prompts] = (
                    x[:, :, :self.num_prompts] +
                    self.visual_prompts[:, :, i].repeat(x.shape[0], x.shape[1], 1, 1)
                )
            data = enc_blk(x)
            x = data["emb"]

        layer_attrs = []
        for enc_blk in self.model.transformer.resblocks:
            layer_attrs.append(
                {
                    **enc_blk.pop_attr()
                }
            )

        embeds = x[:, :, :self.num_prompts].mean(dim=2)

        return dict(
            layer_attrs=layer_attrs,
            embeds=embeds
        )

    def train(self, mode=True):
        super().train(mode)

        if (mode):
            self.model.eval()

        return self


class BinaryLinearClassifier(nn.Module):
    def __init__(
        self,
        *args,
        **kargs,
    ):
        super().__init__()
        self.encoder = PromptedVideoAttrExtractor(
            *args,
            **kargs
        )

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


class PromptedLinearVideoLearner(ODBinaryMetricClassifier):
    def __init__(
        self,
        architecture: str = 'ViT-B/16',
        text_embed: bool = False,
        attn_record: bool = False,
        pretrain: str = None,
        label_weights: List[float] = [1, 1],
        cls_weight: float = 10.0,
        store_attrs: List[str] = [],
        num_prompts=1,
        prompt_mode=PromptMode.DEEP
    ):
        super().__init__()
        self.save_hyperparameters()
        params = dict(
            architecture=architecture,
            text_embed=text_embed,
            attn_record=attn_record,
            pretrain=pretrain,
            store_attrs=store_attrs,
            num_prompts=num_prompts,
            prompt_mode=PromptMode[prompt_mode]
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
        attn_record=False,
        text_embed=False,
        num_prompts=4,
        prompt_mode=PromptMode.SHALLOW
    )
    model.to("cuda")
    result = model(torch.randn(5, frames, 3, 224, 224).to("cuda"))
    logit = result["logits"]
    logit.sum().backward()
    print([m for m, v in model.named_parameters() if v.requires_grad])
    print("done")
