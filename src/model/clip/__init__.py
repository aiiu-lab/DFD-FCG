import torch
import random
import logging
import open_clip
import torch.nn as nn
import src.clip.clip as CLIP


def load_model(architecture, **kargs):
    # architecture parameter split
    # e.g: ViT-B/16|laion2b_s34b_b79k
    params = architecture.split('|')

    clip_arch = params[0]
    if (len(params) == 1):
        model, transform = CLIP.load(
            clip_arch,
            "cpu",
            **kargs
        )

    elif (len(params) == 2):
        open_pretrain = params[1]

        _model, _, _ = open_clip.create_model_and_transforms(
            clip_arch.replace("/", "-"),
            pretrained=open_pretrain,
            device="cpu"
        )

        model, transform = CLIP.load(
            _model.state_dict(),
            "cpu",
            **kargs
        )
        del _model

    elif (len(params) > 2):
        raise NotImplementedError()

    return model, transform


class VideoAttrExtractor(nn.Module):
    def __init__(
        self,
        architecture,
        text_embed,
        store_attrs=[],
        attn_record=False,
        pretrain=None
    ):
        super().__init__()
        self.model, self.transform = load_model(
            architecture,
            store_attrs=store_attrs,
            attn_record=attn_record
        )
        self.model = self.model.visual.float()

        if (pretrain):
            logging.info("Loading image encoder pretrain weights...")
            state_dict = torch.load(pretrain, "cpu")
            try:
                self.model.load_state_dict(state_dict, strict=True)
            except:
                conflicts = self.model.load_state_dict(state_dict, strict=False)
                logging.warning(
                    f"during visual pretrain weights loading, disabling strict mode with conflicts:\n{conflicts}"
                )

        self.model.requires_grad_(False)

        if not text_embed:
            self.model.proj = None
            self.feat_dim = self.model.transformer.width
        else:
            self.feat_dim = self.model.output_dim

    @property
    def n_px(self):
        return self.model.input_resolution

    @property
    def n_layers(self):
        return self.model.transformer.layers

    @property
    def n_heads(self):
        return self.model.transformer.heads

    @property
    def patch_size(self):
        return self.model.patch_size

    @property
    def patch_num(self):
        return self.model.patch_num

    @property
    def n_patch(self):
        return int(self.n_px // self.patch_size)

    @property
    def embed_dim(self):
        return self.feat_dim

    def forward(self, x):
        b, t = x.shape[:2]

        # pass throught for attributes
        embeds = self.model(x)
        # retrieve all layer attributes
        layer_attrs = []
        for blk in self.model.transformer.resblocks:
            attrs = blk.pop_attr()
            layer_attrs.append(attrs)
        return dict(
            layer_attrs=layer_attrs,
            embeds=embeds
        )

    def train(self, mode=True):
        self.model.eval()
        return self


if __name__ == "__main__":
    VideoAttrExtractor(
        "ViT-L/14|laion2b_s32b_b82k",
        text_embed=False
    )
