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


class OpMode(IntFlag):
    S = auto()  # spatial
    T = auto()  # temporal


def call_module(module):
    def fn(*args, **kwargs):
        return module(*args, **kwargs)
    return fn


def get_module(module):
    def fn():
        return module
    return fn


class SynoBlock(nn.Module):
    def __init__(
        self,
        n_synos,
        d_model,
        n_head,
        n_patch,
        n_frames,
        ksize_t,
        ksize_s,
        t_attrs,
        s_k_attr,
        s_v_attr,
        op_mode,
        store_attrs=[],
        attn_record=False
    ):
        super().__init__()

        # parameters
        self.n_patch = n_patch
        self.t_attrs = t_attrs
        self.s_k_attr = s_k_attr
        self.s_v_attr = s_v_attr

        self.op_mode = op_mode

        if (OpMode.T in op_mode):
            # modules
            self.t_conv = self.make_2dconv(
                ksize_t,
                sum([
                    1 if attr in ["out", "emb"] else n_head
                    for attr in t_attrs
                ]),
                1
            )

            self.t_proj = nn.Sequential(
                nn.LayerNorm(n_frames**2),
                nn.Linear(
                    n_frames**2,
                    n_frames
                ),
                nn.GELU(),
                nn.Linear(
                    n_frames,
                    n_frames**2
                )
            )

            self.p_conv = self.make_2dconv(
                ksize_s,
                n_frames ** 2,
                1
            )

        if (OpMode.S in op_mode):

            self.syno_embedding = nn.Parameter(
                torch.zeros(n_synos, d_model)
            )

        # attribute storage
        self.store_attrs = store_attrs
        self.attr = {}

        # attention map recording
        self.attn_record = attn_record
        self.aff = None

    def make_2dconv(self, ksize, in_c, out_c, groups=1):
        conv = nn.Conv2d(
            in_channels=in_c,
            out_channels=out_c,
            kernel_size=ksize,
            stride=1,
            padding=ksize // 2,
            groups=groups,
            bias=True
        )

        nn.init.normal_(conv.weight, std=0.001)
        nn.init.zeros_(conv.bias)

        return conv

    def pop_attr(self):
        ret = self.get_attr()
        self.attr.clear()
        return ret

    def get_attr(self):
        return {k: self.attr[k] for k in self.attr}

    def set_attr(self, **attr):
        self.attr = {
            k: attr[k]
            for k in attr
            if k in self.store_attrs
        }

    def temporal_detection(self, attrs):
        b, t, l, h, d = attrs['q'][:, :, 1:].shape  # ignore cls token
        p = self.n_patch  # p = l ** 0.5

        affs = []
        for attr in self.t_attrs:
            _attr = attrs[attr][:, :, 1:]  # ignore cls token

            if (len(_attr.shape) == 4):
                _attr = _attr.unsqueeze(-2)

            _attr = _attr.permute(0, 2, 1, 3, 4)

            aff = torch.einsum(
                'nlqhc,nlkhc->nlqkh',
                _attr / (_attr.size(-1) ** 0.5),
                _attr
            )

            aff = aff.softmax(dim=-2)

            aff = aff.flatten(0, 1)  # shape = (n*l,t,t,h)
            aff = aff.permute(0, 3, 1, 2)  # shape = (n*l,h,t,t)
            affs.append(aff)

        aff = torch.cat(affs, dim=1)  # shape = (n*l, 3*h, t, t)

        aff = self.t_conv(aff)  # shape = (n*l, r, t, t) where r is number of filters

        aff = aff.unflatten(0, (b, p, p)).flatten(3)  # shape = (n, p, p, t*t)
        aff = aff + self.t_proj(aff)
        aff = aff.permute(0, 3, 1, 2)  # shape = (n, t*t, p, p)

        aff = self.p_conv(aff)  # shape = (n, 1, p, p)

        y = aff.flatten(1)

        return dict(y=y)  # shape = (n, p*p)

    def spatial_detection(self, attrs):
        b, t, l, h, d = attrs['q'][:, :, 1:].shape  # ignore cls token

        _k = attrs[self.s_k_attr][:, :, 1:]  # ignore cls token
        _v = attrs[self.s_v_attr][:, :, 1:]  # ignore cls token

        # prepare query
        s_q = self.syno_embedding.unsqueeze(0).repeat(b, 1, 1)  # shape = (b, synos, width)

        if (len(_k.shape) == 5):
            _k = _k.flatten(-2).contiguous()  # match shape

        if (len(_v.shape) == 5):
            _v = _v.flatten(-2).contiguous()  # match shape

        # ===============================
        # Version 1: Original Attention Module (Square Normalized)
        # s_aff = torch.einsum(
        #     'nqw,ntkw->ntqk',
        #     s_q / (s_q.size(-1) ** 0.5),
        #     _k
        # )

        # s_aff = s_aff.softmax(dim=-1)

        # Version 2: Modified Attention Module (Cosine Similarity)
        s_aff = torch.einsum(
            'nqw,ntkw->ntqk',
            s_q / (s_q.norm(dim=-1, keepdim=True) + 1e-4),
            _k / (_k.norm(dim=-1, keepdim=True) + 1e-4)
        )

        s_aff = (s_aff * 100).softmax(dim=-1)
        # ===============================
        s_mix = torch.einsum(
            'ntql,ntlw->ntqw',
            s_aff,
            _v
        )

        y = s_mix.flatten(1, 2).mean(dim=1)  # shape = (b,w)

        if self.attn_record:
            self.aff = s_aff

        return dict(s_q=s_q, y=y)

    def forward(self, attrs):
        self.pop_attr()
        y_t = 0
        y_s = 0
        if OpMode.T in self.op_mode:
            ret_t = self.temporal_detection(attrs)
            y_t = ret_t.pop('y')
            self.set_attr(
                **ret_t
            )
        if OpMode.S in self.op_mode:
            ret_s = self.spatial_detection(attrs)
            y_s = ret_s.pop('y')
            self.set_attr(
                **ret_s
            )

        return y_t, y_s


class SynoDecoder(nn.Module):
    def __init__(
        self,
        encoder,
        num_synos,
        num_frames,
        ksize_s,
        ksize_t,
        t_attrs,
        s_k_attr,
        s_v_attr,
        op_mode,
        store_attrs=[],
        attn_record=False
    ):
        super().__init__()
        d_model = encoder.transformer.width
        n_head = encoder.transformer.heads
        n_patch = int((encoder.patch_num)**0.5)

        self.encoder = get_module(encoder)

        self.decoder_layers = nn.ModuleList([
            SynoBlock(
                n_synos=num_synos,
                d_model=d_model,
                n_head=n_head,
                n_patch=n_patch,
                n_frames=num_frames,
                ksize_t=ksize_t,
                ksize_s=ksize_s,
                t_attrs=t_attrs,
                s_k_attr=s_k_attr,
                s_v_attr=s_v_attr,
                op_mode=op_mode,
                store_attrs=store_attrs,
                attn_record=attn_record
            )
            for _ in range(encoder.transformer.layers)
        ])
        self.op_mode = op_mode
        self.feat_t_dim = n_patch**2
        self.feat_s_dim = d_model

    @property
    def spatial_dim(self):
        return self.feat_s_dim

    @property
    def temporal_dim(self):
        return self.feat_t_dim

    def forward(self, x):
        b = x.shape[0]
        layer_output = dict(
            y_t=[],
            y_s=[]
        )

        # first, we prepare the encoder before the transformer layers.
        x = self.encoder()._prepare(x)

        # now, we alternate between synoptic and encoder layers
        for enc_blk, dec_blk in zip(
            self.encoder().transformer.resblocks,
            self.decoder_layers
        ):
            data = enc_blk(x)
            x = data["emb"]
            y_t, y_s = dec_blk(data)
            layer_output["y_t"].append(y_t)
            layer_output["y_s"].append(y_s)

        # last, we are done with the encoder, therefore skipping the _finalize step.
        # x =  self.encoder()._finalize(x)

        # aggregate the layer outputs
        y_s = sum(layer_output["y_s"])
        y_t = sum(layer_output["y_t"])

        return y_s, y_t


class SynoVideoAttrExtractor(VideoAttrExtractor):
    def __init__(
        self,
        # VideoAttrExtractor
        architecture,
        text_embed,
        pretrain=None,
        store_attrs=[],
        attn_record=False,
        # synoptic
        ksize_t=3,
        ksize_s=3,
        num_synos=1,
        num_frames=1,
        s_k_attr="k",
        s_v_attr="v",
        op_mode=(OpMode.S | OpMode.T),
        t_attrs=["q", "k", "v"],
    ):
        super(SynoVideoAttrExtractor, self).__init__(
            architecture=architecture,
            text_embed=text_embed,
            store_attrs=store_attrs,
            attn_record=attn_record,
            pretrain=pretrain
        )
        self.decoder = SynoDecoder(
            encoder=self.model,
            num_synos=num_synos,
            num_frames=num_frames,
            t_attrs=t_attrs,
            ksize_t=ksize_t,
            ksize_s=ksize_s,
            s_k_attr=s_k_attr,
            s_v_attr=s_v_attr,
            op_mode=op_mode,
            store_attrs=store_attrs,
            attn_record=attn_record
        )

    @property
    def spatial_dim(self):
        return self.decoder.spatial_dim

    @property
    def temporal_dim(self):
        return self.decoder.temporal_dim

    def forward(self, x):
        syno_s, syno_t = self.decoder(x=x)

        layer_attrs = []
        for enc_blk, dec_blk in zip(self.model.transformer.resblocks, self.decoder.decoder_layers):
            layer_attrs.append(
                {
                    **enc_blk.pop_attr(),
                    **dec_blk.pop_attr()
                }
            )

        return dict(
            layer_attrs=layer_attrs,
            syno_s=syno_s,
            syno_t=syno_t
        )

    def train(self, mode=True):
        super().train(mode)
        if (mode):
            self.model.eval()
            self.decoder.train()
        return self


class BinaryLinearClassifier(nn.Module):
    def __init__(
        self,
        *args,
        op_mode,
        **kargs,

    ):
        super().__init__()
        self.encoder = SynoVideoAttrExtractor(
            *args,
            **kargs,
            op_mode=op_mode
        )
        self.op_mode = op_mode
        if (OpMode.S in self.op_mode):
            self.s_ln = nn.LayerNorm(self.encoder.spatial_dim)
            self.s_head = nn.Linear(self.encoder.spatial_dim, 2)

        if (OpMode.T in self.op_mode):
            self.t_ln = nn.LayerNorm(self.encoder.temporal_dim)
            self.t_head = nn.Linear(self.encoder.temporal_dim, 2)

        if (self.op_mode == (OpMode.T | OpMode.S)):
            self.a_head = nn.Linear(
                self.encoder.temporal_dim + self.encoder.spatial_dim,
                2
            )

    @property
    def transform(self):
        return self.encoder.transform

    @property
    def n_px(self):
        return self.encoder.model.input_resolution

    def forward(self, x, *args, **kargs):
        results = self.encoder(x)
        logits_ = dict()
        if (OpMode.S in self.op_mode):
            _s = self.s_ln(results["syno_s"])
            logits_s = self.s_head(_s)
            logits_["logits_s"] = logits_s
            logits_["logits"] = logits_s

        if (OpMode.T in self.op_mode):
            _t = self.t_ln(results["syno_t"])
            logits_t = self.t_head(_t)
            logits = logits_t
            logits_["logits_t"] = logits_t
            logits_["logits"] = logits_t

        if (self.op_mode == (OpMode.S | OpMode.T)):
            logits_a = self.a_head(torch.cat([_s, _t], dim=-1))
            logits = torch.log(
                (
                    logits_s.softmax(dim=-1) +
                    logits_t.softmax(dim=-1) +
                    logits_a.softmax(dim=-1)
                ) / 3 + 1e-4
            )
            logits_["logits_a"] = logits_a
            logits_["logits"] = logits

        return dict(
            **logits_,
            ** results
        )


class SynoVideoLearner(ODBinaryMetricClassifier):
    def __init__(
        self,
        num_synos: int = 1,
        num_frames: int = 1,
        ksize_s: int = 3,
        ksize_t: int = 3,
        t_attrs: List[str] = ["q", "k", "v"],
        s_k_attr: str = "k",
        s_v_attr: str = "v",
        op_mode: List[str] = ["S", "T"],
        architecture: str = 'ViT-B/16',
        text_embed: bool = False,
        pretrain: str = None,

        attn_record: bool = False,

        store_attrs: List[str] = [],
        is_focal_loss: bool = True,

        cls_weight: float = 10.0,
        label_weights: List[float] = [1, 1],
    ):
        super().__init__()
        self.save_hyperparameters()
        op_mode = reduce(or_, [OpMode[m] for m in op_mode])
        params = dict(
            architecture=architecture,
            text_embed=text_embed,
            attn_record=attn_record,
            pretrain=pretrain,
            num_synos=num_synos,
            num_frames=num_frames,
            ksize_s=ksize_s,
            ksize_t=ksize_t,
            store_attrs=store_attrs,
            t_attrs=t_attrs,
            s_k_attr=s_k_attr,
            s_v_attr=s_v_attr,
            op_mode=op_mode
        )
        self.model = BinaryLinearClassifier(**params)

        self.label_weights = torch.tensor(label_weights)
        self.cls_weight = cls_weight
        self.is_focal_loss = is_focal_loss

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
            logits = [output[k] for k in output if "logits_" in k]
            y = y.repeat(len(logits))
            logits = torch.cat(logits, dim=0)
            if self.is_focal_loss:
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
            else:
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
                weight=None
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


class FFGSynoVideoLearner(SynoVideoLearner):
    def __init__(
        self,
        # ffg
        face_feature_path: str,
        face_parts: List[str] = [
            "lips",
            "skin",
            "eyes",
            "nose"
        ],
        face_attn_attr: str = "k",
        syno_attn_attr: str = "s_q",
        ffg_temper: float = 30,
        ffg_weight: float = 1.5,
        ffg_layers: int = -1,
        ffg_reverse: bool = False,

        # generic
        architecture: str = 'ViT-B/16',
        text_embed: bool = False,
        pretrain: str = None,

        num_frames: int = 1,
        ksize_s: int = 3,
        ksize_t: int = 3,
        t_attrs: List[str] = ["q", "k", "v"],
        s_k_attr: str = "k",
        s_v_attr: str = "v",
        op_mode: List[str] = ["S", "T"],
        attn_record: bool = False,

        store_attrs: List[str] = [],
        is_focal_loss: bool = True,

        cls_weight: float = 10.0,
        label_weights: List[float] = [1, 1],
    ):
        assert 'S' in op_mode, "FFG must include the spatial branch for operation."

        self.num_face_parts = len(face_parts)
        self.face_attn_attr = face_attn_attr
        self.syno_attn_attr = syno_attn_attr
        self.ffg_temper = ffg_temper
        self.ffg_weight = ffg_weight
        self.ffg_layers = ffg_layers
        self.ffg_reverse = ffg_reverse

        super().__init__(
            num_frames=num_frames,
            num_synos=self.num_face_parts,
            ksize_s=ksize_s,
            ksize_t=ksize_t,
            op_mode=op_mode,
            t_attrs=t_attrs,
            s_k_attr=s_k_attr,
            s_v_attr=s_v_attr,
            architecture=architecture,
            text_embed=text_embed,
            attn_record=attn_record,
            pretrain=pretrain,
            store_attrs=set([*store_attrs, self.syno_attn_attr]),
            cls_weight=cls_weight,
            label_weights=label_weights,
            is_focal_loss=is_focal_loss
        )

        self.save_hyperparameters()

        with open(face_feature_path, "rb") as f:
            _face_features = pickle.load(f)
            self.face_features = torch.stack(
                [
                    torch.stack([
                        _face_features[self.face_attn_attr][p][l]
                        for p in face_parts
                    ])
                    for l in range(self.model.encoder.model.transformer.layers)
                ]
            )
            self.face_features = self.face_features.unsqueeze(1)

        for i, dec_blk in enumerate(self.model.encoder.decoder.decoder_layers):
            dec_blk.syno_embedding.data = self.face_features[i].squeeze(0).data.clone()

    def shared_step(self, batch, stage):
        result = super().shared_step(batch, stage)

        if (stage == "train"):
            dts_name = result["dts_name"]
            x = batch["xyz"][0]

            # face feature guided loss
            target_attn_attrs = torch.stack(
                [
                    attrs[self.syno_attn_attr]
                    for attrs in result["output"]["layer_attrs"]
                ]
            )  # qs.shape = [layer,b,t,syno,patch,head]

            if (len(target_attn_attrs.shape) == 4):
                # shape = [l, b, synos, head*width]
                # for: out, emb
                pass
            elif (len(target_attn_attrs.shape) == 5):
                # shape = [l, b, synos, head,width]
                # for: q, k, v
                target_attn_attrs = target_attn_attrs.flatten(-2)
            elif (len(target_attn_attrs.shape) == 6):
                # shape = [l, b, t, synos, head, width]
                # for: q, k, v
                target_attn_attrs = target_attn_attrs.mean(2).flatten(-2)
            else:
                raise NotImplementedError()

            face_features = self.face_features.to(
                dtype=target_attn_attrs.dtype,
                device=target_attn_attrs.device
            )

            if self.ffg_layers == -1:
                pass
            elif self.ffg_layers > 0:
                if (self.ffg_reverse):
                    face_features = face_features[: self.ffg_layers]
                    target_attn_attrs = target_attn_attrs[: self.ffg_layers]
                else:
                    layers = (
                        self.model.encoder.model.transformer.layers -
                        self.ffg_layers
                    )
                    face_features = face_features[layers:]
                    target_attn_attrs = target_attn_attrs[layers:]
            else:
                raise NotImplementedError()

            l, b, q = target_attn_attrs.shape[:3]
            face_features = face_features / face_features.norm(dim=-1, keepdim=True)
            target_attn_attrs = target_attn_attrs / target_attn_attrs.norm(dim=-1, keepdim=True)

            logits = self.ffg_temper * (target_attn_attrs @ face_features.transpose(-1, -2))

            cls_sim = torch.nn.functional.cross_entropy(
                logits.flatten(0, 2),
                (
                    torch.arange(
                        0,
                        self.num_face_parts
                    )
                    .repeat((l * b))
                    .to(x.device)
                ),
                reduction="none"
            ).mean()

            self.log(
                f"{stage}/{dts_name}/syno_sim",
                cls_sim,
                batch_size=x.shape[0]
            )
            result["loss"] += cls_sim * self.ffg_weight

        return result


class FFESynoVideoLearner(SynoVideoLearner):
    def __init__(
        self,
        # ffg
        face_feature_path: str,
        face_parts: List[str] = [
            "lips",
            "skin",
            "eyes",
            "nose"
        ],
        face_attn_attr: str = "k",
        syno_attn_attr: str = "s_q",
        ffg_temper: float = 30,
        ffg_weight: float = 1.5,
        ffg_layers: int = -1,
        ffg_reverse: bool = False,
        ffe_weight: float = 100,
        grad_temper: float = 100,

        # generic
        architecture: str = 'ViT-B/16',
        text_embed: bool = False,
        pretrain: str = None,

        num_frames: int = 1,
        ksize_s: int = 3,
        ksize_t: int = 3,
        t_attrs: List[str] = ["q", "k", "v"],
        s_k_attr: str = "k",
        s_v_attr: str = "v",
        op_mode: List[str] = ["S", "T"],
        attn_record: bool = True,

        store_attrs: List[str] = [],
        is_focal_loss: bool = True,

        cls_weight: float = 10.0,
        label_weights: List[float] = [1, 1],
    ):
        assert 'S' in op_mode, "FFG must include the spatial branch for operation."

        self.num_face_parts = len(face_parts)
        self.face_attn_attr = face_attn_attr
        self.syno_attn_attr = syno_attn_attr
        self.ffg_temper = ffg_temper
        self.ffg_weight = ffg_weight
        self.ffg_layers = ffg_layers
        self.ffg_reverse = ffg_reverse
        self.ffe_weight = ffe_weight
        self.grad_temper = grad_temper

        super().__init__(
            num_frames=num_frames,
            num_synos=self.num_face_parts,
            ksize_s=ksize_s,
            ksize_t=ksize_t,
            op_mode=op_mode,
            t_attrs=t_attrs,
            s_k_attr=s_k_attr,
            s_v_attr=s_v_attr,
            architecture=architecture,
            text_embed=text_embed,
            attn_record=attn_record,
            pretrain=pretrain,
            store_attrs=set([*store_attrs, self.syno_attn_attr]),
            cls_weight=cls_weight,
            label_weights=label_weights,
            is_focal_loss=is_focal_loss
        )

        self.save_hyperparameters()

        with open(face_feature_path, "rb") as f:
            _face_features = pickle.load(f)
            self.face_features = torch.stack(
                [
                    torch.stack([
                        _face_features[self.face_attn_attr][p][l]
                        for p in face_parts
                    ])
                    for l in range(self.model.encoder.model.transformer.layers)
                ]
            )
            self.face_features = self.face_features.unsqueeze(1)

        for i, dec_blk in enumerate(self.model.encoder.decoder.decoder_layers):
            dec_blk.syno_embedding.data = self.face_features[i].squeeze(0).data.clone()

    def shared_step(self, batch, stage):
        result = super().shared_step(batch, stage)

        if (stage == "train"):
            dts_name = result["dts_name"]
            x = batch["xyz"][0]

            # face feature guided loss
            target_attn_attrs = torch.stack(
                [
                    attrs[self.syno_attn_attr]
                    for attrs in result["output"]["layer_attrs"]
                ]
            )  # qs.shape = [layer,b,t,syno,patch,head]

            if (len(target_attn_attrs.shape) == 4):
                # shape = [l, b, synos, head*width]
                # for: out, emb
                pass
            elif (len(target_attn_attrs.shape) == 5):
                # shape = [l, b, synos, head,width]
                # for: q, k, v
                target_attn_attrs = target_attn_attrs.flatten(-2)
            elif (len(target_attn_attrs.shape) == 6):
                # shape = [l, b, t, synos, head, width]
                # for: q, k, v
                target_attn_attrs = target_attn_attrs.mean(2).flatten(-2)
            else:
                raise NotImplementedError()

            face_features = self.face_features.to(
                dtype=target_attn_attrs.dtype,
                device=target_attn_attrs.device
            )

            if self.ffg_layers == -1:
                pass
            elif self.ffg_layers > 0:
                if (self.ffg_reverse):
                    face_features = face_features[: self.ffg_layers]
                    target_attn_attrs = target_attn_attrs[: self.ffg_layers]
                else:
                    layers = (
                        self.model.encoder.model.transformer.layers -
                        self.ffg_layers
                    )
                    face_features = face_features[layers:]
                    target_attn_attrs = target_attn_attrs[layers:]
            else:
                raise NotImplementedError()

            l, b, q = target_attn_attrs.shape[:3]
            face_features = face_features / face_features.norm(dim=-1, keepdim=True)
            target_attn_attrs = target_attn_attrs / target_attn_attrs.norm(dim=-1, keepdim=True)

            logits = self.ffg_temper * (target_attn_attrs @ face_features.transpose(-1, -2))

            cls_sim = torch.nn.functional.cross_entropy(
                logits.flatten(0, 2),
                (
                    torch.arange(
                        0,
                        self.num_face_parts
                    )
                    .repeat((l * b))
                    .to(x.device)
                ),
                reduction="none"
            ).mean()

            self.log(
                f"{stage}/{dts_name}/syno_sim",
                cls_sim,
                batch_size=x.shape[0]
            )

            result["loss"] += cls_sim * self.ffg_weight

            grad_raw_grid = (
                torch.stack(
                    torch.autograd.grad(
                        result["loss"],
                        [layer.aff for layer in self.model.encoder.decoder.decoder_layers],
                        grad_outputs=torch.ones_like(result["loss"]),
                        retain_graph=True,
                        create_graph=True
                    )
                ).abs() * self.grad_temper
            )

            attn_grid = (
                torch.stack(
                    [layer.aff for layer in self.model.encoder.decoder.decoder_layers]
                )
            ).detach()

            grad_align = torch.nn.functional.kl_div(
                torch.nn.functional.log_softmax(grad_raw_grid.flatten(3), dim=-1),
                attn_grid.flatten(3),
            )

            self.log(
                f"{stage}/{dts_name}/grad_align",
                grad_align,
                batch_size=x.shape[0]
            )

            result["loss"] += grad_align * self.ffe_weight

        return result


if __name__ == "__main__":

    frames = 5
    # # AttrExtractor Test
    model = FFGSynoVideoLearner(
        face_feature_path="misc/L14_real_semantic_patches_v2_2000.pickle",
        architecture="ViT-L/14",
        num_frames=frames
    )
    model.to("cuda")
    results = model(torch.randn(9, frames, 3, 224, 224).to("cuda"))
    synos = results["logits"]
    synos.sum().backward()

    # model = GlitchBlock(n_head=12, n_patch=14, n_filt=10, ksize=3, n_frames=frames)
    # model.to("cuda")
    # logits = model({
    #     "q": torch.randn(1, frames, 197, 12, 64).to("cuda"),
    #     "k": torch.randn(1, frames, 197, 12, 64).to("cuda")
    # })
    print("done")
