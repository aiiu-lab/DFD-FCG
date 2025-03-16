import torch
from torch import nn

from collections import OrderedDict
import numpy as np
from typing import Tuple, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

###
from src.model.base import ODBinaryMetricClassifier
from src.model.clip import VideoAttrExtractor
from src.utility.loss import focal_loss

'''
QuickGELU and LayerNorm w/ fp16 from official CLIP repo
(https://github.com/openai/CLIP/blob/3b473b0e682c091a9e53623eebc1ca1657385717/clip/model.py)
'''


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class Attention(nn.Module):
    '''
    A generalized attention module with more flexibility.
    '''

    def __init__(
        self, q_in_dim: int, k_in_dim: int, v_in_dim: int,
        qk_proj_dim: int, v_proj_dim: int, num_heads: int, out_dim: int,
        return_all_features: bool = False,
    ):
        super().__init__()

        self.q_proj = nn.Linear(q_in_dim, qk_proj_dim)
        self.k_proj = nn.Linear(k_in_dim, qk_proj_dim)
        self.v_proj = nn.Linear(v_in_dim, v_proj_dim)
        self.out_proj = nn.Linear(v_proj_dim, out_dim)

        self.num_heads = num_heads
        self.return_all_features = return_all_features
        assert qk_proj_dim % num_heads == 0 and v_proj_dim % num_heads == 0

        self._initialize_weights()

    def _initialize_weights(self):
        for m in (self.q_proj, self.k_proj, self.v_proj, self.out_proj):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        assert q.ndim == 3 and k.ndim == 3 and v.ndim == 3
        N = q.size(0)
        assert k.size(0) == N and v.size(0) == N
        Lq, Lkv = q.size(1), k.size(1)
        assert v.size(1) == Lkv

        q, k, v = self.q_proj(q), self.k_proj(k), self.v_proj(v)

        H = self.num_heads
        Cqk, Cv = q.size(-1) // H, v.size(-1) // H

        q = q.view(N, Lq, H, Cqk)
        k = k.view(N, Lkv, H, Cqk)
        v = v.view(N, Lkv, H, Cv)

        aff = torch.einsum('nqhc,nkhc->nqkh', q / (Cqk ** 0.5), k)
        aff = aff.softmax(dim=-2)
        mix = torch.einsum('nqlh,nlhc->nqhc', aff, v)

        out = self.out_proj(mix.flatten(-2))

        if self.return_all_features:
            return dict(q=q, k=k, v=v, aff=aff, out=out)
        else:
            return out


class PatchEmbed2D(nn.Module):

    def __init__(
        self,
        patch_size: Tuple[int, int] = (16, 16),
        in_channels: int = 3,
        embed_dim: int = 768,
    ):
        super().__init__()

        self.patch_size = patch_size
        self.in_channels = in_channels

        self.proj = nn.Linear(np.prod(patch_size) * in_channels, embed_dim)

    def _initialize_weights(self, x):
        nn.init.kaiming_normal_(self.proj.weight, 0.)
        nn.init.constant_(self.proj.bias, 0.)

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.size()
        pH, pW = self.patch_size

        assert C == self.in_channels and H % pH == 0 and W % pW == 0

        x = x.view(B, C, H // pH, pH, W // pW, pW).permute(0, 2, 4, 1, 3, 5).flatten(3).flatten(1, 2)
        x = self.proj(x)

        return x


class TransformerEncoderLayer(nn.Module):

    def __init__(
        self,
        in_feature_dim: int = 768,
        qkv_dim: int = 768,
        num_heads: int = 12,
        mlp_factor: float = 4.0,
        mlp_dropout: float = 0.0,
        act: nn.Module = QuickGELU,
        return_all_features: bool = False,
    ):
        super().__init__()

        self.return_all_features = return_all_features

        self.attn = Attention(
            q_in_dim=in_feature_dim, k_in_dim=in_feature_dim, v_in_dim=in_feature_dim,
            qk_proj_dim=qkv_dim, v_proj_dim=qkv_dim, num_heads=num_heads, out_dim=in_feature_dim,
            return_all_features=return_all_features,
        )

        mlp_dim = round(mlp_factor * in_feature_dim)
        self.mlp = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(in_feature_dim, mlp_dim)),
            ('act', act()),
            ('dropout', nn.Dropout(mlp_dropout)),
            ('fc2', nn.Linear(mlp_dim, in_feature_dim)),
        ]))

        self.norm1 = LayerNorm(in_feature_dim)
        self.norm2 = LayerNorm(in_feature_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in (self.mlp[0], self.mlp[-1]):
            nn.init.xavier_uniform_(m.weight)
            nn.init.normal_(m.bias, std=1e-6)

    def forward(self, x: torch.Tensor):
        if self.return_all_features:
            ret_dict = {}

            x_norm = self.norm1(x)
            attn_out = self.attn(x_norm, x_norm, x_norm)
            ret_dict['q'] = attn_out['q']
            ret_dict['k'] = attn_out['k']
            ret_dict['v'] = attn_out['v']
            ret_dict['attn_out'] = attn_out['out']
            x = x + attn_out['out']

            x = x + self.mlp(self.norm2(x))
            ret_dict['out'] = x

            return ret_dict

        else:
            x_norm = self.norm1(x)
            x = x + self.attn(x_norm, x_norm, x_norm)
            x = x + self.mlp(self.norm2(x))

            return x


class TransformerDecoderLayer(nn.Module):

    def __init__(
        self,
        in_feature_dim: int = 768,
        qkv_dim: int = 768,
        num_heads: int = 12,
        mlp_factor: float = 4.0,
        mlp_dropout: float = 0.0,
        act: nn.Module = QuickGELU,
    ):
        super().__init__()

        self.attn = Attention(
            q_in_dim=in_feature_dim, k_in_dim=in_feature_dim, v_in_dim=in_feature_dim,
            qk_proj_dim=qkv_dim, v_proj_dim=qkv_dim, num_heads=num_heads, out_dim=in_feature_dim,
        )

        mlp_dim = round(mlp_factor * in_feature_dim)
        self.mlp = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(in_feature_dim, mlp_dim)),
            ('act', act()),
            ('dropout', nn.Dropout(mlp_dropout)),
            ('fc2', nn.Linear(mlp_dim, in_feature_dim)),
        ]))

        self.norm1 = LayerNorm(in_feature_dim)
        self.norm2 = LayerNorm(in_feature_dim)
        self.norm3 = LayerNorm(in_feature_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in (self.mlp[0], self.mlp[-1]):
            nn.init.xavier_uniform_(m.weight)
            nn.init.normal_(m.bias, std=1e-6)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        y_norm = self.norm3(y)
        x = x + self.attn(self.norm1(x), y_norm, y_norm)
        x = x + self.mlp(self.norm2(x))

        return x


class VisionTransformer2D(nn.Module):

    def __init__(
        self,
        feature_dim: int = 768,
        input_size: Tuple[int, int] = (224, 224),
        patch_size: Tuple[int, int] = (16, 16),
        num_heads: int = 12,
        num_layers: int = 12,
        mlp_factor: float = 4.0,
        act: nn.Module = QuickGELU,
        return_all_features: bool = False,
        ln_pre: bool = False,
    ):
        super().__init__()

        self.return_all_features = return_all_features

        self.patch_embed = PatchEmbed2D(patch_size=patch_size, embed_dim=feature_dim)
        self.num_patches = np.prod([x // y for x, y in zip(input_size, patch_size)]) + 1

        self.cls_token = nn.Parameter(torch.zeros([feature_dim]))
        self.pos_embed = nn.Parameter(torch.zeros([self.num_patches, feature_dim]))

        self.blocks = nn.ModuleList([
            TransformerEncoderLayer(
                in_feature_dim=feature_dim, qkv_dim=feature_dim, num_heads=num_heads, mlp_factor=mlp_factor, act=act,
                return_all_features=return_all_features,
            ) for _ in range(num_layers)
        ])

        if ln_pre:
            self.ln_pre = LayerNorm(feature_dim)
        else:
            self.ln_pre = nn.Identity()

        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor):
        dtype = self.patch_embed.proj.weight.dtype
        x = x.to(dtype)

        x = self.patch_embed(x)
        x = torch.cat([self.cls_token.view(1, 1, -1).repeat(x.size(0), 1, 1), x], dim=1)
        x = x + self.pos_embed

        x = self.ln_pre(x)

        if self.return_all_features:
            all_features = []
            for blk in self.blocks:
                x = blk(x)
                all_features.append(x)
                x = x['out']
            return all_features

        else:
            for blk in self.blocks:
                x = blk(x)
            return x


vit_presets = {
    'ViT-B/16-lnpre': dict(
        feature_dim=768,
        input_size=(224, 224),
        patch_size=(16, 16),
        num_heads=12,
        num_layers=12,
        mlp_factor=4.0,
        ln_pre=True,
    ),
    'ViT-L/14-lnpre': dict(
        feature_dim=1024,
        input_size=(224, 224),
        patch_size=(14, 14),
        num_heads=16,
        num_layers=24,
        mlp_factor=4.0,
        ln_pre=True,
    ),
}


class TransformerDecoderLayer(nn.Module):

    def __init__(
        self,
        in_feature_dim: int = 768,
        qkv_dim: int = 768,
        num_heads: int = 12,
        mlp_factor: float = 4.0,
        mlp_dropout: float = 0.0,
        act: nn.Module = QuickGELU,
    ):
        super().__init__()

        self.attn = Attention(
            q_in_dim=in_feature_dim, k_in_dim=in_feature_dim, v_in_dim=in_feature_dim,
            qk_proj_dim=qkv_dim, v_proj_dim=qkv_dim, num_heads=num_heads, out_dim=in_feature_dim,
        )

        mlp_dim = round(mlp_factor * in_feature_dim)
        self.mlp = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(in_feature_dim, mlp_dim)),
            ('act', act()),
            ('dropout', nn.Dropout(mlp_dropout)),
            ('fc2', nn.Linear(mlp_dim, in_feature_dim)),
        ]))

        self.norm1 = LayerNorm(in_feature_dim)
        self.norm2 = LayerNorm(in_feature_dim)
        self.norm3 = LayerNorm(in_feature_dim)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in (self.mlp[0], self.mlp[-1]):
            nn.init.xavier_uniform_(m.weight)
            nn.init.normal_(m.bias, std=1e-6)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        y_norm = self.norm3(y)
        x = x + self.attn(self.norm1(x), y_norm, y_norm)
        x = x + self.mlp(self.norm2(x))

        return x


class TemporalCrossAttention(nn.Module):

    def __init__(
        self,
        spatial_size: Tuple[int, int] = (14, 14),
        feature_dim: int = 768,
    ):
        super().__init__()

        self.spatial_size = spatial_size

        w_size = np.prod([x * 2 - 1 for x in spatial_size])
        self.w1 = nn.Parameter(torch.zeros([w_size, feature_dim]))
        self.w2 = nn.Parameter(torch.zeros([w_size, feature_dim]))

        idx_tensor = torch.zeros([np.prod(spatial_size) for _ in (0, 1)], dtype=torch.long)
        for q in range(np.prod(spatial_size)):
            qi, qj = q // spatial_size[1], q % spatial_size[1]
            for k in range(np.prod(spatial_size)):
                ki, kj = k // spatial_size[1], k % spatial_size[1]
                i_offs = qi - ki + spatial_size[0] - 1
                j_offs = qj - kj + spatial_size[1] - 1
                idx_tensor[q, k] = i_offs * (spatial_size[1] * 2 - 1) + j_offs
        self.idx_tensor = idx_tensor

    def forward_half(self, q: torch.Tensor, k: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        q, k = q[:, :, 1:], k[:, :, 1:]  # remove cls token

        assert q.size() == k.size()
        assert q.size(2) == np.prod(self.spatial_size)

        attn = torch.einsum('ntqhd,ntkhd->ntqkh', q / (q.size(-1) ** 0.5), k)
        attn = attn.softmax(dim=-2).mean(dim=-1)  # L, L, N, T

        self.idx_tensor = self.idx_tensor.to(w.device)
        w_unroll = w[self.idx_tensor]  # L, L, C
        ret = torch.einsum('ntqk,qkc->ntqc', attn, w_unroll)

        return ret

    def forward(self, q: torch.Tensor, k: torch.Tensor):
        N, T, L, H, D = q.size()
        assert L == np.prod(self.spatial_size) + 1

        ret = torch.zeros([N, T, L, self.w1.size(-1)], device='cuda')
        ret[:, 1:, 1:, :] += self.forward_half(q[:, 1:, :, :, :], k[:, :-1, :, :, :], self.w1)
        ret[:, :-1, 1:, :] += self.forward_half(q[:, :-1, :, :, :], k[:, 1:, :, :, :], self.w2)

        return ret


class EVLDecoder(nn.Module):

    def __init__(
        self,
        num_frames: int = 8,
        spatial_size: Tuple[int, int] = (14, 14),
        num_layers: int = 4,
        in_feature_dim: int = 768,
        qkv_dim: int = 768,
        num_heads: int = 12,
        mlp_factor: float = 4.0,
        enable_temporal_conv: bool = True,
        enable_temporal_pos_embed: bool = True,
        enable_temporal_cross_attention: bool = True,
        mlp_dropout: float = 0.5,
    ):
        super().__init__()

        self.enable_temporal_conv = enable_temporal_conv
        self.enable_temporal_pos_embed = enable_temporal_pos_embed
        self.enable_temporal_cross_attention = enable_temporal_cross_attention
        self.num_layers = num_layers

        self.decoder_layers = nn.ModuleList(
            [TransformerDecoderLayer(in_feature_dim, qkv_dim, num_heads, mlp_factor, mlp_dropout)
             for _ in range(num_layers)]
        )

        if enable_temporal_conv:
            self.temporal_conv = nn.ModuleList(
                [nn.Conv1d(in_feature_dim, in_feature_dim, kernel_size=3, stride=1,
                           padding=1, groups=in_feature_dim) for _ in range(num_layers)]
            )
        if enable_temporal_pos_embed:
            self.temporal_pos_embed = nn.ParameterList(
                [nn.Parameter(torch.zeros([num_frames, in_feature_dim])) for _ in range(num_layers)]
            )
        if enable_temporal_cross_attention:
            self.cross_attention = nn.ModuleList(
                [TemporalCrossAttention(spatial_size, in_feature_dim) for _ in range(num_layers)]
            )

        self.cls_token = nn.Parameter(torch.zeros([in_feature_dim]))

    def _initialize_weights(self):
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, in_features: List[Dict[str, torch.Tensor]]):
        N, T, L, C = in_features[0]['out'].size()
        assert len(in_features) == self.num_layers
        x = self.cls_token.view(1, 1, -1).repeat(N, 1, 1)

        for i in range(self.num_layers):
            frame_features = in_features[i]['out']

            if self.enable_temporal_conv:
                feat = in_features[i]['out']
                feat = feat.permute(0, 2, 3, 1).contiguous().flatten(0, 1)  # N * L, C, T
                feat = self.temporal_conv[i](feat)
                feat = feat.view(N, L, C, T).permute(0, 3, 1, 2).contiguous()  # N, T, L, C
                frame_features += feat

            if self.enable_temporal_pos_embed:
                frame_features += self.temporal_pos_embed[i].view(1, T, 1, C)

            if self.enable_temporal_cross_attention:
                frame_features += self.cross_attention[i](in_features[i]['q'], in_features[i]['k'])

            frame_features = frame_features.flatten(1, 2)  # N, T * L, C

            x = self.decoder_layers[i](x, frame_features)

        return x.squeeze(1)


class EVLVideoAttrExtractor(VideoAttrExtractor):

    def __init__(
        self,
        architecture,
        num_frames=1,
        decoder_layers=4,
        attn_record=False
    ):
        super(EVLVideoAttrExtractor, self).__init__(
            architecture=architecture,
            text_embed=False,
            store_attrs=["q", "k", "out"],
            attn_record=attn_record
        )

        self.decoder = EVLDecoder(
            num_frames=num_frames,
            spatial_size=(self.n_patch, self.n_patch),
            num_layers=decoder_layers,
            in_feature_dim=self.embed_dim,
            qkv_dim=self.embed_dim,
            num_heads=self.n_heads,
            mlp_factor=4.0,
            enable_temporal_conv=True,
            enable_temporal_pos_embed=True,
            enable_temporal_cross_attention=True,
            mlp_dropout=0.5
        )
        self.decoder_layers = decoder_layers

    def forward(self, x):
        results = super(EVLVideoAttrExtractor, self).forward(x)

        # remove cls_token attrs
        # layer_attrs = results["layer_attrs"]
        # for i in range(len(layer_attrs)):
        #     for k in layer_attrs[i]:
        #         layer_attrs[i][k] = layer_attrs[i][k][:, :, 1:]

        reps = self.decoder(results["layer_attrs"][-self.decoder_layers:])
        results["reps"] = reps
        return results

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
        self.encoder = EVLVideoAttrExtractor(
            *args,
            **kargs
        )
        self.head = self.make_linear(self.encoder.embed_dim)

    def make_linear(self, embed_dim):
        linear = nn.Linear(
            embed_dim,
            2
        )
        return nn.Sequential(
            nn.LayerNorm(embed_dim),
            linear
        )

    @property
    def transform(self):
        return self.encoder.transform

    @property
    def n_px(self):
        return self.encoder.model.input_resolution

    def forward(self, x, *args, **kargs):
        results = self.encoder(x)
        logits = self.head(results["reps"])
        return dict(
            logits=logits,
            ** results
        )


class EfficientVideoLearner(ODBinaryMetricClassifier):
    def __init__(
        self,
        architecture: str = 'ViT-B/16',
        num_frames: int = 1,
        attn_record: bool = False,
        decoder_layers: int = 4,

        is_focal_loss: bool = True,

        cls_weight: float = 10.0,
        label_weights: List[float] = [1, 1],
    ):
        super().__init__()
        self.save_hyperparameters()
        params = dict(
            architecture=architecture,
            attn_record=attn_record,
            num_frames=num_frames,
            decoder_layers=decoder_layers
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


if __name__ == "__main__":
    frames = 5
    model = EfficientVideoLearner(num_frames=frames, attn_record=True)
    model.to("cuda")
    logit = model(torch.randn(9, frames, 3, 224, 224).to("cuda"))["logits"]
    logit.sum().backward()
    print([k for k, v in model.named_parameters() if v.requires_grad])
    print("done")
