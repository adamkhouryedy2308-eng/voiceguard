"""SSL-AASIST: XLS-R front end + AASIST graph-attention back end.

Ported from https://github.com/TakHemlata/SSL_Anti-spoofing (MIT licence,
Copyright (c) 2022 Hemlata Tak), which builds on AASIST (Jung et al., ICASSP 2022).
The original loads XLS-R through the old `fairseq` library; here XLS-R comes from
Hugging Face `transformers`, and `load_original_checkpoint` renames the authors'
fairseq weights to match. The back end is kept as in the original, including a
quirk in Residual_block (bn1/selu result unused), so the trained weights behave
exactly as they did for the authors.
"""

from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Wav2Vec2Config, Wav2Vec2Model

XLSR_NAME = "facebook/wav2vec2-xls-r-300m"


# --------------------------------------------------------------------------
# Back end (AASIST), as in the original model.py
# --------------------------------------------------------------------------


def _new_param(*size):
    out = nn.Parameter(torch.FloatTensor(*size))
    nn.init.xavier_normal_(out)
    return out


def _pairwise_mul_nodes(x):
    nb_nodes = x.size(1)
    x = x.unsqueeze(2).expand(-1, -1, nb_nodes, -1)
    return x * x.transpose(1, 2)


def _apply_bn(bn, x):
    size = x.size()
    return bn(x.view(-1, size[-1])).view(size)


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, temperature=1.0):
        super().__init__()
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_weight = _new_param(out_dim, 1)
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.input_drop = nn.Dropout(p=0.2)
        self.act = nn.SELU(inplace=True)
        self.temp = temperature

    def forward(self, x):
        x = self.input_drop(x)
        att_map = torch.tanh(self.att_proj(_pairwise_mul_nodes(x)))
        att_map = torch.matmul(att_map, self.att_weight) / self.temp
        att_map = F.softmax(att_map, dim=-2)
        x = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x)) + self.proj_without_att(x)
        return self.act(_apply_bn(self.bn, x))


class HtrgGraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim, temperature=1.0):
        super().__init__()
        self.proj_type1 = nn.Linear(in_dim, in_dim)
        self.proj_type2 = nn.Linear(in_dim, in_dim)
        self.att_proj = nn.Linear(in_dim, out_dim)
        self.att_projM = nn.Linear(in_dim, out_dim)
        self.att_weight11 = _new_param(out_dim, 1)
        self.att_weight22 = _new_param(out_dim, 1)
        self.att_weight12 = _new_param(out_dim, 1)
        self.att_weightM = _new_param(out_dim, 1)
        self.proj_with_att = nn.Linear(in_dim, out_dim)
        self.proj_without_att = nn.Linear(in_dim, out_dim)
        self.proj_with_attM = nn.Linear(in_dim, out_dim)
        self.proj_without_attM = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)
        self.input_drop = nn.Dropout(p=0.2)
        self.act = nn.SELU(inplace=True)
        self.temp = temperature

    def forward(self, x1, x2, master=None):
        n1, n2 = x1.size(1), x2.size(1)
        x = torch.cat([self.proj_type1(x1), self.proj_type2(x2)], dim=1)
        if master is None:
            master = torch.mean(x, dim=1, keepdim=True)
        x = self.input_drop(x)

        # attention between nodes, with separate weights per node-type pair
        att = torch.tanh(self.att_proj(_pairwise_mul_nodes(x)))
        board = torch.zeros_like(att[:, :, :, 0]).unsqueeze(-1)
        board[:, :n1, :n1, :] = torch.matmul(att[:, :n1, :n1, :], self.att_weight11)
        board[:, n1:, n1:, :] = torch.matmul(att[:, n1:, n1:, :], self.att_weight22)
        board[:, :n1, n1:, :] = torch.matmul(att[:, :n1, n1:, :], self.att_weight12)
        board[:, n1:, :n1, :] = torch.matmul(att[:, n1:, :n1, :], self.att_weight12)
        att_map = F.softmax(board / self.temp, dim=-2)

        # update the master node
        att_m = torch.tanh(self.att_projM(x * master))
        att_m = F.softmax(torch.matmul(att_m, self.att_weightM) / self.temp, dim=-2)
        master = self.proj_with_attM(torch.matmul(att_m.squeeze(-1).unsqueeze(1), x)) + self.proj_without_attM(master)

        x = self.proj_with_att(torch.matmul(att_map.squeeze(-1), x)) + self.proj_without_att(x)
        x = self.act(_apply_bn(self.bn, x))
        return x.narrow(1, 0, n1), x.narrow(1, n1, n2), master


class GraphPool(nn.Module):
    def __init__(self, k: float, in_dim: int, p: Union[float, int]):
        super().__init__()
        self.k = k
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()

    def forward(self, h):
        scores = torch.sigmoid(self.proj(self.drop(h)))
        _, n_nodes, n_feat = h.size()
        n_keep = max(int(n_nodes * self.k), 1)
        _, idx = torch.topk(scores, n_keep, dim=1)
        return torch.gather(h * scores, 1, idx.expand(-1, -1, n_feat))


class Residual_block(nn.Module):
    def __init__(self, nb_filts, first=False):
        super().__init__()
        self.first = first
        if not first:
            self.bn1 = nn.BatchNorm2d(nb_filts[0])
        self.conv1 = nn.Conv2d(nb_filts[0], nb_filts[1], kernel_size=(2, 3), padding=(1, 1))
        self.selu = nn.SELU(inplace=True)
        self.bn2 = nn.BatchNorm2d(nb_filts[1])
        self.conv2 = nn.Conv2d(nb_filts[1], nb_filts[1], kernel_size=(2, 3), padding=(0, 1))
        self.downsample = nb_filts[0] != nb_filts[1]
        if self.downsample:
            self.conv_downsample = nn.Conv2d(nb_filts[0], nb_filts[1], kernel_size=(1, 3), padding=(0, 1))

    def forward(self, x):
        identity = x
        out = self.conv1(x)  # original quirk: conv1 takes x, not bn1(x)
        out = self.conv2(self.selu(self.bn2(out)))
        if self.downsample:
            identity = self.conv_downsample(identity)
        return out + identity


# --------------------------------------------------------------------------
# Full model
# --------------------------------------------------------------------------


class SSLAASIST(nn.Module):
    """Raw 16 kHz audio (batch, 64600) -> 2 logits: index 0 = spoof, index 1 = bonafide."""

    def __init__(self, pretrained=False):
        """pretrained=True starts XLS-R from Meta's weights (to train our own model);
        False builds an empty one, to be filled from a saved checkpoint."""
        super().__init__()
        filts = [128, [1, 32], [32, 32], [32, 64], [64, 64]]
        gat_dims = [64, 32]
        pool_ratios = [0.5, 0.5, 0.5, 0.5]
        temperatures = [2.0, 2.0, 100.0, 100.0]

        config = Wav2Vec2Config.from_pretrained(XLSR_NAME)
        config.mask_time_prob = 0.0  # original calls fairseq with mask=False
        config.layerdrop = 0.0
        if pretrained:
            self.ssl = Wav2Vec2Model.from_pretrained(XLSR_NAME, config=config)
        else:
            self.ssl = Wav2Vec2Model(config)
        self.LL = nn.Linear(1024, 128)

        self.first_bn = nn.BatchNorm2d(1)
        self.first_bn1 = nn.BatchNorm2d(64)
        self.drop = nn.Dropout(0.5, inplace=True)
        self.drop_way = nn.Dropout(0.2, inplace=True)
        self.selu = nn.SELU(inplace=True)
        self.encoder = nn.Sequential(
            nn.Sequential(Residual_block(filts[1], first=True)),
            nn.Sequential(Residual_block(filts[2])),
            nn.Sequential(Residual_block(filts[3])),
            nn.Sequential(Residual_block(filts[4])),
            nn.Sequential(Residual_block(filts[4])),
            nn.Sequential(Residual_block(filts[4])),
        )
        self.attention = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(1, 1)),
            nn.SELU(inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 64, kernel_size=(1, 1)),
        )
        self.pos_S = nn.Parameter(torch.randn(1, 42, filts[-1][-1]))
        self.master1 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        self.master2 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        self.GAT_layer_S = GraphAttentionLayer(filts[-1][-1], gat_dims[0], temperatures[0])
        self.GAT_layer_T = GraphAttentionLayer(filts[-1][-1], gat_dims[0], temperatures[1])
        self.HtrgGAT_layer_ST11 = HtrgGraphAttentionLayer(gat_dims[0], gat_dims[1], temperatures[2])
        self.HtrgGAT_layer_ST12 = HtrgGraphAttentionLayer(gat_dims[1], gat_dims[1], temperatures[2])
        self.HtrgGAT_layer_ST21 = HtrgGraphAttentionLayer(gat_dims[0], gat_dims[1], temperatures[2])
        self.HtrgGAT_layer_ST22 = HtrgGraphAttentionLayer(gat_dims[1], gat_dims[1], temperatures[2])
        self.pool_S = GraphPool(pool_ratios[0], gat_dims[0], 0.3)
        self.pool_T = GraphPool(pool_ratios[1], gat_dims[0], 0.3)
        self.pool_hS1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hS2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.out_layer = nn.Linear(5 * gat_dims[1], 2)

    def forward(self, audio):
        # XLS-R: every 20 ms slice -> 1,024 numbers (no averaging this time)
        x = self.ssl(audio).last_hidden_state  # (batch, 201, 1024)
        x = self.LL(x)  # (batch, 201, 128)

        x = x.transpose(1, 2).unsqueeze(1)  # treat it like an image: (batch, 1, 128, 201)
        x = F.max_pool2d(x, (3, 3))
        x = self.selu(self.first_bn(x))
        x = self.encoder(x)
        x = self.selu(self.first_bn1(x))
        w = self.attention(x)

        # spectral nodes and temporal nodes
        e_S = torch.sum(x * F.softmax(w, dim=-1), dim=-1).transpose(1, 2) + self.pos_S
        out_S = self.pool_S(self.GAT_layer_S(e_S))
        e_T = torch.sum(x * F.softmax(w, dim=-2), dim=-2).transpose(1, 2)
        out_T = self.pool_T(self.GAT_layer_T(e_T))

        master1 = self.master1.expand(x.size(0), -1, -1)
        master2 = self.master2.expand(x.size(0), -1, -1)

        out_T1, out_S1, master1 = self.HtrgGAT_layer_ST11(out_T, out_S, master=self.master1)
        out_S1, out_T1 = self.pool_hS1(out_S1), self.pool_hT1(out_T1)
        t_aug, s_aug, m_aug = self.HtrgGAT_layer_ST12(out_T1, out_S1, master=master1)
        out_T1, out_S1, master1 = out_T1 + t_aug, out_S1 + s_aug, master1 + m_aug

        out_T2, out_S2, master2 = self.HtrgGAT_layer_ST21(out_T, out_S, master=self.master2)
        out_S2, out_T2 = self.pool_hS2(out_S2), self.pool_hT2(out_T2)
        t_aug, s_aug, m_aug = self.HtrgGAT_layer_ST22(out_T2, out_S2, master=master2)
        out_T2, out_S2, master2 = out_T2 + t_aug, out_S2 + s_aug, master2 + m_aug

        out_T = torch.max(self.drop_way(out_T1), self.drop_way(out_T2))
        out_S = torch.max(self.drop_way(out_S1), self.drop_way(out_S2))
        master = torch.max(self.drop_way(master1), self.drop_way(master2))

        T_max, _ = torch.max(torch.abs(out_T), dim=1)
        S_max, _ = torch.max(torch.abs(out_S), dim=1)
        hidden = torch.cat([T_max, out_T.mean(dim=1), S_max, out_S.mean(dim=1), master.squeeze(1)], dim=1)
        return self.out_layer(self.drop(hidden))


# --------------------------------------------------------------------------
# Loading the authors' trained weights
# --------------------------------------------------------------------------

# fairseq name -> Hugging Face name (inside the XLS-R part)
_RENAMES = [
    (".self_attn.", ".attention."),
    (".self_attn_layer_norm.", ".layer_norm."),
    (".fc1.", ".feed_forward.intermediate_dense."),
    (".fc2.", ".feed_forward.output_dense."),
]
_UNUSED = ("quantizer.", "project_q.", "final_proj.", "mask_emb")


def _rename_xlsr_key(key):
    if key.startswith(_UNUSED):
        return None
    if key.startswith("feature_extractor.conv_layers."):
        _, _, i, part, *rest = key.split(".")
        if part == "0":
            return f"feature_extractor.conv_layers.{i}.conv.{rest[-1]}"
        return f"feature_extractor.conv_layers.{i}.layer_norm.{rest[-1]}"
    if key.startswith("post_extract_proj."):
        return key.replace("post_extract_proj.", "feature_projection.projection.")
    if key.startswith("layer_norm."):
        return key.replace("layer_norm.", "feature_projection.layer_norm.")
    if key.startswith("encoder.pos_conv.0."):
        name = key.split(".")[-1]
        return {
            "bias": "encoder.pos_conv_embed.conv.bias",
            "weight_g": "encoder.pos_conv_embed.conv.parametrizations.weight.original0",
            "weight_v": "encoder.pos_conv_embed.conv.parametrizations.weight.original1",
        }[name]
    for old, new in _RENAMES:
        key = key.replace(old, new)
    return key


def load_original_checkpoint(path):
    """Build SSLAASIST and load the authors' .pth file (fairseq naming) into it."""
    original = torch.load(path, map_location="cpu", weights_only=True)
    renamed = {}
    for key, value in original.items():
        key = key.removeprefix("module.")
        if key.startswith("ssl_model.model."):
            new = _rename_xlsr_key(key.removeprefix("ssl_model.model."))
            if new is None:
                continue
            renamed["ssl." + new] = value
        else:
            renamed[key] = value
    model = SSLAASIST()
    missing, unexpected = model.load_state_dict(renamed, strict=False)
    missing = [k for k in missing if k != "ssl.masked_spec_embed"]
    if missing or unexpected:
        raise ValueError(f"checkpoint mismatch: missing={missing[:5]} unexpected={unexpected[:5]}")
    return model
