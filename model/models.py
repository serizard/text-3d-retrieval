import torch.nn as nn
import torch
from einops import rearrange
import torch.nn.functional as F
from PointNet_utils import PointNetSetAbstraction


def make_model(config):
    if config.model.name == 'PointBERT':
        return ProjectedEncoder(
            PointPatchTransformer(512, 12, 8, 512*3, 256, 384, 0.2, 64, config.model.in_channel),
            nn.Linear(512, config.model.out_channel)
        )
    else:
        raise ValueError(f"Unsupported model name: {config.model.name}")


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, *extra_args, **kwargs):
        return self.fn(self.norm(x), *extra_args, **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0., rel_pe = False):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        self.rel_pe = rel_pe
        if rel_pe:
            self.pe = nn.Sequential(nn.Conv2d(3, 64, 1), nn.ReLU(), nn.Conv2d(64, 1, 1))

    def forward(self, x, centroid_delta):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        pe = self.pe(centroid_delta) if self.rel_pe else 0
        dots = (torch.matmul(q, k.transpose(-1, -2)) + pe) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0., rel_pe = False):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout, rel_pe = rel_pe)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x, centroid_delta):
        for attn, ff in self.layers:
            x = attn(x, centroid_delta) + x
            x = ff(x) + x
        return x
    

# Original version incorporated torch.redstone library inside the code
# Needs to be revised if this version (w/o torch.redstone) does not work
# https://github.com/Colin97/OpenShape_code/blob/master/src/models/ppat.py#L4
class PointPatchTransformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim, sa_dim, patches, prad, nsamp, in_dim=3, dim_head=64, rel_pe=False, patch_dropout=0) -> None:
        super().__init__()
        self.patches = patches
        self.patch_dropout = patch_dropout
        self.sa = PointNetSetAbstraction(npoint=patches, radius=prad, nsample=nsamp, in_channel=in_dim + 3, mlp=[64, 64, sa_dim], group_all=False)
        self.lift = nn.Sequential(
            nn.Conv1d(sa_dim + 3, dim, 1),
            nn.Lambda(lambda x: torch.permute(x, [0, 2, 1])),
            nn.LayerNorm([dim])
        )
        self.cls_token = nn.Parameter(torch.randn(dim))
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, 0.0, rel_pe)

    def forward(self, xyz: torch.Tensor, features):
        self.sa.npoint = self.patches
        if self.training:
            self.sa.npoint -= self.patch_dropout
        centroids, feature = self.sa(xyz, features)

        x = self.lift(torch.cat([centroids, feature], dim=1))

        # Concat the cls_token to the batch
        cls_token = self.cls_token.unsqueeze(0).expand(x.shape[0], -1).unsqueeze(1)  # Expand for batch size
        x = torch.cat([cls_token, x], dim=1)

        centroids = torch.cat([centroids.new_zeros((centroids.shape[0], 1, centroids.shape[2])), centroids], dim=1)

        centroid_delta = centroids.unsqueeze(-1) - centroids.unsqueeze(-2)
        x = self.transformer(x, centroid_delta)

        return x[:, 0]


class ProjectedEncoder(nn.Module):
    def __init__(self, pp_transformer, head):
        super().__init__()
        self.pp_transformer = pp_transformer
        self.head = head 

    def forward(self, xyz, feature, device, quantization_size):
        output = self.pp_transformer(xyz.transpose(-1,-2).contiguous(), 
                                     feature.transpose(-1,-2).contiguous())
        return self.head(output)