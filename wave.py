import math

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.norm(x))


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class WavePropagationOperator(nn.Module):
    """
    WPO implementation of:
        U_t = IDCT2( DCT2(U_0) * cos(c * |k| * t) )
    with learnable c and fixed t.

    Non-square token counts (e.g. CLS + patches) are mapped to a square
    grid of size `side x side` via a learnable token-dim linear (and back
    after WPO), following the same scheme as `wave_vivit_linear.py`.
    """

    def __init__(self, dim, num_tokens, hidden_dim=None, fixed_t=1.0, init_c=1.0):
        super().__init__()
        hidden_dim = dim if hidden_dim is None else hidden_dim
        if hidden_dim != dim:
            raise ValueError("WavePropagationOperator expects hidden_dim == dim.")
        if num_tokens < 1:
            raise ValueError("num_tokens must be >= 1.")

        self.num_tokens = int(num_tokens)
        self.side = int(math.floor(math.sqrt(self.num_tokens)))
        if self.side < 1:
            raise ValueError("num_tokens must allow a non-empty square grid.")
        self.grid_tokens = self.side * self.side

        self.input_token_linear = nn.Linear(self.num_tokens, self.grid_tokens, bias=True)
        self.output_token_linear = nn.Linear(self.grid_tokens, self.num_tokens, bias=True)

        self.dwconv = nn.Conv2d(dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim)
        self.in_proj = nn.Linear(hidden_dim, 2 * hidden_dim, bias=True)
        self.out_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)

        # c is learnable, t is fixed (user-selected option A)
        # Use a stable positive parameterization for c.
        init_c = max(float(init_c), 1e-6)
        self.raw_c = nn.Parameter(torch.tensor(float(math.log(math.exp(init_c) - 1.0))))
        self.fixed_t = float(fixed_t)

    @staticmethod
    def get_cos_map(n, device, dtype):
        idx_x = (torch.arange(n, device=device, dtype=dtype).view(1, -1) + 0.5) / n
        idx_n = torch.arange(n, device=device, dtype=dtype).view(-1, 1)
        basis = torch.cos(idx_n * idx_x * math.pi) * math.sqrt(2.0 / n)
        basis[0, :] /= math.sqrt(2.0)
        return basis

    @staticmethod
    def get_radial_frequency(h, w, device, dtype):
        kx = torch.linspace(0.0, math.pi, h + 1, device=device, dtype=dtype)[:h].view(-1, 1)
        ky = torch.linspace(0.0, math.pi, w + 1, device=device, dtype=dtype)[:w].view(1, -1)
        radius = torch.sqrt(kx.pow(2) + ky.pow(2))
        return radius.unsqueeze(-1)  # [H, W, 1]

    def _dct2(self, x):
        # x: [B, H, W, C]
        b, h, w, _ = x.shape
        cos_h = self.get_cos_map(h, x.device, x.dtype)
        cos_w = self.get_cos_map(w, x.device, x.dtype)

        x = x.reshape(b, h, -1)
        x = F.conv1d(x, cos_h.view(h, h, 1))
        x = x.reshape(b * h, w, -1)
        x = F.conv1d(x, cos_w.view(w, w, 1))
        return x.reshape(b, h, w, -1)

    def _idct2(self, x):
        # x: [B, H, W, C]
        b, h, w, _ = x.shape
        cos_h = self.get_cos_map(h, x.device, x.dtype)
        cos_w = self.get_cos_map(w, x.device, x.dtype)

        x = x.reshape(b, h, -1)
        x = F.conv1d(x, cos_h.t().view(h, h, 1))
        x = x.reshape(b * h, w, -1)
        x = F.conv1d(x, cos_w.t().view(w, w, 1))
        return x.reshape(b, h, w, -1)

    def forward(self, x):
        # x: [B, N, C], full token sequence (CLS + patches or CLS + frames)
        if x.shape[1] != self.num_tokens:
            raise ValueError(
                f"WavePropagationOperator expected {self.num_tokens} tokens, got {x.shape[1]}."
            )

        x = rearrange(x, "b n d -> b d n")
        x = self.input_token_linear(x)
        x = rearrange(x, "b d n -> b n d")

        side = self.side
        grid = rearrange(x, "b (h w) c -> b c h w", h=side, w=side)
        grid = self.dwconv(grid)
        grid = self.in_proj(grid.permute(0, 2, 3, 1))
        wave_feat, gate_feat = grid.chunk(2, dim=-1)

        # DCT2 -> cosine wave modulation -> IDCT2
        wave_feat = self._dct2(wave_feat)
        radial_k = self.get_radial_frequency(side, side, wave_feat.device, wave_feat.dtype)
        c_speed = F.softplus(self.raw_c) + 1e-6
        phase = c_speed * self.fixed_t * radial_k
        wave_feat = wave_feat * torch.cos(phase)
        wave_feat = self._idct2(wave_feat)

        out = self.out_norm(wave_feat)
        out = out * F.silu(gate_feat)
        out = self.out_proj(out)
        out = rearrange(out, "b h w c -> b (h w) c")

        out = rearrange(out, "b n d -> b d n")
        out = self.output_token_linear(out)
        out = rearrange(out, "b d n -> b n d")
        return out


class WaveTransformer(nn.Module):
    def __init__(self, dim, depth, mlp_dim, num_tokens, dropout=0.0, fixed_t=1.0, init_c=1.0):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            WavePropagationOperator(
                                dim,
                                num_tokens=num_tokens,
                                fixed_t=fixed_t,
                                init_c=init_c,
                            ),
                        ),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        for wpo, ffn in self.layers:
            x = x + wpo(x)
            x = x + ffn(x)
        return self.norm(x)


class ViViT(nn.Module):
    def __init__(
        self,
        image_size=224,
        patch_size=16,
        num_classes=8,
        num_frames=16,
        dim=192,
        depth=4,
        heads=3,
        pool="cls",
        in_channels=3,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        scale_dim=4,
        fixed_t=1.0,
        init_c=1.0,
    ):
        super().__init__()
        del heads, dim_head  # kept for compatibility with existing call sites

        assert pool in {"cls", "mean"}, "pool type must be either cls or mean"
        assert image_size % patch_size == 0, "Image dimensions must be divisible by patch size."

        num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size * patch_size

        self.to_patch_embedding = nn.Sequential(
            Rearrange("b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)", p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_frames, num_patches + 1, dim))
        self.space_token = nn.Parameter(torch.randn(1, 1, dim))
        self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.pool = pool

        self.space_transformer = WaveTransformer(
            dim=dim,
            depth=depth,
            mlp_dim=dim * scale_dim,
            num_tokens=num_patches + 1,
            dropout=dropout,
            fixed_t=fixed_t,
            init_c=init_c,
        )
        self.temporal_transformer = WaveTransformer(
            dim=dim,
            depth=depth,
            mlp_dim=dim * scale_dim,
            num_tokens=num_frames + 1,
            dropout=dropout,
            fixed_t=fixed_t,
            init_c=init_c,
        )

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes),
        )

    def forward(self, x):
        x = self.to_patch_embedding(x)
        b, t, n, _ = x.shape

        cls_space_tokens = repeat(self.space_token, "() n d -> b t n d", b=b, t=t)
        x = torch.cat((cls_space_tokens, x), dim=2)
        x = x + self.pos_embedding[:, :, : n + 1]
        x = self.dropout(x)

        x = rearrange(x, "b t n d -> (b t) n d")
        x = self.space_transformer(x)
        x = rearrange(x[:, 0], "(b t) d -> b t d", b=b)

        cls_temporal_tokens = repeat(self.temporal_token, "() n d -> b n d", b=b)
        x = torch.cat((cls_temporal_tokens, x), dim=1)
        x = self.temporal_transformer(x)

        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]
        return self.mlp_head(x)

