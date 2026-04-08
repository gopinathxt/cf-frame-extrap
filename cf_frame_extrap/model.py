from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def _groupnorm_groups(num_channels: int, max_groups: int = 32) -> int:
    """
    Choose the largest group count <= max_groups that divides num_channels.
    This avoids "num_channels must be divisible by num_groups" for e.g. 48ch.
    """
    n = int(num_channels)
    if n <= 0:
        return 1
    g_max = min(int(max_groups), n)
    # Fast path: gcd guarantees divisibility.
    g = math.gcd(n, g_max)
    if g >= 1:
        return int(g)
    return 1


@dataclass
class ModelConfig:
    k: int = 3  # number of past frames/actions
    action_dim: int = 4  # CTBR
    base_ch: int = 48
    ch_mults: Tuple[int, ...] = (1, 2, 3, 4)  # encoder stages
    num_res_blocks: int = 2
    dropout: float = 0.0
    film_hidden: int = 128


class FiLM(nn.Module):
    """
    Feature-wise linear modulation: y = (1 + gamma) * x + beta
    gamma/beta are produced from an action embedding.
    """

    def __init__(self, cond_dim: int, num_channels: int):
        super().__init__()
        self.to_gb = nn.Linear(cond_dim, 2 * num_channels)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        # x: [B,C,H,W], cond: [B,cond_dim]
        gb = self.to_gb(cond)  # [B,2C]
        gamma, beta = gb.chunk(2, dim=1)
        gamma = gamma[:, :, None, None]
        beta = beta[:, :, None, None]
        return (1.0 + gamma) * x + beta


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, cond_dim: int, dropout: float = 0.0):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.norm1 = nn.GroupNorm(num_groups=_groupnorm_groups(in_ch), num_channels=in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.film1 = FiLM(cond_dim, out_ch)

        self.norm2 = nn.GroupNorm(num_groups=_groupnorm_groups(out_ch), num_channels=out_ch)
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.film2 = FiLM(cond_dim, out_ch)

        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.film1(h, cond)
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        h = self.film2(h, cond)
        return h + self.skip(x)


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        return self.conv(x)


class ActionEncoder(nn.Module):
    """
    Encodes action history (K x action_dim) into a conditioning vector.
    """

    def __init__(self, k: int, action_dim: int, hidden: int):
        super().__init__()
        in_dim = k * action_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
        )

    def forward(self, a_hist: torch.Tensor) -> torch.Tensor:
        # a_hist: [B,K,action_dim]
        b, k, d = a_hist.shape
        return self.net(a_hist.reshape(b, k * d))


class ActionConditionedUNet(nn.Module):
    """
    Predict next frame from K previous frames + K actions.

    Inputs:
      frames: [B,K,H,W] in [0,1]
      actions: [B,K,4] float (CTBR), normalized
    Output:
      next_frame: [B,1,H,W] in [0,1]
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg

        cond_dim = cfg.film_hidden
        self.action_enc = ActionEncoder(cfg.k, cfg.action_dim, cfg.film_hidden)

        in_ch = cfg.k  # stacked grayscale frames as channels
        base = cfg.base_ch

        # encoder (U-Net style): we ONLY store skips before each downsample.
        self.in_conv = nn.Conv2d(in_ch, base, kernel_size=3, padding=1)

        self.down_blocks: nn.ModuleList = nn.ModuleList()
        self.downsamples: nn.ModuleList = nn.ModuleList()
        self.skip_channels: List[int] = []

        cur = base
        for si, mult in enumerate(cfg.ch_mults):
            stage_out = base * mult
            blocks = nn.ModuleList()
            for _ in range(cfg.num_res_blocks):
                blocks.append(ResBlock(cur, stage_out, cond_dim, dropout=cfg.dropout))
                cur = stage_out
                self.skip_channels.append(cur)  # skip after each resblock
            self.down_blocks.append(blocks)
            if si != len(cfg.ch_mults) - 1:
                self.downsamples.append(Downsample(cur))

        # bottleneck
        self.mid1 = ResBlock(cur, cur, cond_dim, dropout=cfg.dropout)
        self.mid2 = ResBlock(cur, cur, cond_dim, dropout=cfg.dropout)

        # decoder: for each stage (reversed), upsample (except bottom), then resblocks with concatenated skip.
        self.up_blocks: nn.ModuleList = nn.ModuleList()
        self.upsamples: nn.ModuleList = nn.ModuleList()

        # We'll consume skip channels in reverse order during forward.
        skip_chs = list(self.skip_channels)

        for si, mult in enumerate(reversed(cfg.ch_mults)):
            stage_out = base * mult
            if si != 0:
                self.upsamples.append(Upsample(cur))
            blocks = nn.ModuleList()
            for _ in range(cfg.num_res_blocks):
                skip = skip_chs.pop()
                blocks.append(ResBlock(cur + skip, stage_out, cond_dim, dropout=cfg.dropout))
                cur = stage_out
            self.up_blocks.append(blocks)

        self.out_norm = nn.GroupNorm(num_groups=_groupnorm_groups(cur), num_channels=cur)
        self.out_conv = nn.Conv2d(cur, 1, kernel_size=3, padding=1)

    def forward(self, frames: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        cond = self.action_enc(actions)  # [B,cond_dim]

        h = self.in_conv(frames)
        skips: List[torch.Tensor] = []

        # encoder
        for si, blocks in enumerate(self.down_blocks):
            for b in blocks:
                h = b(h, cond)
                skips.append(h)
            if si != len(self.down_blocks) - 1:
                h = self.downsamples[si](h)

        # bottleneck
        h = self.mid1(h, cond)
        h = self.mid2(h, cond)

        # decoder
        # The last element of skips is the deepest skip; the first decode resblock uses it.
        for si, blocks in enumerate(self.up_blocks):
            if si != 0:
                h = self.upsamples[si - 1](h)
            for b in blocks:
                s = skips.pop()
                h = torch.cat([h, s], dim=1)
                h = b(h, cond)

        h = F.silu(self.out_norm(h))
        out = self.out_conv(h)
        return torch.sigmoid(out)


def build_model(k: int = 3, base_ch: int = 48, dropout: float = 0.0) -> ActionConditionedUNet:
    cfg = ModelConfig(k=k, base_ch=base_ch, dropout=dropout)
    return ActionConditionedUNet(cfg)

