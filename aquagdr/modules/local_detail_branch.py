from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def _finite_diff_grad(x: torch.Tensor, circular_horizontal: bool = False) -> torch.Tensor:
    if circular_horizontal:
        dx = torch.roll(x, shifts=-1, dims=-1) - x
    else:
        dx = x[:, :, :, 1:] - x[:, :, :, :-1]
        dx = F.pad(dx, (0, 1, 0, 0), mode="replicate")
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    dy = F.pad(dy, (0, 0, 0, 1), mode="replicate")
    return torch.sqrt(dx * dx + dy * dy + 1e-6)


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class CircularConv2d(nn.Module):
    """ERP-aware convolution with horizontal circular padding."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, dilation: int = 1, groups: int = 1,
                 bias: bool = False) -> None:
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        self.pad = dilation * (kernel_size // 2)
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel_size, stride=stride,
            padding=0, dilation=dilation, groups=groups, bias=bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pad > 0:
            x = F.pad(x, (self.pad, self.pad, 0, 0), mode="circular")
            x = F.pad(x, (0, 0, self.pad, self.pad), mode="replicate")
        return self.conv(x)


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, circular_padding: bool = True) -> None:
        spatial_conv = (
            CircularConv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
            if circular_padding
            else nn.Conv2d(
                in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                padding=kernel_size // 2, bias=False,
            )
        )
        super().__init__(
            spatial_conv,
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )


class DepthwiseResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int = 1,
                 circular_padding: bool = True) -> None:
        super().__init__()
        self.depthwise = (
            CircularConv2d(
                channels, channels, kernel_size=3, dilation=dilation, groups=channels,
            )
            if circular_padding
            else nn.Conv2d(
                channels, channels, kernel_size=3, padding=dilation,
                dilation=dilation, groups=channels, bias=False,
            )
        )
        self.norm = nn.GroupNorm(_group_count(channels), channels)
        self.pointwise = nn.Conv2d(channels, channels, kernel_size=1, bias=False)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.depthwise(x)
        y = self.pointwise(self.act(self.norm(y)))
        return self.act(x + y)


class LocalDetailGuidanceBranch(nn.Module):
    """Extract high-resolution local-detail features for depth refinement."""

    def __init__(self, max_depth: float = 10.0, base_channels: int = 12,
                 out_channels: int = 16, num_blocks: int = 2,
                 hard_region_bias: float = -1.5,
                 circular_edge: bool = False,
                 use_edge: bool = True,
                 use_low_resolution: bool = True,
                 circular_padding: bool = True,
                 predict_hard_region: bool = True) -> None:
        super().__init__()
        if base_channels <= 0 or out_channels <= 0:
            raise ValueError("base_channels and out_channels must be positive")
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        self.max_depth = float(max_depth)
        self.out_channels = int(out_channels)
        self.circular_edge = bool(circular_edge)
        self.use_edge = bool(use_edge)
        self.use_low_resolution = bool(use_low_resolution)
        self.circular_padding = bool(circular_padding)
        self.predict_hard_region = bool(predict_hard_region)

        input_channels = 4 + int(self.use_edge)  # RGB (3), normalized D0 (1), optional edge (1).
        self.high_stem = ConvNormAct(
            input_channels, base_channels, kernel_size=3,
            circular_padding=self.circular_padding,
        )
        if self.use_low_resolution:
            self.down = ConvNormAct(
                base_channels, base_channels * 2, kernel_size=3, stride=2,
                circular_padding=self.circular_padding,
            )
            self.blocks = nn.Sequential(*[
                DepthwiseResidualBlock(
                    base_channels * 2, dilation=1 + (i % 2),
                    circular_padding=self.circular_padding,
                )
                for i in range(num_blocks)
            ])
            fuse_in_channels = base_channels * 3
        else:
            self.down = None
            self.blocks = None
            fuse_in_channels = base_channels
        self.fuse = nn.Sequential(
            nn.Conv2d(fuse_in_channels, out_channels, kernel_size=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
            DepthwiseResidualBlock(
                out_channels, circular_padding=self.circular_padding,
            ),
        )
        self.hard_region_head = None
        if self.predict_hard_region:
            self.hard_region_head = nn.Conv2d(out_channels, 1, kernel_size=1)
            nn.init.constant_(self.hard_region_head.bias, float(hard_region_bias))

    def forward(self, rgb: torch.Tensor, initial_depth: torch.Tensor) -> Dict[str, torch.Tensor]:
        if rgb is None:
            raise ValueError("rgb is required by LocalDetailGuidanceBranch")
        if rgb.ndim != 4 or initial_depth.ndim != 4:
            raise ValueError("rgb and initial_depth must be 4D tensors")
        if rgb.shape[-2:] != initial_depth.shape[-2:]:
            rgb = F.interpolate(rgb, size=initial_depth.shape[-2:], mode="bilinear", align_corners=False)

        depth_norm = (initial_depth / self.max_depth).clamp(0.0, 1.0)
        inputs = [rgb, depth_norm]
        if self.use_edge:
            gray = rgb.mean(dim=1, keepdim=True)
            rgb_edge = torch.tanh(
                _finite_diff_grad(gray, circular_horizontal=self.circular_edge)
            )
            inputs.append(rgb_edge)
        x = torch.cat(inputs, dim=1)

        high = self.high_stem(x)
        if self.use_low_resolution:
            low = self.blocks(self.down(high))
            low = F.interpolate(low, size=high.shape[-2:], mode="bilinear", align_corners=False)
            fuse_input = torch.cat([high, low], dim=1)
        else:
            fuse_input = high
        detail_features = self.fuse(fuse_input)
        hard_region_logits = (
            self.hard_region_head(detail_features)
            if self.hard_region_head is not None
            else None
        )
        return {
            "detail_features": detail_features,
            "hard_region_logits": hard_region_logits,
            "hard_region": (
                torch.sigmoid(hard_region_logits)
                if hard_region_logits is not None
                else None
            ),
        }
