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


class ERPAwareConv2d(nn.Conv2d):
    """3x3 convolution with circular horizontal and replicated vertical padding."""

    def __init__(self, in_channels: int, out_channels: int, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size=3, padding=0, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (1, 1, 0, 0), mode="circular")
        x = F.pad(x, (0, 0, 1, 1), mode="replicate")
        return super().forward(x)


class QualityGuidedResidualRefinement(nn.Module):
    """Configurable residual refinement for an initial depth D0.

    The final local-detail model keeps the shared context encoder but disables
    the legacy coarse correction. Historical configs retain that path through
    ``use_coarse_refinement=True``.
    """

    def __init__(
        self,
        max_depth: float = 10.0,
        min_depth: float = 1e-3,
        use_coarse_refinement: bool = True,
        residual_cap: float = 2.0,
        lowpass_k: int = 5,
        use_rgb: bool = True,
        use_edge: bool = True,
        use_quality: bool = False,
        use_global_context: bool = False,
        context_channels: int = 0,
        use_local_detail: bool = False,
        detail_channels: int = 0,
        detail_residual_cap: float = 0.5,
        use_detail_mask: bool = True,
        bounded_detail_residual: bool = True,
        use_hard_gate: bool = True,
        erp_circular: bool = False,
    ):
        super().__init__()
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.use_coarse_refinement = bool(use_coarse_refinement)
        self.residual_cap = residual_cap
        self.lowpass_k = lowpass_k
        self.use_rgb = use_rgb
        self.use_edge = use_edge
        self.use_quality = use_quality
        self.use_global_context = use_global_context
        self.context_channels = int(context_channels)
        self.use_local_detail = use_local_detail
        self.detail_channels = int(detail_channels)
        self.detail_residual_cap = float(detail_residual_cap)
        self.use_detail_mask = bool(use_detail_mask)
        self.bounded_detail_residual = bool(bounded_detail_residual)
        self.use_hard_gate = bool(use_hard_gate)
        self.erp_circular = bool(erp_circular)

        in_ch = 1  # initial depth D0
        if use_quality:
            in_ch += 1
        if use_rgb:
            in_ch += 3
        if use_edge:
            in_ch += 1
        if use_global_context:
            if self.context_channels <= 0:
                raise ValueError("context_channels must be positive when use_global_context=True")
            in_ch += self.context_channels

        spatial_conv = ERPAwareConv2d if self.erp_circular else None
        context_conv1 = (
            spatial_conv(in_ch, 32)
            if spatial_conv is not None
            else nn.Conv2d(in_ch, 32, 3, padding=1)
        )
        context_conv2 = (
            spatial_conv(32, 32)
            if spatial_conv is not None
            else nn.Conv2d(32, 32, 3, padding=1)
        )
        self.encoder = nn.Sequential(
            context_conv1,
            nn.ReLU(inplace=True),
            context_conv2,
            nn.ReLU(inplace=True),
        )
        if self.use_coarse_refinement:
            self.mask_head = nn.Conv2d(32, 1, 1)
            self.delta_head = nn.Conv2d(32, 1, 1)

        if self.use_local_detail:
            if self.detail_channels < 0:
                raise ValueError("detail_channels must be non-negative")
            detail_spatial_conv = (
                ERPAwareConv2d(16, 16, groups=16, bias=False)
                if self.erp_circular
                else nn.Conv2d(16, 16, kernel_size=3, padding=1, groups=16, bias=False)
            )
            self.detail_encoder = nn.Sequential(
                nn.Conv2d(32 + self.detail_channels, 16, kernel_size=1, bias=False),
                nn.GroupNorm(4, 16),
                nn.SiLU(inplace=True),
                detail_spatial_conv,
                nn.GroupNorm(4, 16),
                nn.SiLU(inplace=True),
            )
            self.detail_gate_head = (
                nn.Conv2d(16, 1, kernel_size=1)
                if self.use_detail_mask
                else None
            )
            self.detail_delta_head = nn.Conv2d(16, 1, kernel_size=1)

    def lowpass(self, x: torch.Tensor) -> torch.Tensor:
        if self.lowpass_k <= 1:
            return x
        return F.avg_pool2d(x, kernel_size=self.lowpass_k, stride=1, padding=self.lowpass_k // 2)

    def forward(
        self,
        initial_depth: torch.Tensor,
        quality_map: torch.Tensor = None,
        rgb: torch.Tensor = None,
        valid_mask: torch.Tensor = None,
        global_context: torch.Tensor = None,
        local_detail: torch.Tensor = None,
        hard_region: torch.Tensor = None,
        hard_region_logits: torch.Tensor = None,
    ):
        if quality_map is None:
            quality_for_diag = torch.ones_like(initial_depth)
        elif quality_map.shape[-2:] != initial_depth.shape[-2:]:
            quality_for_diag = F.interpolate(
                quality_map,
                size=initial_depth.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        else:
            quality_for_diag = quality_map
        q_bad = 1.0 - quality_for_diag.clamp(0.0, 1.0)

        if not self.use_coarse_refinement and not self.use_local_detail:
            zero = torch.zeros_like(initial_depth)
            return {
                "refined_depth": initial_depth.clamp(self.min_depth, self.max_depth),
                "residual_delta": zero,
                "refinement_mask": zero,
                "coarse_correction": zero,
                "detail_residual_delta": zero,
                "detail_gate": zero,
                "detail_correction": zero,
                "total_correction": zero,
                "hard_region": hard_region,
                "hard_region_logits": hard_region_logits,
                "q_bad": q_bad,
            }

        feats = [initial_depth]
        if self.use_quality:
            feats.append(q_bad)

        rgb_for_edge = rgb
        if (self.use_rgb or self.use_edge) and rgb_for_edge is None:
            rgb_for_edge = torch.zeros(
                initial_depth.shape[0], 3, initial_depth.shape[2], initial_depth.shape[3],
                device=initial_depth.device, dtype=initial_depth.dtype,
            )
        if rgb_for_edge is not None and rgb_for_edge.shape[-2:] != initial_depth.shape[-2:]:
            rgb_for_edge = F.interpolate(rgb_for_edge, size=initial_depth.shape[-2:], mode="bilinear", align_corners=False)

        if self.use_rgb:
            feats.append(rgb_for_edge)
        if self.use_edge:
            feats.append(
                _finite_diff_grad(
                    rgb_for_edge.mean(dim=1, keepdim=True),
                    circular_horizontal=self.erp_circular,
                )
            )

        if self.use_global_context:
            if global_context is None:
                global_context = torch.zeros(
                    initial_depth.shape[0],
                    self.context_channels,
                    initial_depth.shape[2],
                    initial_depth.shape[3],
                    device=initial_depth.device,
                    dtype=initial_depth.dtype,
                )
            elif global_context.shape[-2:] != initial_depth.shape[-2:]:
                global_context = F.interpolate(
                    global_context,
                    size=initial_depth.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            if global_context.shape[1] != self.context_channels:
                raise ValueError(
                    f"Expected global_context with {self.context_channels} channels, "
                    f"got {global_context.shape[1]}"
                )
            feats.append(global_context)

        x = torch.cat(feats, dim=1)
        f = self.encoder(x)
        if self.use_coarse_refinement:
            refinement_mask = torch.sigmoid(self.mask_head(f))
            residual_delta = self.lowpass(self.residual_cap * torch.tanh(self.delta_head(f)))
            coarse_correction = refinement_mask * residual_delta
        else:
            refinement_mask = torch.zeros_like(initial_depth)
            residual_delta = torch.zeros_like(initial_depth)
            coarse_correction = torch.zeros_like(initial_depth)

        detail_delta = torch.zeros_like(initial_depth)
        detail_gate = torch.zeros_like(initial_depth)
        detail_correction = torch.zeros_like(initial_depth)
        if self.use_local_detail:
            if self.detail_channels > 0 and local_detail is None:
                raise ValueError("local_detail is required when use_local_detail=True")
            if self.use_hard_gate and hard_region is None:
                raise ValueError("hard_region is required when use_hard_gate=True")
            if local_detail is not None and local_detail.shape[-2:] != initial_depth.shape[-2:]:
                local_detail = F.interpolate(
                    local_detail,
                    size=initial_depth.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            if local_detail is not None and local_detail.shape[1] != self.detail_channels:
                raise ValueError(
                    f"Expected local_detail with {self.detail_channels} channels, "
                    f"got {local_detail.shape[1]}"
                )
            if hard_region is not None and hard_region.shape[-2:] != initial_depth.shape[-2:]:
                hard_region = F.interpolate(
                    hard_region,
                    size=initial_depth.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            if hard_region_logits is not None and hard_region_logits.shape[-2:] != initial_depth.shape[-2:]:
                hard_region_logits = F.interpolate(
                    hard_region_logits,
                    size=initial_depth.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )

            detail_input = f if local_detail is None else torch.cat([f, local_detail], dim=1)
            detail_f = self.detail_encoder(detail_input)
            detail_gate = (
                torch.sigmoid(self.detail_gate_head(detail_f))
                if self.use_detail_mask
                else torch.ones_like(initial_depth)
            )
            if self.use_hard_gate:
                detail_gate = hard_region * detail_gate
            raw_detail_delta = self.detail_delta_head(detail_f)
            detail_delta = (
                self.detail_residual_cap * torch.tanh(raw_detail_delta)
                if self.bounded_detail_residual
                else raw_detail_delta
            )
            detail_correction = detail_gate * detail_delta

        total_correction = coarse_correction + detail_correction
        refined_depth = (initial_depth + total_correction).clamp(self.min_depth, self.max_depth)

        return {
            "refined_depth": refined_depth,
            "residual_delta": residual_delta,
            "refinement_mask": refinement_mask,
            "coarse_correction": coarse_correction,
            "detail_residual_delta": detail_delta,
            "detail_gate": detail_gate,
            "detail_correction": detail_correction,
            "total_correction": total_correction,
            "hard_region": hard_region,
            "hard_region_logits": hard_region_logits,
            "q_bad": q_bad,
        }
