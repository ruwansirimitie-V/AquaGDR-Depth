from typing import Dict, Optional, Tuple

import torch


def _safe_valid_values(x: torch.Tensor, mask: Optional[torch.Tensor], min_count: int = 32) -> torch.Tensor:
    # x: [H, W]
    if mask is None:
        vals = x.reshape(-1)
    else:
        m = mask > 0.5
        vals = x[m]
        if vals.numel() < min_count:
            vals = x.reshape(-1)
    return vals


@torch.no_grad()
def median_mad_stats(
    depth: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    depth: [B,1,H,W]
    mask : [B,1,H,W] or None
    return: med[B,1,1,1], mad[B,1,1,1]
    """
    b = depth.shape[0]
    meds = []
    mads = []
    for i in range(b):
        di = depth[i, 0]
        mi = mask[i, 0] if mask is not None else None
        vals = _safe_valid_values(di, mi)
        med = vals.median()
        mad = (vals - med).abs().median().clamp_min(eps)
        meds.append(med)
        mads.append(mad)

    med = torch.stack(meds).view(b, 1, 1, 1)
    mad = torch.stack(mads).view(b, 1, 1, 1)
    return med, mad


@torch.no_grad()
def robust_align_to_reference(
    src_depth: torch.Tensor,
    ref_depth: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
    clamp_min: Optional[float] = 1e-3,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    用 median+MAD 把 src 对齐到 ref：
      aligned = s * src + t
      s = mad_ref / mad_src
      t = med_ref - s * med_src
    """
    med_s, mad_s = median_mad_stats(src_depth, mask=mask, eps=eps)
    med_r, mad_r = median_mad_stats(ref_depth, mask=mask, eps=eps)

    s = mad_r / (mad_s + eps)
    t = med_r - s * med_s
    aligned = s * src_depth + t

    if clamp_min is not None:
        aligned = aligned.clamp_min(clamp_min)

    return aligned, {"scale": s, "shift": t, "med_src": med_s, "med_ref": med_r}


@torch.no_grad()
def normalize_by_median_mad(
    depth: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    med, mad = median_mad_stats(depth, mask=mask, eps=eps)
    return (depth - med) / (mad + eps)