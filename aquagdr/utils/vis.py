from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
from PIL import Image

try:
    import matplotlib.cm as cm
except Exception:
    cm = None


ArrayLike = Union[np.ndarray, torch.Tensor]


def _to_numpy_2d(x: ArrayLike) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().float().numpy()
    x = np.asarray(x)
    if x.ndim == 4:   # [B,1,H,W]
        x = x[0, 0]
    elif x.ndim == 3:
        if x.shape[0] == 1:  # [1,H,W]
            x = x[0]
        elif x.shape[-1] == 1:  # [H,W,1]
            x = x[..., 0]
    return x.astype(np.float32)


def _to_numpy_rgb(x: ArrayLike) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().float().numpy()
    x = np.asarray(x)
    if x.ndim == 4:  # [B,C,H,W]
        x = x[0]
    if x.ndim == 3 and x.shape[0] == 3:  # CHW -> HWC
        x = np.transpose(x, (1, 2, 0))
    x = x.astype(np.float32)

    # 尝试把归一化RGB拉回[0,1]
    if x.min() < 0.0 or x.max() > 1.0:
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        x = x * std[None, None, :] + mean[None, None, :]
    x = np.clip(x, 0.0, 1.0)
    return x


def colorize_depth(depth: ArrayLike, vmin: Optional[float] = None, vmax: Optional[float] = None, cmap: str = "plasma") -> np.ndarray:
    d = _to_numpy_2d(depth)
    valid = np.isfinite(d) & (d > 0)

    if not np.any(valid):
        return np.zeros((d.shape[0], d.shape[1], 3), dtype=np.uint8)

    if vmin is None:
        vmin = float(np.percentile(d[valid], 2))
    if vmax is None:
        vmax = float(np.percentile(d[valid], 98))
    vmax = max(vmax, vmin + 1e-6)

    dn = (d - vmin) / (vmax - vmin)
    dn = np.clip(dn, 0.0, 1.0)

    if cm is not None:
        c = cm.get_cmap(cmap)(dn)[..., :3]
    else:
        c = np.stack([dn, dn, dn], axis=-1)

    c[~valid] = 0.0
    return (c * 255.0).astype(np.uint8)


def save_depth_png(depth: ArrayLike, path: Union[str, Path], vmin: Optional[float] = None, vmax: Optional[float] = None, cmap: str = "plasma"):
    img = colorize_depth(depth, vmin=vmin, vmax=vmax, cmap=cmap)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(str(path))


def save_rgb_depth_triplet(
    rgb: ArrayLike,
    pred_depth: ArrayLike,
    gt_depth: Optional[ArrayLike],
    path: Union[str, Path],
):
    rgb_np = _to_numpy_rgb(rgb)
    pred_np = colorize_depth(pred_depth)
    if gt_depth is not None:
        gt_np = colorize_depth(gt_depth)
        out = np.concatenate([rgb_np * 255.0, pred_np, gt_np], axis=1).astype(np.uint8)
    else:
        out = np.concatenate([rgb_np * 255.0, pred_np], axis=1).astype(np.uint8)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(out).save(str(path))