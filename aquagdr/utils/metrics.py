from typing import Dict, Optional

import torch


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.sum += float(val) * int(n)
        self.count += int(n)

    @property
    def avg(self) -> float:
        if self.count == 0:
            return 0.0
        return self.sum / self.count


def _clamp_depth(x: torch.Tensor, min_depth: float, max_depth: float) -> torch.Tensor:
    return torch.clamp(x, min=min_depth, max=max_depth)


@torch.no_grad()
def compute_depth_metrics(
    gt: torch.Tensor,
    pred: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    min_depth: float = 0.1,
    max_depth: float = 10.0,
    median_align: bool = False,
) -> Dict[str, float]:
    """
    gt/pred: [B,1,H,W]
    """
    if mask is None:
        mask = gt > 0
    else:
        mask = mask.bool()
    mask = mask & torch.isfinite(gt) & torch.isfinite(pred)

    gt_v = gt[mask]
    pred_v = pred[mask]

    if gt_v.numel() == 0:
        return {
            "abs": 0.0,
            "abs_rel": 0.0,
            "sq_rel": 0.0,
            "rmse": 0.0,
            "rmse_log": 0.0,
            "log10": 0.0,
            "a1": 0.0,
            "a2": 0.0,
            "a3": 0.0,
        }

    gt_v = _clamp_depth(gt_v, min_depth, max_depth)
    pred_v = _clamp_depth(pred_v, min_depth, max_depth)

    if median_align:
        scale = torch.median(gt_v) / (torch.median(pred_v) + 1e-6)
        pred_v = pred_v * scale
        pred_v = _clamp_depth(pred_v, min_depth, max_depth)

    thresh = torch.max(gt_v / pred_v, pred_v / gt_v)
    a1 = (thresh < 1.25).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    rmse = torch.sqrt(((gt_v - pred_v) ** 2).mean())
    rmse_log = torch.sqrt(((torch.log10(gt_v) - torch.log10(pred_v)) ** 2).mean())

    abs_ = torch.abs(gt_v - pred_v).mean()
    abs_rel = (torch.abs(gt_v - pred_v) / gt_v).mean()
    sq_rel = (((gt_v - pred_v) ** 2) / gt_v).mean()
    log10 = torch.abs(torch.log10(pred_v / gt_v)).mean()

    return {
        "abs": float(abs_.item()),
        "abs_rel": float(abs_rel.item()),
        "sq_rel": float(sq_rel.item()),
        "rmse": float(rmse.item()),
        "rmse_log": float(rmse_log.item()),
        "log10": float(log10.item()),
        "a1": float(a1.item()),
        "a2": float(a2.item()),
        "a3": float(a3.item()),
    }


class MetricsTracker:
    def __init__(self, keys=None):
        self.keys = keys or ["abs", "abs_rel", "sq_rel", "rmse", "rmse_log", "log10", "a1", "a2", "a3"]
        self.meters = {k: AverageMeter() for k in self.keys}

    def reset(self):
        for m in self.meters.values():
            m.reset()

    def update(self, metrics: Dict[str, float], n: int = 1):
        for k, v in metrics.items():
            if k in self.meters:
                self.meters[k].update(v, n=n)

    def as_dict(self) -> Dict[str, float]:
        return {k: self.meters[k].avg for k in self.keys}


def print_depth_metrics(metrics: Dict[str, float]) -> None:
    keys = ["abs", "abs_rel", "sq_rel", "rmse", "rmse_log", "log10", "a1", "a2", "a3"]
    header = ("{:>9} | " * len(keys)).format(*keys)
    values = ("{: 8.5f} " * len(keys)).format(*(metrics[k] for k in keys))
    print("\n  " + header)
    print("&  " + values)
