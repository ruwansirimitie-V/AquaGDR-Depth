from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def _masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor] = None, eps: float = 1e-6):
    if mask is None:
        return x.mean()
    m = mask.float()
    return (x * m).sum() / (m.sum() + eps)


def _masked_l1(a: torch.Tensor, b: torch.Tensor, mask: Optional[torch.Tensor] = None):
    return _masked_mean((a - b).abs(), mask)


def _grad_xy(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    dx = F.pad(dx, (0, 1, 0, 0), mode="replicate")
    dy = F.pad(dy, (0, 0, 0, 1), mode="replicate")
    return dx, dy


class DepthRefinementLoss(nn.Module):
    """Depth, gradient, and optional D0 hard-region supervision."""

    def __init__(
        self,
        w_depth: float = 1.0,
        w_grad: float = 0.3,
        w_hard: float = 0.0,
        hard_threshold: float = 0.08,
        hard_temperature: float = 0.02,
        pairwise_valid_grad: bool = False,
    ):
        super().__init__()
        self.w_depth = w_depth
        self.w_grad = w_grad
        self.w_hard = w_hard
        self.hard_threshold = hard_threshold
        self.hard_temperature = hard_temperature
        self.pairwise_valid_grad = bool(pairwise_valid_grad)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        gt_depth: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ):
        pred = outputs["pred_depth"]
        loss_terms = {}
        loss_terms["loss_depth"] = _masked_l1(pred, gt_depth, valid_mask)

        pred_dx, pred_dy = _grad_xy(pred)
        gt_dx, gt_dy = _grad_xy(gt_depth)
        grad_mask_x = valid_mask
        grad_mask_y = valid_mask
        if self.pairwise_valid_grad and valid_mask is not None:
            valid = valid_mask.bool()
            grad_mask_x = torch.zeros_like(valid)
            grad_mask_y = torch.zeros_like(valid)
            grad_mask_x[:, :, :, :-1] = valid[:, :, :, :-1] & valid[:, :, :, 1:]
            grad_mask_y[:, :, :-1, :] = valid[:, :, :-1, :] & valid[:, :, 1:, :]
        loss_terms["loss_grad"] = (
            _masked_l1(pred_dx, gt_dx, grad_mask_x)
            + _masked_l1(pred_dy, gt_dy, grad_mask_y)
        )

        loss_hard = pred.new_zeros(())
        hard_target_fraction = pred.new_zeros(())
        hard_pred_fraction = pred.new_zeros(())
        if self.w_hard > 0.0:
            hard_logits = outputs.get("hard_region_logits")
            initial_depth = outputs.get("initial_depth")
            if hard_logits is None or initial_depth is None:
                raise ValueError(
                    "w_hard > 0 requires hard_region_logits and initial_depth in model outputs"
                )
            if hard_logits.shape[-2:] != gt_depth.shape[-2:]:
                hard_logits = F.interpolate(
                    hard_logits,
                    size=gt_depth.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            relative_error = (
                (initial_depth.detach() - gt_depth).abs()
                / gt_depth.clamp_min(1e-3)
            )
            temperature = max(float(self.hard_temperature), 1e-6)
            hard_target = torch.sigmoid(
                (relative_error - float(self.hard_threshold)) / temperature
            )
            hard_bce = F.binary_cross_entropy_with_logits(
                hard_logits,
                hard_target,
                reduction="none",
            )
            loss_hard = _masked_mean(hard_bce, valid_mask)
            hard_target_fraction = _masked_mean(hard_target, valid_mask)
            hard_pred_fraction = _masked_mean(torch.sigmoid(hard_logits), valid_mask)

        loss_terms["loss_hard"] = loss_hard
        loss_terms["hard_target_fraction"] = hard_target_fraction
        loss_terms["hard_pred_fraction"] = hard_pred_fraction
        total = (
            self.w_depth * loss_terms["loss_depth"]
            + self.w_grad * loss_terms["loss_grad"]
            + self.w_hard * loss_terms["loss_hard"]
        )
        loss_terms["loss_total"] = total
        return total, loss_terms


def _apply_loss_overrides(loss: DepthRefinementLoss, overrides: Optional[Dict] = None) -> DepthRefinementLoss:
    if not overrides:
        return loss
    for key, value in overrides.items():
        if not hasattr(loss, key):
            raise ValueError(f"Unknown loss override: {key}")
        setattr(loss, key, value)
    return loss


def build_refinement_loss(stage: str = "stage1_refinement", overrides: Optional[Dict] = None) -> DepthRefinementLoss:
    stage = stage.lower()
    if stage in {"stage_local_addon", "stage_local_refinement", "stage_backbone_finetune"}:
        loss = DepthRefinementLoss(w_depth=1.0, w_grad=0.3)
    elif stage == "stage1_refinement":
        loss = DepthRefinementLoss(w_depth=1.0, w_grad=0.5)
    elif stage == "stage2_joint":
        loss = DepthRefinementLoss(w_depth=1.0, w_grad=0.3)
    elif stage == "stage3_full":
        loss = DepthRefinementLoss(w_depth=1.0, w_grad=0.2)
    else:
        raise ValueError(f"Unknown stage: {stage}")
    return _apply_loss_overrides(loss, overrides)
