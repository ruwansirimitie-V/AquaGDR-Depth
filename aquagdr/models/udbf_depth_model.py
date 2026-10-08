import sys
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn

from aquagdr.modules.qgrr_head import QualityGuidedResidualRefinement


def _append_third_party_path() -> None:
    root = Path(__file__).resolve().parents[2]
    pano_root = root / "third_party" / "unifuse"
    s = str(pano_root)
    while s in sys.path:
        sys.path.remove(s)
    sys.path.insert(0, s)


def build_pano_backbone(
    num_layers: int,
    height: int,
    width: int,
    max_depth: float = 10.0,
    fusion_type: str = "gated",
    se_in_fusion: bool = False,
    pretrained: bool = False,
    use_input_adaptation: bool = True,
):
    _append_third_party_path()
    from networks import UniFuse

    return UniFuse(
        num_layers=num_layers,
        equi_h=height,
        equi_w=width,
        pretrained=pretrained,
        max_depth=max_depth,
        fusion_type=fusion_type,
        se_in_fusion=se_in_fusion,
        use_input_adaptation=use_input_adaptation,
    )


def build_global_context_branch(model_cfg: Dict) -> Optional[nn.Module]:
    """Return no global-context branch in the released final architecture."""
    del model_cfg
    return None


def build_local_detail_branch(model_cfg: Dict) -> Optional[nn.Module]:
    detail_cfg = model_cfg.get("local_detail", {}) or {}
    if not bool(detail_cfg.get("enabled", False)):
        return None
    from aquagdr.modules.local_detail_branch import LocalDetailGuidanceBranch

    return LocalDetailGuidanceBranch(
        max_depth=float(model_cfg.get("max_depth", 10.0)),
        base_channels=int(detail_cfg.get("base_channels", 12)),
        out_channels=int(detail_cfg.get("out_channels", 16)),
        num_blocks=int(detail_cfg.get("num_blocks", 2)),
        hard_region_bias=float(detail_cfg.get("hard_region_bias", -1.5)),
        circular_edge=bool(detail_cfg.get("circular_edge", False)),
        use_edge=bool(detail_cfg.get("use_edge", True)),
        use_low_resolution=bool(detail_cfg.get("use_low_resolution", True)),
        circular_padding=bool(detail_cfg.get("circular_padding", True)),
        predict_hard_region=bool(detail_cfg.get("predict_hard_region", True)),
    )


class AquaGDRDepthModel(nn.Module):
    """AquaGDR-Depth backbone with local-detail residual refinement."""

    def __init__(
        self,
        pano_backbone: nn.Module,
        refinement_head: Optional[nn.Module] = None,
        global_context_branch: Optional[nn.Module] = None,
        local_detail_branch: Optional[nn.Module] = None,
        joint_trainable_keywords=("equi_decoder",),
    ):
        super().__init__()
        self.pano_backbone = pano_backbone
        self.refinement_head = refinement_head or QualityGuidedResidualRefinement()
        self.global_context_branch = global_context_branch
        self.local_detail_branch = local_detail_branch
        self.joint_trainable_keywords = joint_trainable_keywords
        self.train_stage = None

    @staticmethod
    def _set_requires_grad(module: Optional[nn.Module], flag: bool) -> None:
        if module is None:
            return
        for p in module.parameters():
            p.requires_grad = flag

    def set_train_stage(self, stage: str) -> None:
        stage = stage.lower()
        self.train_stage = stage

        if stage == "stage_backbone_finetune":
            self._set_requires_grad(self.pano_backbone, False)
            self._set_requires_grad(self.refinement_head, False)
            self._set_requires_grad(self.global_context_branch, False)
            self._set_requires_grad(self.local_detail_branch, False)
            for name, p in self.pano_backbone.named_parameters():
                if any(k in name for k in self.joint_trainable_keywords):
                    p.requires_grad = True
            return

        if stage == "stage_local_refinement":
            self._set_requires_grad(self.pano_backbone, False)
            self._set_requires_grad(self.refinement_head, False)
            self._set_requires_grad(self.global_context_branch, False)
            self._set_requires_grad(self.local_detail_branch, True)
            detail_modules = [
                getattr(self.refinement_head, name, None)
                for name in ("encoder", "detail_encoder", "detail_gate_head", "detail_delta_head")
            ]
            detail_modules = [module for module in detail_modules if module is not None]
            needs_local_branch = int(getattr(self.refinement_head, "detail_channels", 0)) > 0
            if (needs_local_branch and self.local_detail_branch is None) or not detail_modules:
                raise ValueError(
                    "stage_local_refinement requires its configured local inputs and refinement heads"
                )
            for module in detail_modules:
                self._set_requires_grad(module, True)
            return

        if stage == "stage_local_addon":
            self._set_requires_grad(self.pano_backbone, False)
            self._set_requires_grad(self.refinement_head, False)
            self._set_requires_grad(self.global_context_branch, False)
            self._set_requires_grad(self.local_detail_branch, True)
            detail_modules = [
                getattr(self.refinement_head, name, None)
                for name in ("detail_encoder", "detail_gate_head", "detail_delta_head")
            ]
            if self.local_detail_branch is None or any(module is None for module in detail_modules):
                raise ValueError(
                    "stage_local_addon requires the local-detail branch and all detail refinement heads"
                )
            for module in detail_modules:
                self._set_requires_grad(module, True)
            return

        if stage == "stage1_refinement":
            self._set_requires_grad(self.pano_backbone, False)
            self._set_requires_grad(self.refinement_head, True)
            self._set_requires_grad(self.global_context_branch, True)
            self._set_requires_grad(self.local_detail_branch, True)
            return

        if stage.startswith("stage2_"):
            self._set_requires_grad(self.pano_backbone, False)
            self._set_requires_grad(self.refinement_head, True)
            self._set_requires_grad(self.global_context_branch, True)
            self._set_requires_grad(self.local_detail_branch, True)
            for name, p in self.pano_backbone.named_parameters():
                if any(k in name for k in self.joint_trainable_keywords):
                    p.requires_grad = True
            return

        if stage == "stage3_full":
            self._set_requires_grad(self.pano_backbone, True)
            self._set_requires_grad(self.refinement_head, True)
            self._set_requires_grad(self.global_context_branch, True)
            self._set_requires_grad(self.local_detail_branch, True)
            return

        raise ValueError(f"Unknown stage: {stage}")

    def get_trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]

    def forward(
        self,
        equi_rgb: torch.Tensor,
        cube_rgb: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        **unused,
    ) -> Dict[str, torch.Tensor]:
        backbone_outputs = self.pano_backbone(equi_rgb, cube_rgb)
        initial_depth = backbone_outputs["pred_depth"]
        quality_map = backbone_outputs.get("quality_map")
        global_context = None
        if self.global_context_branch is not None:
            global_context = self.global_context_branch(cube_rgb)

        local_outputs = None
        if self.local_detail_branch is not None:
            # Local guidance treats the frozen initial depth as an input cue.
            local_outputs = self.local_detail_branch(equi_rgb, initial_depth.detach())

        refinement_outputs = self.refinement_head(
            initial_depth=initial_depth,
            quality_map=quality_map,
            rgb=equi_rgb,
            valid_mask=valid_mask,
            global_context=global_context,
            local_detail=None if local_outputs is None else local_outputs["detail_features"],
            hard_region=None if local_outputs is None else local_outputs["hard_region"],
            hard_region_logits=None if local_outputs is None else local_outputs["hard_region_logits"],
        )

        outputs = dict(backbone_outputs)
        outputs["initial_depth"] = initial_depth
        outputs["pred_depth"] = refinement_outputs["refined_depth"]
        outputs["refined_depth"] = refinement_outputs["refined_depth"]
        outputs["residual_delta"] = refinement_outputs["residual_delta"]
        outputs["refinement_mask"] = refinement_outputs["refinement_mask"]
        outputs["coarse_correction"] = refinement_outputs["coarse_correction"]
        outputs["detail_residual_delta"] = refinement_outputs["detail_residual_delta"]
        outputs["detail_gate"] = refinement_outputs["detail_gate"]
        outputs["detail_correction"] = refinement_outputs["detail_correction"]
        outputs["total_correction"] = refinement_outputs["total_correction"]
        outputs["hard_region"] = refinement_outputs["hard_region"]
        outputs["hard_region_logits"] = refinement_outputs["hard_region_logits"]
        outputs["q_bad"] = refinement_outputs["q_bad"]
        if global_context is not None:
            outputs["global_context"] = global_context
        if local_outputs is not None:
            outputs["local_detail"] = local_outputs["detail_features"]
        return outputs


# Compatibility alias for checkpoints and configs created in the research
# workspace, where the wrapper kept its historical class name.
UDBFDepthModel = AquaGDRDepthModel
