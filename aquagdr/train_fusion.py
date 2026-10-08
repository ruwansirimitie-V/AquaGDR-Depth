import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def append_third_party_path() -> Path:
    pano_root = PROJECT_ROOT / "third_party" / "unifuse"
    s = str(pano_root)
    while s in sys.path:
        sys.path.remove(s)
    sys.path.insert(0, s)
    return pano_root


def load_yaml(path: str) -> Dict:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("Missing dependency pyyaml. Please run: pip install pyyaml") from exc
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _normalize_state_dict_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k[7:] if k.startswith("module.") else k: v for k, v in state_dict.items()}


def _adapt_q_free_gate_weight(key: str, value: torch.Tensor, target: torch.Tensor):
    if not key.endswith("gate_net.0.weight"):
        return None
    if value.ndim != 4 or target.ndim != 4:
        return None
    if value.shape[0] != target.shape[0] or value.shape[2:] != target.shape[2:]:
        return None
    if value.shape[1] == target.shape[1] + 1:
        return value[:, : target.shape[1], :, :].contiguous()
    return None


def load_model_weights_flexible(model: torch.nn.Module, ckpt_path: str, device: torch.device) -> int:
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict):
        if isinstance(ckpt.get("state_dict"), dict):
            candidate = ckpt["state_dict"]
        elif isinstance(ckpt.get("model_state_dict"), dict):
            candidate = ckpt["model_state_dict"]
        else:
            candidate = ckpt
    else:
        raise RuntimeError(f"Unsupported checkpoint type in: {ckpt_path}")

    candidate = _normalize_state_dict_keys(candidate)
    model_sd = model.state_dict()
    filtered = {}
    adapted = []
    for k, v in candidate.items():
        if k not in model_sd or not torch.is_tensor(v):
            continue
        if tuple(v.shape) == tuple(model_sd[k].shape):
            filtered[k] = v
            continue
        adapted_v = _adapt_q_free_gate_weight(k, v, model_sd[k])
        if adapted_v is not None:
            filtered[k] = adapted_v
            adapted.append(k)
    if not filtered:
        raise RuntimeError(f"No matched keys found when loading {ckpt_path}")
    model_sd.update(filtered)
    model.load_state_dict(model_sd, strict=True)
    if adapted:
        print(f"[Init] Adapted {len(adapted)} Q-free DAGF gate tensors by dropping the Q input channel.")
    return len(filtered)


def load_resume_state_dict_flexible(model: torch.nn.Module, state_dict: Dict[str, torch.Tensor]) -> Tuple[int, Dict[str, Tuple]]:
    model_sd = model.state_dict()
    loaded = {}
    skipped = {}
    for k, v in _normalize_state_dict_keys(state_dict).items():
        if k not in model_sd or not torch.is_tensor(v):
            continue
        if tuple(v.shape) != tuple(model_sd[k].shape):
            skipped[k] = (tuple(v.shape), tuple(model_sd[k].shape))
            continue
        loaded[k] = v
    model_sd.update(loaded)
    model.load_state_dict(model_sd, strict=True)
    return len(loaded), skipped


def build_dataloaders(cfg: Dict, pano_root: Path):
    import datasets

    dataset_name = cfg["data"]["dataset"]
    height = int(cfg["data"]["height"])
    width = int(cfg["data"]["width"])
    data_path = Path(cfg["data"]["data_path"])
    if not data_path.is_absolute():
        data_path = (PROJECT_ROOT / data_path).resolve()
    data_path = str(data_path)
    batch_size = int(cfg["data"]["batch_size"])
    num_workers = int(cfg["data"]["num_workers"])

    datasets_dict = {
        "3d60": datasets.ThreeD60,
        "panosuncg": datasets.PanoSunCG,
        "stanford2d3d": datasets.Stanford2D3D,
        "matterport3d": datasets.Matterport3D,
    }
    if dataset_name not in datasets_dict:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    dataset_cls = datasets_dict[dataset_name]
    train_txt = cfg["data"].get("train_list", str(pano_root / "datasets" / f"{dataset_name}_train.txt"))
    val_txt = cfg["data"].get("val_list", str(pano_root / "datasets" / f"{dataset_name}_val.txt"))
    train_txt = str((PROJECT_ROOT / train_txt).resolve()) if not Path(train_txt).is_absolute() else train_txt
    val_txt = str((PROJECT_ROOT / val_txt).resolve()) if not Path(val_txt).is_absolute() else val_txt

    train_ds = dataset_cls(
        data_path,
        train_txt,
        height,
        width,
        cfg["data"].get("disable_color_augmentation", False),
        cfg["data"].get("disable_LR_filp_augmentation", False),
        cfg["data"].get("disable_yaw_rotation_augmentation", False),
        is_training=True,
    )
    val_ds = dataset_cls(data_path, val_txt, height, width, True, True, True, is_training=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)
    return train_loader, val_loader, float(train_ds.max_depth_meters)


def set_train_mode_for_stage(model):
    model.train()
    stage = getattr(model, "train_stage", None)

    if stage == "stage_backbone_finetune":
        model.pano_backbone.eval()
        model.refinement_head.eval()
        if getattr(model, "global_context_branch", None) is not None:
            model.global_context_branch.eval()
        if getattr(model, "local_detail_branch", None) is not None:
            model.local_detail_branch.eval()
        for module_name, module in model.pano_backbone.named_modules():
            if any(k in module_name for k in model.joint_trainable_keywords):
                module.train()
        return

    if stage == "stage_local_refinement":
        model.pano_backbone.eval()
        model.refinement_head.eval()
        if getattr(model, "global_context_branch", None) is not None:
            model.global_context_branch.eval()
        if model.local_detail_branch is not None:
            model.local_detail_branch.train()
        for name in ("encoder", "detail_encoder", "detail_gate_head", "detail_delta_head"):
            module = getattr(model.refinement_head, name, None)
            if module is not None:
                module.train()
        return

    if stage == "stage_local_addon":
        model.pano_backbone.eval()
        model.refinement_head.eval()
        if getattr(model, "global_context_branch", None) is not None:
            model.global_context_branch.eval()
        model.local_detail_branch.train()
        for name in ("detail_encoder", "detail_gate_head", "detail_delta_head"):
            getattr(model.refinement_head, name).train()
        return

    if stage == "stage1_refinement":
        model.pano_backbone.eval()
        model.refinement_head.train()
        if getattr(model, "global_context_branch", None) is not None:
            model.global_context_branch.train()
        if getattr(model, "local_detail_branch", None) is not None:
            model.local_detail_branch.train()
        return

    if stage and stage.startswith("stage2_"):
        model.pano_backbone.eval()
        model.refinement_head.train()
        if getattr(model, "global_context_branch", None) is not None:
            model.global_context_branch.train()
        if getattr(model, "local_detail_branch", None) is not None:
            model.local_detail_branch.train()
        for module_name, module in model.pano_backbone.named_modules():
            if any(k in module_name for k in model.joint_trainable_keywords):
                module.train()
        return


def train_one_epoch(model, loader, criterion, optimizer, device, use_amp=False):
    set_train_mode_for_stage(model)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    total = 0.0
    n = 0
    term_totals = {}
    for batch in loader:
        equi = batch["normalized_rgb"].to(device, non_blocking=True)
        cube = batch["normalized_cube_rgb"].to(device, non_blocking=True)
        gt = batch["gt_depth"].to(device, non_blocking=True)
        mask = batch["val_mask"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            outputs = model(equi_rgb=equi, cube_rgb=cube, valid_mask=mask)
            loss, loss_terms = criterion(outputs, gt_depth=gt, valid_mask=mask)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total += float(loss.detach().cpu().item())
        for name, value in loss_terms.items():
            term_totals[name] = term_totals.get(name, 0.0) + float(value.detach().cpu().item())
        n += 1
    denom = max(n, 1)
    return total / denom, {name: value / denom for name, value in term_totals.items()}


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total = 0.0
    n = 0
    term_totals = {}
    for batch in loader:
        equi = batch["normalized_rgb"].to(device, non_blocking=True)
        cube = batch["normalized_cube_rgb"].to(device, non_blocking=True)
        gt = batch["gt_depth"].to(device, non_blocking=True)
        mask = batch["val_mask"].to(device, non_blocking=True)
        outputs = model(equi_rgb=equi, cube_rgb=cube, valid_mask=mask)
        loss, loss_terms = criterion(outputs, gt_depth=gt, valid_mask=mask)
        total += float(loss.detach().cpu().item())
        for name, value in loss_terms.items():
            term_totals[name] = term_totals.get(name, 0.0) + float(value.detach().cpu().item())
        n += 1
    denom = max(n, 1)
    return total / denom, {name: value / denom for name, value in term_totals.items()}


def main():
    parser = argparse.ArgumentParser("Train UDBF-Depth-v3")
    parser.add_argument("--config", type=str, required=True, help="Path to yaml config")
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="Run one forward/backward batch without saving or updating weights.",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(int(cfg.get("seed", 42)))
    pano_root = append_third_party_path()

    from aquagdr.losses.refinement_losses import build_refinement_loss
    from aquagdr.models.udbf_depth_model import (
        UDBFDepthModel,
        build_global_context_branch,
        build_local_detail_branch,
        build_pano_backbone,
    )
    from aquagdr.modules.qgrr_head import QualityGuidedResidualRefinement

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, max_depth = build_dataloaders(cfg, pano_root)

    model_cfg = cfg["model"]
    stage = cfg["train"]["stage"]
    out_dir = Path(cfg["train"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    pano_backbone = build_pano_backbone(
        num_layers=int(model_cfg["num_layers"]),
        height=int(cfg["data"]["height"]),
        width=int(cfg["data"]["width"]),
        max_depth=float(model_cfg.get("max_depth", max_depth)),
        fusion_type=model_cfg.get("fusion_type", "gated"),
        se_in_fusion=bool(model_cfg.get("se_in_fusion", False)),
        pretrained=bool(model_cfg.get("imagenet_pretrained", False)),
        use_input_adaptation=bool(model_cfg.get("input_adaptation", True)),
    ).to(device)

    backbone_weight = cfg.get("pano_backbone", {}).get("weight_file", "")
    if backbone_weight:
        loaded = load_model_weights_flexible(pano_backbone, backbone_weight, device)
        print(f"[Init] Loaded {loaded} params from panoramic-backbone checkpoint: {backbone_weight}")

    qgrr_cfg = model_cfg.get("qgrr", {})
    refinement_head = QualityGuidedResidualRefinement(
        max_depth=float(model_cfg.get("max_depth", max_depth)),
        min_depth=float(qgrr_cfg.get("min_depth", 1e-3)),
        use_coarse_refinement=bool(qgrr_cfg.get("use_coarse_refinement", True)),
        residual_cap=float(qgrr_cfg.get("residual_cap", 2.0)),
        lowpass_k=int(qgrr_cfg.get("lowpass_k", 5)),
        use_rgb=bool(qgrr_cfg.get("use_rgb", True)),
        use_edge=bool(qgrr_cfg.get("use_edge", True)),
        use_quality=bool(qgrr_cfg.get("use_quality", False)),
        use_global_context=bool(qgrr_cfg.get("use_global_context", False)),
        context_channels=int(qgrr_cfg.get("context_channels", 0)),
        use_local_detail=bool(qgrr_cfg.get("use_local_detail", False)),
        detail_channels=int(qgrr_cfg.get("detail_channels", 0)),
        detail_residual_cap=float(qgrr_cfg.get("detail_residual_cap", 0.5)),
        use_detail_mask=bool(qgrr_cfg.get("use_detail_mask", True)),
        bounded_detail_residual=bool(qgrr_cfg.get("bounded_detail_residual", True)),
        use_hard_gate=bool(qgrr_cfg.get("use_hard_gate", True)),
        erp_circular=bool(qgrr_cfg.get("erp_circular", False)),
    )

    global_context_branch = build_global_context_branch(model_cfg)
    if global_context_branch is not None:
        global_context_branch = global_context_branch.to(device)

    local_detail_branch = build_local_detail_branch(model_cfg)
    if local_detail_branch is not None:
        local_detail_branch = local_detail_branch.to(device)

    model = UDBFDepthModel(
        pano_backbone=pano_backbone,
        refinement_head=refinement_head,
        global_context_branch=global_context_branch,
        local_detail_branch=local_detail_branch,
        joint_trainable_keywords=tuple(model_cfg.get("joint_trainable_keywords", ["equi_decoder"])),
    ).to(device)
    model.set_train_stage(stage)

    resume_ckpt = cfg.get("train", {}).get("resume_ckpt", "")
    if resume_ckpt:
        ckpt = torch.load(resume_ckpt, map_location=device)
        sd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt
        loaded, skipped = load_resume_state_dict_flexible(model, sd)
        print(f"[Resume] loaded from: {resume_ckpt}")
        print(f"[Resume] loaded_keys={loaded} skipped_shape_mismatch={len(skipped)}")
        for name, (ckpt_shape, model_shape) in list(skipped.items())[:10]:
            print(f"[Resume] skip {name}: ckpt={ckpt_shape} model={model_shape}")

    if bool(cfg.get("train", {}).get("zero_init_detail_delta", False)):
        if stage not in {"stage_local_addon", "stage_local_refinement"}:
            raise ValueError(
                "zero_init_detail_delta is only supported for "
                "stage_local_addon or stage_local_refinement"
            )
        torch.nn.init.zeros_(model.refinement_head.detail_delta_head.weight)
        if model.refinement_head.detail_delta_head.bias is not None:
            torch.nn.init.zeros_(model.refinement_head.detail_delta_head.bias)
        print("[Init] Zero-initialized the local add-on detail delta head.")

    criterion = build_refinement_loss(stage, cfg.get("loss"))
    optimizer = torch.optim.Adam(
        model.get_trainable_params(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"].get("weight_decay", 0.0)),
    )
    use_amp = bool(cfg["train"].get("amp", False))
    epochs = int(cfg["train"]["epochs"])
    save_every = int(cfg["train"].get("save_every", 1))
    select_by_primary_loss = bool(cfg["train"].get("select_by_primary_loss", False))

    if args.smoke_test:
        set_train_mode_for_stage(model)
        batch = next(iter(train_loader))
        equi = batch["normalized_rgb"].to(device, non_blocking=True)
        cube = batch["normalized_cube_rgb"].to(device, non_blocking=True)
        gt = batch["gt_depth"].to(device, non_blocking=True)
        mask = batch["val_mask"].to(device, non_blocking=True)
        outputs = model(equi_rgb=equi, cube_rgb=cube, valid_mask=mask)
        loss, loss_terms = criterion(outputs, gt_depth=gt, valid_mask=mask)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Smoke-test loss is not finite: {loss}")
        loss.backward()
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[SmokeTest] pred_depth={tuple(outputs['pred_depth'].shape)}")
        hard_region = outputs.get("hard_region")
        hard_shape = None if hard_region is None else tuple(hard_region.shape)
        print(f"[SmokeTest] hard_region={hard_shape}")
        print(f"[SmokeTest] trainable_params={trainable}")
        print(
            "[SmokeTest] "
            + " ".join(
                f"{name}={float(value.detach().cpu().item()):.6f}"
                for name, value in loss_terms.items()
            )
        )
        return

    with open(out_dir / "resolved_config.json", "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

    best_val = float("inf")
    for epoch in range(1, epochs + 1):
        tr_loss, tr_terms = train_one_epoch(model, train_loader, criterion, optimizer, device, use_amp=use_amp)
        va_loss, va_terms = validate(model, val_loader, criterion, device)
        detail_log = ""
        if float(getattr(criterion, "w_hard", 0.0)) > 0.0:
            detail_log = (
                f" val_hard={va_terms.get('loss_hard', 0.0):.6f}"
                f" hard_target={va_terms.get('hard_target_fraction', 0.0):.4f}"
                f" hard_pred={va_terms.get('hard_pred_fraction', 0.0):.4f}"
            )
        selection_loss = va_loss
        if select_by_primary_loss:
            selection_loss = (
                float(getattr(criterion, "w_depth", 1.0)) * va_terms["loss_depth"]
                + float(getattr(criterion, "w_grad", 0.0)) * va_terms["loss_grad"]
            )
        print(
            f"[Epoch {epoch:03d}] train_loss={tr_loss:.6f} "
            f"val_loss={va_loss:.6f} selection_loss={selection_loss:.6f}{detail_log}"
        )

        ckpt = {
            "epoch": epoch,
            "stage": stage,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": tr_loss,
            "val_loss": va_loss,
            "selection_loss": selection_loss,
            "selection_criterion": "primary_depth_gradient" if select_by_primary_loss else "total_loss",
            "train_loss_terms": tr_terms,
            "val_loss_terms": va_terms,
        }
        if epoch % save_every == 0:
            torch.save(ckpt, out_dir / f"epoch_{epoch:03d}.pth")
        if selection_loss < best_val:
            best_val = selection_loss
            torch.save(ckpt, out_dir / "best.pth")


if __name__ == "__main__":
    main()
