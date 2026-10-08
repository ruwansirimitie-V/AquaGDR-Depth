import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict

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


def _is_legacy_quality_key(key: str) -> bool:
    return (
        "stem_equi.quality." in key
        or "stem_cube.quality." in key
    )


def _is_legacy_dead_hard_key(key: str) -> bool:
    return "local_detail_branch.hard_region_head." in key


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


def load_checkpoint_flexible(
    model: torch.nn.Module,
    ckpt_path: str,
    device: torch.device,
    strict: bool = True,
) -> None:
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict):
        if isinstance(ckpt.get("model_state_dict"), dict):
            sd = ckpt["model_state_dict"]
        elif isinstance(ckpt.get("state_dict"), dict):
            sd = ckpt["state_dict"]
        else:
            sd = ckpt
    else:
        raise RuntimeError(f"Unsupported checkpoint format: {ckpt_path}")
    sd = _normalize_state_dict_keys(sd)
    if strict:
        model_keys = set(model.state_dict())
        dropped_quality = [
            key for key in sd
            if key not in model_keys and _is_legacy_quality_key(key)
        ]
        dropped_hard = [
            key for key in sd
            if key not in model_keys and _is_legacy_dead_hard_key(key)
        ]
        if dropped_quality:
            sd = {key: value for key, value in sd.items() if key not in dropped_quality}
            print(
                f"[Checkpoint] Dropped {len(dropped_quality)} legacy dead-Q tensors."
            )
        if dropped_hard:
            sd = {key: value for key, value in sd.items() if key not in dropped_hard}
            print(
                f"[Checkpoint] Dropped {len(dropped_hard)} legacy dead hard-head tensors."
            )
        msg = model.load_state_dict(sd, strict=True)
    else:
        model_sd = model.state_dict()
        selected = {}
        adapted = []
        skipped = []
        for k, v in sd.items():
            if k not in model_sd or not torch.is_tensor(v):
                continue
            if tuple(v.shape) == tuple(model_sd[k].shape):
                selected[k] = v
                continue
            adapted_v = _adapt_q_free_gate_weight(k, v, model_sd[k])
            if adapted_v is not None:
                selected[k] = adapted_v
                adapted.append(k)
            else:
                skipped.append(k)
        if not selected:
            raise RuntimeError(f"No matched keys found when loading {ckpt_path}")
        model_sd.update(selected)
        msg = model.load_state_dict(model_sd, strict=True)
        print(f"[Checkpoint] Loaded compatible tensors: {len(selected)}")
        if adapted:
            print(f"[Checkpoint] Adapted {len(adapted)} Q-free DAGF gate tensors by dropping the Q input channel.")
        if skipped:
            print(f"[Checkpoint] Skipped shape-mismatched tensors: {len(skipped)}")
    print(f"[Checkpoint] Loaded from: {ckpt_path}")
    if msg.missing_keys:
        print(f"[Checkpoint] Missing keys: {len(msg.missing_keys)}")
    if msg.unexpected_keys:
        print(f"[Checkpoint] Unexpected keys: {len(msg.unexpected_keys)}")


def build_dataloader(cfg: Dict, pano_root: Path, split: str):
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

    list_key = {"train": "train_list", "val": "val_list", "test": "test_list"}[split]
    default_list = cfg["data"].get("val_list", str(pano_root / "datasets" / f"{dataset_name}_val.txt"))
    list_file = Path(cfg["data"].get(list_key, default_list))
    if not list_file.is_absolute():
        list_file = (PROJECT_ROOT / list_file).resolve()
    if not list_file.exists():
        raise FileNotFoundError(f"Cannot find split file: {list_file}")

    ds = datasets_dict[dataset_name](data_path, str(list_file), height, width, True, True, True, is_training=False)
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)


def _to_float(x):
    if torch.is_tensor(x):
        return float(x.detach().cpu().item())
    return float(x)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser("Evaluate UDBF-Depth-v3")
    parser.add_argument("--config", type=str, required=True, help="Path to yaml config")
    parser.add_argument("--ckpt", type=str, default="", help="Checkpoint path. Overrides eval.ckpt in config.")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"])
    parser.add_argument("--median_align", action="store_true")
    parser.add_argument(
        "--allow_partial_ckpt",
        action="store_true",
        help="Allow missing or unexpected checkpoint keys. Use only for debugging incompatible checkpoints.",
    )
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(int(cfg.get("seed", 42)))
    pano_root = append_third_party_path()

    from aquagdr.models.udbf_depth_model import (
        UDBFDepthModel,
        build_global_context_branch,
        build_local_detail_branch,
        build_pano_backbone,
    )
    from aquagdr.modules.qgrr_head import QualityGuidedResidualRefinement
    from aquagdr.utils.metrics import MetricsTracker, compute_depth_metrics, print_depth_metrics

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loader = build_dataloader(cfg, pano_root, split=args.split)

    model_cfg = cfg["model"]
    pano_backbone = build_pano_backbone(
        num_layers=int(model_cfg["num_layers"]),
        height=int(cfg["data"]["height"]),
        width=int(cfg["data"]["width"]),
        max_depth=float(model_cfg.get("max_depth", 10.0)),
        fusion_type=model_cfg.get("fusion_type", "gated"),
        se_in_fusion=bool(model_cfg.get("se_in_fusion", False)),
        pretrained=False,
        use_input_adaptation=bool(model_cfg.get("input_adaptation", True)),
    ).to(device)

    backbone_weight = cfg.get("pano_backbone", {}).get("weight_file", "")
    if backbone_weight:
        try:
            load_checkpoint_flexible(pano_backbone, backbone_weight, device, strict=False)
        except Exception as exc:
            print(f"[Warn] Failed to load panoramic-backbone weights, continue with current weights: {exc}")

    qgrr_cfg = model_cfg.get("qgrr", {})
    refinement_head = QualityGuidedResidualRefinement(
        max_depth=float(model_cfg.get("max_depth", 10.0)),
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
    model.set_train_stage(cfg.get("train", {}).get("stage", "stage2_joint"))
    model.eval()

    ckpt_path = args.ckpt or cfg.get("eval", {}).get("ckpt", "")
    if not ckpt_path:
        raise RuntimeError("Please provide --ckpt or set eval.ckpt in config.")
    load_checkpoint_flexible(model, ckpt_path, device, strict=not args.allow_partial_ckpt)

    tracker = MetricsTracker()
    initial_tracker = MetricsTracker()
    sample_count = 0

    for batch in loader:
        equi = batch["normalized_rgb"].to(device, non_blocking=True)
        cube = batch["normalized_cube_rgb"].to(device, non_blocking=True)
        gt = batch["gt_depth"]
        mask = batch["val_mask"]
        out = model(equi_rgb=equi, cube_rgb=cube, valid_mask=mask.to(device, non_blocking=True))
        pred = out["pred_depth"].detach().cpu()
        initial_pred = out["initial_depth"].detach().cpu()
        for i in range(gt.shape[0]):
            metrics = compute_depth_metrics(
                gt[i:i + 1],
                pred[i:i + 1],
                mask[i:i + 1],
                max_depth=float(model_cfg.get("max_depth", 10.0)),
                median_align=args.median_align,
            )
            tracker.update(metrics, n=1)
            initial_metrics = compute_depth_metrics(
                gt[i:i + 1],
                initial_pred[i:i + 1],
                mask[i:i + 1],
                max_depth=float(model_cfg.get("max_depth", 10.0)),
                median_align=args.median_align,
            )
            initial_tracker.update(initial_metrics, n=1)
            sample_count += 1

    result = {k: _to_float(v) for k, v in tracker.as_dict().items()}
    initial_result = {k: _to_float(v) for k, v in initial_tracker.as_dict().items()}
    print("\n[Initial depth D0]")
    print_depth_metrics(initial_result)
    print("\n[Refined depth]")
    print_depth_metrics(result)

    out_dir = Path(cfg.get("eval", {}).get("output_dir", str(PROJECT_ROOT / "outputs" / "metrics")))
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "_median_align" if args.median_align else ""
    out_file = out_dir / f"eval_{args.split}{suffix}.json"
    payload = {
        "config": args.config,
        "ckpt": ckpt_path,
        "split": args.split,
        "median_align": bool(args.median_align),
        "num_samples": sample_count,
        "metric_implementation": "aquagdr.utils.metrics.compute_depth_metrics",
        "initial_metrics": initial_result,
        "metrics": result,
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"[Saved] {out_file}")


if __name__ == "__main__":
    main()
