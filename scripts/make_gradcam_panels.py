"""Genera paneles de Grad-CAM para el paper §Resultados §Interpretabilidad.

Usa el mejor checkpoint del fold 0 (o el que se pase con --ckpt) para producir:
    reports/figures/gradcam_panels.png   — 3x3 con 3 TP + 3 FP + 3 FN
    reports/tables/gradcam_pointing.csv  — pointing-game accuracy + cam_mass_inside_mask por caso
    reports/tables/gradcam_summary.csv   — agregados (mean ± std)

Uso:
    python scripts/make_gradcam_panels.py --ckpt reports/checkpoints_pilot/resnet25d_fold0.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from src.data.cache import load_cached  # noqa: E402
from src.data.inventory import build_inventory  # noqa: E402
from src.data.splits import make_stratified_group_kfold  # noqa: E402
from src.evaluation.gradcam import GradCAM, mass_inside_mask, pointing_game, upsample_to  # noqa: E402
from src.models.cnn_2_5d import build_resnet25d  # noqa: E402


def _build_2_5d_stack(cached, n_slices: int = 3, out_hw: tuple[int, int] = (224, 224)):
    img = cached.image.astype(np.float32)
    mask = cached.mask.astype(np.uint8)
    # Crop alrededor de la máscara.
    if mask.sum() > 0:
        coords = np.argwhere(mask > 0)
        mn = coords.min(axis=0); mx = coords.max(axis=0) + 1
        sl = [slice(max(0, mn[i] - 16), min(img.shape[i], mx[i] + 16)) for i in range(3)]
        img = img[tuple(sl)]; mask = mask[tuple(sl)]
    depth = img.shape[-1]
    if mask.sum() > 0:
        z_with = np.unique(np.argwhere(mask > 0)[:, -1])
        center_z = int(np.median(z_with))
    else:
        center_z = depth // 2
    half = n_slices // 2
    z_start = max(0, center_z - half)
    z_end = min(depth, z_start + n_slices)
    z_start = max(0, z_end - n_slices)
    import torch.nn.functional as F
    stack = []
    for z in range(z_start, z_end):
        t = torch.from_numpy(img[:, :, z]).float().unsqueeze(0).unsqueeze(0)
        t = F.interpolate(t, size=out_hw, mode="bilinear", align_corners=False)
        stack.append(t.squeeze().numpy())
    while len(stack) < n_slices:
        stack.append(stack[-1])
    arr = np.stack(stack, axis=0).astype(np.float32)
    # Máscara del corte central, resize a out_hw.
    m_center = mask[..., center_z if center_z < mask.shape[-1] else mask.shape[-1] - 1].astype(np.float32)
    t = torch.from_numpy(m_center).float().unsqueeze(0).unsqueeze(0)
    t = F.interpolate(t, size=out_hw, mode="nearest")
    mask_center = (t.squeeze().numpy() > 0.5).astype(np.uint8)
    return arr, mask_center


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--fold-id", type=int, default=0)
    parser.add_argument("--n-per-bucket", type=int, default=3)
    parser.add_argument("--preds-csv", type=Path, default=PROJECT / "reports" / "tables" / "dl_2_5d_preds.csv")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_resnet25d(num_classes=1, in_channels=3, pretrained=False).to(device)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()
    cam = GradCAM(model, model.backbone.layer4)

    inv = build_inventory(PROJECT / "data" / "CirrMRI600plus_raw")
    recs = inv.filter(task="binary")
    fold = list(make_stratified_group_kfold(recs, task="binary", n_splits=5, seed=42))[args.fold_id]
    val_recs = [recs[i] for i in fold.val_idx]
    cache_dir = PROJECT / "data" / "cache"

    # Decide qué casos visualizar — usa preds.csv si existe, si no, top probas por bucket.
    bucket_recs: dict[str, list] = {"TP": [], "FP": [], "FN": []}
    if args.preds_csv.exists():
        preds = pd.read_csv(args.preds_csv)
        preds["pred"] = (preds["proba"] >= 0.5).astype(int)
        # Map patient_id → records (cirr es prefijo numérico, healthy es "healthy_<id>").
        recs_by_pid = {r.patient_id: r for r in val_recs}
        for _, row in preds.iterrows():
            pid = str(row["patient_id"]); lbl = int(row["label"]); pred = int(row["pred"])
            if pid not in recs_by_pid:
                continue
            r = recs_by_pid[pid]
            if pred == 1 and lbl == 1:
                bucket_recs["TP"].append((r, row["proba"]))
            elif pred == 1 and lbl == 0:
                bucket_recs["FP"].append((r, row["proba"]))
            elif pred == 0 and lbl == 1:
                bucket_recs["FN"].append((r, 1 - row["proba"]))
    else:
        # Sin preds.csv: solo predecimos en vivo sobre todos los val.
        with torch.no_grad():
            for r in val_recs:
                c = load_cached(cache_dir, r.patient_id, r.modality)
                if c is None: continue
                x, _ = _build_2_5d_stack(c)
                xt = torch.from_numpy(x).float().unsqueeze(0).to(device)
                p = float(torch.sigmoid(model(xt).squeeze()).cpu().item())
                pred = int(p >= 0.5)
                lbl = int(r.is_cirrhotic)
                key = "TP" if (pred == 1 and lbl == 1) else "FP" if (pred == 1 and lbl == 0) else "FN" if (pred == 0 and lbl == 1) else None
                if key:
                    bucket_recs[key].append((r, p))

    # Toma los más confiables.
    for k in bucket_recs:
        bucket_recs[k].sort(key=lambda t: t[1], reverse=True)
        bucket_recs[k] = bucket_recs[k][: args.n_per_bucket]

    rows_pointing = []
    fig, axes = plt.subplots(3, args.n_per_bucket, figsize=(4.5 * args.n_per_bucket, 12))
    axes = np.atleast_2d(axes)
    for i, key in enumerate(("TP", "FP", "FN")):
        for j, (r, p) in enumerate(bucket_recs[key]):
            c = load_cached(cache_dir, r.patient_id, r.modality)
            if c is None: continue
            x_np, mask_center = _build_2_5d_stack(c)
            xt = torch.from_numpy(x_np).float().unsqueeze(0).to(device)
            out = cam(xt)
            cam_full = upsample_to(out.cam, mask_center.shape)
            pi = pointing_game(cam_full, mask_center)
            mm = mass_inside_mask(cam_full, mask_center)
            rows_pointing.append({"patient_id": r.patient_id, "modality": r.modality, "bucket": key,
                                  "score": float(p), "pointing_inside": bool(pi),
                                  "cam_mass_inside_mask": float(mm) if np.isfinite(mm) else float("nan")})
            ax = axes[i, j] if axes.ndim == 2 else axes[j]
            ax.imshow(x_np[len(x_np) // 2], cmap="gray")
            ax.imshow(cam_full, cmap="jet", alpha=0.45)
            ax.contour(mask_center, levels=[0.5], colors="lime", linewidths=1.0)
            ax.set_title(f"{key} · {r.patient_id} ({r.modality}) p={p:.2f}", fontsize=10)
            ax.axis("off")
    fig.suptitle("Grad-CAM sobre layer4 — contorno verde = máscara hepática", y=1.0)
    fig.tight_layout()
    fig.savefig(PROJECT / "reports" / "figures" / "gradcam_panels.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    df_p = pd.DataFrame(rows_pointing)
    df_p.to_csv(PROJECT / "reports" / "tables" / "gradcam_pointing.csv", index=False)
    summary = df_p.groupby("bucket").agg(
        n=("patient_id", "count"),
        pointing_acc=("pointing_inside", "mean"),
        cam_mass_mean=("cam_mass_inside_mask", "mean"),
    ).reset_index()
    summary.to_csv(PROJECT / "reports" / "tables" / "gradcam_summary.csv", index=False)
    print(summary.to_string(index=False))
    print("\nfiguras: reports/figures/gradcam_panels.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
