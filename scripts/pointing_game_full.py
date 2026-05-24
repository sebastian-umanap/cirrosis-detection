"""Pointing-game audit a gran escala (no solo 3 casos).

Para CADA paciente de validación de un fold (o de todos los folds), calcula:
  - pointing_inside: ¿el argmax del Grad-CAM cae dentro de la máscara hepática?
  - cam_mass_inside: fracción de la masa del CAM (>0.5) dentro de la máscara.
  - liver_frac: proporción del corte ocupada por el hígado (baseline aleatorio del pointing-game).

Agrega por clase (cirrótico/sano) y por corrección (TP/FP/TN/FN), dando un N robusto
para sostener (o matizar) la afirmación de shortcut learning.

Uso:
    python scripts/pointing_game_full.py --folds 0 1 2 3 4 --max-per-fold 0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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

CACHE = PROJECT / "data" / "cache"
OUT_HW = (224, 224)
N_SLICES = 3


def _build_stack(cached):
    img = cached.image.astype(np.float32)
    mask = cached.mask.astype(np.uint8)
    if mask.sum() > 0:
        coords = np.argwhere(mask > 0)
        mn = coords.min(axis=0); mx = coords.max(axis=0) + 1
        sl = tuple(slice(max(0, mn[i] - 16), min(img.shape[i], mx[i] + 16)) for i in range(3))
        img = img[sl]; mask = mask[sl]
    depth = img.shape[-1]
    z_with = np.unique(np.argwhere(mask > 0)[:, -1]) if mask.sum() > 0 else np.arange(depth)
    center_z = int(np.median(z_with))
    half = N_SLICES // 2
    z0 = max(0, center_z - half); z1 = min(depth, z0 + N_SLICES); z0 = max(0, z1 - N_SLICES)
    import torch.nn.functional as F
    stack = []
    for z in range(z0, z1):
        t = torch.from_numpy(img[:, :, z]).float().unsqueeze(0).unsqueeze(0)
        stack.append(F.interpolate(t, size=OUT_HW, mode="bilinear", align_corners=False).squeeze().numpy())
    while len(stack) < N_SLICES:
        stack.append(stack[-1])
    arr = np.stack(stack, axis=0).astype(np.float32)
    cz = center_z if center_z < mask.shape[-1] else mask.shape[-1] - 1
    m2 = mask[..., cz].astype(np.float32)
    t = torch.from_numpy(m2).float().unsqueeze(0).unsqueeze(0)
    m2r = (F.interpolate(t, size=OUT_HW, mode="nearest").squeeze().numpy() > 0.5).astype(np.uint8)
    return arr, m2r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, nargs="+", default=[0])
    ap.add_argument("--max-per-fold", type=int, default=0, help="0 = todos")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inv = build_inventory(PROJECT / "data" / "CirrMRI600plus_raw")
    recs = inv.filter(task="binary")
    folds = list(make_stratified_group_kfold(recs, task="binary", n_splits=5, seed=42))

    rows = []
    for fid in args.folds:
        fold = folds[fid]
        ckpt = PROJECT / "reports" / "checkpoints" / f"resnet25d_fold{fid}.pt"
        if not ckpt.exists():
            print(f"skip fold {fid}: no checkpoint"); continue
        model = build_resnet25d(num_classes=1, in_channels=3, pretrained=False).to(device).eval()
        state = torch.load(ckpt, map_location=device)
        model.load_state_dict(state["model"] if "model" in state else state)
        cam = GradCAM(model, model.backbone.layer4)

        val_recs = [recs[i] for i in fold.val_idx]
        if args.max_per_fold:
            val_recs = val_recs[: args.max_per_fold]
        for rec in val_recs:
            c = load_cached(CACHE, rec.patient_id, rec.modality)
            if c is None:
                continue
            x_np, m2r = _build_stack(c)
            xt = torch.from_numpy(x_np).float().unsqueeze(0).to(device)
            with torch.no_grad():
                proba = float(torch.sigmoid(model(xt).squeeze()).cpu())
            out = cam(xt)
            cam_full = upsample_to(out.cam, m2r.shape)
            pi = pointing_game(cam_full, m2r)
            cm = mass_inside_mask(cam_full, m2r)
            liver_frac = float((m2r > 0).mean())
            label = int(rec.is_cirrhotic)
            pred = int(proba >= 0.5)
            rows.append({
                "patient_id": rec.patient_id, "modality": rec.modality, "fold": fid,
                "label": label, "pred": pred, "proba": proba,
                "pointing_inside": bool(pi),
                "cam_mass_inside": float(cm) if np.isfinite(cm) else np.nan,
                "liver_frac": liver_frac,
            })
        print(f"fold {fid}: {len([r for r in rows if r['fold']==fid])} casos evaluados")

    df = pd.DataFrame(rows)
    out_csv = PROJECT / "reports" / "tables" / "pointing_game_full.csv"
    df.to_csv(out_csv, index=False)

    def _summ(sub, name):
        if len(sub) == 0:
            return
        print(f"  {name:28s} n={len(sub):3d}  pointing_acc={sub['pointing_inside'].mean():.3f}  "
              f"cam_mass_inside={sub['cam_mass_inside'].mean():.3f}  liver_frac={sub['liver_frac'].mean():.3f}")

    print("\n=== POINTING-GAME (sobre validación completa) ===")
    _summ(df, "TODOS")
    _summ(df[df.label == 1], "Cirróticos (label=1)")
    _summ(df[df.label == 0], "Sanos (label=0)")
    _summ(df[(df.label == 1) & (df.pred == 1)], "TP (cirr correctos)")
    _summ(df[(df.label == 0) & (df.pred == 1)], "FP (sano->cirr)")
    _summ(df[(df.label == 0) & (df.pred == 0)], "TN (sano correctos)")
    _summ(df[(df.label == 1) & (df.pred == 0)], "FN (cirr->sano)")
    print(f"\nGuardado: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
