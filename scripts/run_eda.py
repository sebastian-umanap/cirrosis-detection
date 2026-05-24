"""EDA scripted sobre CirrMRI600+.

Genera:
    reports/figures/eda_counts_modality_class.png
    reports/figures/eda_demographics.png
    reports/figures/eda_shape_spacing.png
    reports/figures/eda_intensities.png
    reports/figures/eda_axial_panels.png
    reports/figures/eda_mask_quality.png
    reports/figures/eda_severity_distribution.png
    reports/tables/eda_summary.csv
    reports/tables/eda_dimensional.csv
    reports/tables/eda_mask_quality.csv

Más rápido que el notebook porque NO renderiza la salida y NO incluye widgets.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

sns.set_theme(style="whitegrid", context="paper")
log = logging.getLogger("eda")


def _load_inventory(root: Path):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.data.inventory import build_inventory  # noqa: E402
    return build_inventory(root)


def section1_counts(df: pd.DataFrame, fig_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=df, x="modality", hue="is_cirrhotic", ax=ax)
    ax.set_title("Distribución de estudios por modalidad y clase")
    ax.set_ylabel("# estudios"); ax.set_xlabel("modalidad")
    ax.legend(title="Cirrótico", labels=["Sano", "Cirrótico"])
    fig.tight_layout(); fig.savefig(fig_dir / "eda_counts_modality_class.png", dpi=150)
    plt.close(fig)


def section2_demographics(df: pd.DataFrame, fig_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(data=df, x="age", hue="is_cirrhotic", kde=True, bins=25, ax=axes[0],
                 common_norm=False, stat="density")
    axes[0].set_title("Distribución de edad por clase")
    axes[0].set_xlabel("edad (años)"); axes[0].set_ylabel("densidad")
    sns.countplot(data=df.dropna(subset=["sex"]), x="sex", hue="is_cirrhotic", ax=axes[1])
    axes[1].set_title("Distribución de sexo por clase")
    fig.tight_layout(); fig.savefig(fig_dir / "eda_demographics.png", dpi=150)
    plt.close(fig)


def section3_severity(df: pd.DataFrame, fig_dir: Path) -> None:
    sub = df[df["is_cirrhotic"] & df["severity"].notna()].copy()
    sub["severity_label"] = sub["severity"].map({1: "Mild", 2: "Moderate", 3: "Severe"})
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=sub, x="severity_label", hue="modality",
                  order=["Mild", "Moderate", "Severe"], ax=ax)
    ax.set_title("Distribución de severidad radiológica (cirróticos)")
    ax.set_xlabel("Severidad radiológica"); ax.set_ylabel("# estudios")
    fig.tight_layout(); fig.savefig(fig_dir / "eda_severity_distribution.png", dpi=150)
    plt.close(fig)


def section4_dimensional(records: list, fig_dir: Path, tbl_dir: Path, n_sample: int = 60) -> pd.DataFrame:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from src.data.inventory import sanity_check_volume  # noqa: E402

    rng = np.random.default_rng(42)
    sample = rng.choice(records, size=min(n_sample, len(records)), replace=False)
    rows = []
    for r in tqdm(sample, desc="dims"):
        try:
            rows.append(sanity_check_volume(r))
        except Exception as exc:  # noqa: BLE001
            log.warning("sanity %s: %s", r.patient_id, exc)
    dim_df = pd.DataFrame(rows)
    dim_df.to_csv(tbl_dir / "eda_dimensional.csv", index=False)

    shapes = list(dim_df["shape"]); spacings = list(dim_df["spacing"])
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    titles_top = ["shape X", "shape Y", "shape Z"]
    titles_bot = ["spacing X (mm)", "spacing Y (mm)", "spacing Z (mm)"]
    for i, t in enumerate(titles_top):
        sns.histplot([s[i] for s in shapes], ax=axes[0, i], bins=20); axes[0, i].set_title(t)
    for i, t in enumerate(titles_bot):
        sns.histplot([s[i] for s in spacings], ax=axes[1, i], bins=20); axes[1, i].set_title(t)
    fig.suptitle(f"Distribución de shapes y spacings (n={len(dim_df)})")
    fig.tight_layout(); fig.savefig(fig_dir / "eda_shape_spacing.png", dpi=150)
    plt.close(fig)
    return dim_df


def section5_intensities(inventory, fig_dir: Path, n_per_group: int = 6) -> None:
    """Distribución de intensidades antes/después de normalizar — versión sin MONAI.

    Usa nibabel para leer y numpy para clipping de percentil + z-score, equivalente al
    pipeline real (ScaleIntensityRangePercentilesd + NormalizeIntensityd).
    """

    def _normalize(arr: np.ndarray, pct: tuple[float, float] = (0.5, 99.5)) -> np.ndarray:
        lo, hi = np.percentile(arr, pct)
        v = np.clip(arr, lo, hi)
        nz = v > 0
        if nz.any():
            mean = v[nz].mean(); std = v[nz].std() or 1.0
            return (v - mean) / std
        return v

    rng = np.random.default_rng(0)
    rows = []
    for mod in ("T1w", "T2w"):
        for cirr in (True, False):
            pool = [r for r in inventory.records if r.modality == mod and r.is_cirrhotic == cirr]
            recs = rng.choice(pool, size=min(n_per_group, len(pool)), replace=False)
            for rec in tqdm(recs, desc=f"hist {mod} {'cirr' if cirr else 'healthy'}"):
                raw_arr = nib.load(str(rec.image_path)).get_fdata().astype(np.float32)
                norm_arr = _normalize(raw_arr)
                # Sub-sample voxeles para que la KDE no se ahogue.
                raw_vals = raw_arr.ravel()[::500]
                norm_vals = norm_arr.ravel()[::500]
                rows.append({"modality": mod, "is_cirrhotic": cirr, "stage": "raw",  "values": raw_vals})
                rows.append({"modality": mod, "is_cirrhotic": cirr, "stage": "norm", "values": norm_vals})

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for col, mod in enumerate(["T1w", "T2w"]):
        for row, stage in enumerate(["raw", "norm"]):
            ax = axes[row, col]
            for r in rows:
                if r["modality"] != mod or r["stage"] != stage:
                    continue
                v = r["values"]
                v = v[v > np.percentile(v, 0.5)] if stage == "raw" else v
                color = "crimson" if r["is_cirrhotic"] else "steelblue"
                sns.kdeplot(v, ax=ax, lw=1.0, alpha=0.4, color=color,
                            label="cirrótico" if r["is_cirrhotic"] else "sano")
            ax.set_title(f"{mod} — {stage}")
            if row == 1: ax.set_xlim(-3, 3)
            ax.set_xlabel("intensidad"); ax.set_ylabel("densidad")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        if l not in seen: seen[l] = h
    fig.legend(seen.values(), seen.keys(), loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Distribución de intensidades por modalidad y clase — antes vs después de normalizar", y=1.05)
    fig.tight_layout(); fig.savefig(fig_dir / "eda_intensities.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def section6_panels(inventory, fig_dir: Path) -> None:
    def central_axial(rec):
        img = nib.load(str(rec.image_path)).get_fdata()
        if rec.mask_path is not None:
            m = nib.load(str(rec.mask_path)).get_fdata()
            if m.sum() > 0:
                z_with = np.unique(np.argwhere(m > 0)[:, -1])
                z = int(np.median(z_with))
                return img[..., z], m[..., z]
        z = img.shape[-1] // 2
        return img[..., z], None

    rng = np.random.default_rng(7)
    panels = []
    for cirr in (False, True):
        pool = [r for r in inventory.records if r.is_cirrhotic == cirr and r.modality == "T2w"]
        if not pool: continue
        chosen = rng.choice(pool, size=min(4, len(pool)), replace=False)
        panels.append((cirr, chosen))
    fig, axes = plt.subplots(len(panels), 4, figsize=(16, 4 * len(panels)))
    axes = np.atleast_2d(axes)
    for r_i, (cirr, recs) in enumerate(panels):
        for c_i, rec in enumerate(recs):
            img2d, mask2d = central_axial(rec)
            ax = axes[r_i, c_i]
            ax.imshow(img2d, cmap="gray")
            if mask2d is not None:
                ax.imshow(np.ma.masked_where(mask2d == 0, mask2d), cmap="autumn", alpha=0.35)
            ax.set_title(f"{'cirrótico' if cirr else 'sano'} · {rec.patient_id}")
            ax.axis("off")
    fig.suptitle("Cortes axiales centrados en hígado (T2w) — máscara hepática en rojo")
    fig.tight_layout(); fig.savefig(fig_dir / "eda_axial_panels.png", dpi=150)
    plt.close(fig)


def section7_mask_quality(inventory, fig_dir: Path, tbl_dir: Path) -> pd.DataFrame:
    rows = []
    for r in tqdm(inventory.records, desc="masks"):
        if r.mask_path is None:
            continue
        try:
            img = nib.load(str(r.mask_path))
            m = img.get_fdata().astype(np.uint8)
            spacing = img.header.get_zooms()[:3]  # type: ignore[attr-defined]
            vol_ml = float(m.sum() * np.prod(spacing) / 1000.0)
            slices_with = int((m.sum(axis=(0, 1)) > 0).sum()) if m.ndim == 3 else None
            rows.append({
                "patient_id": r.patient_id, "modality": r.modality, "is_cirrhotic": r.is_cirrhotic,
                "liver_volume_ml": vol_ml, "slices_with_liver": slices_with,
            })
        except Exception as exc:  # noqa: BLE001
            log.warning("mask %s: %s", r.patient_id, exc)
    mask_df = pd.DataFrame(rows)
    mask_df.to_csv(tbl_dir / "eda_mask_quality.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(data=mask_df, x="liver_volume_ml", hue="is_cirrhotic", bins=30, kde=True, ax=axes[0])
    axes[0].set_title("Volumen hepático estimado (mL)")
    sns.histplot(data=mask_df, x="slices_with_liver", hue="is_cirrhotic", bins=30, ax=axes[1])
    axes[1].set_title("# cortes con hígado por estudio")
    fig.tight_layout(); fig.savefig(fig_dir / "eda_mask_quality.png", dpi=150)
    plt.close(fig)
    return mask_df


def write_summary(df: pd.DataFrame, dim_df: pd.DataFrame, mask_df: pd.DataFrame, tbl_dir: Path) -> None:
    n_cirr_pid = df[df["is_cirrhotic"]]["patient_id"].nunique()
    n_healthy_pid = df[~df["is_cirrhotic"]]["patient_id"].nunique()
    n_t1 = int((df["modality"] == "T1w").sum())
    n_t2 = int((df["modality"] == "T2w").sum())
    sev_counts = df[df["is_cirrhotic"]]["severity"].value_counts(dropna=False).to_dict()
    summary = {
        "n_studies_total": int(len(df)),
        "n_studies_cirrhotic": int(df["is_cirrhotic"].sum()),
        "n_studies_healthy": int((~df["is_cirrhotic"]).sum()),
        "n_t1w": n_t1,
        "n_t2w": n_t2,
        "n_patients_cirrhotic_unique": int(n_cirr_pid),
        "n_patients_healthy_unique": int(n_healthy_pid),
        "imbalance_cirr_to_healthy": round(int(df["is_cirrhotic"].sum()) / max(1, int((~df["is_cirrhotic"]).sum())), 3),
        "age_mean_cirr": float(df[df["is_cirrhotic"]]["age"].mean()),
        "age_mean_healthy": float(df[~df["is_cirrhotic"]]["age"].mean()),
        "age_std_cirr": float(df[df["is_cirrhotic"]]["age"].std()),
        "age_std_healthy": float(df[~df["is_cirrhotic"]]["age"].std()),
        "severity_mild": int(sev_counts.get(1.0, sev_counts.get(1, 0))),
        "severity_moderate": int(sev_counts.get(2.0, sev_counts.get(2, 0))),
        "severity_severe": int(sev_counts.get(3.0, sev_counts.get(3, 0))),
        "shape_mean_x": float(np.mean([s[0] for s in dim_df["shape"]])),
        "shape_mean_y": float(np.mean([s[1] for s in dim_df["shape"]])),
        "shape_mean_z": float(np.mean([s[2] for s in dim_df["shape"]])),
        "spacing_mean_x": float(np.mean([s[0] for s in dim_df["spacing"]])),
        "spacing_mean_y": float(np.mean([s[1] for s in dim_df["spacing"]])),
        "spacing_mean_z": float(np.mean([s[2] for s in dim_df["spacing"]])),
        "liver_volume_mean_cirr": float(mask_df[mask_df["is_cirrhotic"]]["liver_volume_ml"].mean()),
        "liver_volume_mean_healthy": float(mask_df[~mask_df["is_cirrhotic"]]["liver_volume_ml"].mean()),
        "liver_volume_median_cirr": float(mask_df[mask_df["is_cirrhotic"]]["liver_volume_ml"].median()),
        "liver_volume_median_healthy": float(mask_df[~mask_df["is_cirrhotic"]]["liver_volume_ml"].median()),
    }
    pd.DataFrame([summary]).T.to_csv(tbl_dir / "eda_summary.csv", header=["value"])
    print("\n=== EDA SUMMARY ===")
    for k, v in summary.items():
        print(f"  {k:35s} = {v}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/CirrMRI600plus_raw"))
    parser.add_argument("--skip-intensities", action="store_true", help="Saltar histogramas (lentos).")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    project = Path(__file__).resolve().parents[1]
    fig_dir = project / "reports" / "figures"; fig_dir.mkdir(parents=True, exist_ok=True)
    tbl_dir = project / "reports" / "tables"; tbl_dir.mkdir(parents=True, exist_ok=True)

    inv = _load_inventory(args.root)
    df = inv.to_frame()

    log.info("§1 conteos por modalidad/clase")
    section1_counts(df, fig_dir)
    log.info("§2 demografía")
    section2_demographics(df, fig_dir)
    log.info("§3 severidad")
    section3_severity(df, fig_dir)
    log.info("§4 análisis dimensional (sample 60)")
    dim_df = section4_dimensional(inv.records, fig_dir, tbl_dir, n_sample=60)
    if not args.skip_intensities:
        log.info("§5 distribuciones de intensidad (sample 6 por grupo)")
        section5_intensities(inv, fig_dir, n_per_group=6)
    else:
        log.info("§5 saltado")
    log.info("§6 paneles axiales representativos")
    section6_panels(inv, fig_dir)
    log.info("§7 calidad de máscaras (TODOS los records)")
    mask_df = section7_mask_quality(inv, fig_dir, tbl_dir)
    log.info("§8 resumen agregado")
    write_summary(df, dim_df, mask_df, tbl_dir)
    log.info("Listo. Figuras en %s, tablas en %s", fig_dir, tbl_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
