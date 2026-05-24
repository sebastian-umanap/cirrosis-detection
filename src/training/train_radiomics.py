"""5-fold CV del baseline radiómico — invocable desde la CLI:

    python -m src.training train-radiomics --config configs/radiomics.yaml

Equivale a la versión interactiva del `notebooks/04_classification_baselines.ipynb`.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from src.data.inventory import build_inventory
from src.evaluation.bootstrap import bootstrap_ci, format_with_ci
from src.evaluation.metrics import binary_metrics_at_youden
from src.models.radiomics import RadiomicsPipeline, extract_features_for_record

log = logging.getLogger(__name__)


def _features_or_cache(records, cache_csv: Path) -> pd.DataFrame:
    if cache_csv.exists():
        log.info("Usando cache de features: %s", cache_csv)
        return pd.read_csv(cache_csv)
    log.info("Extrayendo features radiómicos para %d records...", len(records))
    rows = []
    n = len(records)
    for i, r in enumerate(records, 1):
        if r.mask_path is None:
            continue
        try:
            rows.append(extract_features_for_record(r))
        except Exception as exc:  # noqa: BLE001
            log.warning("skip %s: %s", r.patient_id, exc)
        if i % 50 == 0 or i == n:
            log.info("  features %d/%d", i, n)
    df = pd.DataFrame(rows)
    cache_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_csv, index=False)
    log.info("Features guardados en %s (%d filas, %d cols)", cache_csv, len(df), df.shape[1])
    return df


def run_radiomics_cv(cfg: dict) -> int:
    data_cfg = cfg["data"]["dataset"]
    root = Path(data_cfg["root"]).resolve()
    cache_dir = Path(data_cfg["cache_dir"]).resolve()
    out_dir = Path(cfg["output"]["results_csv"]).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    inventory = build_inventory(root)
    records = inventory.filter(task="binary")
    features_csv = cache_dir / "radiomics_features_binary.csv"
    feats = _features_or_cache(records, features_csv)

    y = feats["label"].to_numpy().astype(int)
    groups = feats["patient_id"].astype(str).to_numpy()

    all_models = cfg["classifier"]["models"]
    summary_rows = []
    for model_cfg in all_models:
        name = model_cfg["name"]
        clf_kwargs = model_cfg.get("kwargs", {})
        oof = np.zeros(len(feats), dtype=float)
        fold_aucs = []
        sgkf = StratifiedGroupKFold(n_splits=cfg["data"]["splits"]["n_folds"],
                                    shuffle=True, random_state=cfg["data"]["splits"]["seed"])
        for fold, (tr, va) in enumerate(sgkf.split(feats, y, groups=groups)):
            pipe = RadiomicsPipeline(
                k_features=cfg["selection"]["k_features"],
                selection=cfg["selection"]["method"],
                classifier_name=name,
                classifier_kwargs=clf_kwargs,
                random_state=cfg["data"]["splits"]["seed"],
            )
            pipe.fit(feats.iloc[tr], y[tr])
            proba = pipe.predict_proba(feats.iloc[va])[:, 1]
            oof[va] = proba
            auc = roc_auc_score(y[va], proba)
            fold_aucs.append(auc)
            log.info("%s fold %d: AUC=%.4f", name, fold, auc)
        bm = binary_metrics_at_youden(y, oof)
        boot = bootstrap_ci(y, oof, lambda yt, ys: roc_auc_score(yt, ys), n_resamples=1000, seed=42)
        summary_rows.append({
            "model": f"radiomics_{name}",
            "AUC-ROC": format_with_ci(boot.point, boot.ci_low, boot.ci_high),
            "AUC-PR": f"{bm.auc_pr:.3f}",
            "Sens@Y": f"{bm.sensitivity:.3f}",
            "Spec@Y": f"{bm.specificity:.3f}",
            "F1": f"{bm.f1:.3f}",
            "MCC": f"{bm.mcc:.3f}",
        })
        pred_df = pd.DataFrame({
            "patient_id": feats["patient_id"], "modality": feats["modality"],
            "label": y, "proba": oof,
        })
        pred_df.to_csv(out_dir / f"radiomics_{name}_preds.csv", index=False)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(cfg["output"]["results_csv"], index=False)
    log.info("Resultados guardados en %s", cfg["output"]["results_csv"])
    print(summary.to_string(index=False))
    return 0
