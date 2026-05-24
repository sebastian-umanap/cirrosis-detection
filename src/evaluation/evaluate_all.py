"""Evaluación final: agrega los OOF preds de los modelos, calcula métricas con CIs y DeLong.

Invocable desde CLI:
    python -m src.training evaluate --config configs/eval.yaml
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

from src.evaluation.bootstrap import bootstrap_ci, format_with_ci
from src.evaluation.delong import delong_test_two_aucs
from src.evaluation.metrics import binary_metrics_at_youden

log = logging.getLogger(__name__)


def _load_preds(csv: Path) -> pd.DataFrame:
    df = pd.read_csv(csv)
    if "modality" in df.columns:
        df = df.groupby("patient_id").agg(label=("label", "first"), proba=("proba", "mean")).reset_index()
    return df


def run_evaluation(cfg: dict) -> int:
    out_dir = Path(cfg["output"]["results_csv"]).parent; out_dir.mkdir(parents=True, exist_ok=True)
    fig_path = Path(cfg["output"]["roc_pr_fig"]); fig_path.parent.mkdir(parents=True, exist_ok=True)
    delong_csv = Path(cfg["output"]["delong_csv"])

    models = {m["name"]: _load_preds(Path(m["pred_csv"])) for m in cfg["models_to_compare"]}
    summary_rows = []
    for name, df in models.items():
        y = df["label"].to_numpy(); p = df["proba"].to_numpy()
        bm = binary_metrics_at_youden(y, p)
        boot_auc = bootstrap_ci(y, p, lambda yt, ys: roc_auc_score(yt, ys),
                                n_resamples=cfg["bootstrap"]["n_resamples"], seed=cfg["bootstrap"]["seed"])
        boot_ap = bootstrap_ci(y, p, lambda yt, ys: average_precision_score(yt, ys),
                               n_resamples=cfg["bootstrap"]["n_resamples"], seed=cfg["bootstrap"]["seed"] + 1)
        summary_rows.append({
            "model": name,
            "AUC-ROC": format_with_ci(boot_auc.point, boot_auc.ci_low, boot_auc.ci_high),
            "AUC-PR": format_with_ci(boot_ap.point, boot_ap.ci_low, boot_ap.ci_high),
            "Sens@Youden": f"{bm.sensitivity:.3f}",
            "Spec@Youden": f"{bm.specificity:.3f}",
            "F1": f"{bm.f1:.3f}",
            "MCC": f"{bm.mcc:.3f}",
        })
    pd.DataFrame(summary_rows).to_csv(cfg["output"]["results_csv"], index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for name, df in models.items():
        y = df["label"]; p = df["proba"]
        fpr, tpr, _ = roc_curve(y, p)
        axes[0].plot(fpr, tpr, label=f"{name} AUC={roc_auc_score(y, p):.3f}")
        pr, rc, _ = precision_recall_curve(y, p)
        axes[1].plot(rc, pr, label=f"{name} AP={average_precision_score(y, p):.3f}")
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.3)
    axes[0].set(xlabel="FPR", ylabel="TPR", title="ROC")
    axes[1].set(xlabel="Recall", ylabel="Precision", title="PR")
    axes[0].legend(); axes[1].legend()
    fig.tight_layout(); fig.savefig(fig_path, dpi=150)
    plt.close(fig)

    delong_rows = []
    for a, b in cfg["statistical_tests"]["delong"]["pairs"]:
        da = models[a][["patient_id", "label", "proba"]]
        db = models[b][["patient_id", "proba"]]
        m = da.merge(db, on="patient_id", suffixes=("_a", "_b"))
        if m.empty:
            log.warning("DeLong pair (%s,%s): merge vacío, skip", a, b)
            continue
        res = delong_test_two_aucs(m["label"].to_numpy(), m["proba_a"].to_numpy(), m["proba_b"].to_numpy())
        delong_rows.append({"A": a, "B": b, "AUC_A": res.auc_a, "AUC_B": res.auc_b,
                            "delta": res.auc_b - res.auc_a, "z": res.z, "p_value": res.p_value})
    pd.DataFrame(delong_rows).to_csv(delong_csv, index=False)
    log.info("Resultados, figura y DeLong escritos en %s", out_dir)
    print(pd.DataFrame(summary_rows).to_string(index=False))
    print(pd.DataFrame(delong_rows).to_string(index=False))
    return 0
