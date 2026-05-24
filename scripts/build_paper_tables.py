"""Genera fragmentos LaTeX de las tablas del paper desde los CSV en reports/tables/.

Salidas:
    reports/paper_snippets/table_main.tex
    reports/paper_snippets/table_delong.tex

Compatible con cualquier conjunto de modelos presentes; salta los que falten.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
TBL_DIR = PROJECT / "reports" / "tables"
OUT_DIR = PROJECT / "reports" / "paper_snippets"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _build_main_table() -> str:
    rows = []
    radio_csv = TBL_DIR / "radiomics_results.csv"
    if radio_csv.exists():
        df = pd.read_csv(radio_csv)
        for _, r in df.iterrows():
            label = r["model"].replace("radiomics_", "").upper().replace("LOGREG", "LogReg")
            rows.append({
                "model": f"{label} + radiomics",
                **{k: str(r[k]) for k in ("AUC-ROC", "AUC-PR", "Sens@Y", "Spec@Y", "F1", "MCC")},
            })

    main_csv = TBL_DIR / "main_results.csv"
    if main_csv.exists():
        df = pd.read_csv(main_csv)
        for _, r in df.iterrows():
            name = r["model"]
            if name.startswith("radiomics_"):
                continue   # ya añadidos arriba
            label = name.replace("dl_2_5d_resnet50", "2.5D ResNet-50") \
                        .replace("dl_2_5d", "2.5D ResNet-50")
            rows.append({
                "model": label,
                "AUC-ROC": str(r.get("AUC-ROC", "")),
                "AUC-PR":  str(r.get("AUC-PR", "")),
                "Sens@Y":  str(r.get("Sens@Youden", "")),
                "Spec@Y":  str(r.get("Spec@Youden", "")),
                "F1":      str(r.get("F1", "")),
                "MCC":     str(r.get("MCC", "")),
            })

    if not rows:
        return "% No hay CSV de resultados todavía.\n"

    body = "\n".join(
        " & ".join([r["model"], r["AUC-ROC"], r["AUC-PR"], r["Sens@Y"], r["Spec@Y"], r["F1"]]) + r" \\"
        for r in rows
    )
    return (
        "% Auto-generado por scripts/build_paper_tables.py — NO editar a mano.\n"
        "\\begin{tabular}{lccccc}\n"
        "\\toprule\n"
        "Model & AUC-ROC (95\\% CI) & AUC-PR & Sens@Y & Spec@Y & F1 \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )


def _build_delong_table() -> str:
    p = TBL_DIR / "delong.csv"
    if not p.exists():
        return "% DeLong CSV no disponible aún.\n"
    df = pd.read_csv(p)
    if df.empty:
        return "% DeLong CSV vacío.\n"
    body = "\n".join(
        f"{r['A']} vs {r['B']} & {r['AUC_A']:.3f} & {r['AUC_B']:.3f} & "
        f"{r['delta']:+.3f} & {r['z']:+.2f} & {r['p_value']:.4f} \\\\"
        for _, r in df.iterrows()
    )
    return (
        "\\begin{tabular}{lccccc}\n"
        "\\toprule\n"
        "Comparison & AUC$_A$ & AUC$_B$ & $\\Delta$AUC & $z$ & $p$ \\\\\n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )


def main() -> int:
    main_tex = _build_main_table()
    (OUT_DIR / "table_main.tex").write_text(main_tex, encoding="utf-8")
    print("escrito:", OUT_DIR / "table_main.tex")
    print(main_tex)

    delong_tex = _build_delong_table()
    (OUT_DIR / "table_delong.tex").write_text(delong_tex, encoding="utf-8")
    print("escrito:", OUT_DIR / "table_delong.tex")
    print(delong_tex)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
