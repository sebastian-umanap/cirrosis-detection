"""CLI unificado para los entry points de entrenamiento.

Usos:
    python -m src.training train-radiomics --config configs/radiomics.yaml
    python -m src.training train-dl --config configs/dl_2_5d.yaml
    python -m src.training evaluate --config configs/eval.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

log = logging.getLogger("training-cli")


def _load_config(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # Resolver `defaults: { data: data.yaml }` si está presente.
    if isinstance(cfg, dict) and "defaults" in cfg:
        defs = cfg.pop("defaults", {})
        for key, fname in (defs or {}).items():
            child = path.parent / fname
            with open(child, encoding="utf-8") as f:
                cfg[key] = yaml.safe_load(f)
    return cfg


def _cmd_train_radiomics(cfg: dict) -> int:
    from src.training.train_radiomics import run_radiomics_cv
    return run_radiomics_cv(cfg)


def _cmd_train_dl(cfg: dict) -> int:
    from src.training.train_dl_cv import run_dl_cv
    return run_dl_cv(cfg)


def _cmd_evaluate(cfg: dict) -> int:
    from src.evaluation.evaluate_all import run_evaluation
    return run_evaluation(cfg)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="src.training", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    for name in ("train-radiomics", "train-dl", "evaluate"):
        p = sub.add_parser(name)
        p.add_argument("--config", type=Path, required=True)
        p.add_argument("--log-level", default="INFO")

    args = parser.parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s [%(name)s] %(message)s")
    cfg = _load_config(args.config)

    dispatch = {
        "train-radiomics": _cmd_train_radiomics,
        "train-dl": _cmd_train_dl,
        "evaluate": _cmd_evaluate,
    }
    return dispatch[args.cmd](cfg)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
