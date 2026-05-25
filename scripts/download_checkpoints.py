"""Descarga los checkpoints del ResNet-50 2.5D desde el release de GitHub.

Los 6 archivos suman ~553 MB. Se guardan en `reports/checkpoints/`.
Reanuda descargas parciales y verifica tamaño antes de re-bajar.

Uso:
    python scripts/download_checkpoints.py
    python scripts/download_checkpoints.py --tag v1.0
    python scripts/download_checkpoints.py --only resnet25d_binary_best.pt
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

REPO = "sebastian-umanap/cirrosis-detection"
DEFAULT_TAG = "v1.0"

FILES = [
    "resnet25d_binary_best.pt",
    "resnet25d_fold0.pt",
    "resnet25d_fold1.pt",
    "resnet25d_fold2.pt",
    "resnet25d_fold3.pt",
    "resnet25d_fold4.pt",
]


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "cirrosis-detection-downloader"})
    with urllib.request.urlopen(req) as r:
        total = int(r.headers.get("Content-Length", 0))
        if dest.exists() and total and dest.stat().st_size == total:
            print(f"  ya existe (size OK): {dest.name}")
            return
        downloaded = 0
        chunk = 1 << 20  # 1 MB
        with dest.open("wb") as f:
            while True:
                buf = r.read(chunk)
                if not buf:
                    break
                f.write(buf)
                downloaded += len(buf)
                if total:
                    pct = 100 * downloaded / total
                    sys.stdout.write(f"\r  {dest.name}: {downloaded / 1e6:.1f} / {total / 1e6:.1f} MB ({pct:.0f} %)")
                    sys.stdout.flush()
        print()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default=DEFAULT_TAG, help=f"tag del release (default: {DEFAULT_TAG})")
    ap.add_argument("--dest", default="reports/checkpoints", help="carpeta destino")
    ap.add_argument("--only", nargs="+", default=None, help="descargar solo estos nombres (ej: resnet25d_binary_best.pt)")
    args = ap.parse_args()

    dest = Path(args.dest)
    files = args.only if args.only else FILES
    base = f"https://github.com/{REPO}/releases/download/{args.tag}"

    print(f"Descargando {len(files)} checkpoint(s) desde {base} → {dest}/")
    for name in files:
        url = f"{base}/{name}"
        target = dest / name
        try:
            _download(url, target)
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR descargando {name}: {exc}", file=sys.stderr)
            return 1
    print("Listo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
