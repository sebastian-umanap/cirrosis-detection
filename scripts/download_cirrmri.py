"""Descarga CirrMRI600+ desde OSF (nodo cuk24).

Usa la API HTTP de OSF (no requiere autenticación para nodos públicos).
Soporta reanudación: archivos ya presentes con el tamaño esperado se saltan.

Uso:
    python scripts/download_cirrmri.py --dest data/CirrMRI600plus_raw

Requisitos:
    pip install requests tqdm
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import requests
from tqdm import tqdm

OSF_NODE = "cuk24"
OSF_API_ROOT = f"https://api.osf.io/v2/nodes/{OSF_NODE}/files/osfstorage/"


@dataclass(frozen=True)
class OsfFile:
    name: str
    size: int | None
    download_url: str
    rel_path: str


def _iter_osf(url: str, rel_parent: str = "") -> Iterator[OsfFile]:
    """Recorre recursivamente un nodo OSF storage devolviendo archivos descargables."""
    next_url: str | None = url
    while next_url:
        resp = requests.get(next_url, timeout=60)
        resp.raise_for_status()
        payload = resp.json()
        for item in payload["data"]:
            attr = item["attributes"]
            name = attr["name"]
            kind = attr["kind"]
            rel = f"{rel_parent}/{name}".lstrip("/")
            if kind == "folder":
                child = item["relationships"]["files"]["links"]["related"]["href"]
                yield from _iter_osf(child, rel)
            else:
                yield OsfFile(
                    name=name,
                    size=attr.get("size"),
                    download_url=item["links"]["download"],
                    rel_path=rel,
                )
        next_url = payload["links"].get("next")


def _download_one(f: OsfFile, dest_root: Path, retries: int = 3) -> None:
    out_path = dest_root / f.rel_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and (f.size is None or out_path.stat().st_size == f.size):
        return
    for attempt in range(1, retries + 1):
        try:
            with requests.get(f.download_url, stream=True, timeout=300) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0)) or f.size or 0
                tmp = out_path.with_suffix(out_path.suffix + ".part")
                with open(tmp, "wb") as fh, tqdm(
                    total=total,
                    unit="B",
                    unit_scale=True,
                    desc=f.rel_path[-50:],
                    leave=False,
                ) as bar:
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        if chunk:
                            fh.write(chunk)
                            bar.update(len(chunk))
                tmp.replace(out_path)
            return
        except (requests.RequestException, OSError) as exc:
            if attempt == retries:
                raise
            wait = 2**attempt
            print(f"  ! {f.rel_path}: error {exc}; retry in {wait}s", file=sys.stderr)
            time.sleep(wait)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dest",
        type=Path,
        default=Path("data/CirrMRI600plus_raw"),
        help="Directorio destino (se crea si no existe).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Solo lista archivos, no descarga.",
    )
    args = parser.parse_args()

    args.dest.mkdir(parents=True, exist_ok=True)
    print(f"Enumerando archivos en OSF nodo {OSF_NODE}…")
    files = list(_iter_osf(OSF_API_ROOT))
    total_bytes = sum(f.size or 0 for f in files)
    print(f"  {len(files):,} archivos | total ~{total_bytes / 1e9:.2f} GB")

    if args.dry_run:
        for f in files[:20]:
            print(f"  {f.rel_path}  ({(f.size or 0) / 1e6:.1f} MB)")
        if len(files) > 20:
            print(f"  … y {len(files) - 20} más")
        return 0

    print(f"Descargando a {args.dest.resolve()}")
    for f in tqdm(files, desc="archivos"):
        _download_one(f, args.dest)
    print("Listo.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
