#!/usr/bin/env python3
"""
Download Cloud Optimized GeoTIFFs listed in data/hazard_layers.csv and rewrite
dataset_url entries to local filesystem paths for faster tile/analysis I/O.

Usage (from repo root):
  python scripts/download_hazard_rasters.py
  python scripts/download_hazard_rasters.py --path-prefix /app/data/rasters
  python scripts/download_hazard_rasters.py --dry-run
  python scripts/download_hazard_rasters.py --csv data/hazard_layers.csv --out-dir data/rasters

The script:
  1. Backs up the CSV to data/hazard_layers.remote.csv (once, if missing)
  2. Downloads each http(s) URL into --out-dir (skips files that already exist
     and match Content-Length when available)
  3. Rewrites dataset_url to {path_prefix}/{filename} in the CSV

Docker deployments should use --path-prefix /app/data/rasters so paths match
the volume mount in docker-compose.yml.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
from urllib.parse import urlparse


def _parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parent.parent
    default_csv = repo_root / "data" / "hazard_layers.csv"
    default_out = repo_root / "data" / "rasters"

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path, default=default_csv, help="Path to hazard_layers.csv")
    p.add_argument("--out-dir", type=Path, default=default_out, help="Directory to store GeoTIFFs")
    p.add_argument(
        "--path-prefix",
        type=str,
        default=None,
        help=(
            "Prefix written into dataset_url (default: absolute path of --out-dir). "
            "Use /app/data/rasters for Docker."
        ),
    )
    p.add_argument(
        "--backup",
        type=Path,
        default=None,
        help="Backup path for original remote CSV (default: <csv_dir>/hazard_layers.remote.csv)",
    )
    p.add_argument("--dry-run", action="store_true", help="List downloads without writing")
    p.add_argument("--force", action="store_true", help="Re-download even if local file exists")
    p.add_argument("--timeout", type=int, default=3600, help="Per-file HTTP timeout (seconds)")
    return p.parse_args()


def _read_rows(csv_path: Path) -> Tuple[List[str], List[dict]]:
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f, delimiter=";")
        if reader.fieldnames is None:
            raise SystemExit(f"CSV has no header: {csv_path}")
        fieldnames = list(reader.fieldnames)
        if "hazard" not in fieldnames or "dataset_url" not in fieldnames:
            raise SystemExit(
                f"CSV must include 'hazard' and 'dataset_url' columns (found: {fieldnames})"
            )
        rows = list(reader)
    return fieldnames, rows


def _write_rows(csv_path: Path, fieldnames: Iterable[str], rows: List[dict]) -> None:
    tmp = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames), delimiter=";", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(csv_path)


def _is_remote(url: str) -> bool:
    return url.startswith("http://") or url.startswith("https://")


def _remote_size(url: str, timeout: int) -> Optional[int]:
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            length = resp.headers.get("Content-Length")
            return int(length) if length else None
    except Exception:
        return None


def _download(url: str, dest: Path, timeout: int) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".partial")
    if partial.exists():
        partial.unlink()

    def _reporthook(block_num: int, block_size: int, total_size: int) -> None:
        if total_size <= 0:
            return
        downloaded = block_num * block_size
        pct = min(100.0, 100.0 * downloaded / total_size)
        mb = downloaded / (1024 * 1024)
        total_mb = total_size / (1024 * 1024)
        sys.stdout.write(f"\r    {pct:5.1f}%  ({mb:.1f}/{total_mb:.1f} MiB)")
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, partial, reporthook=_reporthook, timeout=timeout)  # type: ignore[call-arg]
    except TypeError:
        # Python <3.13 urlretrieve has no timeout kwarg — use urlopen
        with urllib.request.urlopen(url, timeout=timeout) as resp, partial.open("wb") as out:
            total = resp.headers.get("Content-Length")
            total_size = int(total) if total else 0
            downloaded = 0
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                downloaded += len(chunk)
                if total_size:
                    pct = min(100.0, 100.0 * downloaded / total_size)
                    sys.stdout.write(
                        f"\r    {pct:5.1f}%  ({downloaded/1024/1024:.1f}/{total_size/1024/1024:.1f} MiB)"
                    )
                    sys.stdout.flush()
    except urllib.error.HTTPError as e:
        if partial.exists():
            partial.unlink()
        raise SystemExit(f"HTTP {e.code} downloading {url}: {e.reason}") from e
    except Exception:
        if partial.exists():
            partial.unlink()
        raise

    partial.replace(dest)
    if sys.stdout.isatty():
        sys.stdout.write("\n")


def main() -> int:
    args = _parse_args()
    csv_path: Path = args.csv.resolve()
    out_dir: Path = args.out_dir.resolve()
    path_prefix = (args.path_prefix or str(out_dir)).rstrip("/")
    backup_path = (args.backup or csv_path.with_name("hazard_layers.remote.csv")).resolve()

    if not csv_path.is_file():
        raise SystemExit(f"CSV not found: {csv_path}")

    fieldnames, rows = _read_rows(csv_path)
    remote_rows = [r for r in rows if _is_remote(r.get("dataset_url", "").strip())]
    already_local = len(rows) - len(remote_rows)

    print(f"Catalog: {csv_path}")
    print(f"Layers:  {len(rows)} ({len(remote_rows)} remote URL(s), {already_local} already local)")
    print(f"Out dir: {out_dir}")
    print(f"Prefix:  {path_prefix}")
    print()

    if not args.dry_run and not backup_path.exists():
        # Prefer backing up a still-remote catalog
        if remote_rows:
            shutil.copy2(csv_path, backup_path)
            print(f"Backed up remote catalog → {backup_path}")
        else:
            print(f"No remote URLs left; skipping backup (use existing {backup_path} if present)")

    failures: List[str] = []

    for i, row in enumerate(rows, start=1):
        hazard = row.get("hazard", f"row_{i}")
        url = (row.get("dataset_url") or "").strip()
        if not url:
            failures.append(f"{hazard}: empty dataset_url")
            continue

        if not _is_remote(url):
            print(f"[{i}/{len(rows)}] SKIP (already local): {hazard}")
            # Normalize prefix if file name matches
            name = Path(url).name
            row["dataset_url"] = f"{path_prefix}/{name}"
            continue

        filename = Path(urlparse(url).path).name
        if not filename:
            failures.append(f"{hazard}: could not parse filename from {url}")
            continue

        dest = out_dir / filename
        local_url = f"{path_prefix}/{filename}"

        print(f"[{i}/{len(rows)}] {hazard}")
        print(f"    URL  → {url}")
        print(f"    File → {dest}")

        if args.dry_run:
            status = "exists" if dest.exists() else "would download"
            print(f"    Dry-run ({status}) → {local_url}")
            row["dataset_url"] = local_url
            continue

        skip_download = False
        if dest.exists() and not args.force:
            remote_len = _remote_size(url, timeout=min(60, args.timeout))
            local_len = dest.stat().st_size
            if remote_len is not None and local_len != remote_len:
                print(
                    f"    Size mismatch (local={local_len}, remote={remote_len}); re-downloading"
                )
            else:
                print(f"    Exists ({local_len / (1024**3):.2f} GiB); skipping download")
                skip_download = True

        if skip_download:
            row["dataset_url"] = local_url
            continue

        try:
            _download(url, dest, timeout=args.timeout)
            size_gb = dest.stat().st_size / (1024**3)
            print(f"    Done ({size_gb:.2f} GiB)")
            row["dataset_url"] = local_url
        except SystemExit as e:
            failures.append(str(e))
            print(f"    FAILED: {e}")
        except Exception as e:
            failures.append(f"{hazard}: {e}")
            print(f"    FAILED: {e}")

    if args.dry_run:
        print("\nDry run only — CSV not modified.")
        return 0 if not failures else 1

    _write_rows(csv_path, fieldnames, rows)
    print(f"\nUpdated catalog → {csv_path}")

    if failures:
        print(f"\n{len(failures)} failure(s):")
        for msg in failures:
            print(f"  - {msg}")
        return 1

    print("All hazard rasters ready for local use.")
    return 0


if __name__ == "__main__":
    # urlretrieve timeout shim for older Python: monkey-patch via opener default
    sys.exit(main())
