#!/usr/bin/env python3
"""Download golden JSON files used by the Run-3 data preselections."""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path


GOLDEN_JSON_URLS = {
    "Cert_Collisions2022_355100_362760_Golden.json": (
        "https://cms-service-dqmdc.web.cern.ch/CAF/certification/Collisions22/"
        "Cert_Collisions2022_355100_362760_Golden.json"
    ),
    "Cert_Collisions2023_366442_370790_Golden.json": (
        "https://cms-service-dqmdc.web.cern.ch/CAF/certification/Collisions23/"
        "Cert_Collisions2023_366442_370790_Golden.json"
    ),
    "Cert_Collisions2024_378981_386951_Golden.json": (
        "https://cms-service-dqmdc.web.cern.ch/CAF/certification/Collisions24/"
        "Cert_Collisions2024_378981_386951_Golden.json"
    ),
    "Cert_Collisions2025_391658_398903_Golden.json": (
        "https://cms-service-dqmdc.web.cern.ch/CAF/certification/Collisions25/"
        "Cert_Collisions2025_391658_398903_Golden.json"
    ),
    "Collisions26_MLEnhancedGolden_Latest.json": (
        "https://cms-service-dqmdc.web.cern.ch/CAF/certification/Collisions26/"
        "Collisions26_MLEnhancedGolden_Latest.json"
    ),
}

ALIASES = {
    "Cert_Collisions2026_Golden.json": "Collisions26_MLEnhancedGolden_Latest.json",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Download CMS golden JSON files.")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("data/golden_jsons"),
        help="Destination directory. Default: data/golden_jsons.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Redownload files that already exist.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, url in GOLDEN_JSON_URLS.items():
        output = args.output_dir / filename
        if output.exists() and not args.overwrite:
            print(f"skip existing {output}")
            continue
        print(f"download {url} -> {output}")
        urllib.request.urlretrieve(url, output)
    for alias, target in ALIASES.items():
        alias_path = args.output_dir / alias
        target_path = args.output_dir / target
        if not target_path.exists():
            print(f"skip alias {alias_path}: missing target {target_path}")
            continue
        if alias_path.exists() or alias_path.is_symlink():
            if not args.overwrite:
                print(f"skip existing alias {alias_path}")
                continue
            alias_path.unlink()
        try:
            alias_path.symlink_to(target_path.name)
            print(f"alias {alias_path} -> {target_path.name}")
        except OSError:
            alias_path.write_bytes(target_path.read_bytes())
            print(f"copy alias {target_path} -> {alias_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
