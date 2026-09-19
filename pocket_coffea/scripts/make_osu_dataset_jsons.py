#!/usr/bin/env python3
"""Create PocketCoffea dataset JSONs from ROOT files copied to OSU storage.

The companion ``copy_filelists_to_remote.py`` script writes one directory per
input filelist and preserves the OSUNano path below
``/store/group/lpcdisapptrks/nano``.  A copied file therefore looks like

    /data/user/mjoyce/disapptrks_nano/Muon_2023C/dev/Muon0/.../nano_1.root

or

    /data/user/mjoyce/disapptrks_nano/Muon_2023C/prod/Muon0_Run2023C_v4/nano_1.root

Special-test files copied from ``dev_v2`` can be selected independently with
``--source-area dev_v2``.  Their dataset names and JSON filenames default to an
``OSUv2`` suffix so they cannot be confused with the standard outputs.

This helper scans those local files, groups them by primary dataset and Run-3
era group, and writes the JSON files consumed by ``submit_osu_dataset.sh``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from disapptrks.datasets import (  # noqa: E402
    ERA_GROUP_BY_LABEL,
    build_dataset_definition,
    group_osunano_files,
    write_dataset_definition,
    write_filelist,
)


def find_root_files(base: Path) -> list[str]:
    return sorted(str(path.resolve()) for path in base.rglob("*.root") if path.is_file())


def write_osu_outputs(
    files: list[str],
    *,
    output_dir: Path,
    filelist_dir: Path | None,
    dataset_name_suffix: str,
    output_filename_suffix: str,
    prod_version_policy: str,
    nano_version: int | None,
    source_areas: tuple[str, ...],
) -> dict[str, dict[str, Path | int | str]]:
    grouped = group_osunano_files(
        files,
        prod_version_policy=prod_version_policy,
        source_areas=source_areas,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    if filelist_dir is not None:
        filelist_dir.mkdir(parents=True, exist_ok=True)

    outputs: dict[str, dict[str, Path | int | str]] = {}
    for (primary, label), paths in sorted(grouped.items()):
        group = ERA_GROUP_BY_LABEL[label]
        sample = f"DATA_{primary}"
        dataset_name = f"Run{label}_{primary}_OSUNano_{dataset_name_suffix}"
        filename_suffix = f"_{output_filename_suffix}" if output_filename_suffix else ""
        dataset_path = output_dir / f"osu_{label}_{primary}{filename_suffix}.json"
        dataset = build_dataset_definition(
            dataset_name=dataset_name,
            files=paths,
            sample=sample,
            year=group.metadata_year,
            era=group.metadata_era,
            primary_dataset=primary,
            is_mc=False,
            nevents="0",
            nano_version=group.nano_version if nano_version is None else nano_version,
        )
        write_dataset_definition(dataset, dataset_path)

        entry: dict[str, Path | int | str] = {
            "dataset_json": dataset_path,
            "dataset_name": dataset_name,
            "n_files": len(paths),
            "sample": sample,
            "year": group.metadata_year,
            "era": group.metadata_era,
        }

        if filelist_dir is not None:
            filelist_path = filelist_dir / f"{primary}_{label}.txt"
            write_filelist(paths, filelist_path)
            entry["filelist"] = filelist_path

        outputs[f"{primary}_{label}"] = entry

    return outputs


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build OSU-local dataset JSONs from copied OSUNano ROOT files."
    )
    parser.add_argument(
        "base",
        nargs="?",
        type=Path,
        default=Path("/data/user/mjoyce/disapptrks_nano"),
        help="Parent directory containing copied filelist-stem subdirectories.",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("datasets"),
        help="Directory where dataset JSONs are written.",
    )
    parser.add_argument(
        "--filelist-dir",
        type=Path,
        help="Optionally also write grouped local filelists to this directory.",
    )
    parser.add_argument(
        "--dataset-name-suffix",
        help="Suffix used in dataset names. Defaults to OSU, or OSUv2 for dev_v2.",
    )
    parser.add_argument(
        "--output-filename-suffix",
        help="Suffix before .json. Defaults to none, or OSUv2 for dev_v2.",
    )
    parser.add_argument(
        "--source-area",
        action="append",
        choices=("dev", "prod", "dev_v2"),
        help="OSUNano area to include. Repeat to include multiple areas. Default: dev and prod.",
    )
    parser.add_argument(
        "--prod-version-policy",
        choices=("latest", "all"),
        default="all",
        help="How to handle prod directories with *_vN suffixes. Default keeps all.",
    )
    parser.add_argument(
        "--nano-version",
        type=int,
        help="Override per-era nano_version metadata. Default: 12 for 2022/2023, 15 for 2024/2025.",
    )
    parser.add_argument(
        "--fail-on-empty",
        action="store_true",
        help="Return nonzero if any primary/year group has zero files.",
    )
    args = parser.parse_args()

    if not args.base.exists():
        raise SystemExit(f"Base directory does not exist: {args.base}")

    files = find_root_files(args.base)
    source_areas = tuple(args.source_area or ("dev", "prod"))
    is_dev_v2_only = source_areas == ("dev_v2",)
    dataset_name_suffix = args.dataset_name_suffix or ("OSUv2" if is_dev_v2_only else "OSU")
    output_filename_suffix = args.output_filename_suffix
    if output_filename_suffix is None:
        output_filename_suffix = "OSUv2" if is_dev_v2_only else ""
    outputs = write_osu_outputs(
        files,
        output_dir=args.output_dir,
        filelist_dir=args.filelist_dir,
        dataset_name_suffix=dataset_name_suffix,
        output_filename_suffix=output_filename_suffix,
        prod_version_policy=args.prod_version_policy,
        nano_version=args.nano_version,
        source_areas=source_areas,
    )

    print(f"Scanned {len(files)} ROOT file(s) under {args.base.resolve()}")
    empty = []
    for key, entry in outputs.items():
        suffix = f", filelist={entry['filelist']}" if "filelist" in entry else ""
        print(
            f"{key:16s} {entry['n_files']:6d} files -> "
            f"{entry['dataset_json']} "
            f"sample={entry['sample']} year={entry['year']}{suffix}"
        )
        if entry["n_files"] == 0:
            empty.append(key)

    if empty:
        print("Warning: empty groups: " + ", ".join(empty), file=sys.stderr)
        return 2 if args.fail_on_empty else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
