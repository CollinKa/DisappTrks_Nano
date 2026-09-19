#!/usr/bin/env python3
"""Create PocketCoffea dataset JSONs for the AMSB signal NanoAOD mass scan.

The signal NanoAOD is produced one file per mass point with OSUNano (which adds
the ``GenChargino`` branches), then copied to the LPC group area, e.g.

    /store/group/lpcdisapptrks/nano/dev/SignalSim/2KEventQuickTest/
        nano_AMSB_Wino_M700GeV_ctau100cm_Run3Summer22.root

This helper writes one dataset definition per mass point, plus a combined
definition holding every mass, so the whole scan can be processed in a single
PocketCoffea run with ``-ps``.

Theory cross sections are not consumed by the workflow (``xsec`` stays at 1.0
like the other signal samples); they are copied into the metadata as
``xsec_theory_pb`` so the limit step can pick them up without a second lookup.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from disapptrks.datasets import (  # noqa: E402
    ERA_GROUP_BY_LABEL,
    build_dataset_definition,
    list_eos_root_files,
    write_dataset_definition,
)

# CRAB writes <eos-dir>/<primaryDataset>/<requestName>/<timestamp>/0000/*.root and the
# MiniAOD campaign only shows up in the request name, so the period has to be read back
# out of the path.  Longest campaign first: "Run3Summer22EE" also contains "Run3Summer22".
CAMPAIGN_TO_ERA_GROUP = (
    ("Run3Summer22EE", "2022EFG"),
    ("Run3Summer23BPix", "2023D"),
    ("Run3Summer22", "2022CD"),
    ("Run3Summer23", "2023C"),
)
MASS_RE = re.compile(r"AMSB_Wino_M(\d+)GeV_ctau(\d+)cm")

DEFAULT_MASSES = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]
DEFAULT_EOS_DIR = (
    "root://cmseos.fnal.gov//store/group/lpcdisapptrks/nano/dev/SignalSim/2KEventQuickTest"
)
DEFAULT_PATTERN = "nano_AMSB_Wino_M{mass}GeV_ctau{ctau}cm_Run3Summer22.root"
DEFAULT_XSEC_JSON = REPO_ROOT / "src/disapptrks/cross_sections/amsb_wino_13p6TeV.json"


def _suffix(tag: str) -> str:
    return f"_{tag}" if tag else ""


def dataset_name(mass: int, ctau: int, tag: str = "", period: str = "") -> str:
    return f"Signal_AMSB_Wino_M{mass}_ctau{ctau}{_suffix(period)}{_suffix(tag)}_OSUNano"


def sample_name(mass: int, ctau: int, tag: str = "", period: str = "") -> str:
    return f"SIG_AMSB_Wino_M{mass}_ctau{ctau}{_suffix(period)}{_suffix(tag)}"


def load_cross_sections(path: Path | None) -> dict[str, dict]:
    if path is None:
        return {}
    if not path.is_file():
        print(f"Warning: cross-section file not found: {path}", file=sys.stderr)
        return {}
    with path.open() as handle:
        return json.load(handle).get("cross_sections", {})


def era_group_from_path(path: str) -> str | None:
    """Return the era-group label ("2022CD", ...) implied by a CRAB output path."""
    for campaign, label in CAMPAIGN_TO_ERA_GROUP:
        if campaign in path:
            return label
    return None


def scan_eos(eos_dir: str, ctau: int) -> dict[tuple[int, str], list[str]]:
    """Group every ROOT file under ``eos_dir`` by (mass, era-group label)."""
    grouped: dict[tuple[int, str], list[str]] = {}
    skipped = 0
    for path in list_eos_root_files(eos_dir, recursive=True):
        mass_match = MASS_RE.search(path)
        label = era_group_from_path(path)
        if mass_match is None or label is None or int(mass_match.group(2)) != ctau:
            skipped += 1
            continue
        grouped.setdefault((int(mass_match.group(1)), label), []).append(path)
    if skipped:
        print(f"Note: skipped {skipped} file(s) that did not match ctau={ctau} cm", file=sys.stderr)
    for files in grouped.values():
        files.sort()
    return grouped


def build_definitions_from_eos(args: argparse.Namespace) -> dict[str, dict[str, dict]]:
    """Return {era-group label: dataset definitions} discovered by listing EOS."""
    cross_sections = load_cross_sections(args.xsec_json)
    grouped = scan_eos(args.eos_dir, args.ctau)
    if not grouped:
        print(f"No signal ROOT files found under {args.eos_dir}", file=sys.stderr)
        return {}

    wanted = set(args.periods) if args.periods else {label for _, label in grouped}
    by_period: dict[str, dict[str, dict]] = {}

    for (mass, label) in sorted(grouped):
        if label not in wanted or mass not in args.masses:
            continue
        group = ERA_GROUP_BY_LABEL[label]
        files = grouped[(mass, label)]
        extra: dict[str, str | int | bool] = {
            "xsec": 1.0,
            "mass_GeV": mass,
            "ctau_cm": args.ctau,
            "osu_version": args.osu_version,
            "era_group": label,
            "nfiles": len(files),
        }
        entry = cross_sections.get(str(mass))
        if entry is not None:
            extra["xsec_theory_pb"] = entry["xsec_total_pb"]
        else:
            print(f"Warning: no theory cross section for M={mass} GeV", file=sys.stderr)

        by_period.setdefault(label, {}).update(
            build_dataset_definition(
                dataset_name=dataset_name(mass, args.ctau, args.tag, label),
                files=files,
                sample=sample_name(mass, args.ctau, args.tag, label),
                year=group.metadata_year,
                era=group.metadata_era,
                primary_dataset=args.primary_dataset,
                is_mc=True,
                nevents=args.nevents,
                nano_version=group.nano_version,
                extra_metadata=extra,
            )
        )

    for label in sorted(by_period):
        total = sum(d["metadata"]["nfiles"] for d in by_period[label].values())
        print(f"{label}: {len(by_period[label])} mass point(s), {total} file(s)")

    return by_period


def build_definitions(args: argparse.Namespace) -> dict[str, dict]:
    cross_sections = load_cross_sections(args.xsec_json)
    definitions: dict[str, dict] = {}

    for mass in args.masses:
        path = f"{args.eos_dir.rstrip('/')}/{args.pattern.format(mass=mass, ctau=args.ctau)}"
        extra: dict[str, str | int | bool] = {
            "xsec": 1.0,
            "mass_GeV": mass,
            "ctau_cm": args.ctau,
            "osu_version": args.osu_version,
        }
        entry = cross_sections.get(str(mass))
        if entry is not None:
            extra["xsec_theory_pb"] = entry["xsec_total_pb"]
        else:
            print(f"Warning: no theory cross section for M={mass} GeV", file=sys.stderr)

        definitions.update(
            build_dataset_definition(
                dataset_name=dataset_name(mass, args.ctau, args.tag),
                files=[path],
                sample=sample_name(mass, args.ctau, args.tag),
                year=args.year,
                era=args.era,
                primary_dataset=args.primary_dataset,
                is_mc=True,
                nevents=args.nevents,
                nano_version=args.nano_version,
                extra_metadata=extra,
            )
        )

    return definitions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eos-dir", default=DEFAULT_EOS_DIR, help="Directory holding the signal NanoAOD files.")
    parser.add_argument("--masses", type=int, nargs="+", default=DEFAULT_MASSES)
    parser.add_argument("--ctau", type=int, default=100, help="Generated proper decay length in cm.")
    parser.add_argument("--pattern", default=DEFAULT_PATTERN, help="File-name pattern with {mass} and {ctau}.")
    parser.add_argument("--year", default="2022_preEE")
    parser.add_argument("--era", default="preEE")
    parser.add_argument("--primary-dataset", default="SignalSim")
    parser.add_argument("--nevents", default="0", help="Metadata only; '0' matches the other EOS-built datasets.")
    parser.add_argument("--nano-version", type=int, default=12)
    parser.add_argument("--osu-version", type=int, default=3, help="OSUNano version that produced the files.")
    parser.add_argument("--xsec-json", type=Path, default=DEFAULT_XSEC_JSON)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "pocket_coffea/datasets")
    parser.add_argument(
        "--tag",
        default="",
        help="Suffix for dataset/sample/file names, e.g. 2K for the 2000-event quick test. "
        "Keeps these definitions from colliding with the full-statistics ones.",
    )
    parser.add_argument("--combined-name", default=None, help="Combined JSON file name; default signal_AMSB_Wino_scan_ctau<ctau><tag>.json.")
    parser.add_argument("--per-mass", action="store_true", help="Also write one JSON per mass point.")
    parser.add_argument(
        "--scan-eos",
        action="store_true",
        help="Discover the files by listing --eos-dir recursively (CRAB output layout) "
        "instead of assuming one file per mass named by --pattern. Writes one combined "
        "JSON per data-taking period.",
    )
    parser.add_argument(
        "--periods",
        nargs="+",
        default=None,
        choices=sorted(ERA_GROUP_BY_LABEL),
        help="With --scan-eos, restrict to these era groups. Default: whatever is on disk.",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite existing dataset JSONs instead of refusing.")
    args = parser.parse_args()

    outputs: list[tuple[Path, dict]] = []

    if args.scan_eos:
        by_period = build_definitions_from_eos(args)
        if not by_period:
            print("No dataset definitions were built.", file=sys.stderr)
            return 1
        for label, definitions in sorted(by_period.items()):
            stem = f"signal_AMSB_Wino_scan_ctau{args.ctau}_{label}{_suffix(args.tag)}"
            outputs.append((args.output_dir / f"{stem}.json", definitions))
            if args.per_mass:
                for mass in args.masses:
                    name = dataset_name(mass, args.ctau, args.tag, label)
                    if name in definitions:
                        outputs.append(
                            (
                                args.output_dir
                                / f"signal_AMSB_Wino_M{mass}_ctau{args.ctau}_{label}{_suffix(args.tag)}.json",
                                {name: definitions[name]},
                            )
                        )
    else:
        definitions = build_definitions(args)
        if not definitions:
            print("No dataset definitions were built.", file=sys.stderr)
            return 1

        combined_name = args.combined_name or f"signal_AMSB_Wino_scan_ctau{args.ctau}{_suffix(args.tag)}.json"
        outputs.append((args.output_dir / combined_name, definitions))

        if args.per_mass:
            for mass in args.masses:
                name = dataset_name(mass, args.ctau, args.tag)
                if name in definitions:
                    outputs.append(
                        (
                            args.output_dir / f"signal_AMSB_Wino_M{mass}_ctau{args.ctau}{_suffix(args.tag)}.json",
                            {name: definitions[name]},
                        )
                    )

    existing = [path for path, _ in outputs if path.exists()]
    if existing and not args.force:
        print("Refusing to overwrite existing dataset JSONs (use --force or --tag):", file=sys.stderr)
        for path in existing:
            print(f"  {path}", file=sys.stderr)
        return 1

    for path, definition in outputs:
        write_dataset_definition(definition, path)
        print(f"wrote {path}  ({len(definition)} dataset(s))")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
