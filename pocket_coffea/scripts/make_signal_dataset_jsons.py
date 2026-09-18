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
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from disapptrks.datasets import (  # noqa: E402
    build_dataset_definition,
    write_dataset_definition,
)

DEFAULT_MASSES = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200]
DEFAULT_EOS_DIR = (
    "root://cmseos.fnal.gov//store/group/lpcdisapptrks/nano/dev/SignalSim/2KEventQuickTest"
)
DEFAULT_PATTERN = "nano_AMSB_Wino_M{mass}GeV_ctau{ctau}cm_Run3Summer22.root"
DEFAULT_XSEC_JSON = REPO_ROOT / "src/disapptrks/cross_sections/amsb_wino_13p6TeV.json"


def _suffix(tag: str) -> str:
    return f"_{tag}" if tag else ""


def dataset_name(mass: int, ctau: int, tag: str = "") -> str:
    return f"Signal_AMSB_Wino_M{mass}_ctau{ctau}{_suffix(tag)}_OSUNano"


def sample_name(mass: int, ctau: int, tag: str = "") -> str:
    return f"SIG_AMSB_Wino_M{mass}_ctau{ctau}{_suffix(tag)}"


def load_cross_sections(path: Path | None) -> dict[str, dict]:
    if path is None:
        return {}
    if not path.is_file():
        print(f"Warning: cross-section file not found: {path}", file=sys.stderr)
        return {}
    with path.open() as handle:
        return json.load(handle).get("cross_sections", {})


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
    parser.add_argument("--force", action="store_true", help="Overwrite existing dataset JSONs instead of refusing.")
    args = parser.parse_args()

    definitions = build_definitions(args)
    if not definitions:
        print("No dataset definitions were built.", file=sys.stderr)
        return 1

    combined_name = args.combined_name or f"signal_AMSB_Wino_scan_ctau{args.ctau}{_suffix(args.tag)}.json"
    outputs: list[tuple[Path, dict]] = [(args.output_dir / combined_name, definitions)]

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
