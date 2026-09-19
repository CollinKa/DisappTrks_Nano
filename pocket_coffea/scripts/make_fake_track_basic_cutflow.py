#!/usr/bin/env python3
"""Write the JetMET BasicSelection cutflow from fake-track coffea outputs.

This is a small convenience wrapper around ``disapptrks.tables`` for the
fake-track normalization side of the estimate.  It expects PocketCoffea outputs
from JetMET fake-track jobs, preferably produced with
``DISAPPTRKS_ENABLE_SEARCH_DIAGNOSTICS=1`` so that the detailed BasicSelection
rows are present.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from disapptrks.tables import write_fake_track_basic_cutflow_latex


def _sum_nested_numeric(left: Any, right: Any) -> Any:
    if isinstance(left, dict) and isinstance(right, dict):
        out = dict(left)
        for key, value in right.items():
            out[key] = _sum_nested_numeric(out[key], value) if key in out else value
        return out
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return left + right
    return right


def _load_merged_cutflow(files: list[Path]) -> dict[str, Any]:
    from coffea.util import load

    merged: dict[str, Any] = {}
    for path in files:
        output = load(path)
        if "cutflow" not in output:
            raise KeyError(f"{path} does not contain a 'cutflow' output")
        merged = _sum_nested_numeric(merged, output["cutflow"])
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Make the fake-track JetMET BasicSelection cutflow table from one "
            "or more fake_tracks .coffea outputs."
        )
    )
    parser.add_argument("files", nargs="+", type=Path, help="JetMET fake_tracks .coffea files")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output LaTeX file")
    parser.add_argument("--dataset", help="Restrict to one dataset key")
    parser.add_argument("--sample", help="Restrict to one sample key, e.g. DATA_JetMET")
    parser.add_argument("--variation", default="nominal", help="Variation to read")
    parser.add_argument(
        "--table-env",
        action="store_true",
        help="Wrap the tabular in a LaTeX table environment",
    )
    args = parser.parse_args()

    cutflow = _load_merged_cutflow(args.files)
    write_fake_track_basic_cutflow_latex(
        cutflow,
        args.output,
        dataset=args.dataset,
        sample=args.sample,
        variation=args.variation,
        include_table_env=args.table_env,
    )
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
