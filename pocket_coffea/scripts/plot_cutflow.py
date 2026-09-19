#!/usr/bin/env python3
"""Make PocketCoffea-native cutflow plots from a DisappTrks coffea output."""

from __future__ import annotations

import argparse
from pathlib import Path

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create PocketCoffea cutflow and sum-of-weights plots from a "
            ".coffea output file."
        )
    )
    parser.add_argument("input", type=Path, help="PocketCoffea .coffea output file")
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("cutflow_plots"),
        help="Directory where plots will be written",
    )
    parser.add_argument(
        "--only-samples",
        nargs="+",
        default=None,
        help="Optional sample names to plot",
    )
    parser.add_argument(
        "--exclude-categories",
        nargs="+",
        default=None,
        help="Optional cutflow categories to skip",
    )
    parser.add_argument("--log-y", action="store_true", help="Use log scale")
    parser.add_argument(
        "--format",
        default="pdf",
        choices=("pdf", "png"),
        help="Output image format",
    )
    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Print the cutflow summary without making plots",
    )
    args = parser.parse_args()

    from coffea.util import load
    from pocket_coffea.utils.cutflow_utils import (
        plot_cutflow_from_output,
        print_cutflow_summary,
    )

    output = load(args.input)
    print_cutflow_summary(
        output,
        exclude_categories=args.exclude_categories,
        only_samples=args.only_samples,
    )

    if args.summary_only:
        return

    saved_files = plot_cutflow_from_output(
        output,
        output_dir=str(args.output_dir),
        exclude_categories=args.exclude_categories,
        only_samples=args.only_samples,
        log_y=args.log_y,
        output_format=args.format,
    )
    n_cutflow = len(saved_files.get("cutflow", ()))
    n_sumw = len(saved_files.get("sumw", ()))
    print(f"Wrote {n_cutflow} cutflow plots and {n_sumw} sum-of-weights plots")


if __name__ == "__main__":
    main()
