from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Union

from . import greet
from .datasets import (
    build_dataset_definition,
    group_osunano_files,
    list_eos_root_files,
    root_files_from_lines,
    scan_eos_bases_for_root_files,
    write_dataset_definition,
    write_grouped_filelists,
)
from .fake_tracks import (
    estimate_fake_track_background,
    estimate_fake_track_background_an,
    fit_dxy_transfer_factor,
    fit_signed_dxy_transfer_factor,
    fixed_an_transfer_factor_fit,
    plot_dxy_transfer_factor,
    summed_hist_counts_edges,
    write_an_fake_track_latex,
    write_fake_track_table34_latex,
    write_fake_track_z_control_latex,
    write_fake_track_latex,
)
from .fiducial import (
    make_fiducial_map_from_outputs,
    plot_fiducial_map_payload,
    write_fiducial_map_payload,
)
from .lepton_backgrounds import (
    estimate_lepton_background,
    write_lepton_background_json,
    write_lepton_background_latex,
)
from .schema import audit_root_file
from .summaries import (
    summarize_ss_subtracted_veto_probability,
    summarize_veto_probability,
)
from .tables import (
    variable_count_sum,
    write_fake_track_basic_cutflow_latex,
    write_lepton_pveto_cutflow_latex,
    write_merged_pveto_latex,
    write_muon_cutflow_latex,
    write_muon_pveto_latex,
)

PairVariableTemplate = Union[str, tuple[str, str]]


def _sum_nested_numeric(left, right):
    if isinstance(left, dict) and isinstance(right, dict):
        out = dict(left)
        for key, value in right.items():
            out[key] = _sum_nested_numeric(out[key], value) if key in out else value
        return out
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return left + right
    return right


def _load_merged_cutflow(files: list[Path]) -> dict:
    from coffea.util import load

    merged = {}
    for path in files:
        output = load(path)
        merged = _sum_nested_numeric(merged, output["cutflow"])
    return merged


def _load_outputs(files: list[Path]) -> list[dict]:
    from coffea.util import load

    return [load(path) for path in files]


def _pair_counts_from_outputs(
    outputs: list[dict],
    *,
    layers: list[str],
    variable_templates: dict[str, PairVariableTemplate],
    dataset: str | None = None,
    sample: str | None = None,
) -> dict[str, dict[str, float]]:
    pair_counts = {}
    for layer in layers:
        suffix = "" if layer == "combinedBins" else f"_{layer}"
        totals = {"den_os": 0.0, "num_os": 0.0, "den_ss": 0.0, "num_ss": 0.0}
        variables = {
            key: _pair_variable_name(template, layer=layer, suffix=suffix)
            for key, template in variable_templates.items()
        }
        for output in outputs:
            output_variables = output.get("variables", {})
            for key, variable in variables.items():
                totals[key] += variable_count_sum(
                    output_variables,
                    variable,
                    dataset=dataset,
                    sample=sample,
                )
        pair_counts[layer] = totals
    return pair_counts


def _has_pair_count_variables(
    outputs: list[dict],
    *,
    layers: list[str],
    variable_templates: dict[str, PairVariableTemplate],
) -> bool:
    wanted = set()
    for layer in layers:
        suffix = "" if layer == "combinedBins" else f"_{layer}"
        wanted.update(
            _pair_variable_name(template, layer=layer, suffix=suffix)
            for template in variable_templates.values()
        )
    return any(wanted.intersection(output.get("variables", {}).keys()) for output in outputs)


def _pair_variable_name(
    template: PairVariableTemplate,
    *,
    layer: str,
    suffix: str,
) -> str:
    if isinstance(template, tuple):
        combined_template, layer_template = template
        template = combined_template if layer == "combinedBins" else layer_template
    return template.format(suffix=suffix)


def _muon_pair_counts_from_outputs(
    outputs: list[dict],
    *,
    layers: list[str],
    dataset: str | None = None,
    sample: str | None = None,
) -> dict[str, dict[str, float]]:
    templates = {
        "den_os": "nMuonVetoTagProbePairZWindow{suffix}",
        "num_os": "nMuonPVetoTagProbePairZWindowPass{suffix}",
        "den_ss": "nMuonVetoTagProbePairSSZWindow{suffix}",
        "num_ss": "nMuonPVetoTagProbePairSSZWindowPass{suffix}",
    }
    if not _has_pair_count_variables(outputs, layers=layers, variable_templates=templates):
        return {}
    return _pair_counts_from_outputs(
        outputs,
        layers=layers,
        variable_templates=templates,
        dataset=dataset,
        sample=sample,
    )


def _sum_pair_count_maps(
    left: dict[str, dict[str, float]],
    right: dict[str, dict[str, float]],
) -> dict[str, dict[str, float]]:
    layers = set(left) | set(right)
    keys = ("den_os", "num_os", "den_ss", "num_ss")
    return {
        layer: {
            key: left.get(layer, {}).get(key, 0.0) + right.get(layer, {}).get(key, 0.0)
            for key in keys
        }
        for layer in layers
    }


LEPTON_PVETO_PAIR_VARIABLES = {
    "electron": {
        "den_os": (
            "nElectronTagProbePairOSMassWindow",
            "nElectronTagProbePairMassWindow{suffix}",
        ),
        "num_os": "nElectronPVetoTagProbePairMassWindowPass{suffix}",
        "den_ss": "nElectronTagProbePairSSMassWindow{suffix}",
        "num_ss": "nElectronPVetoTagProbePairSSMassWindowPass{suffix}",
    },
    "tau_mu": {
        "den_os": (
            "nTauMuTagProbePairOSMassWindow",
            "nTauMuTagProbePairMassWindow{suffix}",
        ),
        "num_os": "nTauMuPVetoTagProbePairMassWindowPass{suffix}",
        "den_ss": "nTauMuTagProbePairSSMassWindow{suffix}",
        "num_ss": "nTauMuPVetoTagProbePairSSMassWindowPass{suffix}",
    },
    "tau_ele": {
        "den_os": (
            "nTauEleTagProbePairOSMassWindow",
            "nTauEleTagProbePairMassWindow{suffix}",
        ),
        "num_os": "nTauElePVetoTagProbePairMassWindowPass{suffix}",
        "den_ss": "nTauEleTagProbePairSSMassWindow{suffix}",
        "num_ss": "nTauElePVetoTagProbePairSSMassWindowPass{suffix}",
    },
}


def _audit_command(args: argparse.Namespace) -> int:
    report = audit_root_file(args.file)
    print(json.dumps(report.as_dict(), indent=2, sort_keys=True))
    return 0 if report.ready_for(args.scope) else 2


def _summarize_pveto_command(args: argparse.Namespace) -> int:
    from coffea.util import load

    output = load(args.file)
    if args.ss_subtract:
        summary = summarize_ss_subtracted_veto_probability(
            output["cutflow"],
            os_denominator_name=args.denominator,
            os_numerator_name=args.numerator,
            ss_denominator_name=args.ss_denominator,
            ss_numerator_name=args.ss_numerator,
            dataset=args.dataset,
            sample=args.sample,
            variation=args.variation,
        )

        if args.json:
            print(json.dumps(summary.as_dict(), indent=2, sort_keys=True))
            return 0

        print(f"OS denominator ({summary.os_denominator_name}): {summary.os_denominator:g}")
        print(f"OS numerator   ({summary.os_numerator_name}): {summary.os_numerator:g}")
        print(f"SS denominator ({summary.ss_denominator_name}): {summary.ss_denominator:g}")
        print(f"SS numerator   ({summary.ss_numerator_name}): {summary.ss_numerator:g}")
        print(f"Subtracted denominator: {summary.denominator:g}")
        print(f"Subtracted numerator:   {summary.numerator:g}")
        print(f"P(pass veto): {summary.probability:.6g} ± {summary.uncertainty:.6g}")
        return 0

    summary = summarize_veto_probability(
        output["cutflow"],
        denominator_name=args.denominator,
        numerator_name=args.numerator,
        dataset=args.dataset,
        sample=args.sample,
        variation=args.variation,
    )

    if args.json:
        print(json.dumps(summary.as_dict(), indent=2, sort_keys=True))
        return 0

    print(f"Denominator ({summary.denominator_name}): {summary.denominator:g}")
    print(f"Numerator   ({summary.numerator_name}): {summary.numerator:g}")
    print(f"P(pass veto): {summary.probability:.6g} ± {summary.uncertainty:.6g}")
    return 0


def _estimate_fake_tracks_command(args: argparse.Namespace) -> int:
    if args.an_control:
        if args.counts_json:
            raise SystemExit("error: --an-control requires coffea files, not --counts-json")
        if not args.files:
            raise SystemExit("error: at least one coffea file is required with --an-control")
        if args.basic_files and not args.basic_yield_category:
            raise SystemExit("error: --basic-files requires --basic-yield-category")

        outputs = _load_outputs(args.files)
        source = _load_merged_cutflow(args.files)
        basic_source = _load_merged_cutflow(args.basic_files) if args.basic_files else None
        if args.basic_cutflow_tex:
            if basic_source is None:
                raise SystemExit(
                    "error: --basic-cutflow-tex requires --basic-files when using --an-control"
                )
            write_fake_track_basic_cutflow_latex(
                basic_source,
                args.basic_cutflow_tex,
                dataset=args.dataset if args.basic_dataset is None else args.basic_dataset,
                sample=args.sample if args.basic_sample is None else args.basic_sample,
                variation=args.variation
                if args.basic_variation is None
                else args.basic_variation,
                include_table_env=args.table_env,
            )
            print(f"Wrote {args.basic_cutflow_tex}")
        control_cfg = {
            "zmumu": {
                "control_region": r"$Z\to\mu\mu$",
                "histogram": "fakeZMuMuFitTrack_absDxy",
                "signed_histogram": "fakeZMuMuFitTrack_dxy",
                "control_category": "fake_zmumu_control",
                "sideband_category": "fake_zmumu_sideband_{layer}",
            },
            "zee": {
                "control_region": r"$Z\to ee$",
                "histogram": "fakeZeeFitTrack_absDxy",
                "signed_histogram": "fakeZeeFitTrack_dxy",
                "control_category": "fake_zee_control",
                "sideband_category": "fake_zee_sideband_{layer}",
            },
        }[args.an_control]

        counts = edges = signed_counts = signed_edges = None
        if args.transfer_factor_source == "fixed":
            if args.fit_plot:
                raise SystemExit(
                    "error: --fit-plot requires --transfer-factor-source fit"
                )
            fit = fixed_an_transfer_factor_fit(args.run_period, args.an_control)
        else:
            counts, edges = summed_hist_counts_edges(
                outputs,
                control_cfg["histogram"],
                dataset=args.dataset,
                sample=args.sample,
            )
            try:
                signed_counts, signed_edges = summed_hist_counts_edges(
                    outputs,
                    control_cfg["signed_histogram"],
                    dataset=args.dataset,
                    sample=args.sample,
                )
            except KeyError:
                signed_counts, signed_edges = None, None

            if signed_counts is None or signed_edges is None:
                fit = fit_dxy_transfer_factor(
                    counts,
                    edges,
                    control_region=control_cfg["control_region"],
                    histogram=control_cfg["histogram"],
                )
            else:
                fit = fit_signed_dxy_transfer_factor(
                    signed_counts,
                    signed_edges,
                    control_region=control_cfg["control_region"],
                    histogram=control_cfg["signed_histogram"],
                )

        if args.fit_plot:
            plot_dxy_transfer_factor(
                counts,
                edges,
                fit,
                args.fit_plot,
                signed_counts=signed_counts,
                signed_edges=signed_edges,
                title=f"{args.run_period} {control_cfg['control_region']} n_layers = 4",
            )
            print(f"Wrote {args.fit_plot}")

        estimates = [
            estimate_fake_track_background_an(
                source,
                layer=layer,
                control_region=control_cfg["control_region"],
                transfer_factor=fit.transfer_factor,
                control_category=control_cfg["control_category"],
                sideband_category=control_cfg["sideband_category"].format(layer=layer),
                basic_yield_category=args.basic_yield_category,
                dataset=args.dataset,
                sample=args.sample,
                variation=args.variation,
                basic_cutflow=basic_source,
                basic_dataset=args.basic_dataset,
                basic_sample=args.basic_sample,
                basic_variation=args.basic_variation,
            )
            for layer in args.layers
        ]

        if args.z_control_tex:
            write_fake_track_z_control_latex(
                estimates,
                args.z_control_tex,
                run_period=args.run_period,
                include_table_env=args.table_env,
            )
            print(f"Wrote {args.z_control_tex}")

        if args.output_json:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "fit": {
                    "control_region": fit.control_region,
                    "histogram": fit.histogram,
                    "numerator_range": fit.numerator_range,
                    "denominator_range": fit.denominator_range,
                    "fit_range": fit.fit_range,
                    "amplitude": fit.amplitude,
                    "sigma": fit.sigma,
                    "constant": fit.constant,
                    "transfer_factor": {
                        "value": fit.transfer_factor.value,
                        "error": fit.transfer_factor.error,
                        "variance": fit.transfer_factor.variance,
                    },
                },
                "estimates": [e.as_dict() for e in estimates],
            }
            args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True))
            print(f"Wrote {args.output_json}")

        if args.output_tex:
            write_an_fake_track_latex(
                estimates,
                fit,
                args.output_tex,
                run_period=args.run_period,
                include_table_env=args.table_env,
            )
            print(f"Wrote {args.output_tex}")

        source_detail = f"{args.transfer_factor_source} transfer factor"
        if args.transfer_factor_source == "fit":
            source_detail += f", fit sigma={fit.sigma:.6g}"
        print(
            f"{control_cfg['control_region']}: "
            f"zeta={fit.transfer_factor.value:.6g} ± {fit.transfer_factor.error:.6g} "
            f"({source_detail})"
        )
        for estimate in estimates:
            fake_yield = (
                "N_fake=--"
                if estimate.fake_yield is None
                else f"N_fake={estimate.fake_yield.value:.6g} ± {estimate.fake_yield.error:.6g}"
            )
            print(
                f"{estimate.layer}: "
                f"N_Z={estimate.control_events.value:.6g}, "
                f"N_sideband={estimate.sideband_events.value:.6g}, "
                f"P_raw={estimate.raw_probability.value:.6g} ± {estimate.raw_probability.error:.6g}, "
                f"P_fake={estimate.fake_probability.value:.6g} ± {estimate.fake_probability.error:.6g}, "
                f"{fake_yield}"
            )
        return 0

    if args.z_control_tex:
        raise SystemExit("error: --z-control-tex requires --an-control")

    if args.counts_json:
        source = json.loads(args.counts_json.read_text())
        source_is_cutflow = False
    else:
        if not args.files:
            raise SystemExit("error: at least one coffea file is required unless --counts-json is used")
        source = _load_merged_cutflow(args.files)
        source_is_cutflow = True

    if args.basic_cutflow_tex:
        if not source_is_cutflow:
            raise SystemExit("error: --basic-cutflow-tex requires coffea input, not --counts-json")
        write_fake_track_basic_cutflow_latex(
            source,
            args.basic_cutflow_tex,
            dataset=args.dataset,
            sample=args.sample,
            variation=args.variation,
            include_table_env=args.table_env,
        )
        print(f"Wrote {args.basic_cutflow_tex}")

    estimates = [
        estimate_fake_track_background(
            source,
            layer=layer,
            transfer_signal_category=args.transfer_signal_category,
            transfer_sideband_category=args.transfer_sideband_category,
            control_category=args.control_category,
            basic_yield_category=args.basic_yield_category,
            z_to_ll_yield_category=args.z_to_ll_yield_category,
            dataset=args.dataset,
            sample=args.sample,
            variation=args.variation,
            source_is_cutflow=source_is_cutflow,
            prescale=args.prescale,
        )
        for layer in args.layers
    ]

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps([e.as_dict() for e in estimates], indent=2, sort_keys=True))
        print(f"Wrote {args.output_json}")

    if args.output_tex:
        write_fake_track_latex(
            estimates,
            args.output_tex,
            run_period=args.run_period,
            include_table_env=args.table_env,
        )
        print(f"Wrote {args.output_tex}")

    for estimate in estimates:
        print(
            f"{estimate.layer}: "
            f"xi={estimate.transfer_factor.value:.6g} ± {estimate.transfer_factor.error:.6g}, "
            f"N_ctrl={estimate.control.value:.6g} ± {estimate.control.error:.6g}, "
            f"N_fake={estimate.estimate.value:.6g} ± {estimate.estimate.error:.6g}"
        )
    return 0


def _make_fake_track_table34_command(args: argparse.Namespace) -> int:
    write_fake_track_table34_latex(
        args.jsons,
        args.output,
        run_period=args.run_period,
        include_table_env=args.table_env,
    )
    print(f"Wrote {args.output}")
    return 0


def _make_dataset_json_command(args: argparse.Namespace) -> int:
    if args.filelist:
        files = root_files_from_lines(args.filelist.read_text().splitlines())
    else:
        if not args.eos_path:
            raise SystemExit("error: eos_path is required unless --filelist is used")
        files = list_eos_root_files(
            args.eos_path,
            xrootd=args.xrootd,
            recursive=args.recursive,
        )

    dataset = build_dataset_definition(
        dataset_name=args.dataset_name,
        files=files,
        sample=args.sample,
        year=args.year,
        era=args.era,
        primary_dataset=args.primary_dataset,
        is_mc=args.is_mc,
        nevents=args.nevents,
        nano_version=args.nano_version,
    )
    write_dataset_definition(dataset, args.output)

    print(f"Wrote {len(files)} ROOT files to {args.output}")
    if not files:
        print("Warning: no ROOT files found")
        return 2
    return 0


def _make_era_filelists_command(args: argparse.Namespace) -> int:
    if args.filelist:
        files = root_files_from_lines(args.filelist.read_text().splitlines())
    else:
        files = scan_eos_bases_for_root_files(args.eos_paths, xrootd=args.xrootd)

    grouped = group_osunano_files(
        files,
        prod_version_policy=args.prod_version_policy,
    )
    outputs = write_grouped_filelists(
        grouped,
        output_dir=args.output_dir,
        dataset_json_dir=args.dataset_json_dir,
        nano_version=args.nano_version,
    )

    print(f"Scanned {len(files)} ROOT file(s)")
    for key, entry in outputs.items():
        suffix = ""
        if "dataset_json" in entry:
            suffix = f", json={entry['dataset_json']}"
        print(f"{key:18s} {entry['n_files']:6d} files -> {entry['filelist']}{suffix}")

    empty = [key for key, entry in outputs.items() if entry["n_files"] == 0]
    if empty:
        print("Warning: empty groups: " + ", ".join(empty))
        return 2 if args.fail_on_empty else 0
    return 0


def _make_pveto_tables_command(args: argparse.Namespace) -> int:
    outputs = _load_outputs(args.files)
    cutflow = {}
    for output in outputs:
        cutflow = _sum_nested_numeric(cutflow, output["cutflow"])
    pair_counts = _muon_pair_counts_from_outputs(
        outputs,
        layers=args.layers,
        dataset=args.dataset,
        sample=args.sample,
    )

    write_muon_cutflow_latex(
        cutflow,
        args.cutflow_tex,
        dataset=args.dataset,
        sample=args.sample,
        variation=args.variation,
        include_table_env=args.table_env,
    )
    summaries = write_muon_pveto_latex(
        cutflow,
        args.pveto_tex,
        run_period=args.run_period,
        flavor=args.flavor,
        layers=args.layers,
        dataset=args.dataset,
        sample=args.sample,
        variation=args.variation,
        os_denominator_name=args.denominator,
        os_numerator_name=args.numerator,
        ss_denominator_name=args.ss_denominator,
        ss_numerator_name=args.ss_numerator,
        include_table_env=args.table_env,
        pair_counts=pair_counts,
    )

    print(f"Wrote {args.cutflow_tex}")
    print(f"Wrote {args.pveto_tex}")
    for layer, summary in summaries.items():
        print(
            f"{layer}: P_veto = "
            f"{summary.central:.6g} +{summary.err_up:.6g} -{summary.err_down:.6g} "
            f"(signed numerator={summary.numerator:g}, denominator={summary.denominator:g})"
        )
    return 0


def _make_lepton_pveto_table_command(args: argparse.Namespace) -> int:
    outputs = _load_outputs(args.files)
    cutflow = {}
    for output in outputs:
        cutflow = _sum_nested_numeric(cutflow, output["cutflow"])

    if args.mode == "tau_mu":
        defaults = {
            "os_denominator_name": "tau_mu_veto_masswindow",
            "os_numerator_name": "tau_mu_pveto_masswindow_pass",
            "ss_denominator_name": "tau_mu_veto_ss_masswindow",
            "ss_numerator_name": "tau_mu_pveto_ss_masswindow_pass",
            "flavor": r"$\tau_{\mu}$",
        }
    elif args.mode == "tau_ele":
        defaults = {
            "os_denominator_name": "tau_ele_veto_masswindow",
            "os_numerator_name": "tau_ele_pveto_masswindow_pass",
            "ss_denominator_name": "tau_ele_veto_ss_masswindow",
            "ss_numerator_name": "tau_ele_pveto_ss_masswindow_pass",
            "flavor": r"$\tau_{e}$",
        }
    elif args.mode == "electron":
        defaults = {
            "os_denominator_name": "electron_veto_zwindow",
            "os_numerator_name": "electron_pveto_zwindow_pass",
            "ss_denominator_name": "electron_veto_ss_zwindow",
            "ss_numerator_name": "electron_pveto_ss_zwindow_pass",
            "flavor": r"$e$",
        }
    else:
        raise ValueError(f"unknown lepton Pveto mode: {args.mode}")

    pair_count_templates = LEPTON_PVETO_PAIR_VARIABLES[args.mode]
    pair_counts = (
        _pair_counts_from_outputs(
            outputs,
            layers=args.layers,
            variable_templates=pair_count_templates,
            dataset=args.dataset,
            sample=args.sample,
        )
        if _has_pair_count_variables(
            outputs,
            layers=args.layers,
            variable_templates=pair_count_templates,
        )
        else {}
    )

    if args.cutflow_tex is not None:
        write_lepton_pveto_cutflow_latex(
            cutflow,
            args.cutflow_tex,
            mode=args.mode,
            dataset=args.dataset,
            sample=args.sample,
            variation=args.variation,
            include_table_env=args.table_env,
            layout=args.cutflow_layout,
        )
        print(f"Wrote {args.cutflow_tex}")

    summaries = write_muon_pveto_latex(
        cutflow,
        args.output,
        run_period=args.run_period,
        flavor=args.flavor or defaults["flavor"],
        layers=args.layers,
        dataset=args.dataset,
        sample=args.sample,
        variation=args.variation,
        os_denominator_name=args.denominator or defaults["os_denominator_name"],
        os_numerator_name=args.numerator or defaults["os_numerator_name"],
        ss_denominator_name=args.ss_denominator or defaults["ss_denominator_name"],
        ss_numerator_name=args.ss_numerator or defaults["ss_numerator_name"],
        include_table_env=args.table_env,
        pair_counts=pair_counts,
    )

    print(f"Wrote {args.output}")
    for layer, summary in summaries.items():
        print(
            f"{layer}: P_veto = "
            f"{summary.central:.6g} +{summary.err_up:.6g} -{summary.err_down:.6g} "
            f"(signed numerator={summary.numerator:g}, denominator={summary.denominator:g})"
        )
    return 0


def _make_tau_pveto_table_command(args: argparse.Namespace) -> int:
    tau_mu_outputs = _load_outputs(args.tau_mu_files)
    tau_ele_outputs = _load_outputs(args.tau_ele_files)

    tau_mu_pair_counts = _pair_counts_from_outputs(
        tau_mu_outputs,
        layers=args.layers,
        variable_templates=LEPTON_PVETO_PAIR_VARIABLES["tau_mu"],
        dataset=args.tau_mu_dataset or args.dataset,
        sample=args.tau_mu_sample,
    )
    tau_ele_pair_counts = _pair_counts_from_outputs(
        tau_ele_outputs,
        layers=args.layers,
        variable_templates=LEPTON_PVETO_PAIR_VARIABLES["tau_ele"],
        dataset=args.tau_ele_dataset or args.dataset,
        sample=args.tau_ele_sample,
    )
    combined_pair_counts = _sum_pair_count_maps(tau_mu_pair_counts, tau_ele_pair_counts)

    summaries = write_muon_pveto_latex(
        {},
        args.output,
        run_period=args.run_period,
        flavor=args.flavor,
        layers=args.layers,
        include_table_env=args.table_env,
        pair_counts=combined_pair_counts,
    )

    print(f"Wrote {args.output}")
    for layer, summary in summaries.items():
        counts = combined_pair_counts.get(layer, {})
        print(
            f"{layer}: P_veto = "
            f"{summary.central:.6g} +{summary.err_up:.6g} -{summary.err_down:.6g} "
            f"(N_OS={counts.get('den_os', 0):g}, N_veto_OS={counts.get('num_os', 0):g}, "
            f"N_SS={counts.get('den_ss', 0):g}, N_veto_SS={counts.get('num_ss', 0):g}; "
            f"signed numerator={summary.numerator:g}, denominator={summary.denominator:g})"
        )
    return 0


def _estimate_lepton_background_command(args: argparse.Namespace) -> int:
    outputs = _load_outputs(args.files)
    cutflow = {}
    for output in outputs:
        cutflow = _sum_nested_numeric(cutflow, output["cutflow"])

    if args.mode == "muon":
        pair_counts = _muon_pair_counts_from_outputs(
            outputs,
            layers=args.layers,
            dataset=args.dataset,
            sample=args.sample,
        )
        flavor = args.flavor or r"$\mu$"
    else:
        pair_counts = _pair_counts_from_outputs(
            outputs,
            layers=args.layers,
            variable_templates=LEPTON_PVETO_PAIR_VARIABLES[args.mode],
            dataset=args.dataset,
            sample=args.sample,
        )
        flavor = args.flavor or {
            "electron": r"$e$",
            "tau_mu": r"$\tau_{\mu}$",
            "tau_ele": r"$\tau_{e}$",
        }[args.mode]

    control_category = args.control_category or f"{args.mode}_background_control_{{layer}}"
    poffline_numerator_category = (
        args.poffline_numerator_category
        or f"{args.mode}_background_offline_{{layer}}"
    )
    poffline_denominator_category = (
        args.poffline_denominator_category
        or control_category
    )
    pmiss_numerator_category = (
        args.pmiss_numerator_category
        or args.ptrigger_numerator_category
        or f"{args.mode}_background_trigger_{{layer}}"
    )
    pmiss_denominator_category = (
        args.pmiss_denominator_category
        or args.ptrigger_denominator_category
        or poffline_numerator_category
    )

    estimates = estimate_lepton_background(
        flavor=flavor,
        layers=args.layers,
        pair_counts=pair_counts,
        cutflow=cutflow,
        control_category=control_category,
        poffline_numerator_category=poffline_numerator_category,
        poffline_denominator_category=poffline_denominator_category,
        pmiss_numerator_category=pmiss_numerator_category,
        pmiss_denominator_category=pmiss_denominator_category,
        dataset=args.dataset,
        sample=args.sample,
        variation=args.variation,
    )

    if args.output_json is not None:
        write_lepton_background_json(estimates, args.output_json)
        print(f"Wrote {args.output_json}")
    if args.output_tex is not None:
        write_lepton_background_latex(
            estimates,
            args.output_tex,
            run_period=args.run_period,
            include_table_env=args.table_env,
        )
        print(f"Wrote {args.output_tex}")

    for estimate in estimates:
        print(
            f"{estimate.layer}: N_lepton = "
            f"{estimate.estimate.value:.6g} ± {estimate.estimate.error:.6g} "
            f"(P_veto={estimate.p_veto.value:.6g}, "
            f"P_offline={estimate.p_offline.value:.6g}, "
            f"P_miss={estimate.p_miss.value:.6g})"
        )
    return 0


def _make_fiducial_map_command(args: argparse.Namespace) -> int:
    outputs = _load_outputs(args.files)
    prefix = {"electron": "electron", "muon": "muon"}[args.flavor]
    before_variable = args.before_variable or f"{prefix}FiducialBefore_eta_phi"
    after_variable = args.after_variable or f"{prefix}FiducialAfter_eta_phi"
    output_npz = args.output_npz
    if output_npz is None and not args.no_npz:
        output_npz = args.output_json.with_suffix(".npz")

    summary, before, after, eta_edges, phi_edges = make_fiducial_map_from_outputs(
        outputs,
        before_variable=before_variable,
        after_variable=after_variable,
        dataset=args.dataset,
        sample=args.sample,
        category=args.category,
        threshold=args.threshold,
    )
    write_fiducial_map_payload(
        summary,
        before=before,
        after=after,
        eta_edges=eta_edges,
        phi_edges=phi_edges,
        output_json=args.output_json,
        output_npz=output_npz,
        metadata={
            "flavor": args.flavor,
            "run_period": args.run_period,
            "before_variable": before_variable,
            "after_variable": after_variable,
            "dataset": args.dataset,
            "sample": args.sample,
            "category": args.category,
            "threshold": args.threshold,
            "input_files": [str(path) for path in args.files],
        },
    )

    print(f"Wrote {args.output_json}")
    if output_npz is not None:
        print(f"Wrote {output_npz}")
    print(
        f"{args.flavor}: mean inefficiency={summary.mean_inefficiency:.6g}, "
        f"stddev={summary.stddev_inefficiency:.6g}, "
        f"hot spots={len(summary.hot_spots)}"
    )
    for hot_spot in summary.hot_spots:
        print(
            f"  eta={hot_spot.eta:.3f}, phi={hot_spot.phi:.3f}, "
            f"radius={hot_spot.radius:.4f}, sigma={hot_spot.sigma:.3f}"
        )
    return 0


def _plot_fiducial_map_command(args: argparse.Namespace) -> int:
    written = plot_fiducial_map_payload(
        args.npz,
        output_prefix=args.output_prefix,
        flavor=args.flavor,
        json_path=args.json,
        run_period=args.run_period,
        lumi_text=args.lumi_text,
        cms_label=args.cms_label,
        formats=args.formats,
        draw_hot_spots=not args.no_hot_spots,
        colormap=args.colormap,
    )
    for path in written:
        print(f"Wrote {path}")
    return 0


def _merge_pveto_tables_command(args: argparse.Namespace) -> int:
    write_merged_pveto_latex(
        args.tables,
        args.output,
        include_table_env=args.table_env,
        keep_combined=args.keep_combined,
        flavor=args.flavor,
        compact_layer_labels=not args.no_compact_layer_labels,
    )
    print(f"Wrote {args.output}")
    return 0


def main():
    parser = argparse.ArgumentParser(prog="disapptrks")
    subparsers = parser.add_subparsers(dest="command")

    audit = subparsers.add_parser(
        "audit-schema",
        help="Check a custom NanoAOD file against the analysis branch contract.",
    )
    audit.add_argument("file", type=Path)
    audit.add_argument(
        "--scope",
        choices=("search", "backgrounds", "fiducial-maps"),
        default="backgrounds",
    )
    audit.set_defaults(func=_audit_command)

    pveto = subparsers.add_parser(
        "summarize-pveto",
        help="Summarize a veto pass probability from a PocketCoffea output file.",
    )
    pveto.add_argument("file", type=Path)
    pveto.add_argument("--dataset", help="Restrict to one dataset key.")
    pveto.add_argument("--sample", help="Restrict to one sample key.")
    pveto.add_argument("--variation", default="nominal")
    pveto.add_argument("--json", action="store_true", help="Print JSON output.")
    pveto.add_argument(
        "--ss-subtract",
        action="store_true",
        help="Compute (OS numerator - SS numerator) / (OS denominator - SS denominator).",
    )
    pveto.add_argument(
        "--denominator",
        default="muon_veto_zwindow",
        help="OS cutflow category used as denominator.",
    )
    pveto.add_argument(
        "--numerator",
        default="muon_pveto_zwindow_pass",
        help="OS cutflow category used as numerator.",
    )
    pveto.add_argument(
        "--ss-denominator",
        default="muon_veto_ss_zwindow",
        help="SS cutflow category subtracted from the denominator.",
    )
    pveto.add_argument(
        "--ss-numerator",
        default="muon_pveto_ss_zwindow_pass",
        help="SS cutflow category subtracted from the numerator.",
    )
    pveto.set_defaults(func=_summarize_pveto_command)

    pveto_tables = subparsers.add_parser(
        "make-pveto-tables",
        help="Write AN-style muon Pveto cutflow and probability LaTeX tables.",
    )
    pveto_tables.add_argument("files", nargs="+", type=Path)
    pveto_tables.add_argument("--dataset", help="Restrict to one dataset key.")
    pveto_tables.add_argument("--sample", help="Restrict to one sample key.")
    pveto_tables.add_argument("--variation", default="nominal")
    pveto_tables.add_argument("--run-period", required=True, help="Run-period label used in the Pveto table.")
    pveto_tables.add_argument("--flavor", default=r"$\mu$", help="Flavor label used in the Pveto table.")
    pveto_tables.add_argument(
        "--layers",
        nargs="+",
        default=["NLayers4", "NLayers5", "NLayers6plus", "combinedBins"],
        help="Layer-bin rows to include in the Pveto table.",
    )
    pveto_tables.add_argument(
        "--cutflow-tex",
        type=Path,
        default=Path("muon_pveto_cutflow.tex"),
        help="Output path for the cutflow LaTeX table.",
    )
    pveto_tables.add_argument(
        "-o",
        "--pveto-tex",
        type=Path,
        default=Path("muon_pveto_table.tex"),
        help="Output path for the Pveto LaTeX table.",
    )
    pveto_tables.add_argument(
        "--table-env",
        action="store_true",
        help="Wrap each tabular in a LaTeX table environment.",
    )
    pveto_tables.add_argument(
        "--denominator",
        default="muon_veto_zwindow",
        help="OS cutflow category used as denominator.",
    )
    pveto_tables.add_argument(
        "--numerator",
        default="muon_pveto_zwindow_pass",
        help="OS cutflow category used as numerator.",
    )
    pveto_tables.add_argument(
        "--ss-denominator",
        default="muon_veto_ss_zwindow",
        help="SS cutflow category subtracted from the denominator.",
    )
    pveto_tables.add_argument(
        "--ss-numerator",
        default="muon_pveto_ss_zwindow_pass",
        help="SS cutflow category subtracted from the numerator.",
    )
    pveto_tables.set_defaults(func=_make_pveto_tables_command)

    lepton_pveto_table = subparsers.add_parser(
        "make-lepton-pveto-table",
        help=(
            "Write an AN-style Pveto probability table for electron or tau "
            "tag-and-probe outputs, merging one or more coffea files."
        ),
    )
    lepton_pveto_table.add_argument("files", nargs="+", type=Path)
    lepton_pveto_table.add_argument(
        "--mode",
        required=True,
        choices=("electron", "tau_mu", "tau_ele"),
        help="Category-name preset to use.",
    )
    lepton_pveto_table.add_argument("--dataset", help="Restrict to one dataset key.")
    lepton_pveto_table.add_argument("--sample", help="Restrict to one sample key.")
    lepton_pveto_table.add_argument("--variation", default="nominal")
    lepton_pveto_table.add_argument(
        "--run-period",
        required=True,
        help="Run-period label used in the Pveto table.",
    )
    lepton_pveto_table.add_argument(
        "--flavor",
        help="Override the flavor label used in the Pveto table.",
    )
    lepton_pveto_table.add_argument(
        "--layers",
        nargs="+",
        default=["NLayers4", "NLayers5", "NLayers6plus", "combinedBins"],
        help="Layer-bin rows to include in the Pveto table.",
    )
    lepton_pveto_table.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output path for the Pveto LaTeX table.",
    )
    lepton_pveto_table.add_argument(
        "--cutflow-tex",
        type=Path,
        help="Also write a compact diagnostic cutflow LaTeX table.",
    )
    lepton_pveto_table.add_argument(
        "--cutflow-layout",
        choices=["diagnostic", "an22_23"],
        default="diagnostic",
        help=(
            "Cutflow row layout. Use an22_23 for tau electron/muon legs in "
            "the compact AN Table 22/23 order."
        ),
    )
    lepton_pveto_table.add_argument(
        "--table-env",
        action="store_true",
        help="Wrap the tabular in a LaTeX table environment.",
    )
    lepton_pveto_table.add_argument(
        "--denominator",
        help="Override the OS cutflow category used as denominator.",
    )
    lepton_pveto_table.add_argument(
        "--numerator",
        help="Override the OS cutflow category used as numerator.",
    )
    lepton_pveto_table.add_argument(
        "--ss-denominator",
        help="Override the SS cutflow category subtracted from the denominator.",
    )
    lepton_pveto_table.add_argument(
        "--ss-numerator",
        help="Override the SS cutflow category subtracted from the numerator.",
    )
    lepton_pveto_table.set_defaults(func=_make_lepton_pveto_table_command)

    tau_pveto_table = subparsers.add_parser(
        "make-tau-pveto-table",
        help=(
            "Write a combined tau Pveto table by summing tau_mu and tau_ele "
            "raw tag-and-probe counts before computing the SS-subtracted probability."
        ),
    )
    tau_pveto_table.add_argument(
        "--tau-mu-files",
        nargs="+",
        type=Path,
        required=True,
        help="tau_mu_pveto PocketCoffea output files.",
    )
    tau_pveto_table.add_argument(
        "--tau-ele-files",
        nargs="+",
        type=Path,
        required=True,
        help="tau_ele_pveto PocketCoffea output files.",
    )
    tau_pveto_table.add_argument(
        "--dataset",
        help="Restrict both tau_mu and tau_ele inputs to one dataset key.",
    )
    tau_pveto_table.add_argument(
        "--tau-mu-dataset",
        help="Restrict tau_mu inputs to one dataset key. Overrides --dataset.",
    )
    tau_pveto_table.add_argument(
        "--tau-ele-dataset",
        help="Restrict tau_ele inputs to one dataset key. Overrides --dataset.",
    )
    tau_pveto_table.add_argument(
        "--tau-mu-sample",
        default="DATA_Muon",
        help="Restrict tau_mu inputs to one sample key.",
    )
    tau_pveto_table.add_argument(
        "--tau-ele-sample",
        default="DATA_EGamma",
        help="Restrict tau_ele inputs to one sample key.",
    )
    tau_pveto_table.add_argument(
        "--run-period",
        required=True,
        help="Run-period label used in the Pveto table.",
    )
    tau_pveto_table.add_argument(
        "--flavor",
        default=r"$\tau_h$",
        help="Flavor label used in the Pveto table.",
    )
    tau_pveto_table.add_argument(
        "--layers",
        nargs="+",
        default=["NLayers4", "NLayers5", "NLayers6plus", "combinedBins"],
        help="Layer-bin rows to include in the Pveto table.",
    )
    tau_pveto_table.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output path for the combined tau Pveto LaTeX table.",
    )
    tau_pveto_table.add_argument(
        "--table-env",
        action="store_true",
        help="Wrap the tabular in a LaTeX table environment.",
    )
    tau_pveto_table.set_defaults(func=_make_tau_pveto_table_command)

    lepton_background = subparsers.add_parser(
        "estimate-lepton-background",
        help=(
            "Compute lepton-background estimates from Pveto pair counts plus "
            "Poffline/Pmiss control-category ratios."
        ),
    )
    lepton_background.add_argument("files", nargs="+", type=Path)
    lepton_background.add_argument(
        "--mode",
        choices=("muon", "electron", "tau_mu", "tau_ele"),
        required=True,
        help="Lepton flavor/control mode to estimate.",
    )
    lepton_background.add_argument("--dataset", help="Restrict to one dataset key.")
    lepton_background.add_argument("--sample", help="Restrict to one sample key.")
    lepton_background.add_argument("--variation", default="nominal")
    lepton_background.add_argument(
        "--run-period",
        required=True,
        help="Run-period label used in the LaTeX table.",
    )
    lepton_background.add_argument(
        "--flavor",
        help="Override the flavor label used in the output table.",
    )
    lepton_background.add_argument(
        "--layers",
        nargs="+",
        default=["NLayers4", "NLayers5", "NLayers6plus", "combinedBins"],
        help="Layer-bin rows to estimate. Category patterns may use {layer}.",
    )
    lepton_background.add_argument(
        "--control-category",
        help=(
            "Control-yield category pattern for N_ctrl. May use {layer}. "
            "Defaults to <mode>_background_control_{layer}."
        ),
    )
    lepton_background.add_argument(
        "--poffline-numerator-category",
        help=(
            "Category pattern for the Poffline numerator. May use {layer}. "
            "Defaults to <mode>_background_offline_{layer}."
        ),
    )
    lepton_background.add_argument(
        "--poffline-denominator-category",
        help=(
            "Category pattern for the Poffline denominator. May use {layer}. "
            "Defaults to the N_ctrl category."
        ),
    )
    lepton_background.add_argument(
        "--ptrigger-numerator-category",
        help=(
            "Compatibility alias for --pmiss-numerator-category."
        ),
    )
    lepton_background.add_argument(
        "--ptrigger-denominator-category",
        help=(
            "Compatibility alias for --pmiss-denominator-category."
        ),
    )
    lepton_background.add_argument(
        "--pmiss-numerator-category",
        help=(
            "Category pattern for the Pmiss numerator. May use {layer}. "
            "Defaults to <mode>_background_trigger_{layer}."
        ),
    )
    lepton_background.add_argument(
        "--pmiss-denominator-category",
        help=(
            "Category pattern for the Pmiss denominator. May use {layer}. "
            "Defaults to the Poffline numerator."
        ),
    )
    lepton_background.add_argument(
        "--output-json",
        type=Path,
        help="Write detailed estimate components to JSON.",
    )
    lepton_background.add_argument(
        "--output-tex",
        type=Path,
        help="Write a LaTeX summary table.",
    )
    lepton_background.add_argument(
        "--table-env",
        action="store_true",
        help="Wrap the LaTeX tabular in a table environment.",
    )
    lepton_background.set_defaults(func=_estimate_lepton_background_command)

    fiducial_map = subparsers.add_parser(
        "make-fiducial-map",
        help=(
            "Build an electron or muon fiducial map from the before/after "
            "eta-phi histograms in fiducial_maps PocketCoffea outputs."
        ),
    )
    fiducial_map.add_argument(
        "files",
        nargs="+",
        type=Path,
        help="PocketCoffea .coffea output files from DISAPPTRKS_CATEGORY_MODE=fiducial_maps.",
    )
    fiducial_map.add_argument(
        "--flavor",
        choices=("electron", "muon"),
        required=True,
        help="Which fiducial-map histogram pair to summarize.",
    )
    fiducial_map.add_argument(
        "--run-period",
        required=True,
        help="Run-period label written into the JSON metadata.",
    )
    fiducial_map.add_argument("--dataset", help="Restrict to one dataset key.")
    fiducial_map.add_argument("--sample", help="Restrict to one sample key.")
    fiducial_map.add_argument(
        "--category",
        default="inclusive",
        help="PocketCoffea category axis value to read.",
    )
    fiducial_map.add_argument(
        "--threshold",
        type=float,
        default=2.0,
        help="Hot-spot threshold in standard deviations above the mean inefficiency.",
    )
    fiducial_map.add_argument(
        "--before-variable",
        help="Override the before-veto histogram variable name.",
    )
    fiducial_map.add_argument(
        "--after-variable",
        help="Override the after-veto histogram variable name.",
    )
    fiducial_map.add_argument(
        "-o",
        "--output-json",
        type=Path,
        required=True,
        help="Output JSON summary path.",
    )
    fiducial_map.add_argument(
        "--output-npz",
        type=Path,
        help="Output NumPy payload path. Defaults to --output-json with .npz suffix.",
    )
    fiducial_map.add_argument(
        "--no-npz",
        action="store_true",
        help="Only write the JSON hot-spot summary.",
    )
    fiducial_map.set_defaults(func=_make_fiducial_map_command)

    fiducial_plot = subparsers.add_parser(
        "plot-fiducial-map",
        help="Draw AN-style fiducial-map plots from a .npz payload.",
    )
    fiducial_plot.add_argument(
        "npz",
        type=Path,
        help="NPZ payload written by make-fiducial-map.",
    )
    fiducial_plot.add_argument(
        "--json",
        type=Path,
        help="JSON summary from make-fiducial-map; used for hot-spot circles.",
    )
    fiducial_plot.add_argument(
        "--flavor",
        choices=("electron", "muon"),
        required=True,
        help="Sets the AN-style z-axis ranges for inefficiency/significance.",
    )
    fiducial_plot.add_argument(
        "--run-period",
        help="Run-period label added to plot titles.",
    )
    fiducial_plot.add_argument(
        "--lumi-text",
        help=r"Top-right luminosity label, e.g. '27.0 fb$^{-1}$ (13.6 TeV)'.",
    )
    fiducial_plot.add_argument(
        "--cms-label",
        default="CMS Preliminary",
        help="Top-left CMS label.",
    )
    fiducial_plot.add_argument(
        "--formats",
        nargs="+",
        default=["pdf", "png"],
        help="Output formats to write.",
    )
    fiducial_plot.add_argument(
        "--colormap",
        default="root56",
        help=(
            "Matplotlib colormap name, or root56 for the AN/ROOT palette-56 "
            "style. Default: root56."
        ),
    )
    fiducial_plot.add_argument(
        "--output-prefix",
        type=Path,
        required=True,
        help=(
            "Output prefix. Files are written as "
            "<prefix>_beforeVeto.<fmt>, <prefix>_afterVeto.<fmt>, "
            "<prefix>_efficiency.<fmt>, and <prefix>_efficiencyInSigma.<fmt>."
        ),
    )
    fiducial_plot.add_argument(
        "--no-hot-spots",
        action="store_true",
        help="Do not draw hot-spot circles on the inefficiency/significance plots.",
    )
    fiducial_plot.set_defaults(func=_plot_fiducial_map_command)

    merge_pveto_tables = subparsers.add_parser(
        "merge-pveto-tables",
        help=(
            "Merge already-written Pveto LaTeX tables into one stacked "
            "AN-style table with run-period blocks."
        ),
    )
    merge_pveto_tables.add_argument(
        "tables",
        nargs="+",
        type=Path,
        help="Input per-period Pveto LaTeX tables, in the order to print them.",
    )
    merge_pveto_tables.add_argument(
        "-o",
        "--output",
        type=Path,
        required=True,
        help="Output path for the merged Pveto LaTeX table.",
    )
    merge_pveto_tables.add_argument(
        "--flavor",
        help="Override the flavor label in every row, e.g. electron, $e$, or $\\mu$.",
    )
    merge_pveto_tables.add_argument(
        "--keep-combined",
        action="store_true",
        help="Keep combined-bin rows. By default they are dropped to match AN summary tables.",
    )
    merge_pveto_tables.add_argument(
        "--no-compact-layer-labels",
        action="store_true",
        help="Keep the original layer labels instead of converting to 4, 5, and $\\geq 6$.",
    )
    merge_pveto_tables.add_argument(
        "--table-env",
        action="store_true",
        help="Wrap the tabular in a LaTeX table environment.",
    )
    merge_pveto_tables.set_defaults(func=_merge_pveto_tables_command)

    fake_tracks = subparsers.add_parser(
        "estimate-fake-tracks",
        help="Compute the AN-style fake-track estimate from coffea cutflow counts.",
    )
    fake_tracks.add_argument(
        "files",
        nargs="*",
        type=Path,
        help="PocketCoffea output files. Their cutflows are summed before estimating.",
    )
    fake_tracks.add_argument(
        "--counts-json",
        type=Path,
        help="Use a flat JSON mapping of category names to counts instead of coffea files.",
    )
    fake_tracks.add_argument("--dataset", help="Restrict to one dataset key for coffea input.")
    fake_tracks.add_argument("--sample", help="Restrict to one sample key for coffea input.")
    fake_tracks.add_argument("--variation", default="nominal")
    fake_tracks.add_argument("--run-period", required=True, help="Run-period label used in the LaTeX table.")
    fake_tracks.add_argument(
        "--an-control",
        choices=["zmumu", "zee"],
        help=(
            "Use the Chapter-5 fake-track method for a Z->ll control region: "
            "P_fake = zeta * N_sideband / N_Z."
        ),
    )
    fake_tracks.add_argument(
        "--transfer-factor-source",
        choices=["fixed", "fit"],
        default="fixed",
        help=(
            "For --an-control, use fixed AN Section-5.2 zeta values by "
            "run period/control region, or refit zeta from the input outputs."
        ),
    )
    fake_tracks.add_argument(
        "--layers",
        nargs="+",
        default=["NLayers4", "NLayers5", "NLayers6plus", "combinedBins"],
        help="Layer-bin rows to estimate. Category patterns may use {layer}.",
    )
    fake_tracks.add_argument(
        "--transfer-signal-category",
        default="fake_basic3hits_d0_signal",
        help="Category/count for the transfer-factor numerator, usually |d0| < 0.02 in the 3-hit/basic sample.",
    )
    fake_tracks.add_argument(
        "--transfer-sideband-category",
        default="fake_basic3hits_d0_sideband",
        help="Category/count for the transfer-factor denominator, usually the sideband 0.05 < |d0| < 0.5.",
    )
    fake_tracks.add_argument(
        "--control-category",
        default="fake_control_{layer}",
        help="Category/count for the target-layer inverted-d0 control yield. May use {layer}.",
    )
    fake_tracks.add_argument(
        "--basic-yield-category",
        help="Optional BasicSelection yield category for normalizing Z->ll to the search sample.",
    )
    fake_tracks.add_argument(
        "--basic-files",
        nargs="+",
        type=Path,
        help=(
            "Optional PocketCoffea output files containing the BasicSelection "
            "normalization, e.g. JetMET outputs. Requires --basic-yield-category."
        ),
    )
    fake_tracks.add_argument(
        "--basic-dataset",
        help="Restrict --basic-files to one dataset key. Defaults to --dataset when omitted.",
    )
    fake_tracks.add_argument(
        "--basic-sample",
        help="Restrict --basic-files to one sample key, e.g. DATA_JetMET. Defaults to --sample when omitted.",
    )
    fake_tracks.add_argument(
        "--basic-variation",
        help="Variation to read from --basic-files. Defaults to --variation when omitted.",
    )
    fake_tracks.add_argument(
        "--z-to-ll-yield-category",
        help="Optional inclusive Z->ll yield category for normalizing Z->ll to the search sample.",
    )
    fake_tracks.add_argument(
        "--prescale",
        type=float,
        default=1.0,
        help="Optional prescale/completion factor applied to N_ctrl.",
    )
    fake_tracks.add_argument(
        "--output-json",
        type=Path,
        help="Write detailed estimate components to JSON.",
    )
    fake_tracks.add_argument(
        "--output-tex",
        type=Path,
        help="Write an AN-style LaTeX summary table.",
    )
    fake_tracks.add_argument(
        "--basic-cutflow-tex",
        type=Path,
        help=(
            "Write a JetMET/basic-selection cutflow table for the fake-track "
            "normalization. With --an-control this is read from --basic-files."
        ),
    )
    fake_tracks.add_argument(
        "--z-control-tex",
        type=Path,
        help=(
            "With --an-control, write a Tables-32/33-style Z-control input "
            "table with N_Z and sideband counts by layer."
        ),
    )
    fake_tracks.add_argument(
        "--fit-plot",
        type=Path,
        help="Write a Figure-26-style signed-dxy transfer-factor fit plot.",
    )
    fake_tracks.add_argument(
        "--table-env",
        action="store_true",
        help="Wrap the LaTeX tabular in a table environment.",
    )
    fake_tracks.set_defaults(func=_estimate_fake_tracks_command)

    fake_track_table34 = subparsers.add_parser(
        "make-fake-track-table34",
        help="Combine Z->mumu and Z->ee fake-track JSON summaries into an AN Table-34-style table.",
    )
    fake_track_table34.add_argument(
        "jsons",
        nargs="+",
        type=Path,
        help="JSON outputs from estimate-fake-tracks, usually one zmumu and one zee file.",
    )
    fake_track_table34.add_argument("--run-period", required=True)
    fake_track_table34.add_argument("-o", "--output", type=Path, required=True)
    fake_track_table34.add_argument(
        "--table-env",
        action="store_true",
        help="Wrap the LaTeX tabular in a table environment.",
    )
    fake_track_table34.set_defaults(func=_make_fake_track_table34_command)

    dataset_json = subparsers.add_parser(
        "make-dataset-json",
        help="Create a PocketCoffea dataset JSON from an EOS directory or filelist.",
    )
    dataset_json.add_argument("eos_path", nargs="?", help="EOS directory, e.g. /store/user/...")
    dataset_json.add_argument("-o", "--output", type=Path, required=True)
    dataset_json.add_argument("--dataset-name", required=True)
    dataset_json.add_argument("--sample", default="DATA_Muon")
    dataset_json.add_argument("--year", required=True)
    dataset_json.add_argument("--era", required=True)
    dataset_json.add_argument("--primary-dataset", default="Muon")
    dataset_json.add_argument("--nevents", default="0")
    dataset_json.add_argument("--nano-version", type=int, default=15)
    dataset_json.add_argument("--is-mc", action="store_true")
    dataset_json.add_argument(
        "--xrootd",
        default="root://cmseos.fnal.gov",
        help="XRootD endpoint passed to xrdfs.",
    )
    dataset_json.add_argument(
        "-R",
        "--recursive",
        action="store_true",
        help="Use xrdfs ls -R -u to recurse through subdirectories.",
    )
    dataset_json.add_argument(
        "--filelist",
        type=Path,
        help="Read ROOT file URLs from a text file instead of calling xrdfs.",
    )
    dataset_json.set_defaults(func=_make_dataset_json_command)

    era_filelists = subparsers.add_parser(
        "make-era-filelists",
        help="Scan OSUNano EOS areas and write Muon/EGamma/JetMET filelists split by Run-3 era group.",
    )
    era_filelists.add_argument(
        "eos_paths",
        nargs="*",
        default=[
            "/store/group/lpcdisapptrks/nano/dev",
            "/store/group/lpcdisapptrks/nano/prod",
        ],
        help="EOS base directories to scan recursively.",
    )
    era_filelists.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("filelists"),
        help="Directory where .txt filelists are written.",
    )
    era_filelists.add_argument(
        "--dataset-json-dir",
        type=Path,
        help="Also write PocketCoffea dataset JSONs to this directory.",
    )
    era_filelists.add_argument(
        "--filelist",
        type=Path,
        help="Classify ROOT files from an existing text file instead of querying EOS.",
    )
    era_filelists.add_argument(
        "--xrootd",
        default="root://cmseos.fnal.gov",
        help="XRootD endpoint passed to xrdfs.",
    )
    era_filelists.add_argument(
        "--prod-version-policy",
        choices=("latest", "all"),
        default="all",
        help="How to handle prod directories with *_vN suffixes. Default treats versions as literal names and keeps all.",
    )
    era_filelists.add_argument(
        "--nano-version",
        type=int,
        default=None,
        help=(
            "Override the per-era NanoAOD version metadata. "
            "By default this uses 12 for 2022/2023 and 15 for 2024/2025."
        ),
    )
    era_filelists.add_argument(
        "--fail-on-empty",
        action="store_true",
        help="Return nonzero if any primary/year group has zero files.",
    )
    era_filelists.set_defaults(func=_make_era_filelists_command)

    args = parser.parse_args()
    if not hasattr(args, "func"):
        parser.print_help()
        return 0
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
