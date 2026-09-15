from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Union

from . import greet
from .datasets import (
    DATASET_JSON_EOS_BASE_DEFAULT,
    OUTPUT_EOS_BASE_DEFAULT,
    OutputAlreadyExistsError,
    build_dataset_definition,
    count_root_events,
    group_signal_files,
    group_osunano_files,
    list_eos_root_files,
    publish_dataset_json,
    publish_output_dir,
    root_files_from_lines,
    scan_eos_bases_for_root_files,
    write_dataset_definition,
    write_grouped_filelists,
)
from .fake_tracks import (
    Count,
    estimate_fake_track_background,
    estimate_fake_track_background_an,
    fit_dxy_transfer_factor,
    fixed_an_transfer_factor_fit,
    plot_fake_sideband_track_diagnostics,
    plot_high_purity_input_distributions,
    plot_signal_dedx_track_distributions,
    plot_dxy_transfer_factor,
    summed_hist_counts_edges,
    write_fake_sideband_event_manifest,
    write_an_fake_track_latex,
    write_combined_fake_track_table34_latex,
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
    legacy_met_probability_components_from_outputs,
    legacy_met_probabilities_from_outputs,
    probability_from_counts,
    read_lepton_background_json,
    trigger_efficiency_from_counts,
    write_combined_lepton_background_latex,
    write_lepton_background_json,
    write_lepton_background_latex,
)
from .schema import audit_root_file
from .summaries import (
    cutflow_count,
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


def _variable_names(output: dict) -> set[str]:
    return {str(name) for name in output.get("variables", {})}


def _has_background_variables(output: dict, *, prefix: str) -> bool:
    background_prefix = f"n{prefix}Background"
    return any(name.startswith(background_prefix) for name in _variable_names(output))


def _has_pveto_pair_variables(output: dict, *, prefix: str) -> bool:
    names = _variable_names(output)
    pair_prefixes = [
        f"n{prefix}TagProbePair",
        f"n{prefix}PVetoTagProbePair",
    ]
    if prefix == "Muon":
        pair_prefixes.extend(
            [
                "nMuonVetoTagProbePair",
                "nMuonPVetoTagProbePair",
            ]
        )
    return any(
        any(name.startswith(pair_prefix) for pair_prefix in pair_prefixes)
        for name in names
    )


def _lepton_background_outputs(outputs: list[dict], *, prefix: str) -> list[dict]:
    """Prefer dedicated Pmiss/Poffline outputs over Pveto outputs.

    Older Pveto productions could also write ``n<Prefix>Background...``
    histograms.  When those files are combined with the dedicated
    ``*_pmiss_poffline`` outputs, blindly summing every input double counts the
    MET histograms.  The dedicated outputs contain the background histograms
    without the Pveto tag-probe pair histograms, so prefer that subset when it
    exists.
    """

    with_background = [
        output for output in outputs if _has_background_variables(output, prefix=prefix)
    ]
    if not with_background:
        return outputs
    dedicated = [
        output
        for output in with_background
        if not _has_pveto_pair_variables(output, prefix=prefix)
    ]
    return dedicated or with_background


def _pair_counts_from_outputs(
    outputs: list[dict],
    *,
    layers: list[str],
    variable_templates: dict[str, PairVariableTemplate],
    category_templates: dict[str, PairVariableTemplate],
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
        categories = {
            key: _pair_variable_name(template, layer=layer, suffix=suffix)
            for key, template in category_templates.items()
        }
        for output in outputs:
            output_variables = output.get("variables", {})
            for key, variable in variables.items():
                totals[key] += variable_count_sum(
                    output_variables,
                    variable,
                    dataset=dataset,
                    sample=sample,
                    category=categories[key],
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
        category_templates=LEPTON_PVETO_PAIR_CATEGORIES["muon"],
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


def _add_counts(left: Count, right: Count) -> Count:
    return Count(left.value + right.value, left.variance + right.variance)


def _sum_named_count_maps(*maps: dict[str, dict[str, Count]]) -> dict[str, dict[str, Count]]:
    out: dict[str, dict[str, Count]] = {}
    for mapping in maps:
        for layer, counts in mapping.items():
            layer_counts = out.setdefault(layer, {})
            for key, count in counts.items():
                layer_counts[key] = (
                    _add_counts(layer_counts[key], count)
                    if key in layer_counts
                    else count
                )
    return out


def _met_probabilities_from_components(
    components: dict[str, dict[str, Count]]
) -> dict[str, tuple[Count, Count]]:
    probabilities = {}
    for layer, counts in components.items():
        probabilities[layer] = (
            probability_from_counts(counts["offline_pass"], counts["control"]),
            probability_from_counts(
                counts["weighted_trigger_pass"],
                counts["offline_total"],
            ),
        )
    return probabilities


def _apply_sparse_tau_met_probability_fallback(
    probabilities: dict[str, tuple[Count, Count]],
    *,
    sparse_layers: tuple[str, ...] = ("NLayers4", "NLayers5"),
    combined_layer: str = "combinedBins",
) -> dict[str, tuple[Count, Count]]:
    """Use the combined tau MET probabilities in sparse layer bins.

    This reproduces the prescription stated in dissertation Table 7.25.
    """

    if combined_layer not in probabilities:
        raise KeyError(
            f"combined tau MET probabilities {combined_layer!r} are required "
            "for the sparse-layer fallback"
        )
    result = dict(probabilities)
    for layer in sparse_layers:
        if layer in result:
            result[layer] = result[combined_layer]
    return result


def _trigger_efficiency_count_components_from_outputs(
    outputs: list[dict],
    *,
    prefix: str,
    layers: list[str],
    dataset: str | None = None,
    sample: str | None = None,
) -> dict[str, dict[str, Count]]:
    count_components = {}
    for layer in layers:
        suffix = "" if layer == "combinedBins" else f"_{layer}"
        variables = {
            "total_os": f"n{prefix}TriggerEffProbesPT55{suffix}",
            "total_ss": f"n{prefix}TriggerEffProbesSSPT55{suffix}",
            "passes_os": f"n{prefix}TriggerEffProbesFiringTrigger{suffix}",
            "passes_ss": f"n{prefix}TriggerEffSSProbesFiringTrigger{suffix}",
        }
        if not any(
            variable in output.get("variables", {})
            for output in outputs
            for variable in variables.values()
        ):
            continue
        counts = {}
        for key, variable in variables.items():
            value = sum(
                variable_count_sum(
                    output.get("variables", {}),
                    variable,
                    dataset=dataset,
                    sample=sample,
                )
                for output in outputs
            )
            counts[key] = Count(value, value)
        count_components[layer] = counts
    return count_components


def _trigger_efficiency_from_outputs(
    outputs: list[dict],
    *,
    prefix: str,
    layers: list[str],
    dataset: str | None = None,
    sample: str | None = None,
) -> dict[str, Count]:
    """Calculate the legacy epsilon divisor from Pveto tag-probe counters."""

    return {
        layer: trigger_efficiency_from_counts(**counts)
        for layer, counts in _trigger_efficiency_count_components_from_outputs(
            outputs,
            prefix=prefix,
            layers=layers,
            dataset=dataset,
            sample=sample,
        ).items()
    }


def _tau_trigger_probability_from_outputs(
    outputs: list[dict],
    *,
    dataset: str | None = None,
    sample: str | None = None,
) -> tuple[Count, Count, Count]:
    numerator = 0.0
    denominator = 0.0
    for output in outputs:
        variables = output.get("variables", {})
        numerator += variable_count_sum(
            variables,
            "nTauTriggerProbabilityNumerator",
            dataset=dataset,
            sample=sample,
        )
        denominator += variable_count_sum(
            variables,
            "nTauTriggerProbabilityDenominator",
            dataset=dataset,
            sample=sample,
        )
    numerator_count = Count(numerator, numerator)
    denominator_count = Count(denominator, denominator)
    if numerator <= 0.0 or denominator <= 0.0:
        probability = Count(0.0, 0.0)
    else:
        # P(tau) = P(muon+tau) / P(muon) = N(cross) / N(muon), both counted
        # independently over the same eta-accepted baseline sample.
        value = numerator / denominator
        variance = value**2 * (1.0 / numerator + 1.0 / denominator)
        probability = Count(value, variance)
    return numerator_count, denominator_count, probability


def _tau_probability_from_json(path: Path) -> Count:
    """Read the measured tau-trigger correction written by the extractor."""

    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    try:
        probability = payload["tau_probability"]
        value = float(probability["value"])
        if "variance" in probability:
            variance = float(probability["variance"])
        else:
            variance = float(probability["error"]) ** 2
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(
            f"invalid tau-probability JSON {path}: expected "
            "tau_probability.value and tau_probability.variance or .error"
        ) from error
    if value <= 0.0 or variance < 0.0:
        raise ValueError(
            f"invalid tau probability in {path}: value must be positive and "
            "variance must be non-negative"
        )
    return Count(value, variance)


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

# PocketCoffea category names for the pair-count histograms above, keyed the
# same way as LEPTON_PVETO_PAIR_VARIABLES. Each pair-count histogram carries a
# "cat" axis spanning every category in the job, not just one value -- reading
# it at the "inclusive" category (variable_count_sum's default) sums the whole
# inclusive event count, not the pair count, since these histograms are only
# meaningfully filled within their own veto/pveto category. Confirmed 2026-09-15
# against real job outputs: muon and electron use a "*_zwindow*" category
# naming (matching their legacy ZWindow-named histograms even though the
# electron pair-count *variable* names above say "MassWindow"), while
# tau_mu/tau_ele use "*_masswindow*" -- the two naming families are not
# interchangeable, so this must stay a per-mode lookup rather than a single
# shared template.
LEPTON_PVETO_PAIR_CATEGORIES = {
    "muon": {
        "den_os": "muon_veto_zwindow{suffix}",
        "num_os": "muon_pveto_zwindow_pass{suffix}",
        "den_ss": "muon_veto_ss_zwindow{suffix}",
        "num_ss": "muon_pveto_ss_zwindow_pass{suffix}",
    },
    "electron": {
        "den_os": "electron_veto_zwindow{suffix}",
        "num_os": "electron_pveto_zwindow_pass{suffix}",
        "den_ss": "electron_veto_ss_zwindow{suffix}",
        "num_ss": "electron_pveto_ss_zwindow_pass{suffix}",
    },
    "tau_mu": {
        "den_os": "tau_mu_veto_masswindow{suffix}",
        "num_os": "tau_mu_pveto_masswindow_pass{suffix}",
        "den_ss": "tau_mu_veto_ss_masswindow{suffix}",
        "num_ss": "tau_mu_pveto_ss_masswindow_pass{suffix}",
    },
    "tau_ele": {
        "den_os": "tau_ele_veto_masswindow{suffix}",
        "num_os": "tau_ele_pveto_masswindow_pass{suffix}",
        "den_ss": "tau_ele_veto_ss_masswindow{suffix}",
        "num_ss": "tau_ele_pveto_ss_masswindow_pass{suffix}",
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

            # Match the dissertation's legacy estimator: fit the folded
            # |d0| histogram with a Poisson likelihood.  The signed histogram
            # is retained only for the Figure-7.15-style visualization.
            fit = fit_dxy_transfer_factor(
                counts,
                edges,
                control_region=control_cfg["control_region"],
                histogram=control_cfg["histogram"],
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


def _standard_coffea_files(directory: Path) -> list[Path]:
    """Resolve one logical PocketCoffea output without double counting shards."""
    if not directory.is_dir():
        raise SystemExit(f"error: standardized output directory not found: {directory}")

    merged = directory / "output_all.coffea"
    if merged.is_file():
        return [merged]

    shards = sorted(directory.glob("output_job_*.coffea"))
    if shards:
        return shards

    outputs = sorted(directory.glob("*.coffea"))
    if len(outputs) == 1:
        return outputs
    if outputs:
        raise SystemExit(
            f"error: ambiguous .coffea outputs in {directory}; expected "
            "output_all.coffea or output_job_*.coffea. Pass explicit files "
            "to avoid double counting."
        )

    raise SystemExit(f"error: no top-level .coffea files found in {directory}")


def _make_standard_fake_track_estimate_command(args: argparse.Namespace) -> int:
    run_periods = (
        [args.run_period] if isinstance(args.run_period, str) else args.run_period
    )
    multiple_periods = len(run_periods) > 1
    output_base = args.output_dir or Path("tables") / "fake_tracks"
    combined_json_paths = {}

    if multiple_periods and (args.basic_files or args.zmumu_files or args.zee_files):
        raise SystemExit(
            "error: explicit --basic-files/--zmumu-files/--zee-files can only "
            "be used with one --run-period"
        )

    for run_period in run_periods:
        input_root = args.input_base / run_period / "fake_tracks"
        output_dir = output_base / run_period if multiple_periods else (
            args.output_dir or output_base / run_period
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        basic_files = args.basic_files or _standard_coffea_files(input_root / "basic")
        control_files = {
            "zmumu": args.zmumu_files or _standard_coffea_files(input_root / "zmumu"),
            "zee": args.zee_files or _standard_coffea_files(input_root / "zee"),
        }
        samples = {"zmumu": "DATA_Muon", "zee": "DATA_EGamma"}

        json_paths = []
        for control in ("zmumu", "zee"):
            json_path = output_dir / f"{control}.json"
            json_paths.append(json_path)
            estimate_args = argparse.Namespace(
                files=control_files[control],
                counts_json=None,
                dataset=None,
                sample=samples[control],
                variation="nominal",
                run_period=run_period,
                an_control=control,
                transfer_factor_source=args.transfer_factor_source,
                layers=["NLayers4", "NLayers5", "NLayers6plus", "combinedBins"],
                transfer_signal_category="fake_basic3hits_d0_signal",
                transfer_sideband_category="fake_basic3hits_d0_sideband",
                control_category="fake_control_{layer}",
                basic_yield_category=args.basic_yield_category,
                basic_files=basic_files,
                basic_dataset=None,
                basic_sample=args.basic_sample,
                basic_variation=None,
                z_to_ll_yield_category=None,
                prescale=1.0,
                output_json=json_path,
                output_tex=output_dir / f"{control}.tex",
                basic_cutflow_tex=(
                    output_dir / "basic_cutflow.tex" if control == "zmumu" else None
                ),
                z_control_tex=output_dir / f"{control}_control.tex",
                fit_plot=(
                    output_dir / f"{control}_fit.pdf"
                    if args.transfer_factor_source == "fit" and args.fit_plots
                    else None
                ),
                table_env=args.table_env,
            )
            _estimate_fake_tracks_command(estimate_args)
            if args.sideband_plots:
                outputs = _load_outputs(control_files[control])
                for plot_path in plot_fake_sideband_track_diagnostics(
                    outputs,
                    output_dir,
                    control=control,
                    sample=samples[control],
                    title_prefix=run_period,
                ):
                    print(f"Wrote {plot_path}")
                manifest_path = output_dir / f"{control}_sideband_events.csv"
                manifest_rows = write_fake_sideband_event_manifest(
                    outputs,
                    manifest_path,
                    control=control,
                    sample=samples[control],
                )
                print(f"Wrote {manifest_path} ({manifest_rows} candidates)")

        table_args = argparse.Namespace(
            jsons=json_paths,
            output=output_dir / "table34.tex",
            run_period=run_period,
            table_env=args.table_env,
        )
        _make_fake_track_table34_command(table_args)
        combined_json_paths[run_period] = json_paths
        print(f"Wrote standardized fake-track products under {output_dir}")

    if multiple_periods:
        combined_path = output_base / "table34_combined.tex"
        write_combined_fake_track_table34_latex(
            combined_json_paths,
            combined_path,
            include_table_env=args.table_env,
        )
        print(f"Wrote {combined_path}")
    return 0


def _period_float_values(values: list[str], periods: list[str], option: str) -> dict[str, float]:
    parsed = {}
    for value in values:
        if "=" in value:
            period, raw = value.split("=", 1)
            if period not in periods:
                raise SystemExit(f"error: {option} has unknown period {period!r}")
            if period in parsed:
                raise SystemExit(f"error: duplicate {option} for {period}")
            try:
                parsed[period] = float(raw)
            except ValueError as exc:
                raise SystemExit(f"error: invalid {option} value {value!r}") from exc
        elif len(periods) == 1:
            try:
                parsed[periods[0]] = float(value)
            except ValueError as exc:
                raise SystemExit(f"error: invalid {option} value {value!r}") from exc
        else:
            raise SystemExit(
                f"error: {option} values must use PERIOD=VALUE for multiple periods"
            )
    missing = [period for period in periods if period not in parsed]
    if missing:
        raise SystemExit(f"error: {option} missing for: {', '.join(missing)}")
    return parsed


def _make_standard_tau_background_command(args: argparse.Namespace) -> int:
    periods = args.run_period
    multiple_periods = len(periods) > 1
    output_base = args.output_dir or Path("tables") / "tau_background"
    trigger_efficiencies = _period_float_values(
        args.trigger_efficiency, periods, "--trigger-efficiency"
    )
    if args.trigger_efficiency_error:
        trigger_errors = _period_float_values(
            args.trigger_efficiency_error,
            periods,
            "--trigger-efficiency-error",
        )
    else:
        trigger_errors = {period: 0.0 for period in periods}

    explicit_inputs = (
        args.tau_control_files,
        args.tau_mu_files,
        args.tau_ele_files,
        args.tau_probability_files,
    )
    if multiple_periods and any(explicit_inputs):
        raise SystemExit(
            "error: explicit tau file overrides can only be used with one --run-period"
        )

    combined_inputs = []
    for period in periods:
        input_root = args.input_base / period
        output_dir = output_base / period if multiple_periods else (
            args.output_dir or output_base / period
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        tau_control_files = args.tau_control_files or _standard_coffea_files(
            input_root / "tau_pmiss_poffline"
        )
        tau_mu_files = args.tau_mu_files or _standard_coffea_files(
            input_root / "tau_mu_pveto"
        )
        tau_ele_files = args.tau_ele_files or _standard_coffea_files(
            input_root / "tau_ele_pveto"
        )
        tau_probability_files = args.tau_probability_files or _standard_coffea_files(
            input_root / "tau_trigger_probability"
        )
        output_json = output_dir / "tau_background.json"

        estimate_args = argparse.Namespace(
            tau_control_files=tau_control_files,
            tau_mu_files=tau_mu_files,
            tau_ele_files=tau_ele_files,
            dataset=None,
            tau_control_dataset=None,
            tau_mu_dataset=None,
            tau_ele_dataset=None,
            tau_control_sample="DATA_Muon",
            tau_mu_sample="DATA_Muon",
            tau_ele_sample="DATA_EGamma",
            variation="nominal",
            run_period=period,
            flavor=r"$\tau_h$",
            layers=["NLayers4", "NLayers5", "NLayers6plus", "combinedBins"],
            control_prescale=args.control_prescale,
            trigger_efficiency=trigger_efficiencies[period],
            trigger_efficiency_error=trigger_errors[period],
            tau_probability=None,
            tau_probability_json=None,
            tau_probability_files=tau_probability_files,
            tau_probability_error=0.0,
            met_cut=args.met_cut,
            phi_cut=args.phi_cut,
            output_json=output_json,
            output_tex=output_dir / "tau_background.tex",
            table_env=args.table_env,
        )
        _estimate_tau_background_command(estimate_args)
        combined_inputs.append((period, read_lepton_background_json(output_json)))
        print(f"Wrote standardized tau-background products under {output_dir}")

    if multiple_periods:
        combined_path = output_base / "tau_background_combined.tex"
        write_combined_lepton_background_latex(
            combined_inputs,
            combined_path,
            include_table_env=args.table_env,
        )
        print(f"Wrote {combined_path}")
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

    if args.is_mc and args.xsec is None:
        raise SystemExit(
            "error: --is-mc requires --xsec "
            "(use --xsec 1.0 for an acceptance-only job)"
        )
    if not args.is_mc and args.xsec is not None:
        raise SystemExit("error: --xsec is only valid together with --is-mc")
    if args.count_events and str(args.nevents) != "0":
        raise SystemExit("error: use either --count-events or --nevents, not both")
    if args.group_signal_points:
        if args.dataset_name:
            raise SystemExit(
                "error: --dataset-name is not used with --group-signal-points; "
                "dataset names come from the directories below "
                f"{args.signal_marker}"
            )
        if not args.is_mc:
            raise SystemExit("error: --group-signal-points requires --is-mc")
        if not args.count_events:
            raise SystemExit(
                "error: --group-signal-points requires --count-events so every "
                "signal point gets its own nevents value"
            )
    elif not args.dataset_name:
        raise SystemExit(
            "error: --dataset-name is required unless --group-signal-points is used"
        )

    nevents = str(args.nevents)
    event_counts: dict[str, int] = {}
    if args.count_events:
        print(f"Counting Events entries in {len(files)} ROOT file(s)...")

        def report(completed, total, path, count):
            print(f"[{completed}/{total}] {count:10d}  {path}")

        total_events, event_counts = count_root_events(
            files,
            max_workers=args.event_count_workers,
            timeout=args.event_count_timeout,
            progress=report,
        )
        nevents = str(total_events)
        print(f"Total Events entries: {nevents}")

    extra_metadata = {"xsec": str(args.xsec)} if args.xsec is not None else None
    if args.group_signal_points:
        grouped = group_signal_files(files, marker=args.signal_marker)
        classified = {path for paths in grouped.values() for path in paths}
        unclassified = [path for path in files if path not in classified]
        if unclassified:
            raise SystemExit(
                f"error: could not identify {args.signal_marker}/<signal-point> for "
                f"{len(unclassified)} file(s), including {unclassified[0]}"
            )
        if not grouped:
            raise SystemExit(
                f"error: no signal points were found below {args.signal_marker}"
            )

        dataset = {}
        for point, point_files in grouped.items():
            point_nevents = sum(event_counts[path] for path in point_files)
            dataset.update(
                build_dataset_definition(
                    dataset_name=point,
                    files=point_files,
                    sample=args.sample,
                    year=args.year,
                    era=args.era,
                    primary_dataset=args.primary_dataset,
                    is_mc=True,
                    nevents=str(point_nevents),
                    nano_version=args.nano_version,
                    extra_metadata=extra_metadata,
                )
            )
            print(f"{point}: {len(point_files)} file(s), {point_nevents} events")
    else:
        dataset = build_dataset_definition(
            dataset_name=args.dataset_name,
            files=files,
            sample=args.sample,
            year=args.year,
            era=args.era,
            primary_dataset=args.primary_dataset,
            is_mc=args.is_mc,
            nevents=nevents,
            nano_version=args.nano_version,
            extra_metadata=extra_metadata,
        )
    write_dataset_definition(dataset, args.output)

    point_summary = f" in {len(dataset)} dataset(s)" if args.group_signal_points else ""
    print(f"Wrote {len(files)} ROOT files{point_summary} to {args.output}")
    if not files:
        print("Warning: no ROOT files found")
        return 2

    if args.publish:
        publish_name = args.publish_name or args.output.stem
        try:
            destination = publish_dataset_json(
                args.output,
                name=publish_name,
                eos_base=args.publish_eos_base,
                force=args.force,
            )
        except FileExistsError as err:
            raise SystemExit(f"error: {err}") from err
        print(f"Published to {destination}")

    return 0


def _publish_output_command(args: argparse.Namespace) -> int:
    if not args.input_dir.is_dir():
        raise SystemExit(f"error: {args.input_dir} is not a directory")
    if args.overwrite and args.suffix:
        raise SystemExit("error: --overwrite and --suffix are mutually exclusive")

    mode = f"{args.mode}_{args.suffix}" if args.suffix else args.mode
    try:
        destination = publish_output_dir(
            args.input_dir,
            period=args.period,
            mode=mode,
            eos_base=args.eos_base,
            overwrite=args.overwrite,
        )
    except OutputAlreadyExistsError as err:
        raise SystemExit(
            f"error: {err.destination} already exists. Re-run with --overwrite to "
            "replace it, or --suffix <label> to publish this run alongside it "
            "without touching the existing copy."
        ) from err
    print(f"Published {args.input_dir} to {destination}")
    return 0


def _make_era_filelists_command(args: argparse.Namespace) -> int:
    if args.filelist:
        files = root_files_from_lines(args.filelist.read_text().splitlines())
    else:
        files = scan_eos_bases_for_root_files(args.eos_paths, xrootd=args.xrootd)

    source_areas = tuple(
        dict.fromkeys(Path(path.rstrip("/")).name for path in args.eos_paths)
    )
    source_areas = tuple(
        area for area in source_areas if area in ("dev", "prod", "dev_v2")
    ) or ("dev", "prod")
    output_suffix = args.output_suffix
    if output_suffix is None:
        output_suffix = "OSUv2" if source_areas == ("dev_v2",) else ""

    grouped = group_osunano_files(
        files,
        prod_version_policy=args.prod_version_policy,
        source_areas=source_areas,
    )
    outputs = write_grouped_filelists(
        grouped,
        output_dir=args.output_dir,
        dataset_json_dir=args.dataset_json_dir,
        nano_version=args.nano_version,
        output_suffix=output_suffix,
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
            category_templates=LEPTON_PVETO_PAIR_CATEGORIES[args.mode],
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
        category_templates=LEPTON_PVETO_PAIR_CATEGORIES["tau_mu"],
        dataset=args.tau_mu_dataset or args.dataset,
        sample=args.tau_mu_sample,
    )
    tau_ele_pair_counts = _pair_counts_from_outputs(
        tau_ele_outputs,
        layers=args.layers,
        variable_templates=LEPTON_PVETO_PAIR_VARIABLES["tau_ele"],
        category_templates=LEPTON_PVETO_PAIR_CATEGORIES["tau_ele"],
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
            category_templates=LEPTON_PVETO_PAIR_CATEGORIES[args.mode],
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
    prefix = {
        "muon": "Muon",
        "electron": "Electron",
        "tau_mu": "TauMu",
        "tau_ele": "TauEle",
    }[args.mode]
    background_outputs = _lepton_background_outputs(outputs, prefix=prefix)
    if background_outputs is not outputs and len(background_outputs) < len(outputs):
        print(
            "Using "
            f"{len(background_outputs)} of {len(outputs)} input output(s) for "
            "Poffline/Pmiss to avoid summing duplicate background histograms "
            "from Pveto outputs."
        )
    background_cutflow = {}
    for output in background_outputs:
        background_cutflow = _sum_nested_numeric(background_cutflow, output["cutflow"])
    control_counts = {
        layer: Count(
            cutflow_count(
                background_cutflow,
                control_category.format(layer=layer),
                dataset=args.dataset,
                sample=args.sample,
                variation=args.variation,
            )
        )
        for layer in args.layers
    }
    met_probabilities = legacy_met_probabilities_from_outputs(
        background_outputs,
        prefix=prefix,
        layers=args.layers,
        control_counts=control_counts,
        dataset=args.dataset,
        sample=args.sample,
        met_cut=args.met_cut,
        phi_cut=args.phi_cut,
    )
    if args.trigger_efficiency is None:
        trigger_efficiency = _trigger_efficiency_from_outputs(
            outputs,
            prefix=prefix,
            layers=args.layers,
            dataset=args.dataset,
            sample=args.sample,
        )
        if trigger_efficiency:
            trigger_efficiency_method = "legacy-tag-probe"
        else:
            trigger_efficiency = Count(1.0, 0.0)
            trigger_efficiency_method = "default"
    else:
        trigger_efficiency = Count(
            args.trigger_efficiency,
            args.trigger_efficiency_error * args.trigger_efficiency_error,
        )
        trigger_efficiency_method = "manual"

    tau_probability = (
        Count(args.tau_probability, args.tau_probability_error * args.tau_probability_error)
        if args.tau_probability is not None
        else None
    )
    if args.low_stat_layers is not None:
        low_stat_layers = args.low_stat_layers
    elif args.mode in ("muon", "tau_mu", "tau_ele"):
        # NLayers4/NLayers5 Poffline/Pmiss/trigger-efficiency control samples
        # are too small for a meaningful per-layer ratio in these modes; fall
        # back to combinedBins by default (mirrors the dissertation Table-7.25
        # sparse-layer prescription that estimate-tau-background's own
        # _apply_sparse_tau_met_probability_fallback already applies).
        low_stat_layers = ["NLayers4", "NLayers5"]
    else:
        low_stat_layers = []
    estimates = estimate_lepton_background(
        flavor=flavor,
        layers=args.layers,
        pair_counts=pair_counts,
        cutflow=background_cutflow,
        control_category=control_category,
        poffline_numerator_category=poffline_numerator_category,
        poffline_denominator_category=poffline_denominator_category,
        pmiss_numerator_category=pmiss_numerator_category,
        pmiss_denominator_category=pmiss_denominator_category,
        control_prescale=args.control_prescale,
        trigger_efficiency=trigger_efficiency,
        tau_probability=tau_probability,
        met_probabilities=met_probabilities,
        low_stat_layers=low_stat_layers,
        combined_layer=args.low_stat_combined_layer,
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
            tau_probability=tau_probability,
        )
        print(f"Wrote {args.output_tex}")

    for estimate in estimates:
        pveto_counts = pair_counts.get(estimate.layer, {})
        met_method = (
            "hist-integrated"
            if estimate.layer in met_probabilities
            else "cutflow-ratio"
        )
        print(
            f"{estimate.layer}: N_lepton = "
            f"{estimate.estimate.value:.6g} ± {estimate.estimate.error:.6g} "
            f"(P_veto={estimate.p_veto.value:.6g}, "
            f"P_veto_counts=[den_os={pveto_counts.get('den_os', 0):g}, "
            f"num_os={pveto_counts.get('num_os', 0):g}, "
            f"den_ss={pveto_counts.get('den_ss', 0):g}, "
            f"num_ss={pveto_counts.get('num_ss', 0):g}], "
            f"P_offline={estimate.p_offline.value:.6g}, "
            f"P_miss={estimate.p_miss.value:.6g}, "
            f"trigger_efficiency={estimate.trigger_efficiency.value:.6g}, "
            f"P_tau={estimate.tau_probability.value:.6g}, "
            f"trigger_efficiency_method={trigger_efficiency_method}, "
            f"met_method={met_method})"
        )
    return 0


def _estimate_tau_background_command(args: argparse.Namespace) -> int:
    tau_mu_outputs = _load_outputs(args.tau_mu_files)
    tau_ele_outputs = _load_outputs(args.tau_ele_files)
    tau_control_outputs = _lepton_background_outputs(
        _load_outputs(args.tau_control_files), prefix="Tau"
    )

    tau_mu_pair_counts = _pair_counts_from_outputs(
        tau_mu_outputs,
        layers=args.layers,
        variable_templates=LEPTON_PVETO_PAIR_VARIABLES["tau_mu"],
        category_templates=LEPTON_PVETO_PAIR_CATEGORIES["tau_mu"],
        dataset=args.tau_mu_dataset or args.dataset,
        sample=args.tau_mu_sample,
    )
    tau_ele_pair_counts = _pair_counts_from_outputs(
        tau_ele_outputs,
        layers=args.layers,
        variable_templates=LEPTON_PVETO_PAIR_VARIABLES["tau_ele"],
        category_templates=LEPTON_PVETO_PAIR_CATEGORIES["tau_ele"],
        dataset=args.tau_ele_dataset or args.dataset,
        sample=args.tau_ele_sample,
    )
    pair_counts = _sum_pair_count_maps(tau_mu_pair_counts, tau_ele_pair_counts)

    def _merged_cutflow(outputs):
        merged = {}
        for output in outputs:
            merged = _sum_nested_numeric(merged, output["cutflow"])
        return merged

    def _leg_counts(cutflow, category_prefix, dataset, sample):
        return {
            kind: {
                layer: Count(
                    cutflow_count(
                        cutflow,
                        f"{category_prefix}_background_{kind}_{layer}",
                        dataset=dataset,
                        sample=sample,
                        variation=args.variation,
                    )
                )
                for layer in args.layers
            }
            for kind in ("control", "offline", "trigger")
        }

    tau_control_cutflow = _merged_cutflow(tau_control_outputs)
    tau_control_counts = _leg_counts(
        tau_control_cutflow,
        "tau_control",
        args.tau_control_dataset or args.dataset,
        args.tau_control_sample,
    )
    # The tau control uses one Muon-dataset sample selected by the
    # year-dependent IsoMu24+tau cross-trigger.
    # The EGamma leg contributes only to the combined P_veto measurement.
    control_counts = tau_control_counts["control"]
    offline_counts = tau_control_counts["offline"]
    trigger_counts = tau_control_counts["trigger"]
    tau_met_components = legacy_met_probability_components_from_outputs(
        tau_control_outputs,
        prefix="Tau",
        layers=args.layers,
        control_counts=tau_control_counts["control"],
        dataset=args.tau_control_dataset or args.dataset,
        sample=args.tau_control_sample,
        met_cut=args.met_cut,
        phi_cut=args.phi_cut,
    )
    met_probabilities = _apply_sparse_tau_met_probability_fallback(
        _met_probabilities_from_components(tau_met_components)
    )

    trigger_efficiency = Count(
        args.trigger_efficiency,
        args.trigger_efficiency_error * args.trigger_efficiency_error,
    )
    trigger_efficiency_method = "manual-cross-trigger-control"

    if args.tau_probability_files is not None:
        if args.tau_probability_error != 0.0:
            raise ValueError(
                "--tau-probability-error cannot be combined with "
                "--tau-probability-files"
            )
        probability_outputs = _load_outputs(args.tau_probability_files)
        probability_numerator, probability_denominator, tau_probability = (
            _tau_trigger_probability_from_outputs(
                probability_outputs,
                dataset=args.tau_control_dataset or args.dataset,
                sample=args.tau_control_sample,
            )
        )
        if probability_denominator.value <= 0.0:
            raise ValueError(
                "tau-probability denominator is zero; verify that the files "
                "were produced in tau_trigger_probability mode"
            )
        print(
            "Calculated tau_probability="
            f"{tau_probability.value:.8g} ± {tau_probability.error:.8g} "
            f"from N_cross={probability_numerator.value:g}, "
            f"N_muon={probability_denominator.value:g}"
        )
    elif args.tau_probability_json is not None:
        if args.tau_probability_error != 0.0:
            raise ValueError(
                "--tau-probability-error cannot be combined with "
                "--tau-probability-json"
            )
        tau_probability = _tau_probability_from_json(args.tau_probability_json)
    elif args.tau_probability is not None:
        tau_probability = Count(
            args.tau_probability,
            args.tau_probability_error * args.tau_probability_error,
        )
    else:
        tau_probability = None
    counts = {}
    for layer in args.layers:
        counts[f"tau_background_control_{layer}"] = control_counts[layer]
        counts[f"tau_background_offline_{layer}"] = offline_counts[layer]
        counts[f"tau_background_trigger_{layer}"] = trigger_counts[layer]
    estimates = estimate_lepton_background(
        flavor=args.flavor,
        layers=args.layers,
        pair_counts=pair_counts,
        counts=counts,
        control_category="tau_background_control_{layer}",
        poffline_numerator_category="tau_background_offline_{layer}",
        poffline_denominator_category="tau_background_control_{layer}",
        pmiss_numerator_category="tau_background_trigger_{layer}",
        pmiss_denominator_category="tau_background_offline_{layer}",
        control_prescale=args.control_prescale,
        trigger_efficiency=trigger_efficiency,
        tau_probability=tau_probability,
        met_probabilities=met_probabilities,
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
            tau_probability=tau_probability,
        )
        print(f"Wrote {args.output_tex}")
    for estimate in estimates:
        met_method = (
            "hist-integrated"
            if estimate.layer in met_probabilities
            else "cutflow-ratio"
        )
        print(
            f"{estimate.layer}: N_tau = "
            f"{estimate.estimate.value:.6g} ± {estimate.estimate.error:.6g} "
            f"(P_veto={estimate.p_veto.value:.6g}, "
            f"P_offline={estimate.p_offline.value:.6g}, "
            f"P_miss={estimate.p_miss.value:.6g}, "
            f"trigger_efficiency={estimate.trigger_efficiency.value:.6g}, "
            f"P_tau={estimate.tau_probability.value:.6g}, "
            f"trigger_efficiency_method={trigger_efficiency_method}, "
            f"met_method={met_method})"
        )
    return 0


def _parse_period_json_input(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(
            f"combined table input {value!r} must be formatted as RUN_PERIOD=path.json"
        )
    period, path = value.split("=", 1)
    period = period.strip()
    if not period:
        raise ValueError(f"combined table input {value!r} has an empty run period")
    return period, Path(path.strip())


def _combine_lepton_background_tables_command(args: argparse.Namespace) -> int:
    period_estimates = []
    for item in args.input:
        run_period, path = _parse_period_json_input(item)
        estimates = read_lepton_background_json(path)
        if args.flavor:
            estimates = [
                estimate
                for estimate in estimates
                if estimate.flavor == args.flavor
            ]
        if not estimates:
            raise ValueError(f"no estimates found in {path} for run period {run_period}")
        period_estimates.append((run_period, estimates))

    tau_probability = (
        Count(args.tau_probability, args.tau_probability_error * args.tau_probability_error)
        if args.tau_probability is not None
        else None
    )
    write_combined_lepton_background_latex(
        period_estimates,
        args.output_tex,
        include_table_env=args.table_env,
        tau_probability=tau_probability,
    )
    print(f"Wrote {args.output_tex}")
    return 0


def _extract_tau_trigger_probability_command(args: argparse.Namespace) -> int:
    outputs = _load_outputs(args.files)
    numerator, denominator, probability = _tau_trigger_probability_from_outputs(
        outputs,
        dataset=args.dataset,
        sample=args.sample,
    )
    if denominator.value <= 0.0:
        raise ValueError(
            "tau-trigger probability denominator is zero. Check that the input "
            "was produced with DISAPPTRKS_CATEGORY_MODE=tau_trigger_probability "
            "and that the NanoAOD contains the single-muon HLT branch."
        )
    payload = {
        "numerator": numerator.as_dict() if hasattr(numerator, "as_dict") else {
            "value": numerator.value,
            "error": numerator.error,
            "variance": numerator.variance,
        },
        "denominator": denominator.as_dict() if hasattr(denominator, "as_dict") else {
            "value": denominator.value,
            "error": denominator.error,
            "variance": denominator.variance,
        },
        "tau_probability": {
            "value": probability.value,
            "error": probability.error,
            "variance": probability.variance,
        },
    }
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        print(f"Wrote {args.output_json}")
    print(
        "tau_probability="
        f"{probability.value:.6g} ± {probability.error:.6g} "
        f"(N_cross={numerator.value:.6g}, "
        f"N_muon={denominator.value:.6g})"
    )
    print(
        "Use with estimate-lepton-background: "
        f"--tau-probability {probability.value:.8g} "
        f"--tau-probability-error {probability.error:.8g}"
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
        stddev_exclude_top=args.stddev_exclude_top,
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
            "stddev_exclude_top": args.stddev_exclude_top,
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
    if summary.stddev_excluded_bins:
        print(
            "Excluded from stddev calculation: "
            f"{len(summary.stddev_excluded_bins)} bin(s)"
        )
        for excluded_bin in summary.stddev_excluded_bins:
            print(
                f"  eta={excluded_bin.eta:.3f}, phi={excluded_bin.phi:.3f}, "
                f"inefficiency={excluded_bin.inefficiency:.6g}"
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


def _plot_high_purity_study_command(args: argparse.Namespace) -> int:
    outputs = _load_outputs(args.files)
    for path in plot_high_purity_input_distributions(
        outputs,
        args.output_dir,
        control=args.control,
        layers=args.layers,
        sample=args.sample,
        title_prefix=args.title_prefix,
    ):
        print(f"Wrote {path}")
    return 0


def _plot_signal_dedx_study_command(args: argparse.Namespace) -> int:
    outputs = _load_outputs(args.files)
    for path in plot_signal_dedx_track_distributions(
        outputs,
        args.output_dir,
        layers=args.layers,
        sample=args.sample,
        title_prefix=args.title_prefix,
    ):
        print(f"Wrote {path}")
    return 0


def _summarize_signal_high_purity_command(args: argparse.Namespace) -> int:
    cutflow = _load_merged_cutflow(args.files)
    rows = (
        ("combined", "combinedBins"),
        ("4 layers", "NLayers4"),
        ("5 layers", "NLayers5"),
        (">=6 layers", "NLayers6plus"),
    )
    print(
        f"{'category':<12} {'without HP':>14} {'with HP':>14} "
        f"{'retained':>12} {'lost':>12}"
    )
    for label, layer in rows:
        without = cutflow_count(
            cutflow,
            f"signal_selection_without_high_purity_{layer}",
            dataset=args.dataset,
            sample=args.sample,
            variation=args.variation,
        )
        with_hp = cutflow_count(
            cutflow,
            f"signal_selection_with_high_purity_{layer}",
            dataset=args.dataset,
            sample=args.sample,
            variation=args.variation,
        )
        retained = with_hp / without if without else float("nan")
        lost = 1.0 - retained
        print(
            f"{label:<12} {without:14.6g} {with_hp:14.6g} "
            f"{retained:11.2%} {lost:11.2%}"
        )

    if not args.full_cutflow:
        return 0

    stages = (
        ("event_metNoMu120", "MET-no-muon >= 120"),
        ("event_leadingJet110", "leading jet pT > 110"),
        ("event_leadingJetEta2p4", "leading jet |eta| < 2.4"),
        ("event_leadingJetTightLepVeto", "leading jet tight-lepton-veto ID"),
        ("event_dijetDphi2p5", "dijet max delta-phi < 2.5"),
        ("event_jetMetDphi0p5", "leading jet/MET delta-phi >= 0.5"),
        ("track_pt55", "track pT > 55"),
        ("track_eta2p1", "track |eta| < 2.1"),
        ("track_noECALCrack", "ECAL crack veto"),
        ("track_noDTWheelGap", "DT wheel-gap veto"),
        ("track_noCSCTransition", "CSC transition veto"),
        ("track_noTOBCrack", "TOB crack veto"),
        ("track_fiducialECAL", "ECAL/electron/muon fiducial vetoes"),
        ("track_pixelHits4", ">= 4 valid pixel hits"),
        ("track_validHits4", ">= 4 valid hits"),
        ("track_noMissingInner", "no missing inner hits"),
        ("track_noMissingMiddle", "no missing middle hits"),
        ("track_chargedIso0p05", "charged isolation < 0.05"),
        ("track_dxy0p02", "|d0| < 0.02"),
        ("track_dz0p5", "|dz| < 0.5"),
        ("track_dRJet0p5", "track/jet delta-R > 0.5"),
        ("track_layers4plus", "layer-bin requirement"),
        ("track_highPurity", "highPurity track ID"),
        ("track_calo10", "calorimeter energy < 10"),
        ("track_missingOuter3", ">= 3 missing outer hits"),
        ("track_electronVeto", "electron veto"),
        ("track_muonVeto", "muon veto"),
        ("track_tauVeto", "tau veto"),
    )
    for layer in args.layers:
        print(f"\n{layer} cumulative cutflow")
        print(
            f"{'selection':<42} {'without HP':>14} {'with HP':>14} "
            f"{'retained':>12}"
        )
        for category, label in (
            ("initial", "initial events"),
            ("skim", "MET-trigger skim"),
            ("presel", "event-quality preselections"),
        ):
            if category not in cutflow:
                continue
            count = cutflow_count(
                cutflow,
                category,
                dataset=args.dataset,
                variation=args.variation,
            )
            print(f"{label:<42} {count:14.6g} {count:14.6g} {1.0:11.2%}")
        reached_high_purity_split = False
        for field, label in stages:
            if field == "track_highPurity":
                reached_high_purity_split = True
            if reached_high_purity_split:
                without_category = (
                    f"signal_cutflow_without_high_purity_{layer}_{field}"
                )
                with_category = f"signal_cutflow_with_high_purity_{layer}_{field}"
            elif field == "track_layers4plus":
                without_category = with_category = (
                    f"signal_cutflow_layer_{layer}_track_layers4plus"
                )
            else:
                # Before the layer-bin split, the two variants and all layer
                # bins are identical. PocketCoffea stores these once as common
                # StandardSelection categories.
                without_category = with_category = f"signal_cutflow_common_{field}"
            without = cutflow_count(
                cutflow,
                without_category,
                dataset=args.dataset,
                sample=args.sample,
                variation=args.variation,
            )
            with_hp = cutflow_count(
                cutflow,
                with_category,
                dataset=args.dataset,
                sample=args.sample,
                variation=args.variation,
            )
            retained = with_hp / without if without else float("nan")
            print(
                f"{label:<42} {without:14.6g} {with_hp:14.6g} "
                f"{retained:11.2%}"
            )
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

    tau_trigger_probability = subparsers.add_parser(
        "extract-tau-trigger-probability",
        help=(
            "Extract the AN tau-trigger correction from "
            "DISAPPTRKS_CATEGORY_MODE=tau_trigger_probability outputs."
        ),
    )
    tau_trigger_probability.add_argument("files", nargs="+", type=Path)
    tau_trigger_probability.add_argument(
        "--dataset",
        help="Restrict to one dataset key.",
    )
    tau_trigger_probability.add_argument(
        "--sample",
        help="Restrict to one sample key.",
    )
    tau_trigger_probability.add_argument(
        "--output-json",
        type=Path,
        help="Write numerator, denominator, and tau_probability to JSON.",
    )
    tau_trigger_probability.set_defaults(
        func=_extract_tau_trigger_probability_command
    )

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
        "--control-prescale",
        type=float,
        default=1.0,
        help=(
            "Scale factor applied to N_ctrl before computing N_lepton. "
            "Use this for the legacy MET/EGamma luminosity or prescale correction."
        ),
    )
    lepton_background.add_argument(
        "--trigger-efficiency",
        type=float,
        help=(
            "Manual epsilon trigger-efficiency divisor override. By default "
            "epsilon is calculated from legacy-style Pveto tag-probe trigger "
            "counters when available."
        ),
    )
    lepton_background.add_argument(
        "--trigger-efficiency-error",
        type=float,
        default=0.0,
        help="Absolute uncertainty on --trigger-efficiency.",
    )
    lepton_background.add_argument(
        "--tau-probability",
        type=float,
        help=(
            "Optional data-derived IsoMu24+tau/IsoMu24 trigger scale P(tau). "
            "This scales N_ctrl and is stored in JSON."
        ),
    )
    lepton_background.add_argument(
        "--tau-probability-error",
        type=float,
        default=0.0,
        help="Absolute uncertainty on --tau-probability.",
    )
    lepton_background.add_argument(
        "--low-stat-layers",
        nargs="*",
        default=None,
        help=(
            "Layer bins whose Poffline/Pmiss and trigger efficiency are too "
            "low-statistics to use their own per-layer control categories; "
            "draw them from --low-stat-combined-layer's categories instead. "
            "N_ctrl and Pveto still use each layer's own categories. Defaults "
            "to 'NLayers4 NLayers5' for --mode muon/tau_mu/tau_ele, and to "
            "none (every layer uses its own categories) for --mode electron. "
            "Pass with no values to force-disable for any mode."
        ),
    )
    lepton_background.add_argument(
        "--low-stat-combined-layer",
        default="combinedBins",
        help="Layer whose Poffline/Pmiss/trigger-efficiency categories back --low-stat-layers.",
    )
    lepton_background.add_argument(
        "--met-cut",
        type=float,
        default=120.0,
        help="Offline lepton-removed MET threshold used for Poffline/Pmiss integration.",
    )
    lepton_background.add_argument(
        "--phi-cut",
        type=float,
        default=0.5,
        help="Delta-phi threshold used for Poffline/Pmiss integration.",
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

    tau_background = subparsers.add_parser(
        "estimate-tau-background",
        help=(
            "Compute the tau background using the Muon-dataset muon+tau "
            "cross-trigger control and combined Muon/EGamma P_veto legs."
        ),
    )
    tau_background.add_argument(
        "--tau-control-files",
        nargs="+",
        type=Path,
        required=True,
        help=(
            "Muon-data tau_pmiss_poffline outputs selected by both the "
            "muon+tau cross trigger and IsoMu24; supplies N_ctrl, P_offline, "
            "and P_trigger."
        ),
    )
    tau_background.add_argument(
        "--tau-mu-files",
        nargs="+",
        type=Path,
        required=True,
        help=(
            "Muon-data tau_mu_pveto outputs used for P_veto."
        ),
    )
    tau_background.add_argument(
        "--tau-ele-files",
        nargs="+",
        type=Path,
        required=True,
        help=(
            "EGamma-data tau_ele_pveto outputs used only for P_veto."
        ),
    )
    tau_background.add_argument(
        "--dataset",
        help="Default dataset-key restriction for all tau inputs.",
    )
    tau_background.add_argument(
        "--tau-control-dataset",
        help="Restrict the cross-trigger tau-control input to one dataset key.",
    )
    tau_background.add_argument(
        "--tau-mu-dataset",
        help="Restrict tau_mu inputs to one dataset key. Overrides --dataset.",
    )
    tau_background.add_argument(
        "--tau-ele-dataset",
        help="Restrict tau_ele inputs to one dataset key. Overrides --dataset.",
    )
    tau_background.add_argument(
        "--tau-control-sample",
        default="DATA_Muon",
        help="Restrict the cross-trigger tau-control input to one sample key.",
    )
    tau_background.add_argument(
        "--tau-mu-sample",
        default="DATA_Muon",
        help="Restrict tau_mu inputs to one sample key.",
    )
    tau_background.add_argument(
        "--tau-ele-sample",
        default="DATA_EGamma",
        help="Restrict tau_ele inputs to one sample key.",
    )
    tau_background.add_argument("--variation", default="nominal")
    tau_background.add_argument(
        "--run-period",
        required=True,
        help="Run-period label used in the LaTeX table.",
    )
    tau_background.add_argument(
        "--flavor",
        default=r"$\tau_h$",
        help="Flavor label used in the output table.",
    )
    tau_background.add_argument(
        "--layers",
        nargs="+",
        default=["NLayers4", "NLayers5", "NLayers6plus", "combinedBins"],
        help="Layer-bin rows to estimate.",
    )
    tau_background.add_argument(
        "--control-prescale",
        type=float,
        default=1.0,
        help="Effective prescale/luminosity factor applied to tau N_ctrl.",
    )
    tau_background.add_argument(
        "--trigger-efficiency",
        type=float,
        required=True,
        help=(
            "Effective trigger-efficiency divisor for the tau control sample."
        ),
    )
    tau_background.add_argument(
        "--trigger-efficiency-error",
        type=float,
        default=0.0,
        help="Absolute uncertainty on --trigger-efficiency.",
    )
    tau_probability_source = tau_background.add_mutually_exclusive_group()
    tau_probability_source.add_argument(
        "--tau-probability",
        type=float,
        help=(
            "Data-derived N_cross/N_(cross and IsoMu24) trigger scale P(tau). "
            "This scales the cross-and-IsoMu24 tau-control N_ctrl back to the "
            "full cross-trigger population and is stored in JSON."
        ),
    )
    tau_probability_source.add_argument(
        "--tau-probability-json",
        type=Path,
        help=(
            "Read P(tau) and its uncertainty from the JSON written by "
            "extract-tau-trigger-probability."
        ),
    )
    tau_probability_source.add_argument(
        "--tau-probability-files",
        nargs="+",
        type=Path,
        help=(
            "Calculate P(tau)=N_cross/N_(cross and IsoMu24) directly from one "
            "or more Pocket Coffea outputs "
            "produced with DISAPPTRKS_CATEGORY_MODE=tau_trigger_probability."
        ),
    )
    tau_background.add_argument(
        "--tau-probability-error",
        type=float,
        default=0.0,
        help="Absolute uncertainty on --tau-probability.",
    )
    tau_background.add_argument(
        "--met-cut",
        type=float,
        default=120.0,
        help="Offline lepton-removed MET threshold used for Poffline/Pmiss integration.",
    )
    tau_background.add_argument(
        "--phi-cut",
        type=float,
        default=0.5,
        help="Delta-phi threshold used for Poffline/Pmiss integration.",
    )
    tau_background.add_argument(
        "--output-json",
        type=Path,
        help="Write detailed estimate components to JSON.",
    )
    tau_background.add_argument(
        "--output-tex",
        type=Path,
        help="Write a LaTeX summary table.",
    )
    tau_background.add_argument(
        "--table-env",
        action="store_true",
        help="Wrap the LaTeX tabular in a table environment.",
    )
    tau_background.set_defaults(func=_estimate_tau_background_command)

    standard_tau_background = subparsers.add_parser(
        "make-standard-tau-background",
        help=(
            "Build per-period and combined tau-background estimates from the "
            "standardized analysis_output directory tree."
        ),
    )
    standard_tau_background.add_argument(
        "--run-period",
        nargs="+",
        required=True,
        help="One or more run periods, e.g. 2022CD 2022EFG.",
    )
    standard_tau_background.add_argument(
        "--trigger-efficiency",
        nargs="+",
        required=True,
        help=(
            "Effective trigger efficiency. A bare value is accepted for one "
            "period; use PERIOD=VALUE for multiple periods."
        ),
    )
    standard_tau_background.add_argument(
        "--trigger-efficiency-error",
        nargs="+",
        help=(
            "Absolute trigger-efficiency uncertainty. Defaults to zero; use "
            "PERIOD=VALUE for multiple periods."
        ),
    )
    standard_tau_background.add_argument(
        "--input-base",
        type=Path,
        default=Path("analysis_output"),
        help="Input base containing <period>/<mode> (default: analysis_output).",
    )
    standard_tau_background.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "For one period, the output directory; for multiple periods, the "
            "base containing per-period directories and the combined table."
        ),
    )
    standard_tau_background.add_argument("--tau-control-files", nargs="+", type=Path)
    standard_tau_background.add_argument("--tau-mu-files", nargs="+", type=Path)
    standard_tau_background.add_argument("--tau-ele-files", nargs="+", type=Path)
    standard_tau_background.add_argument(
        "--tau-probability-files", nargs="+", type=Path
    )
    standard_tau_background.add_argument("--control-prescale", type=float, default=1.0)
    standard_tau_background.add_argument("--met-cut", type=float, default=120.0)
    standard_tau_background.add_argument("--phi-cut", type=float, default=0.5)
    standard_tau_background.add_argument("--table-env", action="store_true")
    standard_tau_background.set_defaults(func=_make_standard_tau_background_command)

    combined_lepton_background = subparsers.add_parser(
        "combine-lepton-background-tables",
        help="Combine per-period lepton-background JSON summaries into one AN-style LaTeX table.",
    )
    combined_lepton_background.add_argument(
        "--input",
        action="append",
        required=True,
        help=(
            "Run-period label and JSON path formatted as RUN_PERIOD=path.json. "
            "Repeat this option in the desired table order."
        ),
    )
    combined_lepton_background.add_argument(
        "--output-tex",
        type=Path,
        required=True,
        help="Write the combined LaTeX table.",
    )
    combined_lepton_background.add_argument(
        "--flavor",
        help="Optional exact flavor label filter, e.g. '$e$' or '$\\mu$'.",
    )
    combined_lepton_background.add_argument(
        "--tau-probability",
        type=float,
        help=(
            "Deprecated display override retained for compatibility. The "
            "LaTeX background estimate table no longer includes a P(tau) column."
        ),
    )
    combined_lepton_background.add_argument(
        "--tau-probability-error",
        type=float,
        default=0.0,
        help="Absolute uncertainty on --tau-probability.",
    )
    combined_lepton_background.add_argument(
        "--table-env",
        action="store_true",
        help="Wrap the LaTeX tabular in a table environment.",
    )
    combined_lepton_background.set_defaults(func=_combine_lepton_background_tables_command)

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
        "--stddev-exclude-top",
        type=int,
        default=0,
        help=(
            "Exclude the N highest-inefficiency occupied eta-phi bins from the "
            "stddev calculation only. Default 0 preserves legacy behavior."
        ),
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

    high_purity_study = subparsers.add_parser(
        "plot-high-purity-study",
        help="Overlay fake-sideband track inputs before and after highPurity.",
    )
    high_purity_study.add_argument("files", nargs="+", type=Path)
    high_purity_study.add_argument("--control", choices=("zmumu", "zee"), required=True)
    high_purity_study.add_argument("--sample")
    high_purity_study.add_argument(
        "--layers", nargs="+", default=["NLayers4"],
        choices=("NLayers4", "NLayers5", "NLayers6plus", "combinedBins"),
    )
    high_purity_study.add_argument("--output-dir", type=Path, default=Path("plots/high_purity_study"))
    high_purity_study.add_argument("--title-prefix", default="")
    high_purity_study.set_defaults(func=_plot_high_purity_study_command)

    signal_dedx_study = subparsers.add_parser(
        "plot-signal-dedx-study",
        help=(
            "Plot per-track dE/dx summaries after the full disappearing-track "
            "signal selection."
        ),
    )
    signal_dedx_study.add_argument("files", nargs="+", type=Path)
    signal_dedx_study.add_argument("--sample")
    signal_dedx_study.add_argument(
        "--layers",
        nargs="+",
        default=["NLayers4", "NLayers5", "NLayers6plus"],
        choices=("NLayers4", "NLayers5", "NLayers6plus", "combinedBins"),
    )
    signal_dedx_study.add_argument(
        "--output-dir",
        type=Path,
        default=Path("plots/signal_dedx_study"),
    )
    signal_dedx_study.add_argument("--title-prefix", default="")
    signal_dedx_study.set_defaults(func=_plot_signal_dedx_study_command)

    signal_high_purity = subparsers.add_parser(
        "summarize-signal-high-purity",
        help=(
            "Compare the full signal selection before and after the nominal "
            "highPurity requirement in every layer bin."
        ),
    )
    signal_high_purity.add_argument("files", nargs="+", type=Path)
    signal_high_purity.add_argument("--dataset", help="Restrict to one dataset key.")
    signal_high_purity.add_argument("--sample", help="Restrict to one sample key.")
    signal_high_purity.add_argument("--variation", default="nominal")
    signal_high_purity.add_argument(
        "--full-cutflow",
        action="store_true",
        help="Print every cumulative event and track selection row.",
    )
    signal_high_purity.add_argument(
        "--layers",
        nargs="+",
        default=["NLayers4", "NLayers5", "NLayers6plus", "combinedBins"],
        choices=("NLayers4", "NLayers5", "NLayers6plus", "combinedBins"),
        help="Layer-bin cutflows to print with --full-cutflow.",
    )
    signal_high_purity.set_defaults(func=_summarize_signal_high_purity_command)

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

    standard_fake_tracks = subparsers.add_parser(
        "make-standard-fake-track-estimate",
        help=(
            "Build zmumu, zee, and combined fake-track estimates from the "
            "standardized analysis_output directory tree."
        ),
    )
    standard_fake_tracks.add_argument(
        "--run-period",
        nargs="+",
        required=True,
        help="One or more run-period directory/table labels, e.g. 2022CD 2022EFG.",
    )
    standard_fake_tracks.add_argument(
        "--input-base",
        type=Path,
        default=Path("analysis_output"),
        help="Input base containing <period>/fake_tracks (default: analysis_output).",
    )
    standard_fake_tracks.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "For one period, the output directory; for multiple periods, the "
            "base containing per-period directories and table34_combined.tex."
        ),
    )
    standard_fake_tracks.add_argument("--basic-files", nargs="+", type=Path)
    standard_fake_tracks.add_argument("--zmumu-files", nargs="+", type=Path)
    standard_fake_tracks.add_argument("--zee-files", nargs="+", type=Path)
    standard_fake_tracks.add_argument(
        "--basic-yield-category",
        default="basic_selection",
        help="Basic/JetMET normalization category (default: basic_selection).",
    )
    standard_fake_tracks.add_argument(
        "--basic-sample",
        default="DATA_JetMET",
        help="Basic-output sample key (default: DATA_JetMET).",
    )
    standard_fake_tracks.add_argument(
        "--transfer-factor-source",
        choices=["fixed", "fit"],
        default="fit",
        help="Transfer-factor source (default: fit).",
    )
    standard_fake_tracks.add_argument(
        "--fit-plots",
        action="store_true",
        help="With --transfer-factor-source fit, write control fit PDFs.",
    )
    standard_fake_tracks.add_argument(
        "--sideband-plots",
        action="store_true",
        help=(
            "Write candidate hit-pattern and per-layer dE/dx PDFs for the "
            "Z-control events entering N_sideband."
        ),
    )
    standard_fake_tracks.add_argument("--table-env", action="store_true")
    standard_fake_tracks.set_defaults(func=_make_standard_fake_track_estimate_command)

    dataset_json = subparsers.add_parser(
        "make-dataset-json",
        help="Create a PocketCoffea dataset JSON from an EOS directory or filelist.",
    )
    dataset_json.add_argument("eos_path", nargs="?", help="EOS directory, e.g. /store/user/...")
    dataset_json.add_argument("-o", "--output", type=Path, required=True)
    dataset_json.add_argument(
        "--dataset-name",
        help="Dataset key (required unless --group-signal-points is used).",
    )
    dataset_json.add_argument("--sample", default="DATA_Muon")
    dataset_json.add_argument("--year", required=True)
    dataset_json.add_argument("--era", required=True)
    dataset_json.add_argument("--primary-dataset", default="Muon")
    dataset_json.add_argument("--nevents", default="0")
    dataset_json.add_argument("--nano-version", type=int, default=15)
    dataset_json.add_argument("--is-mc", action="store_true")
    dataset_json.add_argument(
        "--xsec",
        help=(
            "MC cross section written to metadata. Required with --is-mc; "
            "use 1.0 when only within-sample acceptance ratios are needed."
        ),
    )
    dataset_json.add_argument(
        "--count-events",
        action="store_true",
        help="Open each ROOT file and set nevents to the exact total Events entries.",
    )
    dataset_json.add_argument(
        "--group-signal-points",
        action="store_true",
        help=(
            "Create one dataset entry per <signal-marker>/<signal-point> directory. "
            "Requires --is-mc and --count-events."
        ),
    )
    dataset_json.add_argument(
        "--signal-marker",
        default="SignalSim",
        help=(
            "Parent directory whose immediate children define signal points "
            "for --group-signal-points. Default: SignalSim."
        ),
    )
    dataset_json.add_argument(
        "--event-count-workers",
        type=int,
        default=12,
        help="Concurrent ROOT metadata reads for --count-events. Default: 12.",
    )
    dataset_json.add_argument(
        "--event-count-timeout",
        type=int,
        default=120,
        help="Per-file XRootD open timeout in seconds. Default: 120.",
    )
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
    dataset_json.add_argument(
        "--publish",
        action="store_true",
        help=(
            "After writing --output locally, also xrdcp it to the group's shared "
            "canonical dataset-JSON space, so other checkouts can resolve it by name "
            "via DISAPPTRKS_DATASET_JSON."
        ),
    )
    dataset_json.add_argument(
        "--publish-name",
        help=(
            "Canonical name to publish under (no directory or .json suffix). "
            "Default: the --output filename stem."
        ),
    )
    dataset_json.add_argument(
        "--publish-eos-base",
        default=DATASET_JSON_EOS_BASE_DEFAULT,
        help="Shared EOS base directory for --publish.",
    )
    dataset_json.add_argument(
        "--force",
        action="store_true",
        help=(
            "With --publish, overwrite an existing canonical dataset JSON of the "
            "same name instead of refusing."
        ),
    )
    dataset_json.set_defaults(func=_make_dataset_json_command)

    publish_output = subparsers.add_parser(
        "publish-output",
        help=(
            "Copy a local analysis_output/<period>/<mode> directory to the group's "
            "shared EOS output space."
        ),
    )
    publish_output.add_argument(
        "input_dir",
        type=Path,
        help="Local directory to publish, e.g. analysis_output/2025/fake_tracks/basic.",
    )
    publish_output.add_argument(
        "--period", required=True, help="Run period label, e.g. 2025."
    )
    publish_output.add_argument(
        "--mode",
        required=True,
        help="Mode/category label, e.g. fake_tracks/basic.",
    )
    publish_output.add_argument(
        "--eos-base",
        default=OUTPUT_EOS_BASE_DEFAULT,
        help="Shared EOS base directory for the published output.",
    )
    publish_output.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing EOS copy at the same period/mode path.",
    )
    publish_output.add_argument(
        "--suffix",
        help=(
            "Publish under <mode>_<suffix> instead, alongside any existing copy at "
            "the unsuffixed path. Mutually exclusive with --overwrite."
        ),
    )
    publish_output.set_defaults(func=_publish_output_command)

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
        "--output-suffix",
        help=(
            "Suffix for generated filelists, dataset JSON filenames, and dataset names. "
            "Defaults to OSUv2 when scanning only dev_v2, otherwise no suffix."
        ),
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
