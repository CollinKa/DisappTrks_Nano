"""Lepton-background estimates built from Pveto-style control counts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from math import floor, log10, sqrt
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .fake_tracks import Count, _control_column_key, _count_from_payload, _layer_label
from .summaries import cutflow_count
from .tables import (
    CountWithVariance,
    POISSON_ZERO_UPPER_68,
    format_asymmetric_latex,
    format_pm_latex,
    format_value_with_uncertainty,
    pveto_with_asymmetric_uncertainty,
)


@dataclass(frozen=True)
class LeptonBackgroundEstimate:
    """One layer-bin lepton-background estimate."""

    flavor: str
    layer: str
    control_raw: Count
    control: Count
    p_veto: Count
    p_offline: Count
    p_miss: Count
    trigger_efficiency: Count
    tau_probability: Count
    estimate: Count
    control_category: str
    poffline_numerator_category: str
    poffline_denominator_category: str
    pmiss_numerator_category: str
    pmiss_denominator_category: str

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        for key in (
            "control_raw",
            "control",
            "p_veto",
            "p_offline",
            "p_miss",
            "trigger_efficiency",
            "tau_probability",
            "estimate",
        ):
            value = out[key]
            out[key] = {
                "value": value["value"],
                "error": sqrt(max(value["variance"], 0.0)),
                "variance": value["variance"],
            }
        return out

    @property
    def p_trigger(self) -> Count:
        """Backward-compatible alias for the AN ``Pmiss`` factor."""

        return self.p_miss


def _category_name(pattern: str, layer: str) -> str:
    return pattern.format(layer=layer)


def _count_from_mapping(counts: Mapping[str, Any], category: str) -> Count:
    if category not in counts:
        raise KeyError(f"category {category!r} not found")
    value = counts[category]
    if isinstance(value, Count):
        return value
    if isinstance(value, Mapping):
        if "value" not in value:
            raise KeyError(f"category {category!r} mapping must contain a 'value' key")
        return Count(float(value["value"]), float(value.get("variance", value["value"])))
    return Count(float(value))


def _count_from_cutflow(
    cutflow: dict[str, Any],
    category: str,
    *,
    dataset: str | None = None,
    sample: str | None = None,
    variation: str = "nominal",
) -> Count:
    value = cutflow_count(
        cutflow,
        category,
        dataset=dataset,
        sample=sample,
        variation=variation,
    )
    return Count(value)


def _count_from_source(
    *,
    counts: Mapping[str, Any] | None,
    cutflow: dict[str, Any] | None,
    category: str,
    dataset: str | None,
    sample: str | None,
    variation: str,
) -> Count:
    if counts is not None:
        return _count_from_mapping(counts, category)
    if cutflow is None:
        raise ValueError("either counts or cutflow must be provided")
    return _count_from_cutflow(
        cutflow,
        category,
        dataset=dataset,
        sample=sample,
        variation=variation,
    )


def probability_from_counts(numerator: Count, denominator: Count) -> Count:
    """Return a binomial probability with a conservative zero-denominator guard."""

    if denominator.value <= 0.0:
        return Count(0.0, 0.0)
    value = numerator.value / denominator.value
    clipped = min(max(value, 0.0), 1.0)
    variance = clipped * (1.0 - clipped) / denominator.value
    return Count(value, variance)


def trigger_efficiency_from_counts(
    *,
    total_os: Count,
    total_ss: Count,
    passes_os: Count,
    passes_ss: Count,
) -> Count:
    """Legacy OS-minus-SS trigger-efficiency ratio helper."""

    passes = Count(
        passes_os.value - passes_ss.value,
        passes_os.variance + passes_ss.variance,
    )
    total = Count(
        total_os.value - total_ss.value,
        total_os.variance + total_ss.variance,
    )
    return probability_from_counts(passes, total)


def _multiply_counts_at_physical_boundary(*factors: Count) -> Count:
    """Multiply lepton factors without losing an uncertainty at value zero.

    The shared ``Count`` arithmetic is intentionally left unchanged because it
    is also used by the established fake-track estimate.  Lepton ``Pveto`` can
    instead sit at the physical boundary with a nonzero upper uncertainty, so
    propagate absolute derivatives locally for this product.
    """

    result = Count(1.0, 0.0)
    for factor in factors:
        value = result.value * factor.value
        variance = (
            factor.value * factor.value * float(result.variance)
            + result.value * result.value * float(factor.variance)
        )
        result = Count(value, variance)
    return result


def _divide_counts_at_physical_boundary(
    numerator: Count,
    denominator: Count,
) -> Count:
    """Divide while retaining numerator uncertainty at a zero boundary."""

    if denominator.value <= 0.0:
        raise ValueError("denominator must be positive")
    value = numerator.value / denominator.value
    variance = (
        float(numerator.variance) / (denominator.value * denominator.value)
        + numerator.value
        * numerator.value
        * float(denominator.variance)
        / (denominator.value**4)
    )
    return Count(value, variance)


def _walk_hists(value: Any):
    if hasattr(value, "axes") and hasattr(value, "values"):
        yield value
    elif isinstance(value, Mapping):
        for nested in value.values():
            yield from _walk_hists(nested)


def _iter_variable_hists(
    outputs: Sequence[Mapping[str, Any]],
    variable: str,
    *,
    dataset: str | None = None,
    sample: str | None = None,
):
    for output in outputs:
        variables = output.get("variables", {})
        if variable not in variables:
            continue
        value: Any = variables[variable]
        if sample is not None and isinstance(value, Mapping):
            value = value.get(sample, {})
        if dataset is not None and sample is None and isinstance(value, Mapping):
            nested_values = value.values()
        elif dataset is not None and isinstance(value, Mapping):
            nested_values = [value.get(dataset, {})]
        else:
            nested_values = [value]
        for nested in nested_values:
            yield from _walk_hists(nested)


def _strip_category_axes(hist_obj: Any, *, category: str = "inclusive") -> Any | None:
    try:
        axis_names = [axis.name for axis in hist_obj.axes]
    except AttributeError:
        return None

    for axis_name in ("cat", "variation", "sample"):
        if axis_name in axis_names:
            try:
                selector = category if axis_name == "cat" else "nominal"
                hist_obj = hist_obj[{axis_name: selector}]
            except Exception:
                if axis_name == "cat":
                    return None
    return hist_obj


def _hist_counts_edges_nd(hist_obj: Any, *, category: str = "inclusive"):
    hist_obj = _strip_category_axes(hist_obj, category=category)
    if hist_obj is None:
        return None
    axes = [axis for axis in hist_obj.axes if getattr(axis, "name", "") != "cat"]
    try:
        counts = np.asarray(hist_obj.values(flow=False), dtype=float)
        edges = [np.asarray(axis.edges, dtype=float) for axis in axes]
    except Exception:
        return None
    if counts.ndim != len(edges):
        return None
    return counts, edges


def _summed_hist_counts_edges_nd(
    outputs: Sequence[Mapping[str, Any]],
    variable: str,
    *,
    dataset: str | None = None,
    sample: str | None = None,
    category: str = "inclusive",
):
    total_counts = None
    total_edges = None
    for hist_obj in _iter_variable_hists(
        outputs,
        variable,
        dataset=dataset,
        sample=sample,
    ):
        result = _hist_counts_edges_nd(hist_obj, category=category)
        if result is None:
            continue
        counts, edges = result
        if total_counts is None:
            total_counts = counts.copy()
            total_edges = [edge.copy() for edge in edges]
        else:
            if counts.shape != total_counts.shape:
                raise ValueError(f"incompatible histogram shapes for {variable}")
            total_counts += counts
    if total_counts is None or total_edges is None:
        return None
    return total_counts, total_edges


def _hist_integral(
    counts: np.ndarray,
    edges: Sequence[np.ndarray],
    *,
    met_cut: float,
    phi_cut: float,
) -> float:
    if counts.ndim != 2 or len(edges) != 2:
        return 0.0
    met_centers = 0.5 * (edges[0][:-1] + edges[0][1:])
    phi_centers = 0.5 * (edges[1][:-1] + edges[1][1:])
    mask = (met_centers[:, None] >= met_cut) & (phi_centers[None, :] >= phi_cut)
    return float(counts[mask].sum())


def _weighted_met_trigger_integral(
    met_phi_counts: np.ndarray,
    met_phi_edges: Sequence[np.ndarray],
    trigger_total_counts: np.ndarray,
    trigger_pass_counts: np.ndarray,
    trigger_edges: Sequence[np.ndarray],
    *,
    met_cut: float,
    phi_cut: float,
) -> tuple[float, float]:
    if met_phi_counts.ndim != 2 or len(met_phi_edges) != 2:
        return 0.0, 0.0
    if trigger_total_counts.ndim != 1 or trigger_pass_counts.ndim != 1:
        return 0.0, 0.0

    met_centers = 0.5 * (met_phi_edges[0][:-1] + met_phi_edges[0][1:])
    phi_centers = 0.5 * (met_phi_edges[1][:-1] + met_phi_edges[1][1:])
    trigger_centers = 0.5 * (trigger_edges[0][:-1] + trigger_edges[0][1:])
    with np.errstate(divide="ignore", invalid="ignore"):
        turn_on = np.divide(
            trigger_pass_counts,
            trigger_total_counts,
            out=np.zeros_like(trigger_pass_counts, dtype=float),
            where=trigger_total_counts > 0.0,
        )
    met_bin_turn_on = np.interp(
        met_centers,
        trigger_centers,
        turn_on,
        left=0.0,
        right=turn_on[-1] if len(turn_on) else 0.0,
    )
    offline_mask = (met_centers[:, None] >= met_cut) & (
        phi_centers[None, :] >= phi_cut
    )
    offline_total = float(met_phi_counts[offline_mask].sum())
    weighted = float((met_phi_counts * met_bin_turn_on[:, None])[offline_mask].sum())
    return weighted, offline_total


def legacy_met_probability_components_from_outputs(
    outputs: Sequence[Mapping[str, Any]],
    *,
    prefix: str,
    layers: Sequence[str],
    control_counts: Mapping[str, Count],
    dataset: str | None = None,
    sample: str | None = None,
    category: str = "inclusive",
    met_cut: float = 120.0,
    phi_cut: float = 0.5,
) -> dict[str, dict[str, Count]]:
    """Return raw components used for the legacy Poffline/Pmiss ratios."""

    components: dict[str, dict[str, Count]] = {}
    for layer in layers:
        met_variable = f"n{prefix}BackgroundMetNoMuPt_{layer}"
        trig_variable = f"n{prefix}BackgroundMetNoMuPtTrig_{layer}"
        fallback_met_variable = f"n{prefix}BackgroundMetMinusOnePt_{layer}"
        fallback_trig_variable = f"n{prefix}BackgroundMetMinusOnePtTrig_{layer}"
        met_phi_variable = (
            f"n{prefix}BackgroundDeltaPhiMetJetLeadingVsMetMinusOnePt_{layer}"
        )
        met_hist = _summed_hist_counts_edges_nd(
            outputs,
            met_variable,
            dataset=dataset,
            sample=sample,
            category=category,
        )
        trig_hist = _summed_hist_counts_edges_nd(
            outputs,
            trig_variable,
            dataset=dataset,
            sample=sample,
            category=category,
        )
        if met_hist is None or trig_hist is None:
            met_hist = _summed_hist_counts_edges_nd(
                outputs,
                fallback_met_variable,
                dataset=dataset,
                sample=sample,
                category=category,
            )
            trig_hist = _summed_hist_counts_edges_nd(
                outputs,
                fallback_trig_variable,
                dataset=dataset,
                sample=sample,
                category=category,
            )
        met_phi_hist = _summed_hist_counts_edges_nd(
            outputs,
            met_phi_variable,
            dataset=dataset,
            sample=sample,
            category=category,
        )
        if met_hist is None or trig_hist is None or met_phi_hist is None:
            continue
        met_counts, met_edges = met_hist
        trig_counts, _ = trig_hist
        met_phi_counts, met_phi_edges = met_phi_hist

        offline_pass = _hist_integral(
            met_phi_counts,
            met_phi_edges,
            met_cut=met_cut,
            phi_cut=phi_cut,
        )
        weighted_trigger_pass, offline_total = _weighted_met_trigger_integral(
            met_phi_counts,
            met_phi_edges,
            met_counts,
            trig_counts,
            met_edges,
            met_cut=met_cut,
            phi_cut=phi_cut,
        )
        components[layer] = {
            "control": control_counts[layer],
            "offline_pass": Count(offline_pass, offline_pass),
            "weighted_trigger_pass": Count(weighted_trigger_pass, weighted_trigger_pass),
            "offline_total": Count(offline_total, offline_total),
        }
    return components


def legacy_met_probabilities_from_outputs(
    outputs: Sequence[Mapping[str, Any]],
    *,
    prefix: str,
    layers: Sequence[str],
    control_counts: Mapping[str, Count],
    dataset: str | None = None,
    sample: str | None = None,
    category: str = "inclusive",
    met_cut: float = 120.0,
    phi_cut: float = 0.5,
) -> dict[str, tuple[Count, Count]]:
    """Compute Poffline and Pmiss with the legacy MET-trigger turn-on method."""

    components = legacy_met_probability_components_from_outputs(
        outputs,
        prefix=prefix,
        layers=layers,
        control_counts=control_counts,
        dataset=dataset,
        sample=sample,
        category=category,
        met_cut=met_cut,
        phi_cut=phi_cut,
    )
    probabilities: dict[str, tuple[Count, Count]] = {}
    for layer, values in components.items():
        poffline = probability_from_counts(
            values["offline_pass"],
            values["control"],
        )
        pmiss = probability_from_counts(
            values["weighted_trigger_pass"],
            values["offline_total"],
        )
        probabilities[layer] = (poffline, pmiss)
    return probabilities


def pveto_count_from_pair_counts(
    pair_counts: Mapping[str, float],
    *,
    use_two_lepton_denominator: bool = False,
) -> Count:
    """Convert OS/SS Pveto pair counts into a symmetric Count approximation.

    The legacy histogram-based path, used by the 2022/2023 background scripts,
    uses the same-sign-subtracted direct ratio.  The two-lepton denominator is
    kept for reproducing the older non-histogram fallback path.
    """

    if use_two_lepton_denominator:
        numerator = pair_counts.get("num_os", 0.0) - pair_counts.get("num_ss", 0.0)
        denominator = (
            2.0 * (pair_counts.get("den_os", 0.0) - pair_counts.get("den_ss", 0.0))
            - numerator
        )
        if denominator <= 0.0:
            return Count(0.0, 0.0)
        if numerator <= 0.0:
            return Count(0.0, (POISSON_ZERO_UPPER_68 / denominator) ** 2)
        value = numerator / denominator
        variance = value * (1.0 - min(max(value, 0.0), 1.0)) / denominator
        return Count(value, variance)

    summary = pveto_with_asymmetric_uncertainty(
        den_os=CountWithVariance(pair_counts.get("den_os", 0.0), pair_counts.get("den_os", 0.0)),
        num_os=CountWithVariance(pair_counts.get("num_os", 0.0), pair_counts.get("num_os", 0.0)),
        den_ss=CountWithVariance(pair_counts.get("den_ss", 0.0), pair_counts.get("den_ss", 0.0)),
        num_ss=CountWithVariance(pair_counts.get("num_ss", 0.0), pair_counts.get("num_ss", 0.0)),
    )
    uncertainty = max(summary.err_down, summary.err_up)
    return Count(summary.central, uncertainty * uncertainty)


def estimate_lepton_background(
    *,
    flavor: str,
    layers: Sequence[str],
    pair_counts: Mapping[str, Mapping[str, float]],
    cutflow: dict[str, Any] | None = None,
    counts: Mapping[str, Any] | None = None,
    control_category: str,
    poffline_numerator_category: str,
    poffline_denominator_category: str,
    pmiss_numerator_category: str | None = None,
    pmiss_denominator_category: str | None = None,
    ptrigger_numerator_category: str | None = None,
    ptrigger_denominator_category: str | None = None,
    control_prescale: float = 1.0,
    trigger_efficiency: Count | Mapping[str, Count] | None = None,
    tau_probability: Count | None = None,
    met_probabilities: Mapping[str, tuple[Count, Count]] | None = None,
    low_stat_layers: Sequence[str] = (),
    combined_layer: str = "combinedBins",
    dataset: str | None = None,
    sample: str | None = None,
    variation: str = "nominal",
) -> list[LeptonBackgroundEstimate]:
    """Compute one lepton-background estimate per requested layer bin.

    ``low_stat_layers`` names layer bins whose Poffline/Pmiss and trigger
    efficiency are drawn from ``combined_layer`` instead of their own
    per-layer categories -- N_ctrl and Pveto still use the requested layer's
    own categories. Use this for layer bins (e.g. NLayers4/NLayers5) whose
    own Poffline/Pmiss control samples are too small to give a meaningful
    per-layer ratio; leave it empty (the default) to use every layer's own
    categories, as before.
    """

    estimates = []
    pmiss_numerator_category = pmiss_numerator_category or ptrigger_numerator_category
    pmiss_denominator_category = pmiss_denominator_category or ptrigger_denominator_category
    if pmiss_numerator_category is None or pmiss_denominator_category is None:
        raise ValueError(
            "pmiss_numerator_category and pmiss_denominator_category are required"
        )
    trigger_efficiency = trigger_efficiency or Count(1.0, 0.0)
    tau_probability = tau_probability or Count(1.0, 0.0)
    for layer in layers:
        poffline_pmiss_layer = combined_layer if layer in low_stat_layers else layer
        control_name = _category_name(control_category, layer)
        poffline_num_name = _category_name(poffline_numerator_category, poffline_pmiss_layer)
        poffline_den_name = _category_name(poffline_denominator_category, poffline_pmiss_layer)
        pmiss_num_name = _category_name(pmiss_numerator_category, poffline_pmiss_layer)
        pmiss_den_name = _category_name(pmiss_denominator_category, poffline_pmiss_layer)

        control_raw = _count_from_source(
            counts=counts,
            cutflow=cutflow,
            category=control_name,
            dataset=dataset,
            sample=sample,
            variation=variation,
        )
        if control_raw.value == 0.0:
            # A raw Count built directly from an observed event count
            # defaults to Poisson variance = value (see Count.__post_init__),
            # which is itself 0 at value == 0 -- an observed zero is not an
            # infinitely precise zero, so quote the standard zero-count 68%
            # CL Poisson upper limit instead. Without this, the final
            # estimate collapses to exactly 0 +/- 0 downstream even when
            # Pveto has a real, nonzero uncertainty.
            control_raw = Count(0.0, POISSON_ZERO_UPPER_68**2)
        # Boundary-safe, not plain Count multiplication: Count.__mul__'s
        # relative-variance formula collapses to zero uncertainty whenever
        # an intermediate value is exactly zero (see control_raw above), so
        # a plain `control_raw * control_prescale * tau_probability` would
        # discard the Poisson upper bound just set on control_raw.
        control = _multiply_counts_at_physical_boundary(
            control_raw, Count(control_prescale, 0.0), tau_probability
        )
        if met_probabilities is not None and poffline_pmiss_layer in met_probabilities:
            poffline, pmiss = met_probabilities[poffline_pmiss_layer]
        else:
            poffline = probability_from_counts(
                _count_from_source(
                    counts=counts,
                    cutflow=cutflow,
                    category=poffline_num_name,
                    dataset=dataset,
                    sample=sample,
                    variation=variation,
                ),
                _count_from_source(
                    counts=counts,
                    cutflow=cutflow,
                    category=poffline_den_name,
                    dataset=dataset,
                    sample=sample,
                    variation=variation,
                ),
            )
            pmiss = probability_from_counts(
                _count_from_source(
                    counts=counts,
                    cutflow=cutflow,
                    category=pmiss_num_name,
                    dataset=dataset,
                    sample=sample,
                    variation=variation,
                ),
                _count_from_source(
                    counts=counts,
                    cutflow=cutflow,
                    category=pmiss_den_name,
                    dataset=dataset,
                    sample=sample,
                    variation=variation,
                ),
            )
        p_veto = pveto_count_from_pair_counts(
            pair_counts.get(layer, {}),
            use_two_lepton_denominator=False,
        )
        layer_trigger_efficiency = (
            trigger_efficiency.get(poffline_pmiss_layer, Count(1.0, 0.0))
            if isinstance(trigger_efficiency, Mapping)
            else trigger_efficiency
        )
        if layer_trigger_efficiency.value <= 0.0:
            raise ValueError(
                f"trigger efficiency must be positive for layer {layer}; "
                f"got {layer_trigger_efficiency.value}"
            )
        # Use the boundary-safe product/quotient for every flavor, not just
        # tau: it is mathematically identical to plain Count arithmetic
        # (control * p_veto * poffline * pmiss / trigger_efficiency) away
        # from zero, but -- unlike Count.__mul__/__truediv__, whose relative-
        # variance formula is undefined and guarded to 0.0 at a zero central
        # value -- it keeps a factor's real uncertainty (e.g. control_raw
        # observing zero raw control events, or a zero Pveto numerator) from
        # silently collapsing the whole estimate to 0 +/- 0.
        estimate_numerator = _multiply_counts_at_physical_boundary(
            control,
            p_veto,
            poffline,
            pmiss,
        )
        estimate = _divide_counts_at_physical_boundary(
            estimate_numerator,
            layer_trigger_efficiency,
        )
        estimates.append(
            LeptonBackgroundEstimate(
                flavor=flavor,
                layer=layer,
                control_raw=control_raw,
                control=control,
                p_veto=p_veto,
                p_offline=poffline,
                p_miss=pmiss,
                trigger_efficiency=layer_trigger_efficiency,
                tau_probability=tau_probability,
                estimate=estimate,
                control_category=control_name,
                poffline_numerator_category=poffline_num_name,
                poffline_denominator_category=poffline_den_name,
                pmiss_numerator_category=pmiss_num_name,
                pmiss_denominator_category=pmiss_den_name,
            )
        )
    return estimates


def write_lepton_background_json(
    estimates: Sequence[LeptonBackgroundEstimate],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps([estimate.as_dict() for estimate in estimates], indent=2, sort_keys=True))


def _count_from_serialized(payload: Mapping[str, Any]) -> Count:
    variance = payload.get("variance")
    if variance is None:
        error = float(payload.get("error", 0.0))
        variance = error * error
    return Count(float(payload["value"]), float(variance))


def read_lepton_background_json(path: Path) -> list[LeptonBackgroundEstimate]:
    payload = json.loads(path.read_text())
    estimates = []
    for item in payload:
        estimates.append(
            LeptonBackgroundEstimate(
                flavor=item["flavor"],
                layer=item["layer"],
                control_raw=_count_from_serialized(item["control_raw"]),
                control=_count_from_serialized(item["control"]),
                p_veto=_count_from_serialized(item["p_veto"]),
                p_offline=_count_from_serialized(item["p_offline"]),
                p_miss=_count_from_serialized(item["p_miss"]),
                trigger_efficiency=_count_from_serialized(item["trigger_efficiency"]),
                tau_probability=_count_from_serialized(
                    item.get("tau_probability", {"value": 1.0, "variance": 0.0})
                ),
                estimate=_count_from_serialized(item["estimate"]),
                control_category=item.get("control_category", ""),
                poffline_numerator_category=item.get("poffline_numerator_category", ""),
                poffline_denominator_category=item.get("poffline_denominator_category", ""),
                pmiss_numerator_category=item.get("pmiss_numerator_category", ""),
                pmiss_denominator_category=item.get("pmiss_denominator_category", ""),
            )
        )
    return estimates


def _an_layer_label(layer: str) -> str:
    return {
        "NLayers4": "4",
        "NLayers5": "5",
        "NLayers6plus": r"$\geq 6$",
        "combinedBins": "combined",
    }.get(layer, layer)


def _an_background_title(flavor: str) -> str:
    normalized = flavor.replace("$", "").replace("\\", "").lower()
    if "tau" in normalized:
        return "tau background"
    if "mu" in normalized:
        return "muon background"
    if "e" in normalized:
        return "electron background"
    return f"{flavor} background"


def _format_an_pm(value: float, error: float, *, scientific_threshold: float = 1.0e-3) -> str:
    scale_value = max(abs(float(value)), abs(float(error)))
    if scale_value == 0.0:
        return r"$0 \pm 0$"
    if scale_value < scientific_threshold:
        exponent = int(floor(log10(scale_value)))
        scale = 10.0**exponent
        value_text, error_text = format_value_with_uncertainty(
            value / scale,
            error / scale,
        )
        return rf"$({value_text} \pm {error_text}) \times 10^{{{exponent}}}$"
    return format_pm_latex(value, error)


def _format_tau_boundary(value: float, error: float) -> str:
    if value == 0.0 and error > 0.0:
        return format_asymmetric_latex(0.0, error, 0.0)
    return _format_an_pm(value, error)


def _format_tau_probability(value: float, error: float) -> str:
    """Format P(tau) in scientific notation, matching the exponent used for its error."""

    if value == 0.0:
        return format_pm_latex(value, error)
    exponent = int(floor(log10(abs(value))))
    scale = 10.0**exponent
    value_text, error_text = format_value_with_uncertainty(value / scale, error / scale)
    return rf"$({value_text} \pm {error_text}) \times 10^{{{exponent}}}$"


def _write_lepton_background_latex_body(
    out,
    period_estimates: Sequence[tuple[str, Sequence[LeptonBackgroundEstimate]]],
    *,
    tau_probability: Count | None = None,
) -> None:
    first_estimates = period_estimates[0][1] if period_estimates else []
    title = _an_background_title(first_estimates[0].flavor) if first_estimates else "lepton background"
    is_tau = bool(first_estimates and "tau" in first_estimates[0].flavor.lower())
    n_columns = 9 if is_tau else 8
    column_spec = "ll" + ("c" * (n_columns - 2))
    out.write(r"\begin{tiny}" + "\n")
    out.write(r"\begin{tabular}{" + column_spec + r"}" + "\n")
    out.write(r"\hline" + "\n")
    out.write(r"\multicolumn{" + str(n_columns) + r"}{c}{" + title + r"} \\" + "\n")
    header = (
        r"run period & $n_{\mathrm{layers}}$ & "
        r"$\epsilon_{\mathrm{trigger}}^{\ell}$ & "
    )
    if is_tau:
        header += r"$P(\tau)$ & "
    header += (
        r"$N_{\mathrm{ctrl}}^{\ell}$ & $P_{\mathrm{veto}}$ & "
        r"$P_{\mathrm{offline}}$ & $P_{\mathrm{trigger}}$ & estimate \\"
    )
    out.write(header + "\n")
    out.write(r"\hline" + "\n")
    for period_index, (run_period, estimates) in enumerate(period_estimates):
        if period_index > 0:
            out.write(r"\hline" + "\n")
        n_rows = len(estimates)
        for index, estimate in enumerate(estimates):
            run_period_cell = rf"\multirow{{{n_rows}}}{{*}}{{{run_period}}}" if index == 0 else ""
            sparse_tau_layer = is_tau and estimate.layer in ("NLayers4", "NLayers5")
            poffline_text = (
                r"---"
                if sparse_tau_layer
                else format_pm_latex(estimate.p_offline.value, estimate.p_offline.error)
            )
            ptrigger_text = (
                r"---"
                if sparse_tau_layer
                else format_pm_latex(estimate.p_miss.value, estimate.p_miss.error)
            )
            row = (
                f"{run_period_cell} & {_an_layer_label(estimate.layer)} & "
                f"{format_pm_latex(estimate.trigger_efficiency.value, estimate.trigger_efficiency.error)} & "
            )
            if is_tau:
                row += (
                    f"{_format_tau_probability(estimate.tau_probability.value, estimate.tau_probability.error)} & "
                )
            row += (
                f"{format_pm_latex(estimate.control_raw.value, estimate.control_raw.error)} & "
                f"{(_format_tau_boundary if is_tau else _format_an_pm)(estimate.p_veto.value, estimate.p_veto.error)} & "
                f"{poffline_text} & "
                f"{ptrigger_text} & "
                f"{(_format_tau_boundary if is_tau else _format_an_pm)(estimate.estimate.value, estimate.estimate.error)} "
                r"\\" + "\n"
            )
            out.write(row)
    out.write(r"\hline" + "\n")
    out.write(r"\end{tabular}" + "\n")
    out.write(r"\end{tiny}" + "\n")


def write_lepton_background_latex(
    estimates: Sequence[LeptonBackgroundEstimate],
    path: Path,
    *,
    run_period: str,
    include_table_env: bool = False,
    tau_probability: Count | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    title = _an_background_title(estimates[0].flavor) if estimates else "lepton background"
    with path.open("w") as out:
        if include_table_env:
            out.write(r"\begin{table}" + "\n")
            out.write(r"\centering" + "\n")
            out.write(r"\caption{" + title.capitalize() + r" estimate.}" + "\n")
            out.write(r"\label{tab:lepton_background_estimate}" + "\n")
        _write_lepton_background_latex_body(
            out,
            [(run_period, estimates)],
            tau_probability=tau_probability,
        )
        if include_table_env:
            out.write(r"\end{table}" + "\n")


def write_combined_lepton_background_latex(
    period_estimates: Sequence[tuple[str, Sequence[LeptonBackgroundEstimate]]],
    path: Path,
    *,
    include_table_env: bool = False,
    tau_probability: Count | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    first_estimates = period_estimates[0][1] if period_estimates else []
    title = _an_background_title(first_estimates[0].flavor) if first_estimates else "lepton background"
    with path.open("w") as out:
        if include_table_env:
            out.write(r"\begin{table}" + "\n")
            out.write(r"\centering" + "\n")
            out.write(r"\caption{" + title.capitalize() + r" estimate.}" + "\n")
            out.write(r"\label{tab:lepton_background_estimate}" + "\n")
        _write_lepton_background_latex_body(
            out,
            period_estimates,
            tau_probability=tau_probability,
        )
        if include_table_env:
            out.write(r"\end{table}" + "\n")


_TOTAL_BACKGROUND_LAYER_ORDER = ("NLayers4", "NLayers5", "NLayers6plus")


def _add_counts(counts: Sequence[Count]) -> Count:
    value = sum(count.value for count in counts)
    variance = sum(count.variance for count in counts)
    return Count(value, variance)


def _fake_yield_by_layer(path: Path, *, control_region: str) -> dict[str, Count]:
    payload = json.loads(path.read_text())
    by_layer: dict[str, Count] = {}
    for estimate in payload.get("estimates", []):
        if _control_column_key(estimate["control_region"]) != control_region:
            continue
        by_layer[estimate["layer"]] = _count_from_payload(estimate["fake_yield"])
    return by_layer


def write_combined_total_background_latex(
    periods: Sequence[str],
    muon_json: Mapping[str, Path],
    electron_json: Mapping[str, Path],
    tau_json: Mapping[str, Path],
    fake_json: Mapping[str, Path],
    path: Path,
    *,
    include_table_env: bool = False,
    fake_control_region: str = "zmumu",
) -> None:
    """Write a Leptons / Spurious Tracks / Total table spanning multiple run periods.

    Sums the electron, muon, and tau lepton-background estimates ("Leptons")
    and combines them with the fake-track estimate for a single nominal
    control region ("Spurious Tracks", Z->mu mu by default -- Z->ee is a
    cross-check, per the dissertation's convention, not folded in here).
    Statistical uncertainties only, propagated in quadrature (independent
    measurements: variances add).
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as out:
        if include_table_env:
            out.write(r"\begin{table}[htbp]" + "\n")
            out.write(r"\centering" + "\n")
            out.write(
                r"\caption{Expected backgrounds: leptons, spurious tracks, and "
                r"total, per run period and layer bin (statistical uncertainties "
                r"only).}" + "\n"
            )
            out.write(r"\label{tab:total_background_estimate}" + "\n")
        out.write(r"\begin{tabular}{llccc}" + "\n")
        out.write(r"\hline" + "\n")
        out.write(r" & & \multicolumn{3}{c}{Expected Backgrounds} \\" + "\n")
        out.write(
            r"Run Period & $n_{\mathrm{layers}}$ & Leptons & "
            r"Spurious Tracks & Total \\" + "\n"
        )
        out.write(r"\hline" + "\n")
        for period_index, period in enumerate(periods):
            muon = {
                estimate.layer: estimate.estimate
                for estimate in read_lepton_background_json(muon_json[period])
            }
            electron = {
                estimate.layer: estimate.estimate
                for estimate in read_lepton_background_json(electron_json[period])
            }
            tau = {
                estimate.layer: estimate.estimate
                for estimate in read_lepton_background_json(tau_json[period])
            }
            fake = _fake_yield_by_layer(fake_json[period], control_region=fake_control_region)

            missing_layers = [
                layer
                for layer in _TOTAL_BACKGROUND_LAYER_ORDER
                if layer not in muon or layer not in electron or layer not in tau or layer not in fake
            ]
            if missing_layers:
                raise ValueError(
                    f"run period {period!r} is missing layer bin(s) {missing_layers} "
                    "in its muon/electron/tau/fake-track inputs"
                )

            if period_index > 0:
                out.write(r"\hline" + "\n")
            for layer_index, layer in enumerate(_TOTAL_BACKGROUND_LAYER_ORDER):
                lepton_sum = _add_counts([muon[layer], electron[layer], tau[layer]])
                spurious = fake[layer]
                total = _add_counts([lepton_sum, spurious])
                period_text = period if layer_index == 0 else ""
                out.write(
                    f"{period_text} & {_layer_label(layer)} & "
                    f"{_format_tau_boundary(lepton_sum.value, lepton_sum.error)} & "
                    f"{_format_tau_boundary(spurious.value, spurious.error)} & "
                    f"{_format_tau_boundary(total.value, total.error)} \\\\\n"
                )
        out.write(r"\hline" + "\n")
        out.write(r"\end{tabular}" + "\n")
        if include_table_env:
            out.write(r"\end{table}" + "\n")
