"""Lepton-background estimates built from Pveto-style control counts."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from math import sqrt
from pathlib import Path
from typing import Any, Mapping, Sequence

from .fake_tracks import Count
from .summaries import cutflow_count
from .tables import (
    CountWithVariance,
    format_count,
    format_pm_latex,
    format_value_with_uncertainty,
    pveto_with_asymmetric_uncertainty,
)


@dataclass(frozen=True)
class LeptonBackgroundEstimate:
    """One layer-bin lepton-background estimate."""

    flavor: str
    layer: str
    control: Count
    p_veto: Count
    p_offline: Count
    p_miss: Count
    estimate: Count
    control_category: str
    poffline_numerator_category: str
    poffline_denominator_category: str
    pmiss_numerator_category: str
    pmiss_denominator_category: str

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        for key in ("control", "p_veto", "p_offline", "p_miss", "estimate"):
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


def pveto_count_from_pair_counts(pair_counts: Mapping[str, float]) -> Count:
    """Convert OS/SS Pveto pair counts into a symmetric Count approximation."""

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
    dataset: str | None = None,
    sample: str | None = None,
    variation: str = "nominal",
) -> list[LeptonBackgroundEstimate]:
    estimates = []
    pmiss_numerator_category = pmiss_numerator_category or ptrigger_numerator_category
    pmiss_denominator_category = pmiss_denominator_category or ptrigger_denominator_category
    if pmiss_numerator_category is None or pmiss_denominator_category is None:
        raise ValueError(
            "pmiss_numerator_category and pmiss_denominator_category are required"
        )
    for layer in layers:
        control_name = _category_name(control_category, layer)
        poffline_num_name = _category_name(poffline_numerator_category, layer)
        poffline_den_name = _category_name(poffline_denominator_category, layer)
        pmiss_num_name = _category_name(pmiss_numerator_category, layer)
        pmiss_den_name = _category_name(pmiss_denominator_category, layer)

        control = _count_from_source(
            counts=counts,
            cutflow=cutflow,
            category=control_name,
            dataset=dataset,
            sample=sample,
            variation=variation,
        )
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
        p_veto = pveto_count_from_pair_counts(pair_counts.get(layer, {}))
        estimates.append(
            LeptonBackgroundEstimate(
                flavor=flavor,
                layer=layer,
                control=control,
                p_veto=p_veto,
                p_offline=poffline,
                p_miss=pmiss,
                estimate=control * p_veto * poffline * pmiss,
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


def write_lepton_background_latex(
    estimates: Sequence[LeptonBackgroundEstimate],
    path: Path,
    *,
    run_period: str,
    include_table_env: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as out:
        if include_table_env:
            out.write(r"\begin{table}" + "\n")
            out.write(r"\centering" + "\n")
            out.write(r"\caption{Lepton-background estimate.}" + "\n")
            out.write(r"\label{tab:lepton_background_estimate}" + "\n")
        out.write(r"\begin{tabular}{llrrrrr}" + "\n")
        out.write(r"\hline" + "\n")
        out.write(
            r"run period & flavor/layer & $N_{\mathrm{ctrl}}$ & $P_{\mathrm{veto}}$ & "
            r"$P_{\mathrm{offline}}$ & $P_{\mathrm{miss}}$ & $N_{\ell}$ \\" + "\n"
        )
        out.write(r"\hline" + "\n")
        for estimate in estimates:
            out.write(
                f"{run_period} & {estimate.flavor} {estimate.layer} & "
                f"{format_count(estimate.control.value)} & "
                f"{format_pm_latex(estimate.p_veto.value, estimate.p_veto.error)} & "
                f"{format_pm_latex(estimate.p_offline.value, estimate.p_offline.error)} & "
                f"{format_pm_latex(estimate.p_miss.value, estimate.p_miss.error)} & "
                f"{format_value_with_uncertainty(estimate.estimate.value, estimate.estimate.error)} "
                r"\\" + "\n"
            )
        out.write(r"\hline" + "\n")
        out.write(r"\end{tabular}" + "\n")
        if include_table_env:
            out.write(r"\end{table}" + "\n")
