"""Small summary helpers for PocketCoffea output files."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import sqrt
from numbers import Number
from typing import Any


@dataclass(frozen=True)
class VetoProbabilitySummary:
    denominator_name: str
    numerator_name: str
    denominator: float
    numerator: float
    probability: float
    uncertainty: float

    def as_dict(self) -> dict[str, float | str]:
        return asdict(self)


@dataclass(frozen=True)
class SSSubtractedVetoProbabilitySummary:
    os_denominator_name: str
    os_numerator_name: str
    ss_denominator_name: str
    ss_numerator_name: str
    os_denominator: float
    os_numerator: float
    ss_denominator: float
    ss_numerator: float
    denominator: float
    numerator: float
    probability: float
    uncertainty: float

    def as_dict(self) -> dict[str, float | str]:
        return asdict(self)


def _sum_numeric_leaves(value: Any) -> float:
    """Sum all numeric leaves in a nested PocketCoffea cutflow object."""
    if isinstance(value, Number):
        return float(value)
    if isinstance(value, dict):
        return sum(_sum_numeric_leaves(v) for v in value.values())
    return 0.0


def cutflow_count(
    cutflow: dict[str, Any],
    category: str,
    *,
    dataset: str | None = None,
    sample: str | None = None,
    variation: str = "nominal",
) -> float:
    """Extract a category count from a nested PocketCoffea cutflow.

    PocketCoffea cutflows are nested differently depending on the stage.  The
    analysis categories we use here normally look like
    ``category -> dataset -> sample -> variation``.  This helper keeps the CLI
    forgiving: if filters are omitted it sums all matching numeric leaves.
    """
    if category not in cutflow:
        raise KeyError(f"category {category!r} not found in cutflow")

    value: Any = cutflow[category]
    if isinstance(value, Number):
        return float(value)
    if dataset is not None:
        value = value[dataset]
    if isinstance(value, Number):
        return float(value)
    if sample is not None:
        if isinstance(value, dict) and sample in value:
            value = value[sample]
        elif dataset is None and isinstance(value, dict):
            value = {
                key: nested[sample]
                for key, nested in value.items()
                if isinstance(nested, dict) and sample in nested
            }
        else:
            value = value[sample]
    if isinstance(value, dict) and variation in value:
        value = value[variation]
    return _sum_numeric_leaves(value)


def summarize_veto_probability(
    cutflow: dict[str, Any],
    *,
    denominator_name: str,
    numerator_name: str,
    dataset: str | None = None,
    sample: str | None = None,
    variation: str = "nominal",
) -> VetoProbabilitySummary:
    denominator = cutflow_count(
        cutflow,
        denominator_name,
        dataset=dataset,
        sample=sample,
        variation=variation,
    )
    numerator = cutflow_count(
        cutflow,
        numerator_name,
        dataset=dataset,
        sample=sample,
        variation=variation,
    )

    probability = numerator / denominator if denominator > 0.0 else 0.0
    uncertainty = (
        sqrt(probability * (1.0 - probability) / denominator)
        if denominator > 0.0
        else 0.0
    )

    return VetoProbabilitySummary(
        denominator_name=denominator_name,
        numerator_name=numerator_name,
        denominator=denominator,
        numerator=numerator,
        probability=probability,
        uncertainty=uncertainty,
    )


def summarize_ss_subtracted_veto_probability(
    cutflow: dict[str, Any],
    *,
    os_denominator_name: str,
    os_numerator_name: str,
    ss_denominator_name: str,
    ss_numerator_name: str,
    dataset: str | None = None,
    sample: str | None = None,
    variation: str = "nominal",
) -> SSSubtractedVetoProbabilitySummary:
    os_denominator = cutflow_count(
        cutflow,
        os_denominator_name,
        dataset=dataset,
        sample=sample,
        variation=variation,
    )
    os_numerator = cutflow_count(
        cutflow,
        os_numerator_name,
        dataset=dataset,
        sample=sample,
        variation=variation,
    )
    ss_denominator = cutflow_count(
        cutflow,
        ss_denominator_name,
        dataset=dataset,
        sample=sample,
        variation=variation,
    )
    ss_numerator = cutflow_count(
        cutflow,
        ss_numerator_name,
        dataset=dataset,
        sample=sample,
        variation=variation,
    )

    denominator = os_denominator - ss_denominator
    numerator = os_numerator - ss_numerator
    probability = numerator / denominator if denominator > 0.0 else 0.0

    # First-pass statistical uncertainty: propagate Poisson variances for
    # OS/SS numerator and denominator through p = (OS_num-SS_num)/(OS_den-SS_den).
    if denominator > 0.0:
        numerator_variance = os_numerator + ss_numerator
        denominator_variance = os_denominator + ss_denominator
        uncertainty = sqrt(
            numerator_variance / denominator**2
            + (numerator**2 * denominator_variance) / denominator**4
        )
    else:
        uncertainty = 0.0

    return SSSubtractedVetoProbabilitySummary(
        os_denominator_name=os_denominator_name,
        os_numerator_name=os_numerator_name,
        ss_denominator_name=ss_denominator_name,
        ss_numerator_name=ss_numerator_name,
        os_denominator=os_denominator,
        os_numerator=os_numerator,
        ss_denominator=ss_denominator,
        ss_numerator=ss_numerator,
        denominator=denominator,
        numerator=numerator,
        probability=probability,
        uncertainty=uncertainty,
    )
