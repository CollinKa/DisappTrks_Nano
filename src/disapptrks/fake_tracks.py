"""Fake-track background estimate helpers.

The Run-3 fake-track estimate follows the legacy DisappTrks convention

    N_fake = xi * N_ctrl

where ``xi`` is a d0 transfer factor measured from a 3-layer/basic-track d0
shape and ``N_ctrl`` is the yield in the target disappearing-track sideband.
When the control region is a Z->ll sample, ``N_ctrl`` is first normalized by the
ratio of the basic-search yield to the inclusive Z->ll yield.
"""

from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from math import erf, sqrt
from pathlib import Path
from typing import Any, Mapping, Sequence

from .summaries import cutflow_count
from .tables import (
    POISSON_ZERO_UPPER_68,
    format_count,
    format_pm_latex,
    format_value_with_uncertainty,
)


SIDEBAND_MANIFEST_TRACK_FIELDS = (
    "isoTrackIdx",
    "pt",
    "eta",
    "phi",
    "charge",
    "dxy",
    "dz",
    "isHighPurityTrack",
    "hp_nValidHits",
    "hp_nValidPixelHits",
    "hp_trackerLayersWithMeasurement",
    "missingInnerHits",
    "missingMiddleHits",
    "missingOuterHits",
    "pfRelIso03_chg",
    "caloEnergy",
    "dEdxPixel",
    "dEdxStrip",
)


@dataclass(frozen=True)
class Count:
    value: float
    variance: float | None = None

    def __post_init__(self) -> None:
        if self.variance is None:
            object.__setattr__(self, "variance", max(float(self.value), 0.0))

    @property
    def error(self) -> float:
        return sqrt(max(float(self.variance), 0.0))

    def __mul__(self, other: "Count | float") -> "Count":
        if isinstance(other, Count):
            value = self.value * other.value
            variance = value * value * (_relative_variance(self) + _relative_variance(other))
            return Count(value, variance)
        return Count(self.value * other, float(self.variance) * other * other)

    def __truediv__(self, other: "Count | float") -> "Count":
        if isinstance(other, Count):
            if other.value == 0.0:
                return Count(0.0, 0.0)
            value = self.value / other.value
            variance = value * value * (_relative_variance(self) + _relative_variance(other))
            return Count(value, variance)
        if other == 0.0:
            return Count(0.0, 0.0)
        return Count(self.value / other, float(self.variance) / (other * other))


@dataclass(frozen=True)
class FakeTrackEstimate:
    layer: str
    transfer_signal: Count
    transfer_sideband: Count
    transfer_factor: Count
    control_raw: Count
    normalization: Count
    control: Count
    estimate: Count
    p_fake_raw: Count | None = None
    transfer_signal_category: str = ""
    transfer_sideband_category: str = ""
    control_category: str = ""
    basic_yield_category: str | None = None
    z_to_ll_yield_category: str | None = None

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        for key in (
            "transfer_signal",
            "transfer_sideband",
            "transfer_factor",
            "control_raw",
            "normalization",
            "control",
            "estimate",
            "p_fake_raw",
        ):
            value = out[key]
            if value is not None:
                out[key] = {
                    "value": value["value"],
                    "error": sqrt(max(value["variance"], 0.0)),
                    "variance": value["variance"],
                }
        return out


@dataclass(frozen=True)
class DxyTransferFactorFit:
    """Fit-derived d0 transfer factor used by the AN fake-track method."""

    control_region: str
    histogram: str
    numerator_range: tuple[float, float]
    denominator_range: tuple[float, float]
    fit_range: tuple[float, float]
    amplitude: float
    sigma: float
    constant: float
    transfer_factor: Count


@dataclass(frozen=True)
class ANFakeTrackEstimate:
    """Chapter-5 fake-track estimate components for one layer bin."""

    control_region: str
    layer: str
    control_events: Count
    sideband_events: Count
    basic_events: Count | None
    raw_probability: Count
    transfer_factor: Count
    fake_probability: Count
    fake_yield: Count | None
    control_category: str
    sideband_category: str
    basic_yield_category: str | None = None

    def as_dict(self) -> dict[str, Any]:
        out = asdict(self)
        for key in (
            "control_events",
            "sideband_events",
            "basic_events",
            "raw_probability",
            "transfer_factor",
            "fake_probability",
            "fake_yield",
        ):
            value = out[key]
            if value is not None:
                out[key] = {
                    "value": value["value"],
                    "error": sqrt(max(value["variance"], 0.0)),
                    "variance": value["variance"],
                }
        return out


AN_FIXED_TRANSFER_FACTORS = {
    ("2022CD", "zmumu"): Count(0.10, 0.06**2),
    ("2022CD", "zee"): Count(0.11, 0.16**2),
    ("2022EFG", "zmumu"): Count(0.09, 0.04**2),
    ("2022EFG", "zee"): Count(0.10, 0.04**2),
    ("2023C", "zmumu"): Count(0.08, 0.04**2),
    ("2023C", "zee"): Count(0.07, 0.07**2),
    ("2023D", "zmumu"): Count(0.09, 0.05**2),
    ("2023D", "zee"): Count(0.09, 0.04**2),
}

AN_RUN_PERIOD_ALIASES = {
    "2022CD": "2022CD",
    "2022C": "2022CD",
    "2022D": "2022CD",
    "2022PREEE": "2022CD",
    "2022_PRE_EE": "2022CD",
    "2022_PREFIRE": "2022CD",
    "2022EFG": "2022EFG",
    "2022E": "2022EFG",
    "2022F": "2022EFG",
    "2022G": "2022EFG",
    "2022POSTEE": "2022EFG",
    "2022_POST_EE": "2022EFG",
    "2023C": "2023C",
    "2023PREBPIX": "2023C",
    "2023_PRE_BPIX": "2023C",
    "2023D": "2023D",
    "2023POSTBPIX": "2023D",
    "2023_POST_BPIX": "2023D",
}

AN_CONTROL_REGION_ALIASES = {
    "ZMUMU": "zmumu",
    "ZMM": "zmumu",
    "MUMU": "zmumu",
    "MUON": "zmumu",
    "MUONS": "zmumu",
    "ZEE": "zee",
    "EE": "zee",
    "EGAMMA": "zee",
    "ELECTRON": "zee",
    "ELECTRONS": "zee",
}


def _normalize_fixed_transfer_factor_key(value: str) -> str:
    return "".join(char for char in value.upper() if char.isalnum() or char == "_")


def fixed_an_transfer_factor_fit(run_period: str, control_region: str) -> DxyTransferFactorFit:
    """Return the AN Section-5.2 fixed transfer factor for a period/control region."""

    period_key = _normalize_fixed_transfer_factor_key(run_period)
    control_key = _normalize_fixed_transfer_factor_key(control_region)
    canonical_period = AN_RUN_PERIOD_ALIASES.get(period_key)
    canonical_control = AN_CONTROL_REGION_ALIASES.get(control_key)
    if canonical_period is None or canonical_control is None:
        supported = ", ".join(
            f"{period}/{control}" for period, control in sorted(AN_FIXED_TRANSFER_FACTORS)
        )
        raise KeyError(
            "No fixed AN fake-track transfer factor for "
            f"run_period={run_period!r}, control_region={control_region!r}. "
            f"Supported canonical combinations: {supported}"
        )

    transfer_factor = AN_FIXED_TRANSFER_FACTORS[(canonical_period, canonical_control)]
    return DxyTransferFactorFit(
        control_region=control_region,
        histogram=f"fixed:{canonical_period}:{canonical_control}",
        numerator_range=(0.0, 0.02),
        denominator_range=(0.05, 0.50),
        fit_range=(0.10, 0.50),
        amplitude=0.0,
        sigma=0.0,
        constant=0.0,
        transfer_factor=transfer_factor,
    )


def _relative_variance(count: Count) -> float:
    if count.value == 0.0:
        return 0.0
    return float(count.variance) / (count.value * count.value)


def _category_name(pattern: str | None, layer: str) -> str | None:
    if pattern is None:
        return None
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


def _find_hist_axis(hist_obj: Any):
    axes = list(getattr(hist_obj, "axes", []))
    for axis in axes:
        name = getattr(axis, "name", "")
        if name not in {"cat", "variation", "sample"}:
            return axis
    return axes[0] if axes else None


def _hist_counts_edges(hist_obj: Any, *, category: str = "inclusive") -> tuple[Any, Any] | None:
    """Return counts and bin edges from a one-dimensional PocketCoffea histogram."""

    try:
        axis_names = [axis.name for axis in hist_obj.axes]
    except AttributeError:
        return None

    if "cat" in axis_names:
        try:
            hist_obj = hist_obj[{"cat": category}]
        except Exception:
            return None

    axis = _find_hist_axis(hist_obj)
    if axis is None:
        return None

    try:
        import numpy as np

        values = hist_obj.values(flow=False)
        counts = np.asarray(values, dtype=float)
        edges = np.asarray(axis.edges, dtype=float)
    except Exception:
        return None

    while counts.ndim > 1:
        counts = counts.sum(axis=0)

    if counts.ndim != 1 or len(edges) != len(counts) + 1:
        return None
    return counts, edges


def _hist_counts_edges_with_flow(
    hist_obj: Any, *, category: str = "inclusive"
) -> tuple[Any, Any, float, float] | None:
    """Return regular-bin counts, edges, underflow, and overflow."""

    try:
        axis_names = [axis.name for axis in hist_obj.axes]
    except AttributeError:
        return None

    if "cat" in axis_names:
        try:
            hist_obj = hist_obj[{"cat": category}]
        except Exception:
            return None

    axis = _find_hist_axis(hist_obj)
    if axis is None:
        return None

    try:
        import numpy as np

        values = hist_obj.values(flow=True)
        counts_with_flow = np.asarray(values, dtype=float)
        edges = np.asarray(axis.edges, dtype=float)
    except Exception:
        return None

    while counts_with_flow.ndim > 1:
        counts_with_flow = counts_with_flow.sum(axis=0)

    if counts_with_flow.ndim != 1 or len(counts_with_flow) != len(edges) + 1:
        return None
    return (
        counts_with_flow[1:-1],
        edges,
        float(counts_with_flow[0]),
        float(counts_with_flow[-1]),
    )


def _walk_hists(value: Any):
    if hasattr(value, "axes") and hasattr(value, "values"):
        yield value
    elif isinstance(value, Mapping):
        for nested in value.values():
            yield from _walk_hists(nested)


def summed_hist_counts_edges(
    outputs: Sequence[Mapping[str, Any]],
    variable: str,
    *,
    dataset: str | None = None,
    sample: str | None = None,
    category: str = "inclusive",
) -> tuple[Any, Any]:
    """Sum one histogram variable across PocketCoffea output files."""

    import numpy as np

    total_counts = None
    total_edges = None
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
            for hist_obj in _walk_hists(nested):
                result = _hist_counts_edges(hist_obj, category=category)
                if result is None:
                    continue
                counts, edges = result
                if total_counts is None:
                    total_counts = counts.copy()
                    total_edges = edges.copy()
                else:
                    if len(edges) != len(total_edges) or not np.allclose(edges, total_edges):
                        raise ValueError(f"histogram {variable!r} has inconsistent binning")
                    total_counts += counts

    if total_counts is None or total_edges is None:
        raise KeyError(f"histogram variable {variable!r} not found")
    return total_counts, total_edges


def summed_hist_counts_edges_with_flow(
    outputs: Sequence[Mapping[str, Any]],
    variable: str,
    *,
    dataset: str | None = None,
    sample: str | None = None,
    category: str = "inclusive",
) -> tuple[Any, Any, float, float]:
    """Sum a one-dimensional histogram, retaining flow-bin counts."""

    import numpy as np

    total_counts = None
    total_edges = None
    total_underflow = 0.0
    total_overflow = 0.0
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
            for hist_obj in _walk_hists(nested):
                result = _hist_counts_edges_with_flow(hist_obj, category=category)
                if result is None:
                    continue
                counts, edges, underflow, overflow = result
                if total_counts is None:
                    total_counts = counts.copy()
                    total_edges = edges.copy()
                else:
                    if len(edges) != len(total_edges) or not np.allclose(edges, total_edges):
                        raise ValueError(f"histogram {variable!r} has inconsistent binning")
                    total_counts += counts
                total_underflow += underflow
                total_overflow += overflow

    if total_counts is None or total_edges is None:
        raise KeyError(f"histogram variable {variable!r} not found")
    return total_counts, total_edges, total_underflow, total_overflow


def _hist_has_non_sentinel_entries(
    counts: Any,
    edges: Any,
    underflow: float,
    overflow: float,
    *,
    sentinel: float,
) -> bool:
    """Whether a histogram contains anything beyond one missing-value sentinel."""

    import numpy as np

    total = float(np.asarray(counts, dtype=float).sum()) + underflow + overflow
    if not total:
        return False
    if sentinel < edges[0]:
        sentinel_count = underflow
    elif sentinel >= edges[-1]:
        sentinel_count = overflow
    else:
        index = int(np.searchsorted(edges, sentinel, side="right") - 1)
        sentinel_count = float(counts[index])
    return total > sentinel_count


def _outputs_have_histogram(outputs: Sequence[Mapping[str, Any]], variable: str) -> bool:
    return any(variable in output.get("variables", {}) for output in outputs)


def summed_hist_counts_edges_2d(
    outputs: Sequence[Mapping[str, Any]],
    variable: str,
    *,
    dataset: str | None = None,
    sample: str | None = None,
    category: str = "inclusive",
) -> tuple[Any, Any, Any]:
    """Sum one two-dimensional histogram across PocketCoffea outputs."""

    import numpy as np

    total_counts = None
    total_edges = None
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
            for hist_obj in _walk_hists(nested):
                try:
                    axis_names = [axis.name for axis in hist_obj.axes]
                    if "cat" in axis_names:
                        hist_obj = hist_obj[{"cat": category}]
                    for axis_name in ("variation", "sample"):
                        if axis_name in [axis.name for axis in hist_obj.axes]:
                            hist_obj = hist_obj[{axis_name: "nominal"}]
                    axes = list(hist_obj.axes)
                    counts = np.asarray(hist_obj.values(flow=False), dtype=float)
                    edges = [np.asarray(axis.edges, dtype=float) for axis in axes]
                except Exception:
                    continue
                if counts.ndim != 2 or len(edges) != 2:
                    continue
                if total_counts is None:
                    total_counts = counts.copy()
                    total_edges = [edge.copy() for edge in edges]
                else:
                    if counts.shape != total_counts.shape or any(
                        len(edge) != len(total_edge)
                        or not np.allclose(edge, total_edge)
                        for edge, total_edge in zip(edges, total_edges)
                    ):
                        raise ValueError(f"histogram {variable!r} has inconsistent binning")
                    total_counts += counts

    if total_counts is None or total_edges is None:
        raise KeyError(f"histogram variable {variable!r} not found")
    return total_counts, total_edges[0], total_edges[1]


def write_fake_sideband_event_manifest(
    outputs: Sequence[Mapping[str, Any]],
    path: Path,
    *,
    control: str,
    sample: str,
) -> int:
    """Write run/lumi/event and candidate details from PocketCoffea columns."""

    control_key = {"zmumu": "ZMuMu", "zee": "Zee"}[control]
    layers = ("NLayers4", "NLayers5", "NLayers6plus")
    fieldnames = [
        "control",
        "layer_category",
        "sample",
        "dataset",
        "run",
        "luminosityBlock",
        "event",
        "candidate_index_in_event",
        *SIDEBAND_MANIFEST_TRACK_FIELDS,
    ]
    rows = []

    def values(columns, name):
        value = columns[name]
        return value.value if hasattr(value, "value") else value

    for output in outputs:
        sample_columns = output.get("columns", {}).get(sample, {})
        for dataset, categories in sample_columns.items():
            for layer in layers:
                category = f"fake_{control}_sideband_{layer}"
                nominal = categories.get(category, {}).get("nominal", {})
                if not nominal:
                    continue
                collection = f"Fake{control_key}SidebandTrack_{layer}"
                sizes = values(nominal, f"{collection}_N")
                runs = values(nominal, "events_run")
                lumis = values(nominal, "events_luminosityBlock")
                event_ids = values(nominal, "events_event")
                track_values = {
                    field: values(nominal, f"{collection}_{field}")
                    for field in SIDEBAND_MANIFEST_TRACK_FIELDS
                }
                offset = 0
                for event_index, size in enumerate(sizes):
                    for candidate_index in range(int(size)):
                        flat_index = offset + candidate_index
                        row = {
                            "control": control,
                            "layer_category": layer,
                            "sample": sample,
                            "dataset": dataset,
                            "run": int(runs[event_index]),
                            "luminosityBlock": int(lumis[event_index]),
                            "event": int(event_ids[event_index]),
                            "candidate_index_in_event": candidate_index,
                        }
                        for field, array in track_values.items():
                            item = array[flat_index]
                            row[field] = item.item() if hasattr(item, "item") else item
                        rows.append(row)
                    offset += int(size)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def _gauss_plus_constant_integral(amplitude: float, sigma: float, constant: float, lo: float, hi: float) -> float:
    if sigma <= 0.0:
        return constant * (hi - lo)
    gaussian = amplitude * sigma * sqrt(3.141592653589793 / 2.0)
    gaussian *= erf(hi / (sqrt(2.0) * sigma)) - erf(lo / (sqrt(2.0) * sigma))
    return gaussian + constant * (hi - lo)


def _fit_gauss_plus_constant(x, y):
    """Fit binned counts with the legacy Poisson-likelihood prescription.

    The OSUT3 ``TH1::Fit`` call used the ROOT ``L`` option.  Fit positive
    parameters in log space so the equivalent Poisson likelihood remains
    well behaved at the physical boundaries, then obtain the asymptotic
    covariance from the expected Fisher information.
    """
    import numpy as np
    from scipy.optimize import minimize

    def model(xvals, amplitude, sigma, constant):
        return amplitude * np.exp(-0.5 * (xvals / sigma) ** 2) + constant

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    p0 = np.asarray(
        [max(float(y.max() - y.min()), 1.0), 0.20, max(float(y.min()), 1.0e-3)],
        dtype=float,
    )

    def nll(log_params):
        params = np.exp(log_params)
        expected = np.maximum(model(x, *params), 1.0e-12)
        # The omitted log(y!) term is independent of the fit parameters.
        return float(np.sum(expected - y * np.log(expected)))

    result = minimize(
        nll,
        np.log(p0),
        method="L-BFGS-B",
        bounds=[(-30.0, 30.0), (np.log(1.0e-4), 0.0), (-30.0, 30.0)],
        options={"maxiter": 20000, "ftol": 1.0e-12, "gtol": 1.0e-8},
    )
    if not result.success:
        raise RuntimeError(f"Poisson-likelihood d0 fit failed: {result.message}")

    popt = np.exp(result.x)
    amplitude, sigma, _ = popt
    expected = np.maximum(model(x, *popt), 1.0e-12)
    gaussian = np.exp(-0.5 * (x / sigma) ** 2)
    # Derivatives of the expected bin counts with respect to A, sigma, C.
    jacobian = np.column_stack(
        (
            gaussian,
            amplitude * gaussian * x * x / sigma**3,
            np.ones_like(x),
        )
    )
    information = jacobian.T @ ((1.0 / expected)[:, None] * jacobian)
    pcov = np.linalg.pinv(information)
    return popt, pcov


def _transfer_factor_from_fit(
    popt,
    pcov,
    *,
    numerator_ranges: Sequence[tuple[float, float]],
    denominator_ranges: Sequence[tuple[float, float]],
) -> Count:
    import numpy as np

    def ratio(params) -> float:
        numerator = sum(_gauss_plus_constant_integral(*params, *r) for r in numerator_ranges)
        denominator = sum(_gauss_plus_constant_integral(*params, *r) for r in denominator_ranges)
        return 0.0 if denominator == 0.0 else numerator / denominator

    value = ratio(popt)
    variance = 0.0
    try:
        grad = []
        for i, param in enumerate(popt):
            step = max(abs(float(param)) * 1.0e-5, 1.0e-7)
            shifted_hi = list(popt)
            shifted_lo = list(popt)
            shifted_hi[i] += step
            shifted_lo[i] -= step
            grad.append((ratio(shifted_hi) - ratio(shifted_lo)) / (2.0 * step))
        grad = np.asarray(grad)
        variance = float(grad @ pcov @ grad)
    except Exception:
        variance = 0.0
    return Count(value, max(variance, 0.0))


def fit_dxy_transfer_factor(
    counts,
    edges,
    *,
    control_region: str,
    histogram: str,
    numerator_range: tuple[float, float] = (0.0, 0.02),
    denominator_range: tuple[float, float] = (0.05, 0.50),
    fit_range: tuple[float, float] = (0.10, 0.50),
) -> DxyTransferFactorFit:
    """Fit folded |dxy| sideband with Gaussian(0)+constant and return zeta.

    This follows the legacy estimator: a binned Poisson-likelihood fit is
    performed in 0.10 <= |dxy| < 0.50 cm and extrapolated into the signal
    window.
    """

    import numpy as np

    centers = 0.5 * (edges[:-1] + edges[1:])
    fit_mask = (centers >= fit_range[0]) & (centers < fit_range[1])
    x = centers[fit_mask]
    y = counts[fit_mask]
    if len(x) < 3 or float(y.sum()) <= 0.0:
        raise ValueError(f"not enough entries to fit {histogram!r} for {control_region}")

    popt, pcov = _fit_gauss_plus_constant(x, y)
    amplitude, sigma, constant = (float(v) for v in popt)
    transfer_factor = _transfer_factor_from_fit(
        popt,
        pcov,
        numerator_ranges=[numerator_range],
        denominator_ranges=[denominator_range],
    )

    return DxyTransferFactorFit(
        control_region=control_region,
        histogram=histogram,
        numerator_range=numerator_range,
        denominator_range=denominator_range,
        fit_range=fit_range,
        amplitude=amplitude,
        sigma=sigma,
        constant=constant,
        transfer_factor=transfer_factor,
    )


def fit_signed_dxy_transfer_factor(
    counts,
    edges,
    *,
    control_region: str,
    histogram: str,
    numerator_abs_range: tuple[float, float] = (0.0, 0.02),
    denominator_abs_range: tuple[float, float] = (0.05, 0.50),
    fit_abs_range: tuple[float, float] = (0.10, 0.50),
) -> DxyTransferFactorFit:
    """Fit signed dxy sidebands, excluding central |dxy| < fit_abs_range[0]."""

    import numpy as np

    centers = 0.5 * (edges[:-1] + edges[1:])
    abs_centers = np.abs(centers)
    fit_mask = (abs_centers >= fit_abs_range[0]) & (abs_centers < fit_abs_range[1])
    x = centers[fit_mask]
    y = counts[fit_mask]
    if len(x) < 3 or float(y.sum()) <= 0.0:
        raise ValueError(f"not enough entries to fit {histogram!r} for {control_region}")

    popt, pcov = _fit_gauss_plus_constant(x, y)
    amplitude, sigma, constant = (float(v) for v in popt)
    numerator_ranges = [
        (-numerator_abs_range[1], -numerator_abs_range[0]),
        numerator_abs_range,
    ]
    denominator_ranges = [
        (-denominator_abs_range[1], -denominator_abs_range[0]),
        denominator_abs_range,
    ]
    transfer_factor = _transfer_factor_from_fit(
        popt,
        pcov,
        numerator_ranges=numerator_ranges,
        denominator_ranges=denominator_ranges,
    )

    return DxyTransferFactorFit(
        control_region=control_region,
        histogram=histogram,
        numerator_range=numerator_abs_range,
        denominator_range=denominator_abs_range,
        fit_range=fit_abs_range,
        amplitude=amplitude,
        sigma=sigma,
        constant=constant,
        transfer_factor=transfer_factor,
    )


def _dxy_fit_model(xvals, amplitude: float, sigma: float, constant: float):
    import numpy as np

    return amplitude * np.exp(-0.5 * (xvals / sigma) ** 2) + constant


def plot_dxy_transfer_factor(
    counts,
    edges,
    fit: DxyTransferFactorFit,
    path: Path,
    *,
    signed_counts=None,
    signed_edges=None,
    title: str | None = None,
) -> None:
    """Write a Figure-26-style signed-dxy histogram and transfer-factor fit plot.

    New outputs fit the signed ``dxy`` sidebands on both sides of zero while
    excluding the central fit gap.  Old outputs without signed histograms fall
    back to a mirrored visualization of the folded ``|dxy|`` histogram.
    """

    import numpy as np

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required to write --fit-plot outputs") from exc

    path.parent.mkdir(parents=True, exist_ok=True)
    if signed_counts is None or signed_edges is None:
        # Backward-compatible fallback for old outputs.  The sign information is
        # not recoverable from |dxy|, so split the folded counts evenly into the
        # two signed halves.  New fake-track outputs should provide signed dxy.
        positive_edges = np.asarray(edges, dtype=float)
        negative_edges = -positive_edges[::-1]
        signed_edges = np.concatenate([negative_edges[:-1], positive_edges])
        signed_counts = 0.5 * np.concatenate([counts[::-1], counts])

    signed_counts = np.asarray(signed_counts, dtype=float)
    signed_edges = np.asarray(signed_edges, dtype=float)
    centers = 0.5 * (signed_edges[:-1] + signed_edges[1:])
    xfit = np.linspace(float(signed_edges[0]), float(signed_edges[-1]), 1000)
    yfit = _dxy_fit_model(np.abs(xfit), fit.amplitude, fit.sigma, fit.constant)
    if signed_counts is not None and signed_edges is not None and "absDxy" in fit.histogram:
        # The folded histogram contains both signs.  Convert its counts/bin
        # model to the signed histogram's counts/bin before drawing it.
        folded_width = float(np.median(np.diff(np.asarray(edges, dtype=float))))
        signed_width = float(np.median(np.diff(signed_edges)))
        yfit *= signed_width / (2.0 * folded_width)

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.stairs(signed_counts, signed_edges, label="data", color="black", linewidth=1.5)
    ax.errorbar(
        centers,
        signed_counts,
        yerr=np.sqrt(np.maximum(signed_counts, 1.0)),
        fmt="o",
        color="black",
        markersize=3,
        linewidth=1,
    )
    ax.plot(
        xfit,
        yfit,
        color="tab:red",
        linewidth=2,
        label="Gaussian + constant fit",
    )

    for index, (lo, hi) in enumerate(
        ((-fit.numerator_range[1], -fit.numerator_range[0]), fit.numerator_range)
    ):
        ax.axvspan(
            lo,
            hi,
            color="tab:green",
            alpha=0.18,
            label="signal window" if index == 0 else None,
        )
    for index, (lo, hi) in enumerate(
        ((-fit.denominator_range[1], -fit.denominator_range[0]), fit.denominator_range)
    ):
        ax.axvspan(
            lo,
            hi,
            color="tab:blue",
            alpha=0.10,
            label="sideband" if index == 0 else None,
        )

    for boundary in (-fit.fit_range[1], -fit.fit_range[0], fit.fit_range[0], fit.fit_range[1]):
        ax.axvline(boundary, color="tab:red", linestyle="--", linewidth=1)

    ax.set_xlabel(r"track $d_{0}$ [cm]")
    ax.set_ylabel("Number of tracks / 0.04 cm")
    ax.set_xlim(float(signed_edges[0]), float(signed_edges[-1]))
    ax.set_ylim(bottom=0.0)
    ax.set_title(title or f"{fit.control_region} transfer-factor fit")
    ax.text(
        0.98,
        0.95,
        rf"$\zeta = {fit.transfer_factor.value:.3g} \pm {fit.transfer_factor.error:.2g}$"
        "\n"
        rf"$\sigma = {fit.sigma:.3g}$ cm",
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8, "edgecolor": "0.8"},
    )
    ax.legend(loc="upper right", bbox_to_anchor=(1.0, 0.78), fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_fake_sideband_track_diagnostics(
    outputs: Sequence[Mapping[str, Any]],
    output_dir: Path,
    *,
    control: str,
    sample: str | None = None,
    title_prefix: str = "",
) -> list[Path]:
    """Plot hit-pattern and dE/dx diagnostics for N_sideband candidates."""

    import numpy as np

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for sideband diagnostic plots") from exc

    control_key = {"zmumu": "ZMuMu", "zee": "Zee"}[control]
    control_label = {"zmumu": r"$Z\to\mu\mu$", "zee": r"$Z\to ee$"}[control]
    layers = (
        ("NLayers4", "4 layers"),
        ("NLayers5", "5 layers"),
        ("NLayers6plus", r"$\geq6$ layers"),
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    hit_fields = (
        ("hp_pixelBarrelLayersWithMeasurement", "Pixel barrel"),
        ("hp_pixelEndcapLayersWithMeasurement", "Pixel endcap"),
        ("hp_stripTIBLayersWithMeasurement", "TIB"),
        ("hp_stripTIDLayersWithMeasurement", "TID"),
        ("hp_stripTOBLayersWithMeasurement", "TOB"),
        ("hp_stripTECLayersWithMeasurement", "TEC"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5), sharey=False)
    for ax, (field, detector) in zip(axes.flat, hit_fields):
        for layer, layer_label in layers:
            counts, edges = summed_hist_counts_edges(
                outputs,
                f"fake{control_key}Sideband_{layer}_{field}",
                sample=sample,
            )
            ax.stairs(counts, edges, label=layer_label, linewidth=1.5)
        ax.set_title(detector)
        ax.set_xlabel("Layers with measurement")
        ax.set_ylabel("Sideband candidates")
        ax.set_yscale("log")
        ax.set_ylim(bottom=0.7)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(f"{title_prefix} {control_label} fake-track sideband hit pattern".strip())
    fig.tight_layout()
    hit_path = output_dir / f"{control}_sideband_hit_pattern.pdf"
    fig.savefig(hit_path)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, (field, detector) in zip(
        axes,
        (("dEdxPixel", "Pixel"), ("dEdxStrip", "Strip")),
    ):
        for layer, layer_label in layers:
            counts, edges = summed_hist_counts_edges(
                outputs,
                f"fake{control_key}Sideband_{layer}_{field}",
                sample=sample,
            )
            integral = float(np.sum(counts))
            density = counts / integral if integral > 0.0 else counts
            ax.stairs(density, edges, label=layer_label, linewidth=1.5)
        ax.set_title(detector)
        ax.set_xlabel(r"d$E$/d$x$ [MeV/mm]")
        ax.set_ylabel("Fraction of sideband candidates")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{title_prefix} {control_label} fake-track sideband dE/dx".strip())
    fig.tight_layout()
    dedx_path = output_dir / f"{control}_sideband_dedx.pdf"
    fig.savefig(dedx_path)
    plt.close(fig)

    correlation_paths = []
    correlation_fields = (
        ("hp_pixelBarrelLayersWithMeasurement", "Pixel barrel", "dEdxPixel"),
        ("hp_pixelEndcapLayersWithMeasurement", "Pixel endcap", "dEdxPixel"),
        ("hp_stripTIBLayersWithMeasurement", "TIB", "dEdxStrip"),
        ("hp_stripTIDLayersWithMeasurement", "TID", "dEdxStrip"),
        ("hp_stripTOBLayersWithMeasurement", "TOB", "dEdxStrip"),
        ("hp_stripTECLayersWithMeasurement", "TEC", "dEdxStrip"),
    )
    for layer, layer_label in layers:
        fig, axes = plt.subplots(2, 3, figsize=(11, 6.8))
        for ax, (hit_field, detector, dedx_field) in zip(
            axes.flat, correlation_fields
        ):
            counts, hit_edges, dedx_edges = summed_hist_counts_edges_2d(
                outputs,
                (
                    f"fake{control_key}Sideband_{layer}_{dedx_field}"
                    f"_vs_{hit_field}"
                ),
                sample=sample,
            )
            mesh = ax.pcolormesh(
                hit_edges,
                dedx_edges,
                counts.T,
                shading="auto",
                cmap="viridis",
            )
            ax.set_title(detector)
            ax.set_xlabel("Layers with measurement")
            ax.set_ylabel(r"d$E$/d$x$")
            fig.colorbar(mesh, ax=ax, label="Sideband candidates")
        fig.suptitle(
            (
                f"{title_prefix} {control_label} fake-track sideband "
                f"dE/dx vs hit pattern ({layer_label})"
            ).strip()
        )
        fig.tight_layout()
        correlation_path = (
            output_dir / f"{control}_sideband_dedx_vs_hit_pattern_{layer}.pdf"
        )
        fig.savefig(correlation_path)
        plt.close(fig)
        correlation_paths.append(correlation_path)

    paths = [hit_path, dedx_path, *correlation_paths]
    return paths


def plot_high_purity_input_distributions(
    outputs: Sequence[Mapping[str, Any]], output_dir: Path, *, control: str,
    layers: Sequence[str] = ("NLayers4",), sample: str | None = None,
    title_prefix: str = "",
) -> list[Path]:
    """Write multipage before/after-highPurity sideband comparison PDFs."""
    import numpy as np
    try:
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for high-purity study plots") from exc

    control_key = {"zmumu": "ZMuMu", "zee": "Zee"}[control]
    control_label = {"zmumu": r"$Z\to\mu\mu$", "zee": r"$Z\to ee$"}[control]
    features = {
        "pt": r"track $p_T$ [GeV]", "eta": r"track $\eta$", "phi": r"track $\phi$",
        "trackPtErr": r"$\delta p_T$ [GeV]", "trackEtaErr": r"$\delta\eta$", "trackPhiErr": r"$\delta\phi$ [rad]",
        "innerPx": r"inner-state $p_x$ [GeV]", "innerPy": r"inner-state $p_y$ [GeV]",
        "innerPz": r"inner-state $p_z$ [GeV]", "innerPt": r"inner-state $p_T$ [GeV]",
        "outerPx": r"outer-state $p_x$ [GeV]", "outerPy": r"outer-state $p_y$ [GeV]",
        "outerPz": r"outer-state $p_z$ [GeV]", "outerPt": r"outer-state $p_T$ [GeV]",
        "dxyBS": r"$d_0$ (beamspot) [cm]", "dzBS": r"$d_z$ (beamspot) [cm]",
        "dxyClosestPV": r"$d_0$ (closest PV) [cm]", "dzClosestPV": r"$d_z$ (closest PV) [cm]",
        "dxyBSErr": r"$\delta d_0$ (beamspot) [cm]", "dzBSErr": r"$\delta d_z$ (beamspot) [cm]",
        "dxyClosestPVErr": r"$\delta d_0$ (closest PV) [cm]", "dzClosestPVErr": r"$\delta d_z$ (closest PV) [cm]",
        "trackChi2": r"track $\chi^2$", "trackNdof": "track ndof",
        "trackNormalizedChi2": r"track $\chi^2$/ndof",
        "hp_nValidPixelHits": "valid pixel hits", "hp_nValidStripHits": "valid strip hits",
        "hp_nLostHitsInner": "missing hits before innermost hit", "hp_nLostHitsOuter": "missing hits after outermost hit",
        "hp_trackerLayersTotallyOffOrBadInner": "inactive layers before innermost hit",
        "hp_trackerLayersTotallyOffOrBadOuter": "inactive layers after outermost hit",
        "missingMiddleHits": "layers without hits on track body",
        "trackAlgo": "track algorithm / iteration flag",
        "trackOriginalAlgo": "original track algorithm / iteration flag",
    }
    layer_labels = {"NLayers4": "4 layers", "NLayers5": "5 layers", "NLayers6plus": r"$\geq6$ layers", "combinedBins": r"$\geq4$ layers"}
    state_fields = {
        "innerPx", "innerPy", "innerPz", "innerPt",
        "outerPx", "outerPy", "outerPz", "outerPt",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for layer in layers:
        path = output_dir / f"{control}_high_purity_inputs_{layer}.pdf"
        with PdfPages(path) as pdf:
            for field, xlabel in features.items():
                variable = f"highPurityStudy{control_key}_{field}"
                try:
                    all_counts, edges, all_underflow, all_overflow = summed_hist_counts_edges_with_flow(
                        outputs,
                        variable,
                        sample=sample,
                        category=f"high_purity_before_{layer}",
                    )
                    pass_counts, pass_edges, pass_underflow, pass_overflow = summed_hist_counts_edges_with_flow(
                        outputs,
                        variable,
                        sample=sample,
                        category=f"high_purity_pass_{layer}",
                    )
                except KeyError:
                    # Read outputs made before the native Cartesian category
                    # layout was introduced.
                    prefix = f"highPurityStudy{control_key}_{layer}"
                    all_counts, edges, all_underflow, all_overflow = summed_hist_counts_edges_with_flow(
                        outputs, f"{prefix}_All_{field}", sample=sample
                    )
                    pass_counts, pass_edges, pass_underflow, pass_overflow = summed_hist_counts_edges_with_flow(
                        outputs, f"{prefix}_Pass_{field}", sample=sample
                    )
                if not np.allclose(edges, pass_edges):
                    raise ValueError(f"before/after binning differs for {field}")
                n_all_in_range = float(all_counts.sum())
                n_pass_in_range = float(pass_counts.sum())
                n_all = n_all_in_range + all_underflow + all_overflow
                n_pass = n_pass_in_range + pass_underflow + pass_overflow

                # Older custom NanoAOD schemas represent unavailable fitted
                # inner/outer states with -999.  Do not create a PDF page when
                # that sentinel is the entire population, even if it happens
                # to lie inside the configured histogram range.
                if field in state_fields and not _hist_has_non_sentinel_entries(
                    all_counts,
                    edges,
                    all_underflow,
                    all_overflow,
                    sentinel=-999.0,
                ):
                    continue
                if not n_all:
                    continue

                def legend_label(name, total, underflow, overflow):
                    label = f"{name} (N={total:g}"
                    if underflow or overflow:
                        label += f", UF={underflow:g}, OF={overflow:g}"
                    return label + ")"

                fig, ax = plt.subplots(figsize=(7.2, 5.0))
                if n_all:
                    ax.stairs(
                        all_counts / n_all,
                        edges,
                        linewidth=1.7,
                        label=legend_label(
                            "Before highPurity", n_all, all_underflow, all_overflow
                        ),
                    )
                if n_pass:
                    ax.stairs(
                        pass_counts / n_pass,
                        edges,
                        linewidth=1.7,
                        label=legend_label(
                            "With highPurity", n_pass, pass_underflow, pass_overflow
                        ),
                    )
                if not n_all_in_range:
                    ax.text(
                        0.5,
                        0.5,
                        "No entries inside the plotted range; see flow counts",
                        transform=ax.transAxes,
                        ha="center",
                        va="center",
                    )
                if field == "pt":
                    ax.set_xscale("log")
                elif field == "trackPtErr":
                    ax.set_xscale("symlog", linthresh=10.0)
                ax.set(xlabel=xlabel, ylabel="Fraction of sideband candidates")
                ax.set_title(f"{title_prefix} {control_label} fake-track sideband, {layer_labels.get(layer, layer)}".strip())
                if n_all or n_pass:
                    ax.legend(fontsize=9)
                fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
        paths.append(path)
    paths.extend(
        _plot_high_purity_dedx_hit_distributions(
            outputs,
            output_dir,
            control=control,
            layers=layers,
            sample=sample,
            title_prefix=title_prefix,
        )
    )
    return paths


def _plot_high_purity_dedx_hit_distributions(
    outputs: Sequence[Mapping[str, Any]],
    output_dir: Path,
    *,
    control: str,
    layers: Sequence[str],
    sample: str | None,
    title_prefix: str,
) -> list[Path]:
    """Write optional per-hit dE/dx diagnostics for the high-purity study."""

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.colors import LogNorm

    control_key = {"zmumu": "ZMuMu", "zee": "Zee"}[control]
    control_label = {"zmumu": r"$Z\to\mu\mu$", "zee": r"$Z\to ee$"}[control]
    base = f"highPurityStudy{control_key}DeDxHit"
    track_base = f"highPurityStudy{control_key}DeDxTrack"
    if not _outputs_have_histogram(
        outputs, f"{base}_nIsoTrackDeDxHit"
    ):
        return []

    one_dimensional = {
        "nHits": "associated retained dE/dx hits per event",
        "isoTrackIdx": "source IsoTrack row index",
        "hitIdx": "index in DeDxHitInfo payload",
        "detId": "raw tracker detector ID",
        "subdet": "tracker subdetector code",
        "layer": "barrel layer or endcap disk/wheel",
        "side": "tracker side code",
        "isPixel": "is pixel hit",
        "type": "DeDxHitInfo hit type",
        "passesStripShapeSelection": "passes strip-shape selection (pixel=true)",
        "charge": "cluster charge",
        "pathLength": "path length through active material",
        "dEdx": r"per-hit dE/dx [MeV/mm]",
        "localX": "hit local x",
        "localY": "hit local y",
        "pixelSize": "pixel cluster size",
        "pixelSizeX": "pixel cluster size in local x",
        "pixelSizeY": "pixel cluster size in local y",
    }
    per_track = {
        "nRetainedDeDxHits": "retained dE/dx hits on track",
        "nRetainedDeDxHitsMinusLayers": (
            "retained dE/dx hits minus measured layers"
        ),
        "dEdxMedian": r"median per-hit dE/dx [MeV/mm]",
        "dEdxTruncatedMeanDropMaximum": (
            r"mean per-hit dE/dx after dropping maximum [MeV/mm]"
        ),
        "dEdxMaximum": r"maximum per-hit dE/dx [MeV/mm]",
        "dEdxStdDev": r"per-track dE/dx standard deviation [MeV/mm]",
        "dEdxRange": r"per-track dE/dx range [MeV/mm]",
        "dEdxMaximumOverMedian": "maximum / median per-hit dE/dx",
        "nDeDxHitsAbove10": r"dE/dx hits $\geq10$ MeV/mm",
        "nDeDxHitsAbove20": r"dE/dx hits $\geq20$ MeV/mm",
        "nStripDeDxHits": "retained strip dE/dx hits",
        "nStripShapeFailures": "strip hits failing shape selection",
        "stripShapeFailureFraction": (
            "fraction of strip hits failing shape selection"
        ),
    }
    two_dimensional = {
        "subdet_vs_layer": (
            "tracker subdetector code",
            "layer/disk/wheel number",
        ),
        "type_vs_detectorLayer": (
            "detector layer",
            "DeDxHitInfo hit type",
        ),
        "stripPassesShapeSelection_vs_detectorLayer": (
            "detector layer",
            "strip hit passes shape selection",
        ),
        "charge_vs_detectorLayer": ("detector layer", "cluster charge"),
        "pathLength_vs_detectorLayer": (
            "detector layer",
            "path length through active material",
        ),
        "dEdx_vs_detectorLayer": (
            "detector layer",
            r"per-hit dE/dx [MeV/mm]",
        ),
        "localX_vs_detectorLayer": ("detector layer", "hit local x"),
        "localY_vs_detectorLayer": ("detector layer", "hit local y"),
        "pixelSize_vs_detectorLayer": ("detector layer", "pixel cluster size"),
        "pixelSizeX_vs_detectorLayer": (
            "detector layer",
            "pixel cluster size in local x",
        ),
        "pixelSizeY_vs_detectorLayer": (
            "detector layer",
            "pixel cluster size in local y",
        ),
    }
    layer_labels = {
        "NLayers4": "4 layers",
        "NLayers5": "5 layers",
        "NLayers6plus": r"$\geq6$ layers",
        "combinedBins": r"$\geq4$ layers",
    }
    detector_ticks = (
        [(10 + layer, f"PXB{layer}") for layer in range(1, 5)]
        + [(20 + layer, f"PXF{layer}") for layer in range(1, 4)]
        + [(30 + layer, f"TIB{layer}") for layer in range(1, 5)]
        + [(40 + layer, f"TID{layer}") for layer in range(1, 4)]
        + [(50 + layer, f"TOB{layer}") for layer in range(1, 7)]
        + [(60 + layer, f"TEC{layer}") for layer in range(1, 10)]
    )

    def flow_label(name, total, underflow, overflow):
        label = f"{name} (N={total:g}"
        if underflow or overflow:
            label += f", UF={underflow:g}, OF={overflow:g}"
        return label + ")"

    paths = []
    for layer in layers:
        path = output_dir / f"{control}_high_purity_dedx_hits_{layer}.pdf"
        with PdfPages(path) as pdf:
            raw_counts, raw_edges, raw_underflow, raw_overflow = (
                summed_hist_counts_edges_with_flow(
                    outputs,
                    f"{base}_nIsoTrackDeDxHit",
                    sample=sample,
                    category="inclusive",
                )
            )
            raw_total = float(raw_counts.sum()) + raw_underflow + raw_overflow
            if raw_total:
                fig, ax = plt.subplots(figsize=(7.2, 5.0))
                ax.stairs(
                    raw_counts / raw_total,
                    raw_edges,
                    linewidth=1.7,
                    label=flow_label(
                        "Selected events", raw_total, raw_underflow, raw_overflow
                    ),
                )
                ax.set(
                    xlabel="N(IsoTrackDeDxHit rows in sideband event)",
                    ylabel="Fraction of selected events",
                )
                ax.set_title(
                    f"{title_prefix} {control_label} sideband raw dE/dx-hit multiplicity".strip()
                )
                ax.legend(fontsize=9)
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

            for field, xlabel in per_track.items():
                variable = f"{track_base}_pass_{layer}_{field}"
                if not _outputs_have_histogram(outputs, variable):
                    continue
                counts, edges, underflow, overflow = (
                    summed_hist_counts_edges_with_flow(
                        outputs,
                        variable,
                        sample=sample,
                        category="inclusive",
                    )
                )
                total = float(counts.sum()) + underflow + overflow
                if not total:
                    continue
                fig, ax = plt.subplots(figsize=(7.2, 5.0))
                ax.stairs(
                    counts / total,
                    edges,
                    linewidth=1.7,
                    label=flow_label(
                        "With highPurity", total, underflow, overflow
                    ),
                )
                ax.set(
                    xlabel=xlabel,
                    ylabel="Fraction of sideband candidates",
                )
                ax.set_title(
                    f"{title_prefix} {control_label} per-track dE/dx, "
                    f"{layer_labels.get(layer, layer)}".strip()
                )
                ax.legend(fontsize=8)
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

            for field, xlabel in one_dimensional.items():
                pass_var = f"{base}_pass_{layer}_{field}"
                if not _outputs_have_histogram(outputs, pass_var):
                    continue
                passed, edges, pass_uf, pass_of = (
                    summed_hist_counts_edges_with_flow(
                        outputs,
                        pass_var,
                        sample=sample,
                        category="inclusive",
                    )
                )
                pass_total = float(passed.sum()) + pass_uf + pass_of
                if not pass_total:
                    continue
                fig, ax = plt.subplots(figsize=(7.2, 5.0))
                ax.stairs(
                    passed / pass_total,
                    edges,
                    linewidth=1.7,
                    label=flow_label(
                        "With highPurity", pass_total, pass_uf, pass_of
                    ),
                )
                ax.set(xlabel=xlabel, ylabel="Fraction")
                if field == "subdet":
                    ax.set_xticks(range(1, 7))
                    ax.set_xticklabels(["PXB", "PXF", "TIB", "TID", "TOB", "TEC"])
                elif field == "side":
                    ax.set_xticks((0, 1, 2))
                    ax.set_xticklabels(("barrel", "-z", "+z"))
                elif field == "isPixel":
                    ax.set_xticks((0, 1))
                    ax.set_xticklabels(("strip", "pixel"))
                elif field == "passesStripShapeSelection":
                    ax.set_xticks((0, 1))
                    ax.set_xticklabels(("fail", "pass"))
                ax.set_title(
                    f"{title_prefix} {control_label} dE/dx hits, "
                    f"{layer_labels.get(layer, layer)}".strip()
                )
                ax.legend(fontsize=8)
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)

            for field, (xlabel, ylabel) in two_dimensional.items():
                pass_var = f"{base}_pass_{layer}_{field}"
                if not _outputs_have_histogram(outputs, pass_var):
                    continue
                passed, xedges, yedges = summed_hist_counts_edges_2d(
                    outputs, pass_var, sample=sample, category="inclusive"
                )
                pass_total = float(passed.sum())
                if not pass_total:
                    continue
                pass_fraction = passed / pass_total
                positive = pass_fraction[pass_fraction > 0]
                norm = None
                if positive.size:
                    vmin = float(positive.min())
                    vmax = float(positive.max())
                    if vmax > vmin:
                        norm = LogNorm(vmin=vmin, vmax=vmax)
                fig = plt.figure(figsize=(8.0, 5.2), constrained_layout=True)
                grid = fig.add_gridspec(
                    1, 2, width_ratios=(1.0, 0.045), wspace=0.04
                )
                ax = fig.add_subplot(grid[0, 0])
                colorbar_ax = fig.add_subplot(grid[0, 1])
                mesh = ax.pcolormesh(
                    xedges,
                    yedges,
                    pass_fraction.T,
                    shading="auto",
                    norm=norm,
                )
                ax.set(xlabel=xlabel, ylabel=ylabel)
                plot_title = (
                    f"{title_prefix} {control_label} dE/dx hits, "
                    f"{layer_labels.get(layer, layer)}\n"
                    f"With highPurity (in-range hits={pass_total:g})"
                ).strip()
                ax.set_title(
                    plot_title
                )
                if "detectorLayer" in field:
                    ax.set_xticks([item[0] for item in detector_ticks])
                    ax.set_xticklabels(
                        [item[1] for item in detector_ticks],
                        rotation=90,
                        fontsize=6,
                    )
                elif field == "subdet_vs_layer":
                    ax.set_xticks(range(1, 7))
                    ax.set_xticklabels(
                        ["PXB", "PXF", "TIB", "TID", "TOB", "TEC"]
                    )
                fig.colorbar(
                    mesh,
                    cax=colorbar_ax,
                    label="Fraction of retained hits",
                )
                pdf.savefig(fig)
                plt.close(fig)
        paths.append(path)
    return paths


def plot_signal_dedx_track_distributions(
    outputs: Sequence[Mapping[str, Any]],
    output_dir: Path,
    *,
    layers: Sequence[str] = ("NLayers4", "NLayers5", "NLayers6plus"),
    sample: str | None = None,
    title_prefix: str = "",
) -> list[Path]:
    """Plot dE/dx summaries for tracks passing the full signal selection."""

    try:
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_pdf import PdfPages
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for signal dE/dx plots") from exc

    features = {
        "nRetainedDeDxHits": "retained dE/dx hits on track",
        "nRetainedDeDxHitsMinusLayers": (
            "retained dE/dx hits minus measured layers"
        ),
        "dEdxMedian": r"median per-hit dE/dx [MeV/mm]",
        "dEdxTruncatedMeanDropMaximum": (
            r"mean per-hit dE/dx after dropping maximum [MeV/mm]"
        ),
        "dEdxMaximum": r"maximum per-hit dE/dx [MeV/mm]",
        "dEdxStdDev": r"per-track dE/dx standard deviation [MeV/mm]",
        "dEdxRange": r"per-track dE/dx range [MeV/mm]",
        "dEdxMaximumOverMedian": "maximum / median per-hit dE/dx",
        "nDeDxHitsAbove10": r"dE/dx hits $\geq10$ MeV/mm",
        "nDeDxHitsAbove20": r"dE/dx hits $\geq20$ MeV/mm",
        "nStripDeDxHits": "retained strip dE/dx hits",
        "nStripShapeFailures": "strip hits failing shape selection",
        "stripShapeFailureFraction": (
            "fraction of strip hits failing shape selection"
        ),
    }
    layer_labels = {
        "NLayers4": "4 layers",
        "NLayers5": "5 layers",
        "NLayers6plus": r"$\geq6$ layers",
        "combinedBins": r"$\geq4$ layers",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for layer in layers:
        first_variable = f"signalDeDxTrack_{layer}_nRetainedDeDxHits"
        if not _outputs_have_histogram(outputs, first_variable):
            continue
        path = output_dir / f"signal_dedx_track_summary_{layer}.pdf"
        with PdfPages(path) as pdf:
            for field, xlabel in features.items():
                variable = f"signalDeDxTrack_{layer}_{field}"
                counts, edges, underflow, overflow = (
                    summed_hist_counts_edges_with_flow(
                        outputs,
                        variable,
                        sample=sample,
                        category="inclusive",
                    )
                )
                total = float(counts.sum()) + underflow + overflow
                fig, ax = plt.subplots(figsize=(7.2, 5.0))
                if total:
                    label = f"Full selection + highPurity (N={total:g}"
                    if underflow or overflow:
                        label += f", UF={underflow:g}, OF={overflow:g}"
                    label += ")"
                    ax.stairs(counts / total, edges, linewidth=1.7, label=label)
                    ax.legend(fontsize=9)
                else:
                    ax.text(
                        0.5,
                        0.5,
                        "No selected signal tracks",
                        transform=ax.transAxes,
                        ha="center",
                        va="center",
                    )
                ax.set(
                    xlabel=xlabel,
                    ylabel="Fraction of selected signal tracks",
                )
                ax.set_title(
                    (
                        f"{title_prefix} signal after full disappearing-track "
                        f"selection, {layer_labels.get(layer, layer)}"
                    ).strip()
                )
                fig.tight_layout()
                pdf.savefig(fig)
                plt.close(fig)
        paths.append(path)
    return paths


def estimate_fake_track_background_an(
    cutflow: Mapping[str, Any],
    *,
    layer: str,
    control_region: str,
    transfer_factor: Count,
    control_category: str,
    sideband_category: str,
    basic_yield_category: str | None = None,
    dataset: str | None = None,
    sample: str | None = None,
    variation: str = "nominal",
    basic_cutflow: Mapping[str, Any] | None = None,
    basic_dataset: str | None = None,
    basic_sample: str | None = None,
    basic_variation: str | None = None,
) -> ANFakeTrackEstimate:
    control_events = _count_from_cutflow(
        dict(cutflow),
        control_category,
        dataset=dataset,
        sample=sample,
        variation=variation,
    )
    sideband_events = _count_from_cutflow(
        dict(cutflow),
        sideband_category,
        dataset=dataset,
        sample=sample,
        variation=variation,
    )
    raw_probability = sideband_events / control_events
    fake_probability = raw_probability * transfer_factor

    # Count's multiplicative error propagation (variance = value^2 * sum of
    # relative variances) collapses to exactly zero uncertainty whenever the
    # central value is zero, since a "relative" variance is undefined at
    # zero and the codebase's _relative_variance() guards it to 0.0 there.
    # A zero observed sideband count is not an infinitely precise zero,
    # though -- it's a Poisson measurement whose true rate could still be as
    # high as the standard zero-count 68% CL upper limit. Quote that upper
    # bound instead of a bare 0 +/- 0, propagated through the same
    # transfer-factor/basic-yield multiplications while treating those
    # (much better-measured) factors as exact, mirroring the zero-numerator
    # branch of tables.pveto_with_asymmetric_uncertainty.
    if sideband_events.value == 0.0 and control_events.value > 0.0:
        raw_probability = Count(
            0.0, (POISSON_ZERO_UPPER_68 / control_events.value) ** 2
        )
        fake_probability = Count(
            0.0, (raw_probability.error * transfer_factor.value) ** 2
        )

    basic_events = None
    fake_yield = None
    if basic_yield_category:
        basic_source = cutflow if basic_cutflow is None else basic_cutflow
        basic_events = _count_from_cutflow(
            dict(basic_source),
            basic_yield_category,
            dataset=dataset if basic_dataset is None else basic_dataset,
            sample=sample if basic_sample is None else basic_sample,
            variation=variation if basic_variation is None else basic_variation,
        )
        fake_yield = fake_probability * basic_events
        if sideband_events.value == 0.0 and control_events.value > 0.0:
            fake_yield = Count(
                0.0, (fake_probability.error * basic_events.value) ** 2
            )

    return ANFakeTrackEstimate(
        control_region=control_region,
        layer=layer,
        control_events=control_events,
        sideband_events=sideband_events,
        basic_events=basic_events,
        raw_probability=raw_probability,
        transfer_factor=transfer_factor,
        fake_probability=fake_probability,
        fake_yield=fake_yield,
        control_category=control_category,
        sideband_category=sideband_category,
        basic_yield_category=basic_yield_category,
    )


def estimate_fake_track_background(
    source: Mapping[str, Any],
    *,
    layer: str,
    transfer_signal_category: str = "fake_basic3hits_d0_signal",
    transfer_sideband_category: str = "fake_basic3hits_d0_sideband",
    control_category: str = "fake_control_{layer}",
    basic_yield_category: str | None = None,
    z_to_ll_yield_category: str | None = None,
    dataset: str | None = None,
    sample: str | None = None,
    variation: str = "nominal",
    source_is_cutflow: bool = True,
    prescale: float = 1.0,
) -> FakeTrackEstimate:
    """Compute the fake-track estimate for one layer bin.

    ``source`` can either be a PocketCoffea cutflow object or a flat mapping of
    category names to counts.  Category strings may include ``{layer}``, which is
    replaced by the current layer name.
    """

    def get_count(category_pattern: str) -> tuple[str, Count]:
        category = _category_name(category_pattern, layer)
        assert category is not None
        if source_is_cutflow:
            return category, _count_from_cutflow(
                source, category, dataset=dataset, sample=sample, variation=variation
            )
        return category, _count_from_mapping(source, category)

    transfer_signal_name, transfer_signal = get_count(transfer_signal_category)
    transfer_sideband_name, transfer_sideband = get_count(transfer_sideband_category)
    control_name, control_raw = get_count(control_category)

    transfer_factor = transfer_signal / transfer_sideband
    normalization = Count(1.0, 0.0)
    p_fake_raw = None
    basic_yield_name = _category_name(basic_yield_category, layer)
    z_to_ll_yield_name = _category_name(z_to_ll_yield_category, layer)

    if basic_yield_name is not None and z_to_ll_yield_name is not None:
        if source_is_cutflow:
            basic_yield = _count_from_cutflow(
                source,
                basic_yield_name,
                dataset=dataset,
                sample=sample,
                variation=variation,
            )
            z_to_ll_yield = _count_from_cutflow(
                source,
                z_to_ll_yield_name,
                dataset=dataset,
                sample=sample,
                variation=variation,
            )
        else:
            basic_yield = _count_from_mapping(source, basic_yield_name)
            z_to_ll_yield = _count_from_mapping(source, z_to_ll_yield_name)

        normalization = basic_yield / z_to_ll_yield
        p_fake_raw = control_raw / z_to_ll_yield

    control = control_raw * normalization * prescale
    estimate = control * transfer_factor

    return FakeTrackEstimate(
        layer=layer,
        transfer_signal=transfer_signal,
        transfer_sideband=transfer_sideband,
        transfer_factor=transfer_factor,
        control_raw=control_raw,
        normalization=normalization,
        control=control,
        estimate=estimate,
        p_fake_raw=p_fake_raw,
        transfer_signal_category=transfer_signal_name,
        transfer_sideband_category=transfer_sideband_name,
        control_category=control_name,
        basic_yield_category=basic_yield_name,
        z_to_ll_yield_category=z_to_ll_yield_name,
    )


def write_fake_track_latex(
    estimates: Sequence[FakeTrackEstimate],
    path: Path,
    *,
    run_period: str,
    include_table_env: bool = False,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as out:
        if include_table_env:
            out.write(r"\begin{table}[htbp]" + "\n")
            out.write(r"\centering" + "\n")
            out.write(r"\caption{Fake-track background estimate.}" + "\n")
            out.write(r"\label{tab:fake_track_estimate}" + "\n")

        out.write(r"\begin{tabular}{lrrrrr}" + "\n")
        out.write(r"\hline" + "\n")
        out.write(
            r"run period & $n_{\mathrm{layers}}$ & $N_{\mathrm{ctrl}}$ & "
            r"$\xi$ & $P_{\mathrm{fake}}^{\mathrm{raw}}$ & "
            r"$N_{\mathrm{fake}}$ \\" + "\n"
        )
        out.write(r"\hline" + "\n")
        for estimate in estimates:
            p_fake = (
                "--"
                if estimate.p_fake_raw is None
                else format_pm_latex(estimate.p_fake_raw.value, estimate.p_fake_raw.error)
            )
            out.write(
                f"{run_period} & {estimate.layer} & "
                f"{format_pm_latex(estimate.control.value, estimate.control.error)} & "
                f"{format_pm_latex(estimate.transfer_factor.value, estimate.transfer_factor.error)} & "
                f"{p_fake} & "
                f"{format_pm_latex(estimate.estimate.value, estimate.estimate.error)} \\\\\n"
            )
        out.write(r"\hline" + "\n")
        out.write(r"\end{tabular}" + "\n")
        if include_table_env:
            out.write(r"\end{table}" + "\n")


def write_an_fake_track_latex(
    estimates: Sequence[ANFakeTrackEstimate],
    fit: DxyTransferFactorFit,
    path: Path,
    *,
    run_period: str,
    include_table_env: bool = False,
) -> None:
    """Write the Chapter-5 fake-track estimate table."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as out:
        if include_table_env:
            out.write(r"\begin{table}[htbp]" + "\n")
            out.write(r"\centering" + "\n")
            out.write(r"\caption{Fake-track background estimate.}" + "\n")
            out.write(r"\label{tab:fake_track_estimate}" + "\n")

        has_yield = any(estimate.fake_yield is not None for estimate in estimates)
        if has_yield:
            out.write(r"\begin{tabular}{llrrrrrr}" + "\n")
            header = (
                r"run period & control & $n_{\mathrm{layers}}$ & $N_Z$ & "
                r"$N_{\mathrm{sideband}}$ & $\zeta$ & $P_{\mathrm{fake}}$ & "
                r"$N_{\mathrm{fake}}$ \\"
            )
        else:
            out.write(r"\begin{tabular}{llrrrrr}" + "\n")
            header = (
                r"run period & control & $n_{\mathrm{layers}}$ & $N_Z$ & "
                r"$N_{\mathrm{sideband}}$ & $\zeta$ & $P_{\mathrm{fake}}$ \\"
            )
        out.write(r"\hline" + "\n")
        out.write(header + "\n")
        out.write(r"\hline" + "\n")

        for estimate in estimates:
            zeta = format_pm_latex(
                estimate.transfer_factor.value,
                estimate.transfer_factor.error,
            )
            p_fake = format_pm_latex(
                estimate.fake_probability.value,
                estimate.fake_probability.error,
            )
            row = (
                f"{run_period} & {estimate.control_region} & {estimate.layer} & "
                f"{format_count(estimate.control_events.value)} & "
                f"{format_count(estimate.sideband_events.value)} & "
                f"{zeta} & {p_fake}"
            )
            if has_yield:
                if estimate.fake_yield is None:
                    row += " & --"
                else:
                    row += f" & {format_pm_latex(estimate.fake_yield.value, estimate.fake_yield.error)}"
            out.write(row + r" \\" + "\n")

        out.write(r"\hline" + "\n")
        out.write(r"\end{tabular}" + "\n")
        if fit.histogram.startswith("fixed:"):
            out.write(
                "% Transfer factor: fixed AN Section-5.2 value for this "
                "run period and control region.\n"
            )
        else:
            out.write(
                "% Fit: Gaussian with mean fixed to zero plus constant, "
                f"{fit.fit_range[0]} <= |dxy| < {fit.fit_range[1]} cm. "
                f"sigma={fit.sigma:.4g}, constant={fit.constant:.4g}.\n"
            )
        if include_table_env:
            out.write(r"\end{table}" + "\n")


def write_fake_track_z_control_latex(
    estimates: Sequence[ANFakeTrackEstimate],
    path: Path,
    *,
    run_period: str,
    include_table_env: bool = False,
) -> None:
    """Write the Z-control-region inputs used by the AN fake-track estimate.

    These are the Tables-32/33-style ingredients: the inclusive Z control count
    and the disappearing-track sideband counts in each layer bin.  They are
    intentionally separate from the JetMET/basic-selection cutflow because they
    are not a sequential selection on the JetMET sample.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as out:
        if include_table_env:
            out.write(r"\begin{table}[htbp]" + "\n")
            out.write(r"\centering" + "\n")
            out.write(r"\caption{Fake-track Z-control-region inputs.}" + "\n")
            out.write(r"\label{tab:fake_track_z_control}" + "\n")

        out.write(r"\begin{tabular}{llcrrr}" + "\n")
        out.write(r"\hline" + "\n")
        out.write(
            r"run period & control & $n_{\mathrm{layers}}$ & "
            r"$N_Z$ & $N_{\mathrm{sideband}}$ & "
            r"$N_{\mathrm{sideband}}/N_Z$ \\" + "\n"
        )
        out.write(r"\hline" + "\n")
        first = True
        for estimate in estimates:
            raw_probability = format_pm_latex(
                estimate.raw_probability.value,
                estimate.raw_probability.error,
            )
            out.write(
                f"{run_period if first else ''} & "
                f"{estimate.control_region if first else ''} & "
                f"{_layer_label(estimate.layer)} & "
                f"{format_count(estimate.control_events.value)} & "
                f"{format_count(estimate.sideband_events.value)} & "
                f"{raw_probability} \\\\\n"
            )
            first = False
        out.write(r"\hline" + "\n")
        out.write(r"\end{tabular}" + "\n")
        if include_table_env:
            out.write(r"\end{table}" + "\n")


def _count_from_payload(value: Mapping[str, Any]) -> Count:
    return Count(float(value["value"]), float(value.get("variance", value.get("error", 0.0) ** 2)))


def _layer_label(layer: str) -> str:
    return {
        "NLayers4": "4",
        "NLayers5": "5",
        "NLayers6plus": r"$\geq 6$",
        "combinedBins": "combined",
    }.get(layer, layer)


def _control_column_key(control_region: str) -> str:
    lower = control_region.lower()
    if r"\mu" in lower or "mumu" in lower or "mu" in lower:
        return "zmumu"
    if "ee" in lower or "electron" in lower:
        return "zee"
    return control_region


def _format_scientific_pm(value: float, error: float) -> str:
    if value == 0.0 and error == 0.0:
        return r"$0 \pm 0$"

    import math

    scale_value = abs(value) if value != 0.0 else abs(error)
    exponent = int(math.floor(math.log10(scale_value))) if scale_value > 0.0 else 0
    mantissa = value / (10.0**exponent)
    mantissa_error = error / (10.0**exponent)
    mantissa_text, error_text = format_value_with_uncertainty(
        mantissa,
        mantissa_error,
    )
    return rf"$({mantissa_text} \pm {error_text}) \times 10^{{{exponent}}}$"


def _format_yield_pm(value: float, error: float) -> str:
    value_text, error_text = format_value_with_uncertainty(value, error)
    return rf"${value_text} \pm {error_text}$"


def write_fake_track_table34_latex(
    json_paths: Sequence[Path],
    path: Path,
    *,
    run_period: str,
    include_table_env: bool = False,
) -> None:
    """Write an AN Table-34-style comparison of Z->mumu and Z->ee estimates."""

    write_combined_fake_track_table34_latex(
        {run_period: json_paths},
        path,
        include_table_env=include_table_env,
    )


def write_combined_fake_track_table34_latex(
    period_json_paths: Mapping[str, Sequence[Path]],
    path: Path,
    *,
    include_table_env: bool = False,
) -> None:
    """Write one Table-34-style comparison spanning multiple run periods."""

    by_period: dict[str, dict[str, dict[str, dict[str, Count]]]] = {}
    for run_period, json_paths in period_json_paths.items():
        by_layer = by_period.setdefault(run_period, {})
        for json_path in json_paths:
            payload = json.loads(json_path.read_text())
            for estimate in payload.get("estimates", []):
                layer = estimate["layer"]
                control = _control_column_key(estimate["control_region"])
                by_layer.setdefault(layer, {})[control] = {
                    "p_fake": _count_from_payload(estimate["fake_probability"]),
                    "n_fake": _count_from_payload(estimate["fake_yield"])
                    if estimate.get("fake_yield") is not None
                    else Count(0.0, 0.0),
                }

    layer_order = ["NLayers4", "NLayers5", "NLayers6plus", "combinedBins"]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as out:
        if include_table_env:
            out.write(r"\begin{table}[htbp]" + "\n")
            out.write(r"\centering" + "\n")
            out.write(r"\caption{Comparison of fake-track event probabilities and background estimates.}" + "\n")
            out.write(r"\label{tab:fake_track_table34}" + "\n")

        out.write(r"\begin{tabular}{llrrrr}" + "\n")
        out.write(r"\hline" + "\n")
        out.write(
            r"run period & $n_{\mathrm{layers}}$ & "
            r"$P_{\mathrm{fake}}(Z\to\mu\mu)$ & "
            r"$P_{\mathrm{fake}}(Z\to ee)$ & "
            r"$N^{\mathrm{fake}}_{\mathrm{est}}(Z\to\mu\mu)$ & "
            r"$N^{\mathrm{fake}}_{\mathrm{est}}(Z\to ee)$ \\" + "\n"
        )
        out.write(r"\hline" + "\n")
        for run_period, by_layer in by_period.items():
            layers = [layer for layer in layer_order if layer in by_layer] + [
                layer for layer in by_layer if layer not in layer_order
            ]
            for i, layer in enumerate(layers):
                controls = by_layer[layer]
                zmumu = controls.get("zmumu")
                zee = controls.get("zee")

                def get(control: dict[str, Count] | None, key: str, formatter) -> str:
                    if control is None:
                        return "--"
                    count = control[key]
                    return formatter(count.value, count.error)

                out.write(
                    f"{run_period if i == 0 else ''} & {_layer_label(layer)} & "
                    f"{get(zmumu, 'p_fake', _format_scientific_pm)} & "
                    f"{get(zee, 'p_fake', _format_scientific_pm)} & "
                    f"{get(zmumu, 'n_fake', _format_yield_pm)} & "
                    f"{get(zee, 'n_fake', _format_yield_pm)} \\\\\n"
                )
        out.write(r"\hline" + "\n")
        out.write(r"\end{tabular}" + "\n")
        if include_table_env:
            out.write(r"\end{table}" + "\n")
