import json

import numpy as np
import pytest

from disapptrks.fake_tracks import (
    _hist_counts_edges_with_flow,
    _hist_has_non_sentinel_entries,
    estimate_fake_track_background,
    fit_dxy_transfer_factor,
    fixed_an_transfer_factor_fit,
    write_combined_fake_track_table34_latex,
    write_fake_track_latex,
)


def test_hist_counts_edges_with_flow_keeps_out_of_range_entries():
    hist = pytest.importorskip("hist")
    histogram = hist.Hist(hist.axis.Regular(2, 0.0, 2.0, name="value"))
    histogram.fill(value=[-1.0, 0.25, 1.25, 3.0])

    counts, edges, underflow, overflow = _hist_counts_edges_with_flow(histogram)

    assert counts.tolist() == [1.0, 1.0]
    assert edges.tolist() == [0.0, 1.0, 2.0]
    assert underflow == 1.0
    assert overflow == 1.0


@pytest.mark.parametrize(
    ("counts", "underflow", "expected"),
    [
        ([0.0, 0.0], 4.0, False),
        ([1.0, 0.0], 4.0, True),
    ],
)
def test_hist_sentinel_availability_when_sentinel_is_underflow(
    counts, underflow, expected
):
    assert _hist_has_non_sentinel_entries(
        np.asarray(counts),
        np.asarray([-600.0, 0.0, 600.0]),
        underflow,
        0.0,
        sentinel=-999.0,
    ) is expected


def test_hist_sentinel_availability_when_sentinel_is_in_range():
    assert not _hist_has_non_sentinel_entries(
        np.asarray([0.0, 7.0, 0.0]),
        np.asarray([-1600.0, -1200.0, -800.0, 0.0]),
        0.0,
        0.0,
        sentinel=-999.0,
    )


def test_folded_dxy_poisson_likelihood_fit_recovers_shape():
    edges = np.linspace(0.0, 0.5, 51)
    centers = 0.5 * (edges[:-1] + edges[1:])
    amplitude = 120.0
    sigma = 0.23
    constant = 8.0
    counts = amplitude * np.exp(-0.5 * (centers / sigma) ** 2) + constant

    fit = fit_dxy_transfer_factor(
        counts,
        edges,
        control_region=r"$Z\to\mu\mu$",
        histogram="fakeZMuMuFitTrack_absDxy",
    )

    assert fit.amplitude == pytest.approx(amplitude, rel=2e-3)
    assert fit.sigma == pytest.approx(sigma, rel=2e-3)
    assert fit.constant == pytest.approx(constant, rel=2e-2)
    assert fit.transfer_factor.value > 0.0
    assert fit.transfer_factor.error > 0.0


def test_fake_track_estimate_from_cutflow_counts():
    cutflow = {
        "fake_basic3hits_d0_signal": {"dataset": {"sample": {"nominal": 20.0}}},
        "fake_basic3hits_d0_sideband": {"dataset": {"sample": {"nominal": 100.0}}},
        "fake_control_NLayers4": {"dataset": {"sample": {"nominal": 50.0}}},
    }

    estimate = estimate_fake_track_background(
        cutflow,
        layer="NLayers4",
        dataset="dataset",
        sample="sample",
    )

    assert estimate.transfer_factor.value == 0.2
    assert estimate.control.value == 50.0
    assert estimate.estimate.value == 10.0
    assert estimate.estimate.error > 0.0


def test_fake_track_estimate_applies_z_to_ll_normalization():
    counts = {
        "fake_basic3hits_d0_signal": 20.0,
        "fake_basic3hits_d0_sideband": 100.0,
        "fake_control_NLayers5": 50.0,
        "basic_yield": 2000.0,
        "ztoll_yield": 1000.0,
    }

    estimate = estimate_fake_track_background(
        counts,
        layer="NLayers5",
        basic_yield_category="basic_yield",
        z_to_ll_yield_category="ztoll_yield",
        source_is_cutflow=False,
    )

    assert estimate.normalization.value == 2.0
    assert estimate.control.value == 100.0
    assert estimate.estimate.value == 20.0
    assert estimate.p_fake_raw.value == 0.05


def test_write_fake_track_latex(tmp_path):
    counts = {
        "fake_basic3hits_d0_signal": 20.0,
        "fake_basic3hits_d0_sideband": 100.0,
        "fake_control_NLayers6plus": 50.0,
    }
    estimate = estimate_fake_track_background(
        counts,
        layer="NLayers6plus",
        source_is_cutflow=False,
    )

    path = tmp_path / "fake.tex"
    write_fake_track_latex([estimate], path, run_period="2023C")
    text = path.read_text()

    assert "run period" in text
    assert "2023C" in text
    assert "NLayers6plus" in text


def test_fake_track_estimate_json_serializable():
    counts = {
        "fake_basic3hits_d0_signal": 20.0,
        "fake_basic3hits_d0_sideband": 100.0,
        "fake_control_combinedBins": 50.0,
    }
    estimate = estimate_fake_track_background(
        counts,
        layer="combinedBins",
        source_is_cutflow=False,
    )

    json.dumps(estimate.as_dict())


def test_fixed_an_transfer_factor_accepts_run_period_aliases():
    fit = fixed_an_transfer_factor_fit("2022 CD", "zmumu")

    assert fit.histogram == "fixed:2022CD:zmumu"
    assert fit.transfer_factor.value == 0.10
    assert fit.transfer_factor.error == 0.06


def test_combined_table34_contains_each_run_period(tmp_path):
    period_paths = {}
    for period in ("2022CD", "2022EFG"):
        json_paths = []
        for control_region in (r"$Z\to\mu\mu$", r"$Z\to ee$"):
            path = tmp_path / f"{period}_{len(json_paths)}.json"
            path.write_text(
                json.dumps(
                    {
                        "estimates": [
                            {
                                "layer": "NLayers4",
                                "control_region": control_region,
                                "fake_probability": {"value": 0.001, "variance": 1e-8},
                                "fake_yield": {"value": 2.0, "variance": 0.25},
                            }
                        ]
                    }
                )
            )
            json_paths.append(path)
        period_paths[period] = json_paths

    output = tmp_path / "combined.tex"
    write_combined_fake_track_table34_latex(period_paths, output)

    text = output.read_text()
    assert "2022CD" in text
    assert "2022EFG" in text
