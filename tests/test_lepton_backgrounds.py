from pathlib import Path

import numpy as np

from disapptrks.cli import (
    _apply_sparse_tau_met_probability_fallback,
    _lepton_background_outputs,
    _met_probabilities_from_components,
    _sum_named_count_maps,
    _tau_probability_from_json,
    _tau_trigger_probability_from_outputs,
    _trigger_efficiency_count_components_from_outputs,
    _trigger_efficiency_from_outputs,
)
from disapptrks.fake_tracks import Count
from disapptrks.lepton_backgrounds import (
    _divide_counts_at_physical_boundary,
    _format_an_pm,
    _format_tau_probability,
    _multiply_counts_at_physical_boundary,
    estimate_lepton_background,
    legacy_met_probabilities_from_outputs,
    read_lepton_background_json,
    trigger_efficiency_from_counts,
    write_combined_lepton_background_latex,
    write_lepton_background_json,
    write_lepton_background_latex,
)


def test_tau_product_preserves_uncertainty_at_zero_pveto_boundary():
    result = _multiply_counts_at_physical_boundary(
        Count(100.0, 100.0),
        Count(0.0, 0.17**2),
        Count(0.5, 0.0),
        Count(0.8, 0.0),
    )

    assert result.value == 0.0
    assert np.isclose(result.error, 6.8)


def test_tau_division_preserves_uncertainty_at_zero_boundary():
    result = _divide_counts_at_physical_boundary(
        Count(0.0, 6.8**2),
        Count(0.9, 0.006**2),
    )

    assert result.value == 0.0
    assert np.isclose(result.error, 6.8 / 0.9)


def test_tau_probability_formats_tiny_uncertainty_scientifically():
    text = _format_tau_probability(1.2967107, 6.8849848e-5)

    assert "1.296711" in text
    assert "6.9 \\times 10^{-5}" in text


class FakeAxis:
    def __init__(self, name, edges):
        self.name = name
        self.edges = np.asarray(edges, dtype=float)


class FakeHist:
    def __init__(self, counts, axes):
        self._counts = np.asarray(counts, dtype=float)
        self.axes = axes

    def values(self, flow=False):
        return self._counts


def test_estimate_lepton_background_uses_an_product():
    counts = {
        "electron_background_control_NLayers4": 100.0,
        "electron_background_offline_NLayers4": 25.0,
        "electron_background_trigger_NLayers4": 20.0,
    }
    pair_counts = {
        "NLayers4": {
            "den_os": 50.0,
            "num_os": 5.0,
            "den_ss": 10.0,
            "num_ss": 1.0,
        }
    }

    estimates = estimate_lepton_background(
        flavor=r"$e$",
        layers=["NLayers4"],
        pair_counts=pair_counts,
        counts=counts,
        control_category="electron_background_control_{layer}",
        poffline_numerator_category="electron_background_offline_{layer}",
        poffline_denominator_category="electron_background_control_{layer}",
        pmiss_numerator_category="electron_background_trigger_{layer}",
        pmiss_denominator_category="electron_background_offline_{layer}",
    )

    estimate = estimates[0]
    assert estimate.p_veto.value == 4.0 / 40.0
    assert estimate.p_offline.value == 0.25
    assert estimate.p_miss.value == 0.8
    assert estimate.p_trigger.value == 0.8
    assert estimate.estimate.value == 100.0 * (4.0 / 40.0) * 0.25 * 0.8


def test_estimate_lepton_background_applies_control_prescale_and_trigger_efficiency():
    estimates = estimate_lepton_background(
        flavor=r"$e$",
        layers=["NLayers4"],
        pair_counts={
            "NLayers4": {
                "den_os": 50.0,
                "num_os": 5.0,
                "den_ss": 10.0,
                "num_ss": 1.0,
            }
        },
        counts={
            "control_NLayers4": 100.0,
            "offline_NLayers4": 25.0,
            "trigger_NLayers4": 20.0,
        },
        control_category="control_{layer}",
        poffline_numerator_category="offline_{layer}",
        poffline_denominator_category="control_{layer}",
        pmiss_numerator_category="trigger_{layer}",
        pmiss_denominator_category="offline_{layer}",
        control_prescale=2.0,
        trigger_efficiency=Count(0.5, 0.0),
    )

    estimate = estimates[0]
    assert estimate.control_raw.value == 100.0
    assert estimate.control.value == 200.0
    assert estimate.trigger_efficiency.value == 0.5
    assert estimate.estimate.value == 200.0 * (4.0 / 40.0) * 0.25 * 0.8 / 0.5


def test_estimate_lepton_background_applies_tau_probability_scale():
    estimates = estimate_lepton_background(
        flavor=r"$\tau$",
        layers=["NLayers4"],
        pair_counts={
            "NLayers4": {
                "den_os": 50.0,
                "num_os": 5.0,
                "den_ss": 10.0,
                "num_ss": 1.0,
            }
        },
        counts={
            "control_NLayers4": 100.0,
            "offline_NLayers4": 25.0,
            "trigger_NLayers4": 20.0,
        },
        control_category="control_{layer}",
        poffline_numerator_category="offline_{layer}",
        poffline_denominator_category="control_{layer}",
        pmiss_numerator_category="trigger_{layer}",
        pmiss_denominator_category="offline_{layer}",
        tau_probability=Count(1.5, 0.01),
    )

    estimate = estimates[0]
    assert estimate.control_raw.value == 100.0
    assert estimate.control.value == 150.0
    assert estimate.tau_probability.value == 1.5
    assert estimate.estimate.value == 150.0 * (4.0 / 40.0) * 0.25 * 0.8


def test_estimate_lepton_background_accepts_layer_trigger_efficiencies():
    pair_counts = {
        "NLayers4": {"den_os": 50.0, "num_os": 5.0, "den_ss": 10.0, "num_ss": 1.0},
        "NLayers5": {"den_os": 50.0, "num_os": 5.0, "den_ss": 10.0, "num_ss": 1.0},
    }
    counts = {
        "control_NLayers4": 100.0,
        "offline_NLayers4": 25.0,
        "trigger_NLayers4": 20.0,
        "control_NLayers5": 100.0,
        "offline_NLayers5": 25.0,
        "trigger_NLayers5": 20.0,
    }

    estimates = estimate_lepton_background(
        flavor=r"$e$",
        layers=["NLayers4", "NLayers5"],
        pair_counts=pair_counts,
        counts=counts,
        control_category="control_{layer}",
        poffline_numerator_category="offline_{layer}",
        poffline_denominator_category="control_{layer}",
        pmiss_numerator_category="trigger_{layer}",
        pmiss_denominator_category="offline_{layer}",
        trigger_efficiency={
            "NLayers4": Count(0.5, 0.0),
            "NLayers5": Count(1.0, 0.0),
        },
    )

    assert estimates[0].trigger_efficiency.value == 0.5
    assert estimates[1].trigger_efficiency.value == 1.0
    assert estimates[0].estimate.value == 2.0 * estimates[1].estimate.value


def test_estimate_lepton_background_low_stat_layers_use_combined_poffline_pmiss():
    """NLayers4/NLayers5 Poffline/Pmiss control samples can be too small for a
    meaningful per-layer ratio -- low_stat_layers substitutes combined_layer's
    Poffline/Pmiss and trigger efficiency for those layers, while leaving
    N_ctrl, Pveto, and any layer not listed using their own categories.
    """

    pair_counts = {
        "NLayers4": {"den_os": 50.0, "num_os": 5.0, "den_ss": 10.0, "num_ss": 1.0},
        "NLayers5": {"den_os": 60.0, "num_os": 6.0, "den_ss": 12.0, "num_ss": 1.0},
        "combinedBins": {"den_os": 500.0, "num_os": 50.0, "den_ss": 100.0, "num_ss": 10.0},
    }
    counts = {
        # NLayers4's own Poffline/Pmiss inputs are deliberately implausible
        # (offline > control) so the test fails loudly if they leak through
        # instead of being replaced by combinedBins.
        "control_NLayers4": 100.0,
        "offline_NLayers4": 999.0,
        "trigger_NLayers4": 999.0,
        # NLayers5 is not in low_stat_layers -- must keep using its own.
        "control_NLayers5": 100.0,
        "offline_NLayers5": 25.0,
        "trigger_NLayers5": 20.0,
        "control_combinedBins": 1000.0,
        "offline_combinedBins": 300.0,
        "trigger_combinedBins": 270.0,
    }

    estimates = estimate_lepton_background(
        flavor=r"$\mu$",
        layers=["NLayers4", "NLayers5", "combinedBins"],
        pair_counts=pair_counts,
        counts=counts,
        control_category="control_{layer}",
        poffline_numerator_category="offline_{layer}",
        poffline_denominator_category="control_{layer}",
        pmiss_numerator_category="trigger_{layer}",
        pmiss_denominator_category="offline_{layer}",
        trigger_efficiency={
            "NLayers4": Count(0.5, 0.0),
            "NLayers5": Count(1.0, 0.0),
            "combinedBins": Count(2.0, 0.0),
        },
        low_stat_layers=["NLayers4"],
    )
    nlayers4, nlayers5, combined = estimates

    # N_ctrl and Pveto still come from NLayers4's own categories.
    assert nlayers4.control_raw.value == 100.0
    assert nlayers4.p_veto.value == (5.0 - 1.0) / (50.0 - 10.0)

    # Poffline/Pmiss/trigger efficiency come from combinedBins instead.
    assert nlayers4.p_offline.value == 300.0 / 1000.0
    assert nlayers4.p_miss.value == 270.0 / 300.0
    assert nlayers4.trigger_efficiency.value == 2.0
    assert nlayers4.poffline_numerator_category == "offline_combinedBins"
    assert nlayers4.poffline_denominator_category == "control_combinedBins"
    assert nlayers4.pmiss_numerator_category == "trigger_combinedBins"
    assert nlayers4.pmiss_denominator_category == "offline_combinedBins"

    # NLayers5 was not listed in low_stat_layers -- unaffected.
    assert nlayers5.p_offline.value == 25.0 / 100.0
    assert nlayers5.p_miss.value == 20.0 / 25.0
    assert nlayers5.trigger_efficiency.value == 1.0
    assert nlayers5.poffline_numerator_category == "offline_NLayers5"

    # combinedBins itself is unaffected either way.
    assert combined.p_offline.value == 300.0 / 1000.0
    assert combined.trigger_efficiency.value == 2.0


def test_estimate_lepton_background_low_stat_layers_default_disabled():
    """The default (empty low_stat_layers) must reproduce prior behavior --
    every layer uses its own Poffline/Pmiss/trigger-efficiency categories."""

    estimates = estimate_lepton_background(
        flavor=r"$\mu$",
        layers=["NLayers4"],
        pair_counts={
            "NLayers4": {"den_os": 50.0, "num_os": 5.0, "den_ss": 10.0, "num_ss": 1.0},
        },
        counts={
            "control_NLayers4": 100.0,
            "offline_NLayers4": 25.0,
            "trigger_NLayers4": 20.0,
            "control_combinedBins": 1000.0,
            "offline_combinedBins": 300.0,
            "trigger_combinedBins": 270.0,
        },
        control_category="control_{layer}",
        poffline_numerator_category="offline_{layer}",
        poffline_denominator_category="control_{layer}",
        pmiss_numerator_category="trigger_{layer}",
        pmiss_denominator_category="offline_{layer}",
    )

    assert estimates[0].p_offline.value == 0.25
    assert estimates[0].p_miss.value == 0.8


def test_trigger_efficiency_uses_same_sign_subtraction():
    efficiency = trigger_efficiency_from_counts(
        total_os=Count(100.0, 100.0),
        total_ss=Count(20.0, 20.0),
        passes_os=Count(70.0, 70.0),
        passes_ss=Count(10.0, 10.0),
    )

    assert efficiency.value == 60.0 / 80.0


def test_trigger_efficiency_from_outputs_uses_legacy_counters():
    output = {
        "variables": {
            "nElectronTriggerEffProbesPT55_NLayers4": {
                "sample": {"dataset": FakeHist([0.0, 100.0], [FakeAxis("n", [-0.5, 0.5, 1.5])])}
            },
            "nElectronTriggerEffProbesSSPT55_NLayers4": {
                "sample": {"dataset": FakeHist([0.0, 20.0], [FakeAxis("n", [-0.5, 0.5, 1.5])])}
            },
            "nElectronTriggerEffProbesFiringTrigger_NLayers4": {
                "sample": {"dataset": FakeHist([0.0, 70.0], [FakeAxis("n", [-0.5, 0.5, 1.5])])}
            },
            "nElectronTriggerEffSSProbesFiringTrigger_NLayers4": {
                "sample": {"dataset": FakeHist([0.0, 10.0], [FakeAxis("n", [-0.5, 0.5, 1.5])])}
            },
            "nElectronTriggerEffProbesPT55_NLayers5": {
                "sample": {"dataset": FakeHist([0.0, 50.0], [FakeAxis("n", [-0.5, 0.5, 1.5])])}
            },
            "nElectronTriggerEffProbesSSPT55_NLayers5": {
                "sample": {"dataset": FakeHist([0.0, 10.0], [FakeAxis("n", [-0.5, 0.5, 1.5])])}
            },
            "nElectronTriggerEffProbesFiringTrigger_NLayers5": {
                "sample": {"dataset": FakeHist([0.0, 30.0], [FakeAxis("n", [-0.5, 0.5, 1.5])])}
            },
            "nElectronTriggerEffSSProbesFiringTrigger_NLayers5": {
                "sample": {"dataset": FakeHist([0.0, 2.0], [FakeAxis("n", [-0.5, 0.5, 1.5])])}
            },
        }
    }

    efficiencies = _trigger_efficiency_from_outputs(
        [output],
        prefix="Electron",
        layers=["NLayers4", "NLayers5"],
        dataset="dataset",
        sample="sample",
    )

    assert efficiencies["NLayers4"].value == 60.0 / 80.0
    assert efficiencies["NLayers5"].value == 28.0 / 40.0


def test_count_component_helpers_sum_raw_tau_legs_before_ratios():
    tau_mu_components = {
        "NLayers4": {
            "control": Count(100.0, 100.0),
            "offline_pass": Count(20.0, 20.0),
            "weighted_trigger_pass": Count(10.0, 10.0),
            "offline_total": Count(25.0, 25.0),
        }
    }
    tau_ele_components = {
        "NLayers4": {
            "control": Count(300.0, 300.0),
            "offline_pass": Count(180.0, 180.0),
            "weighted_trigger_pass": Count(30.0, 30.0),
            "offline_total": Count(75.0, 75.0),
        }
    }

    probabilities = _met_probabilities_from_components(
        _sum_named_count_maps(tau_mu_components, tau_ele_components)
    )

    poffline, pmiss = probabilities["NLayers4"]
    assert poffline.value == (20.0 + 180.0) / (100.0 + 300.0)
    assert pmiss.value == (10.0 + 30.0) / (25.0 + 75.0)


def test_sparse_tau_bins_use_combined_met_probabilities():
    probabilities = {
        "NLayers4": (Count(0.6, 0.01), Count(0.0, 0.0)),
        "NLayers5": (Count(1.0, 0.0), Count(0.0, 0.0)),
        "NLayers6plus": (Count(0.25, 0.01), Count(0.55, 0.01)),
        "combinedBins": (Count(0.273, 0.0009), Count(0.570, 0.0041)),
    }

    result = _apply_sparse_tau_met_probability_fallback(probabilities)

    assert result["NLayers4"] == result["combinedBins"]
    assert result["NLayers5"] == result["combinedBins"]
    assert result["NLayers6plus"] == probabilities["NLayers6plus"]


def test_trigger_efficiency_components_can_be_combined_before_ratio():
    tau_mu_output = {
        "variables": {
            "nTauMuTriggerEffProbesPT55_NLayers4": {
                "sample": {"dataset": FakeHist([0.0, 100.0], [FakeAxis("n", [-0.5, 0.5, 1.5])])}
            },
            "nTauMuTriggerEffProbesSSPT55_NLayers4": {
                "sample": {"dataset": FakeHist([0.0, 20.0], [FakeAxis("n", [-0.5, 0.5, 1.5])])}
            },
            "nTauMuTriggerEffProbesFiringTrigger_NLayers4": {
                "sample": {"dataset": FakeHist([0.0, 70.0], [FakeAxis("n", [-0.5, 0.5, 1.5])])}
            },
            "nTauMuTriggerEffSSProbesFiringTrigger_NLayers4": {
                "sample": {"dataset": FakeHist([0.0, 10.0], [FakeAxis("n", [-0.5, 0.5, 1.5])])}
            },
        }
    }
    tau_ele_output = {
        "variables": {
            "nTauEleTriggerEffProbesPT55_NLayers4": {
                "sample": {"dataset": FakeHist([0.0, 50.0], [FakeAxis("n", [-0.5, 0.5, 1.5])])}
            },
            "nTauEleTriggerEffProbesSSPT55_NLayers4": {
                "sample": {"dataset": FakeHist([0.0, 10.0], [FakeAxis("n", [-0.5, 0.5, 1.5])])}
            },
            "nTauEleTriggerEffProbesFiringTrigger_NLayers4": {
                "sample": {"dataset": FakeHist([0.0, 30.0], [FakeAxis("n", [-0.5, 0.5, 1.5])])}
            },
            "nTauEleTriggerEffSSProbesFiringTrigger_NLayers4": {
                "sample": {"dataset": FakeHist([0.0, 2.0], [FakeAxis("n", [-0.5, 0.5, 1.5])])}
            },
        }
    }

    combined = _sum_named_count_maps(
        _trigger_efficiency_count_components_from_outputs(
            [tau_mu_output],
            prefix="TauMu",
            layers=["NLayers4"],
            dataset="dataset",
            sample="sample",
        ),
        _trigger_efficiency_count_components_from_outputs(
            [tau_ele_output],
            prefix="TauEle",
            layers=["NLayers4"],
            dataset="dataset",
            sample="sample",
        ),
    )
    efficiency = trigger_efficiency_from_counts(**combined["NLayers4"])

    assert efficiency.value == ((70.0 + 30.0) - (10.0 + 2.0)) / (
        (100.0 + 50.0) - (20.0 + 10.0)
    )


def test_tau_trigger_probability_from_outputs_uses_mode_counters():
    output = {
        "variables": {
            "nTauTriggerProbabilityNumerator": {
                "sample": {"dataset": FakeHist([0.0, 146.0], [FakeAxis("n", [0.0, 1.0, 2.0])])}
            },
            "nTauTriggerProbabilityDenominator": {
                "sample": {"dataset": FakeHist([0.0, 100.0], [FakeAxis("n", [0.0, 1.0, 2.0])])}
            },
        }
    }

    numerator, denominator, probability = _tau_trigger_probability_from_outputs(
        [output],
        dataset="dataset",
        sample="sample",
    )

    assert numerator.value == 146.0
    assert denominator.value == 100.0
    assert probability.value == 1.46
    assert np.isclose(
        probability.variance,
        1.46**2 * (1.0 / 146.0 + 1.0 / 100.0),
    )


def test_tau_probability_json_roundtrip(tmp_path: Path):
    path = tmp_path / "tau_probability.json"
    path.write_text(
        '{"tau_probability": {"value": 1.2967107, '
        '"error": 0.000068849848, "variance": 4.7403e-9}}'
    )

    probability = _tau_probability_from_json(path)

    assert probability.value == 1.2967107
    assert probability.variance == 4.7403e-9


def test_legacy_met_probabilities_integrate_trigger_turn_on():
    met_edges = [0.0, 120.0, 240.0]
    phi_edges = [0.0, 0.5, 1.0]
    output = {
        "variables": {
            "nElectronBackgroundMetMinusOnePt_NLayers4": {
                "sample": {
                    "dataset": FakeHist(
                        [10.0, 20.0],
                        [FakeAxis("met", met_edges)],
                    )
                }
            },
            "nElectronBackgroundMetMinusOnePtTrig_NLayers4": {
                "sample": {
                    "dataset": FakeHist(
                        [0.0, 10.0],
                        [FakeAxis("met", met_edges)],
                    )
                }
            },
            "nElectronBackgroundDeltaPhiMetJetLeadingVsMetMinusOnePt_NLayers4": {
                "sample": {
                    "dataset": FakeHist(
                        [[5.0, 5.0], [4.0, 6.0]],
                        [FakeAxis("met", met_edges), FakeAxis("dphi", phi_edges)],
                    )
                }
            },
        }
    }

    probabilities = legacy_met_probabilities_from_outputs(
        [output],
        prefix="Electron",
        layers=["NLayers4"],
        control_counts={"NLayers4": Count(40.0, 40.0)},
        dataset="dataset",
        sample="sample",
        met_cut=120.0,
        phi_cut=0.5,
    )

    poffline, pmiss = probabilities["NLayers4"]
    assert poffline.value == 6.0 / 40.0
    assert pmiss.value == 0.5


def test_legacy_met_probabilities_prefer_met_no_mu_turn_on():
    met_edges = [0.0, 120.0, 240.0]
    phi_edges = [0.0, 0.5, 1.0]
    output = {
        "variables": {
            "nElectronBackgroundMetNoMuPt_NLayers4": {
                "sample": {"dataset": FakeHist([10.0, 20.0], [FakeAxis("met", met_edges)])}
            },
            "nElectronBackgroundMetNoMuPtTrig_NLayers4": {
                "sample": {"dataset": FakeHist([0.0, 5.0], [FakeAxis("met", met_edges)])}
            },
            "nElectronBackgroundMetMinusOnePt_NLayers4": {
                "sample": {"dataset": FakeHist([10.0, 20.0], [FakeAxis("met", met_edges)])}
            },
            "nElectronBackgroundMetMinusOnePtTrig_NLayers4": {
                "sample": {"dataset": FakeHist([0.0, 20.0], [FakeAxis("met", met_edges)])}
            },
            "nElectronBackgroundDeltaPhiMetJetLeadingVsMetMinusOnePt_NLayers4": {
                "sample": {
                    "dataset": FakeHist(
                        [[5.0, 5.0], [4.0, 8.0]],
                        [FakeAxis("met", met_edges), FakeAxis("dphi", phi_edges)],
                    )
                }
            },
        }
    }

    probabilities = legacy_met_probabilities_from_outputs(
        [output],
        prefix="Electron",
        layers=["NLayers4"],
        control_counts={"NLayers4": Count(40.0, 40.0)},
        dataset="dataset",
        sample="sample",
        met_cut=120.0,
        phi_cut=0.5,
    )

    _, pmiss = probabilities["NLayers4"]
    assert pmiss.value == 0.25


def test_lepton_background_outputs_ignore_pveto_duplicates():
    pveto_output = {
        "cutflow": {},
        "variables": {
            "nElectronTagProbePairMassWindow_NLayers4": {},
            "nElectronBackgroundMetNoMuPt_NLayers4": {},
        },
    }
    dedicated_output = {
        "cutflow": {},
        "variables": {
            "nElectronBackgroundMetNoMuPt_NLayers4": {},
            "nElectronBackgroundDeltaPhiMetJetLeadingVsMetMinusOnePt_NLayers4": {},
        },
    }

    assert _lepton_background_outputs(
        [pveto_output, dedicated_output],
        prefix="Electron",
    ) == [dedicated_output]


def test_write_lepton_background_outputs(tmp_path: Path):
    estimates = estimate_lepton_background(
        flavor=r"$\mu$",
        layers=["NLayers4"],
        pair_counts={
            "NLayers4": {
                "den_os": 10.0,
                "num_os": 1.0,
                "den_ss": 0.0,
                "num_ss": 0.0,
            }
        },
        counts={
            "control_NLayers4": 10.0,
            "offline_NLayers4": 5.0,
            "trigger_NLayers4": 4.0,
        },
        control_category="control_{layer}",
        poffline_numerator_category="offline_{layer}",
        poffline_denominator_category="control_{layer}",
        pmiss_numerator_category="trigger_{layer}",
        pmiss_denominator_category="offline_{layer}",
    )

    json_path = tmp_path / "lepton.json"
    tex_path = tmp_path / "lepton.tex"
    write_lepton_background_json(estimates, json_path)
    write_lepton_background_latex(estimates, tex_path, run_period="2023D")

    assert '"p_miss"' in json_path.read_text()
    text = tex_path.read_text()
    assert "muon background" in text
    assert "\\multirow{1}{*}{2023D}" in text
    assert "n_{\\mathrm{layers}}" in text
    assert "\\epsilon_{\\mathrm{trigger}}^{\\ell}" in text
    assert "P_{\\mathrm{offline}}" in text
    assert "P_{\\mathrm{trigger}}" in text
    assert "2023D" in text
    assert "('" not in text

    tau_tex_path = tmp_path / "tau_lepton.tex"
    write_lepton_background_latex(
        estimates,
        tau_tex_path,
        run_period="2023D",
        tau_probability=Count(1.46, 0.00239 * 0.00239),
    )
    tau_text = tau_tex_path.read_text()
    assert "P(\\tau)" not in tau_text
    assert "1.4600 $\\pm$ 0.0024" not in tau_text

    scaled_estimates = estimate_lepton_background(
        flavor=r"$\tau$",
        layers=["NLayers4"],
        pair_counts={
            "NLayers4": {
                "den_os": 10.0,
                "num_os": 1.0,
                "den_ss": 0.0,
                "num_ss": 0.0,
            }
        },
        counts={
            "control_NLayers4": 10.0,
            "offline_NLayers4": 5.0,
            "trigger_NLayers4": 4.0,
        },
        control_category="control_{layer}",
        poffline_numerator_category="offline_{layer}",
        poffline_denominator_category="control_{layer}",
        pmiss_numerator_category="trigger_{layer}",
        pmiss_denominator_category="offline_{layer}",
        tau_probability=Count(1.46, 0.00239 * 0.00239),
    )
    scaled_json_path = tmp_path / "tau_scaled.json"
    scaled_tex_path = tmp_path / "tau_scaled.tex"
    write_lepton_background_json(scaled_estimates, scaled_json_path)
    scaled_roundtrip = read_lepton_background_json(scaled_json_path)
    assert scaled_roundtrip[0].tau_probability.value == 1.46
    write_lepton_background_latex(scaled_roundtrip, scaled_tex_path, run_period="2023D")
    scaled_text = scaled_tex_path.read_text()
    assert "P(\\tau)" in scaled_text
    assert "1.4600 $\\pm$ 0.0024" in scaled_text
    assert " & --- & --- & " in scaled_text


def test_write_combined_lepton_background_latex(tmp_path: Path):
    estimates = estimate_lepton_background(
        flavor=r"$e$",
        layers=["NLayers4", "combinedBins"],
        pair_counts={
            "NLayers4": {"den_os": 10.0, "num_os": 1.0, "den_ss": 0.0, "num_ss": 0.0},
            "combinedBins": {"den_os": 20.0, "num_os": 1.0, "den_ss": 0.0, "num_ss": 0.0},
        },
        counts={
            "control_NLayers4": 10.0,
            "offline_NLayers4": 5.0,
            "trigger_NLayers4": 4.0,
            "control_combinedBins": 20.0,
            "offline_combinedBins": 10.0,
            "trigger_combinedBins": 8.0,
        },
        control_category="control_{layer}",
        poffline_numerator_category="offline_{layer}",
        poffline_denominator_category="control_{layer}",
        pmiss_numerator_category="trigger_{layer}",
        pmiss_denominator_category="offline_{layer}",
    )
    first_json = tmp_path / "electron_2022CD.json"
    second_json = tmp_path / "electron_2022EFG.json"
    write_lepton_background_json(estimates, first_json)
    write_lepton_background_json(estimates, second_json)
    output = tmp_path / "combined.tex"

    write_combined_lepton_background_latex(
        [
            ("2022 CD", read_lepton_background_json(first_json)),
            ("2022 EFG", read_lepton_background_json(second_json)),
        ],
        output,
    )

    text = output.read_text()
    assert text.count(r"\multirow{2}{*}") == 2
    assert "2022 CD" in text
    assert "2022 EFG" in text
    assert "electron background" in text


def test_an_pm_formatter_uses_scientific_notation_for_small_values():
    assert _format_an_pm(1.62e-5, 0.31e-5) == r"$(1.62 \pm 0.31) \times 10^{-5}$"
    assert _format_an_pm(0.642, 0.016) == r"0.642 $\pm$ 0.016"
