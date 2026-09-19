from pathlib import Path

from disapptrks.tables import (
    CountWithVariance,
    POISSON_ZERO_UPPER_68,
    pveto_with_asymmetric_uncertainty,
    write_lepton_pveto_cutflow_latex,
    write_fake_track_basic_cutflow_latex,
    write_muon_cutflow_latex,
    write_muon_pveto_latex,
)


def test_pveto_with_asymmetric_uncertainty_uses_ss_subtraction():
    summary = pveto_with_asymmetric_uncertainty(
        den_os=CountWithVariance(100.0, 100.0),
        num_os=CountWithVariance(10.0, 10.0),
        den_ss=CountWithVariance(20.0, 20.0),
        num_ss=CountWithVariance(2.0, 2.0),
    )

    assert summary.denominator == 80.0
    assert summary.numerator == 8.0
    assert summary.central == 0.1
    assert summary.err_down > 0.0
    assert summary.err_up > 0.0


def test_pveto_with_negative_subtracted_numerator_quotes_upward_only():
    summary = pveto_with_asymmetric_uncertainty(
        den_os=CountWithVariance(100.0, 100.0),
        num_os=CountWithVariance(1.0, 1.0),
        den_ss=CountWithVariance(20.0, 20.0),
        num_ss=CountWithVariance(2.0, 2.0),
    )

    assert summary.numerator == -1.0
    assert summary.central == 0.0
    assert summary.err_down == 0.0
    assert summary.err_up == POISSON_ZERO_UPPER_68 / 80.0


def test_negative_pveto_boundary_does_not_use_os_ss_quadrature_error():
    summary = pveto_with_asymmetric_uncertainty(
        den_os=CountWithVariance(100.0, 100.0),
        num_os=CountWithVariance(40.0, 40.0),
        den_ss=CountWithVariance(20.0, 20.0),
        num_ss=CountWithVariance(41.0, 41.0),
    )

    assert summary.central == 0.0
    assert summary.err_down == 0.0
    assert summary.err_up == POISSON_ZERO_UPPER_68 / 80.0


def test_pveto_with_zero_subtracted_numerator_quotes_poisson_upper():
    summary = pveto_with_asymmetric_uncertainty(
        den_os=CountWithVariance(100.0, 100.0),
        num_os=CountWithVariance(0.0, 0.0),
        den_ss=CountWithVariance(0.0, 0.0),
        num_ss=CountWithVariance(0.0, 0.0),
    )

    assert summary.numerator == 0.0
    assert summary.denominator == 100.0
    assert summary.central == 0.0
    assert summary.err_down == 0.0
    assert summary.err_up > 0.0


def test_write_muon_latex_tables(tmp_path: Path):
    cutflow = {
        "inclusive": {"dataset": {"sample": {"nominal": 100.0}}},
        "muon_veto_tag": {"dataset": {"sample": {"nominal": 50.0}}},
        "muon_veto_probe": {"dataset": {"sample": {"nominal": 20.0}}},
        "muon_veto_pair": {"dataset": {"sample": {"nominal": 10.0}}},
        "muon_veto_pair_os_mass10": {"dataset": {"sample": {"nominal": 8.0}}},
        "muon_veto_zwindow": {"dataset": {"sample": {"nominal": 6.0}}},
        "muon_pveto_zwindow_pass": {"dataset": {"sample": {"nominal": 1.0}}},
        "muon_veto_ss_zwindow": {"dataset": {"sample": {"nominal": 2.0}}},
        "muon_pveto_ss_zwindow_pass": {"dataset": {"sample": {"nominal": 0.0}}},
        "muon_veto_zwindow_NLayers4": {"dataset": {"sample": {"nominal": 3.0}}},
        "muon_pveto_zwindow_pass_NLayers4": {"dataset": {"sample": {"nominal": 1.0}}},
        "muon_veto_ss_zwindow_NLayers4": {"dataset": {"sample": {"nominal": 1.0}}},
        "muon_pveto_ss_zwindow_pass_NLayers4": {"dataset": {"sample": {"nominal": 0.0}}},
    }

    cutflow_path = tmp_path / "cutflow.tex"
    pveto_path = tmp_path / "pveto.tex"
    write_muon_cutflow_latex(
        cutflow,
        cutflow_path,
        dataset="dataset",
        sample="sample",
    )
    summaries = write_muon_pveto_latex(
        cutflow,
        pveto_path,
        run_period="2024F",
        layers=["NLayers4", "combinedBins"],
        dataset="dataset",
        sample="sample",
    )

    assert "Cut & Events" in cutflow_path.read_text()
    assert "run period & flavor" in pveto_path.read_text()
    assert "2024F" in pveto_path.read_text()
    assert "N_{\\mathrm{layers}}=4" in pveto_path.read_text()
    assert summaries["combinedBins"].denominator == 4.0
    assert summaries["combinedBins"].numerator == 1.0
    assert summaries["NLayers4"].denominator == 2.0
    assert summaries["NLayers4"].numerator == 1.0


def test_write_tau_pveto_an_cutflow_layout_groups_fiducial_rows(tmp_path: Path):
    cutflow = {
        "tau_pveto_diag_tau_ele_event_trigger": {"dataset": {"sample": {"nominal": 100.0}}},
        "tau_pveto_diag_tau_ele_tag_pt": {"dataset": {"sample": {"nominal": 90.0}}},
        "tau_pveto_diag_tau_ele_tag_eta2p1": {"dataset": {"sample": {"nominal": 80.0}}},
        "tau_pveto_diag_tau_ele_tag_tight_id": {"dataset": {"sample": {"nominal": 70.0}}},
        "tau_pveto_diag_tau_ele_tag_low_mt": {"dataset": {"sample": {"nominal": 60.0}}},
        "tau_pveto_diag_tau_ele_track_pt30": {"dataset": {"sample": {"nominal": 50.0}}},
        "tau_pveto_diag_tau_ele_track_eta2p1": {"dataset": {"sample": {"nominal": 49.0}}},
        "tau_pveto_diag_tau_ele_track_fiducialECAL": {"dataset": {"sample": {"nominal": 40.0}}},
        "tau_pveto_diag_tau_ele_track_pixelHits4": {"dataset": {"sample": {"nominal": 30.0}}},
        "tau_pveto_diag_tau_ele_track_noMissingInner": {"dataset": {"sample": {"nominal": 20.0}}},
        "tau_pveto_diag_tau_ele_track_noMissingMiddle": {"dataset": {"sample": {"nominal": 10.0}}},
        "tau_pveto_diag_tau_ele_track_chargedIso0p05": {"dataset": {"sample": {"nominal": 9.0}}},
        "tau_pveto_diag_tau_ele_track_dxy0p02": {"dataset": {"sample": {"nominal": 8.0}}},
        "tau_pveto_diag_tau_ele_track_dz0p5": {"dataset": {"sample": {"nominal": 7.0}}},
        "tau_pveto_diag_tau_ele_track_electronVeto": {"dataset": {"sample": {"nominal": 6.0}}},
        "tau_pveto_diag_tau_ele_track_muonVeto": {"dataset": {"sample": {"nominal": 5.0}}},
        "tau_pveto_diag_tau_ele_pair_masswindow": {"dataset": {"sample": {"nominal": 4.0}}},
        "tau_pveto_diag_tau_ele_pair_os": {"dataset": {"sample": {"nominal": 3.0}}},
        "tau_pveto_diag_tau_ele_layer_combinedBins": {"dataset": {"sample": {"nominal": 2.0}}},
    }

    path = tmp_path / "tau_ele_cutflow.tex"
    write_lepton_pveto_cutflow_latex(
        cutflow,
        path,
        mode="tau_ele",
        dataset="dataset",
        sample="sample",
        layout="an22_23",
    )

    text = path.read_text()
    assert "passing fiducial selections" in text
    assert "tracks $|\\eta| < 2.1$" not in text
    assert "event passes MET filters" not in text


def test_write_fake_track_basic_cutflow_uses_an_table17_row_names(tmp_path: Path):
    cutflow = {
        "initial": {"dataset": {"sample": {"nominal": 100.0}}},
        "skim": {"dataset": {"sample": {"nominal": 90.0}}},
        "presel": {"dataset": {"sample": {"nominal": 80.0}}},
        "inclusive": {"dataset": {"sample": {"nominal": 70.0}}},
        "diag_event_metNoMu120": {"dataset": {"sample": {"nominal": 60.0}}},
        "diag_event_leadingJet110": {"dataset": {"sample": {"nominal": 50.0}}},
        "diag_event_jetMetDphi0p5": {"dataset": {"sample": {"nominal": 40.0}}},
        "diag_event_dijetDphi2p5": {"dataset": {"sample": {"nominal": 30.0}}},
        "basic_selection": {"dataset": {"sample": {"nominal": 20.0}}},
        "fake_control_combinedBins": {"dataset": {"sample": {"nominal": 10.0}}},
    }

    path = tmp_path / "basic_cutflow.tex"
    write_fake_track_basic_cutflow_latex(
        cutflow,
        path,
        dataset="dataset",
        sample="sample",
    )

    labels = [
        line.split(" & ", maxsplit=1)[0]
        for line in path.read_text().splitlines()
        if " & " in line and not line.startswith("Cut/category")
    ]

    assert labels == [
        r"initial events",
        r"event passes JetMET/MET triggers",
        r"event passes golden JSON, MET filters, and jet veto map",
        r"inclusive analysis category after preselections",
        r"$p_T^{\mathrm{miss,no}\,\mu}>120~\mathrm{GeV}$",
        r"leading jet $p_T>110~\mathrm{GeV}$",
        r"$\Delta\phi(\mathrm{leading~jet},\vec{p}_T^{\mathrm{miss,no}\,\mu})>0.5$",
        r"maximum dijet $\Delta\phi<2.5$",
    ]
