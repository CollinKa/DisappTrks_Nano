"""LaTeX table helpers for AN-style tag-and-probe summaries."""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from pathlib import Path
from typing import Any, Sequence

from .summaries import cutflow_count

POISSON_ZERO_UPPER_68 = 1.1394342831883648


@dataclass(frozen=True)
class CountWithVariance:
    value: float
    variance: float


@dataclass(frozen=True)
class AsymmetricVetoProbability:
    central: float
    err_down: float
    err_up: float
    numerator: float
    denominator: float


MUON_CUTFLOW_ROWS = [
    ("muon_table16_event_singlemu_trigger", r"event passes SingleMuon triggers"),
    ("muon_table16_event_met_filters", r"event passes MET filters"),
    ("muon_table16_event_jet_veto_map", r"event passes jet veto map filter"),
    ("muon_table16_muon_pt26", r"$\geq 1$ muons $p_T > 26~\mathrm{GeV}$"),
    ("muon_table16_muon_eta2p1", r"$\geq 1$ muons $|\eta| < 2.1$"),
    ("muon_table16_muon_tight_id", r"$\geq 1$ muons passing tight muon ID"),
    (
        "muon_table16_muon_selected_tag",
        r"exactly one passing muon chosen randomly",
    ),
    ("muon_table16_track_pt30", r"$\geq 1$ tracks $p_T > 30~\mathrm{GeV}$"),
    ("muon_table16_track_eta2p1", r"$\geq 1$ tracks $|\eta| < 2.1$"),
    (
        "muon_table16_track_noDTWheelGap",
        r"$\geq 1$ tracks $|\eta| < 0.15$ OR $|\eta| > 0.35$",
    ),
    (
        "muon_table16_track_noECALCrack",
        r"$\geq 1$ tracks $|\eta| < 1.42$ OR $|\eta| > 1.65$",
    ),
    (
        "muon_table16_track_noCSCTransition",
        r"$\geq 1$ tracks $|\eta| < 1.55$ OR $|\eta| > 1.85$",
    ),
    (
        "muon_table16_track_fiducialECAL",
        r"$\geq 1$ tracks min $\Delta R_{\mathrm{track,noisy/dead~ECAL~ch.}}>0.05$",
    ),
    (
        "muon_table16_track_dzOrLambda",
        r"$\geq 1$ tracks $|d_z| > 0.5~\mathrm{cm}$ OR $|\lambda|>10^{-3}$",
    ),
    (
        "muon_table16_track_pixelHits4",
        r"$\geq 1$ tracks number of pixel hits $\geq 4$",
    ),
    (
        "muon_table16_track_noMissingInner",
        r"$\geq 1$ tracks missing inner hits $=0$",
    ),
    (
        "muon_table16_track_noMissingMiddle",
        r"$\geq 1$ tracks missing middle hits $=0$",
    ),
    (
        "muon_table16_track_chargedIso0p05",
        r"$\geq 1$ tracks rel. PF-based iso. $<0.05$",
    ),
    ("muon_table16_track_dxy0p02", r"$\geq 1$ tracks $|d_{xy}|<0.02~\mathrm{cm}$"),
    ("muon_table16_track_dz0p5", r"$\geq 1$ tracks $|d_z|<0.5~\mathrm{cm}$"),
    (
        "muon_table16_track_dRJet0p5",
        r"$\geq 1$ track--jet pairs $\Delta R_{\mathrm{track,jet}}>0.5$",
    ),
    (
        "muon_table16_pair_mass10",
        r"$\geq 1$ track--muon pairs $M_{\mathrm{track},\mu}>10~\mathrm{GeV}$",
    ),
    (
        "muon_table16_track_electronVeto",
        r"$\geq 1$ tracks min $\Delta R_{\mathrm{track,electron}}>0.15$",
    ),
    (
        "muon_table16_track_tauVeto",
        r"$\geq 1$ tracks min $\Delta R_{\mathrm{track,had.~tau}}>0.15$",
    ),
    ("muon_table16_track_calo10", r"$\geq 1$ tracks $E_{\mathrm{calo}}<10~\mathrm{GeV}$"),
    (
        "muon_table16_track_probe_before_layer",
        r"$\geq 1$ passing probe tracks before layer selection",
    ),
    (
        "muon_table16_pair_zwindow",
        r"$=1$ track--muon pairs $|M_{\mathrm{track},\mu}-M_Z|<10~\mathrm{GeV}$",
    ),
    (
        "muon_table16_pair_os",
        r"$=1$ track--muon pairs $q_{\mathrm{track}}\cdot q_{\mu}<0$",
    ),
    (
        "muon_table16_layer_combinedBins",
        r"$\geq 1$ track $n_{\mathrm{layers}}\geq 4$ (three signal region bins)",
    ),
]

SIGNAL_SEARCH_PARALLEL_CUTFLOW_ROWS = {
    "diag_eventKinematics_track_layers4",
    "diag_eventKinematics_track_layers5",
    "diag_eventKinematics_track_layers6plus",
}


SIGNAL_SEARCH_CUTFLOW_ROWS = [
    ("initial", r"raw events before configured preselections"),
    ("inclusive", r"events entering signal-search diagnostics"),
    ("diag_event_metTrigger", r"event passes the MET-trigger OR"),
    ("diag_event_metFilters", r"event passes MET filters excluding the ECAL bad-calib row"),
    (
        "diag_event_passEcalBadCalibFilterUpdate",
        r"event passes the ECAL bad-calibration filter update",
    ),
    ("diag_event_goodPV", r"event has a good primary vertex"),
    ("diag_event_metNoMu120", r"$p_T^{\mathrm{miss,no-\mu}} \geq 120~\mathrm{GeV}$"),
    ("diag_event_leadingJet110", r"$\geq 1$ jet with $p_T > 110~\mathrm{GeV}$"),
    ("diag_event_leadingJetEta2p4", r"$\geq 1$ jet with $p_T > 110~\mathrm{GeV}$ and $|\eta| < 2.4$"),
    (
        "diag_event_leadingJetTightLepVeto",
        r"$\geq 1$ jet with $p_T > 110~\mathrm{GeV}$, $|\eta| < 2.4$, and TightLepVeto ID",
    ),
    (
        "diag_event_jetMetDphi0p5",
        r"$\Delta\phi(p_T^{\mathrm{miss,no-\mu}},\mathrm{jet}) \geq 0.5$",
    ),
    ("diag_event_dijetDphi2p5", r"veto dijet pairs with $\Delta\phi > 2.5$"),
    (
        "diag_eventKinematics_track_pt55",
        r"$\geq 1$ track $p_T > 55~\mathrm{GeV}$",
    ),
    ("diag_eventKinematics_track_eta2p1", r"$\geq 1$ track $|\eta| < 2.1$"),
    (
        "diag_eventKinematics_track_noECALCrack",
        r"$\geq 1$ track outside the ECAL barrel-endcap gap",
    ),
    (
        "diag_eventKinematics_track_noDTWheelGap",
        r"$\geq 1$ track outside the DT wheel gap",
    ),
    (
        "diag_eventKinematics_track_noCSCTransition",
        r"$\geq 1$ track outside the CSC transition region",
    ),
    ("diag_eventKinematics_track_noTOBCrack", r"$\geq 1$ track outside the TOB crack"),
    (
        "diag_eventKinematics_track_fiducialECAL",
        r"$\geq 1$ track passing the ECAL fiducial veto",
    ),
    (
        "diag_eventKinematics_track_fiducialElectron",
        r"$\geq 1$ track passing the electron fiducial map",
    ),
    (
        "diag_eventKinematics_track_fiducialMuon",
        r"$\geq 1$ track passing the muon fiducial map",
    ),
    (
        "diag_eventKinematics_track_pixelHits4",
        r"$\geq 1$ track with $\geq 4$ valid pixel hits",
    ),
    (
        "diag_eventKinematics_track_validHits4",
        r"$\geq 1$ track with $\geq 4$ valid tracker hits",
    ),
    (
        "diag_eventKinematics_track_noMissingInner",
        r"$\geq 1$ track with no missing inner hits",
    ),
    (
        "diag_eventKinematics_track_noMissingMiddle",
        r"$\geq 1$ track with no missing middle hits",
    ),
    (
        "diag_eventKinematics_track_chargedIso0p05",
        r"$\geq 1$ track with relative charged isolation $<0.05$",
    ),
    ("diag_eventKinematics_track_dxy0p02", r"$\geq 1$ track $|d_{xy}| < 0.02~\mathrm{cm}$"),
    ("diag_eventKinematics_track_dz0p5", r"$\geq 1$ track $|d_z| < 0.5~\mathrm{cm}$"),
    (
        "diag_eventKinematics_track_dRJet0p5",
        r"$\geq 1$ track with $\Delta R(\mathrm{track},\mathrm{jet}) > 0.5$",
    ),
    (
        "diag_eventKinematics_track_layers4plus",
        r"$\geq 1$ track with tracker layers with measurement $\geq 4$",
    ),
    (
        "diag_eventKinematics_track_layers4",
        r"$\geq 1$ track with tracker layers with measurement $=4$",
    ),
    (
        "diag_eventKinematics_track_layers5",
        r"$\geq 1$ track with tracker layers with measurement $=5$",
    ),
    (
        "diag_eventKinematics_track_layers6plus",
        r"$\geq 1$ track with tracker layers with measurement $\geq 6$",
    ),
    (
        "diag_eventKinematics_track_calo10",
        r"$\geq 1$ track with calorimeter energy $<10~\mathrm{GeV}$",
    ),
    (
        "diag_eventKinematics_track_missingOuter3",
        r"$\geq 1$ track with missing outer hits $\geq 3$",
    ),
    (
        "diag_eventKinematics_track_electronVeto",
        r"$\geq 1$ track passing the electron veto",
    ),
    (
        "diag_eventKinematics_track_muonVeto",
        r"$\geq 1$ track passing the muon veto",
    ),
    (
        "diag_eventKinematics_track_tauVeto",
        r"$\geq 1$ track passing the hadronic tau veto",
    ),
    ("search", r"event search kinematics and $\geq 1$ disappearing track"),
]

LEPTON_PVETO_CUTFLOW_ROWS = {
    "electron": [
        ("electron_pveto_diag_event_singleele_trigger", r"event passes SingleElectron triggers"),
        ("electron_pveto_diag_event_met_filters", r"event passes MET filters"),
        ("electron_pveto_diag_event_jet_veto_map", r"event passes jet veto map filter"),
        ("electron_pveto_diag_electron_pt35", r"$\geq 1$ electrons $p_T > 35~\mathrm{GeV}$"),
        ("electron_pveto_diag_electron_eta2p1", r"$\geq 1$ electrons $|\eta| < 2.1$"),
        (
            "electron_pveto_diag_electron_tight_id",
            r"$\geq 1$ electrons passing tight electron ID",
        ),
        (
            "electron_pveto_diag_electron_dxy",
            r"$\geq 1$ electrons passing barrel/endcap $d_{xy}$ cuts",
        ),
        (
            "electron_pveto_diag_electron_dz",
            r"$\geq 1$ electrons passing barrel/endcap $d_z$ cuts",
        ),
        ("electron_pveto_diag_electron_selected_tag", r"$\geq 1$ selected electron tags"),
        ("electron_pveto_diag_track_pt30", r"$\geq 1$ tracks $p_T > 30~\mathrm{GeV}$"),
        ("electron_pveto_diag_track_eta2p1", r"$\geq 1$ tracks $|\eta| < 2.1$"),
        (
            "electron_pveto_diag_track_noDTWheelGap",
            r"$\geq 1$ tracks $|\eta| < 0.15$ OR $|\eta| > 0.35$",
        ),
        (
            "electron_pveto_diag_track_noECALCrack",
            r"$\geq 1$ tracks $|\eta| < 1.42$ OR $|\eta| > 1.65$",
        ),
        (
            "electron_pveto_diag_track_noCSCTransition",
            r"$\geq 1$ tracks $|\eta| < 1.55$ OR $|\eta| > 1.85$",
        ),
        (
            "electron_pveto_diag_track_fiducialECAL",
            r"$\geq 1$ tracks min $\Delta R_{\mathrm{track,noisy/dead~ECAL~ch.}}>0.05$",
        ),
        (
            "electron_pveto_diag_track_dzOrLambda",
            r"$\geq 1$ tracks $|d_z| > 0.5~\mathrm{cm}$ OR $|\lambda|>10^{-3}$",
        ),
        (
            "electron_pveto_diag_track_pixelHits4",
            r"$\geq 1$ tracks number of pixel hits $\geq 4$",
        ),
        (
            "electron_pveto_diag_track_noMissingInner",
            r"$\geq 1$ tracks missing inner hits $=0$",
        ),
        (
            "electron_pveto_diag_track_noMissingMiddle",
            r"$\geq 1$ tracks missing middle hits $=0$",
        ),
        (
            "electron_pveto_diag_track_chargedIso0p05",
            r"$\geq 1$ tracks rel. PF-based iso. $<0.05$",
        ),
        (
            "electron_pveto_diag_track_dxy0p02",
            r"$\geq 1$ tracks $|d_{xy}|<0.02~\mathrm{cm}$",
        ),
        (
            "electron_pveto_diag_track_dz0p5",
            r"$\geq 1$ tracks $|d_z|<0.5~\mathrm{cm}$",
        ),
        (
            "electron_pveto_diag_track_dRJet0p5",
            r"$\geq 1$ track--jet pairs $\Delta R_{\mathrm{track,jet}}>0.5$",
        ),
        (
            "electron_pveto_diag_pair_mass10",
            r"$\geq 1$ electron-tag--probe-track pairs $M_{\mathrm{track},e}>10~\mathrm{GeV}$",
        ),
        (
            "electron_pveto_diag_track_muonVeto",
            r"$\geq 1$ tracks min $\Delta R_{\mathrm{track,\mu}}>0.15$",
        ),
        (
            "electron_pveto_diag_track_tauVeto",
            r"$\geq 1$ tracks min $\Delta R_{\mathrm{track,had.~tau}}>0.15$",
        ),
        ("electron_pveto_diag_track_calo10", r"$\geq 1$ tracks $E_{\mathrm{calo}}<10~\mathrm{GeV}$"),
        (
            "electron_pveto_diag_track_probe_before_layer",
            r"$\geq 1$ passing probe tracks before layer selection",
        ),
        (
            "electron_pveto_diag_pair_zwindow",
            r"$\geq 1$ electron-tag--probe-track pairs $|M_{\mathrm{track},e}-M_Z|<10~\mathrm{GeV}$",
        ),
        (
            "electron_pveto_diag_pair_os",
            r"$\geq 1$ electron-tag--probe-track pairs $q_{\mathrm{track}}q_e<0$",
        ),
        (
            "electron_pveto_diag_layer_combinedBins",
            r"$\geq 1$ track $n_{\mathrm{layers}}\geq 4$ (three signal region bins)",
        ),
        (
            "electron_pveto_diag_pair_pass_electron_pveto",
            r"OS $Z$-window pairs passing electron veto",
        ),
    ],
    "tau_mu": [
        ("tau_pveto_diag_tau_mu_event_trigger", r"event passes SingleMuon triggers"),
        ("tau_pveto_diag_tau_mu_event_met_filters", r"event passes MET filters"),
        ("tau_pveto_diag_tau_mu_event_jet_veto_map", r"event passes jet veto map filter"),
        ("tau_pveto_diag_tau_mu_tag_pt", r"$\geq 1$ muons $p_T > 26~\mathrm{GeV}$"),
        ("tau_pveto_diag_tau_mu_tag_eta2p1", r"$\geq 1$ muons $|\eta| < 2.1$"),
        ("tau_pveto_diag_tau_mu_tag_tight_id", r"$\geq 1$ muons passing tight muon ID"),
        ("tau_pveto_diag_tau_mu_tag_low_mt", r"$\geq 1$ muons $M_T(p_T^{\mathrm{miss}},\mu)<40~\mathrm{GeV}$"),
        ("tau_pveto_diag_tau_mu_track_pt30", r"$\geq 1$ tracks $p_T > 30~\mathrm{GeV}$"),
        ("tau_pveto_diag_tau_mu_track_eta2p1", r"$\geq 1$ tracks $|\eta| < 2.1$"),
        ("tau_pveto_diag_tau_mu_track_noDTWheelGap", r"$\geq 1$ tracks $|\eta| < 0.15$ OR $|\eta| > 0.35$"),
        ("tau_pveto_diag_tau_mu_track_noECALCrack", r"$\geq 1$ tracks $|\eta| < 1.42$ OR $|\eta| > 1.65$"),
        ("tau_pveto_diag_tau_mu_track_noCSCTransition", r"$\geq 1$ tracks $|\eta| < 1.55$ OR $|\eta| > 1.85$"),
        ("tau_pveto_diag_tau_mu_track_fiducialECAL", r"$\geq 1$ tracks fiducial to the ECAL"),
        ("tau_pveto_diag_tau_mu_track_pixelHits4", r"$\geq 1$ tracks number of pixel hits $\geq 4$"),
        ("tau_pveto_diag_tau_mu_track_noMissingInner", r"$\geq 1$ tracks missing inner hits $=0$"),
        ("tau_pveto_diag_tau_mu_track_noMissingMiddle", r"$\geq 1$ tracks missing middle hits $=0$"),
        ("tau_pveto_diag_tau_mu_track_chargedIso0p05", r"$\geq 1$ tracks rel. PF-based iso. $<0.05$"),
        ("tau_pveto_diag_tau_mu_track_dxy0p02", r"$\geq 1$ tracks $|d_{xy}|<0.02~\mathrm{cm}$"),
        ("tau_pveto_diag_tau_mu_track_dz0p5", r"$\geq 1$ tracks $|d_z|<0.5~\mathrm{cm}$"),
        ("tau_pveto_diag_tau_mu_track_electronVeto", r"$\geq 1$ tracks min $\Delta R_{\mathrm{track,electron}}>0.15$"),
        ("tau_pveto_diag_tau_mu_track_muonVeto", r"$\geq 1$ tracks min $\Delta R_{\mathrm{track,\mu}}>0.15$"),
        ("tau_pveto_diag_tau_mu_pair_masswindow", r"$\geq 1$ track--muon pairs $15<M_Z-M_{\mathrm{track},\mu}<50~\mathrm{GeV}$"),
        ("tau_pveto_diag_tau_mu_pair_os", r"$\geq 1$ track--muon pairs $q_{\mathrm{track}}q_\mu<0$"),
        ("tau_pveto_diag_tau_mu_layer_combinedBins", r"$\geq 1$ track $n_{\mathrm{layers}}\geq 4$ (three signal region bins)"),
        ("tau_pveto_diag_tau_mu_pair_pass_tau_pveto", r"OS mass-window pairs passing tau veto"),
        ("tau_pveto_diag_tau_mu_pair_ss_masswindow", r"SS tag--probe pairs in the tau mass window"),
        ("tau_pveto_diag_tau_mu_pair_ss_pass_tau_pveto", r"SS mass-window pairs passing tau veto"),
    ],
    "tau_ele": [
        ("tau_pveto_diag_tau_ele_event_trigger", r"event passes SingleElectron/EGamma triggers"),
        ("tau_pveto_diag_tau_ele_event_met_filters", r"event passes MET filters"),
        ("tau_pveto_diag_tau_ele_event_jet_veto_map", r"event passes jet veto map filter"),
        ("tau_pveto_diag_tau_ele_tag_pt", r"$\geq 1$ electrons $p_T > 32~\mathrm{GeV}$"),
        ("tau_pveto_diag_tau_ele_tag_eta2p1", r"$\geq 1$ electrons $|\eta| < 2.1$"),
        ("tau_pveto_diag_tau_ele_tag_tight_id", r"$\geq 1$ electrons passing tight electron ID"),
        ("tau_pveto_diag_tau_ele_tag_low_mt", r"$\geq 1$ electrons $M_T(p_T^{\mathrm{miss}},e)<40~\mathrm{GeV}$"),
        ("tau_pveto_diag_tau_ele_track_pt30", r"$\geq 1$ tracks $p_T > 30~\mathrm{GeV}$"),
        ("tau_pveto_diag_tau_ele_track_eta2p1", r"$\geq 1$ tracks $|\eta| < 2.1$"),
        ("tau_pveto_diag_tau_ele_track_noDTWheelGap", r"$\geq 1$ tracks $|\eta| < 0.15$ OR $|\eta| > 0.35$"),
        ("tau_pveto_diag_tau_ele_track_noECALCrack", r"$\geq 1$ tracks $|\eta| < 1.42$ OR $|\eta| > 1.65$"),
        ("tau_pveto_diag_tau_ele_track_noCSCTransition", r"$\geq 1$ tracks $|\eta| < 1.55$ OR $|\eta| > 1.85$"),
        ("tau_pveto_diag_tau_ele_track_fiducialECAL", r"$\geq 1$ tracks fiducial to the ECAL"),
        ("tau_pveto_diag_tau_ele_track_pixelHits4", r"$\geq 1$ tracks number of pixel hits $\geq 4$"),
        ("tau_pveto_diag_tau_ele_track_noMissingInner", r"$\geq 1$ tracks missing inner hits $=0$"),
        ("tau_pveto_diag_tau_ele_track_noMissingMiddle", r"$\geq 1$ tracks missing middle hits $=0$"),
        ("tau_pveto_diag_tau_ele_track_chargedIso0p05", r"$\geq 1$ tracks rel. PF-based iso. $<0.05$"),
        ("tau_pveto_diag_tau_ele_track_dxy0p02", r"$\geq 1$ tracks $|d_{xy}|<0.02~\mathrm{cm}$"),
        ("tau_pveto_diag_tau_ele_track_dz0p5", r"$\geq 1$ tracks $|d_z|<0.5~\mathrm{cm}$"),
        ("tau_pveto_diag_tau_ele_track_electronVeto", r"$\geq 1$ tracks min $\Delta R_{\mathrm{track,electron}}>0.15$"),
        ("tau_pveto_diag_tau_ele_track_muonVeto", r"$\geq 1$ tracks min $\Delta R_{\mathrm{track,\mu}}>0.15$"),
        ("tau_pveto_diag_tau_ele_pair_masswindow", r"$\geq 1$ track--electron pairs $15<M_Z-M_{\mathrm{track},e}<50~\mathrm{GeV}$"),
        ("tau_pveto_diag_tau_ele_pair_os", r"$\geq 1$ track--electron pairs $q_{\mathrm{track}}q_e<0$"),
        ("tau_pveto_diag_tau_ele_layer_combinedBins", r"$\geq 1$ track $n_{\mathrm{layers}}\geq 4$ (three signal region bins)"),
        ("tau_pveto_diag_tau_ele_pair_pass_tau_pveto", r"OS mass-window pairs passing tau veto"),
        ("tau_pveto_diag_tau_ele_pair_ss_masswindow", r"SS tag--probe pairs in the tau mass window"),
        ("tau_pveto_diag_tau_ele_pair_ss_pass_tau_pveto", r"SS mass-window pairs passing tau veto"),
    ],
}

DISPLAY_LAYER = {
    "NLayers4": r"$N_{\mathrm{layers}}=4$",
    "NLayers5": r"$N_{\mathrm{layers}}=5$",
    "NLayers6plus": r"$N_{\mathrm{layers}}\geq 6$",
    "combinedBins": r"combined",
}

PVETO_TABLE_LAYERS = ("NLayers4", "NLayers5", "NLayers6plus", "combinedBins")


def format_count(value: float) -> str:
    if abs(value - round(value)) < 1.0e-9:
        return f"{round(value):d}"
    return f"{value:.3g}"


def format_pveto_latex(summary: AsymmetricVetoProbability) -> str:
    return (
        rf"${summary.central:.4g}^{{+{summary.err_up:.2g}}}"
        rf"_{{-{summary.err_down:.2g}}}$"
    )


def pveto_with_asymmetric_uncertainty(
    *,
    num_os: CountWithVariance,
    num_ss: CountWithVariance,
    den_os: CountWithVariance,
    den_ss: CountWithVariance,
) -> AsymmetricVetoProbability:
    """Compute the AN Pveto convention with SS subtraction.

    The convention follows the legacy table helper:

    ``P_veto = (N_veto_OS - N_veto_SS)/(N_OS - N_SS)``.

    If the signed numerator is non-positive, the central value is set to zero
    and only an upward one-sigma uncertainty is quoted.  For exactly zero
    numerator variance, use the legacy 68% Poisson upper interval,
    ``0.5 * ChiSquareQuantile(0.68, 2)``.
    """
    numerator = num_os.value - num_ss.value
    denominator = den_os.value - den_ss.value

    if denominator <= 0.0:
        return AsymmetricVetoProbability(0.0, 0.0, 0.0, numerator, denominator)

    numerator_variance = max(num_os.variance + num_ss.variance, 0.0)
    denominator_variance = max(den_os.variance + den_ss.variance, 0.0)
    sigma_numerator = sqrt(numerator_variance)
    sigma_denominator = sqrt(denominator_variance)

    if numerator <= 0.0:
        upper_numerator = max(sigma_numerator, POISSON_ZERO_UPPER_68)
        return AsymmetricVetoProbability(
            0.0,
            0.0,
            upper_numerator / denominator,
            numerator,
            denominator,
        )

    central = numerator / denominator
    rel2 = 0.0
    if numerator > 0.0:
        rel2 += (sigma_numerator / numerator) ** 2
    if denominator > 0.0:
        rel2 += (sigma_denominator / denominator) ** 2
    uncertainty = abs(central) * sqrt(rel2)
    return AsymmetricVetoProbability(
        central,
        uncertainty,
        uncertainty,
        numerator,
        denominator,
    )


def combined_pveto_from_layer_counts(
    layer_counts: Sequence[dict[str, float]],
) -> AsymmetricVetoProbability:
    """Combine layer bins without letting negative SS fluctuations cancel bins.

    The displayed table columns retain the raw OS and SS totals.  For the
    combined-row probability, however, each layer contributes a non-negative
    SS-subtracted veto numerator.  This avoids a downward same-sign fluctuation
    in one layer bin erasing a positive veto observation in another bin.
    """

    denominator = sum(
        counts["den_os"] - counts["den_ss"] for counts in layer_counts
    )
    numerator = sum(
        max(counts["num_os"] - counts["num_ss"], 0.0)
        for counts in layer_counts
    )

    if denominator <= 0.0:
        return AsymmetricVetoProbability(0.0, 0.0, 0.0, numerator, denominator)

    denominator_variance = sum(
        max(counts["den_os"] + counts["den_ss"], 0.0)
        for counts in layer_counts
    )
    positive_layers = [
        counts for counts in layer_counts if counts["num_os"] - counts["num_ss"] > 0.0
    ]
    numerator_variance = sum(
        max(counts["num_os"] + counts["num_ss"], 0.0)
        for counts in positive_layers
    )

    sigma_numerator = sqrt(numerator_variance)
    sigma_denominator = sqrt(max(denominator_variance, 0.0))
    if numerator <= 0.0:
        upper_numerator = max(sigma_numerator, POISSON_ZERO_UPPER_68)
        return AsymmetricVetoProbability(
            0.0,
            0.0,
            upper_numerator / denominator,
            numerator,
            denominator,
        )

    central = numerator / denominator
    rel2 = (sigma_numerator / numerator) ** 2
    rel2 += (sigma_denominator / denominator) ** 2
    uncertainty = abs(central) * sqrt(rel2)
    return AsymmetricVetoProbability(
        central,
        uncertainty,
        uncertainty,
        numerator,
        denominator,
    )


def _category_count(
    cutflow: dict[str, Any],
    category: str,
    *,
    dataset: str | None = None,
    sample: str | None = None,
    variation: str = "nominal",
) -> float:
    try:
        return cutflow_count(
            cutflow,
            category,
            dataset=dataset,
            sample=sample,
            variation=variation,
        )
    except KeyError:
        return 0.0


def _hist_count_sum(hist_obj: Any, *, category: str = "inclusive") -> float:
    """Return sum of an integer multiplicity from a PocketCoffea histogram.

    The ``n*`` variables are filled as integer event quantities into regular
    unit-width bins starting at zero.  To recover the sum of pair multiplicities
    from old outputs, multiply each bin content by the bin's lower edge.
    """
    try:
        axis_names = [axis.name for axis in hist_obj.axes]
    except AttributeError:
        return 0.0

    if "cat" in axis_names:
        try:
            hist_obj = hist_obj[{"cat": category}]
        except Exception:
            return 0.0

    axes = list(hist_obj.axes)
    if len(axes) != 1:
        return 0.0

    axis = axes[0]
    try:
        counts = hist_obj.values(flow=False)
        weights = axis.edges[:-1]
    except Exception:
        return 0.0
    return float((counts * weights).sum())


def _walk_hists(value: Any):
    if hasattr(value, "axes") and hasattr(value, "values"):
        yield value
    elif isinstance(value, dict):
        for nested in value.values():
            yield from _walk_hists(nested)


def variable_count_sum(
    variables: dict[str, Any],
    variable: str,
    *,
    dataset: str | None = None,
    sample: str | None = None,
    category: str = "inclusive",
) -> float:
    """Sum an event multiplicity variable from PocketCoffea histograms."""
    if variable not in variables:
        return 0.0

    value: Any = variables[variable]
    if sample is not None and isinstance(value, dict):
        value = value.get(sample, {})
    if dataset is not None and sample is None and isinstance(value, dict):
        return sum(
            variable_count_sum(
                {variable: nested},
                variable,
                dataset=dataset,
                category=category,
            )
            for nested in value.values()
        )
    if dataset is not None and isinstance(value, dict):
        value = value.get(dataset, {})

    return sum(_hist_count_sum(hist, category=category) for hist in _walk_hists(value))


def write_muon_cutflow_latex(
    cutflow: dict[str, Any],
    path: Path,
    *,
    dataset: str | None = None,
    sample: str | None = None,
    variation: str = "nominal",
    include_table_env: bool = False,
) -> None:
    """Write an AN-style muon tag-and-probe cutflow table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        (
            label,
            _category_count(
                cutflow,
                category,
                dataset=dataset,
                sample=sample,
                variation=variation,
            ),
        )
        for category, label in MUON_CUTFLOW_ROWS
    ]

    with path.open("w") as out:
        if include_table_env:
            out.write(r"\begin{table}[htbp]" + "\n")
            out.write(r"\centering" + "\n")
            out.write(r"\caption{Muon tag-and-probe cutflow.}" + "\n")
            out.write(r"\label{tab:muon_tp_cutflow}" + "\n")

        out.write(r"\begin{tabular}{lrrr}" + "\n")
        out.write(r"\hline" + "\n")
        out.write(
            r"Cut & Events & $\epsilon_{\mathrm{prev}}$ & "
            r"$\epsilon_{\mathrm{total}}$ \\" + "\n"
        )
        out.write(r"\hline" + "\n")

        first = None
        previous = None
        for label, value in rows:
            if first is None:
                first = value
            eff_prev = value / previous if previous else 1.0
            eff_total = value / first if first else 0.0
            out.write(
                f"{label} & {format_count(value)} & "
                f"{eff_prev:.4f} & {eff_total:.4f} \\\\\n"
            )
            previous = value

        out.write(r"\hline" + "\n")
        out.write(r"\end{tabular}" + "\n")
        if include_table_env:
            out.write(r"\end{table}" + "\n")


def write_signal_search_cutflow_latex(
    cutflow: dict[str, Any],
    path: Path,
    *,
    dataset: str | None = None,
    sample: str | None = None,
    variation: str = "nominal",
    include_table_env: bool = False,
) -> None:
    """Write the current signal-search diagnostic cutflow table."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        (
            category,
            label,
            _category_count(
                cutflow,
                category,
                dataset=dataset,
                sample=sample,
                variation=variation,
            ),
        )
        for category, label in SIGNAL_SEARCH_CUTFLOW_ROWS
    ]

    with path.open("w") as out:
        if include_table_env:
            out.write(r"\begin{table}[htbp]" + "\n")
            out.write(r"\centering" + "\n")
            out.write(r"\caption{Signal-search diagnostic cutflow.}" + "\n")
            out.write(r"\label{tab:signal_search_cutflow}" + "\n")

        out.write(r"\begin{tabular}{lrrr}" + "\n")
        out.write(r"\hline" + "\n")
        out.write(
            r"Cut & Events & $\epsilon_{\mathrm{prev}}$ & "
            r"$\epsilon_{\mathrm{total}}$ \\" + "\n"
        )
        out.write(r"\hline" + "\n")

        first = None
        previous = None
        for category, label, value in rows:
            if first is None:
                first = value
            eff_prev = value / previous if previous else 1.0
            eff_total = value / first if first else 0.0
            out.write(
                f"{label} & {format_count(value)} & "
                f"{eff_prev:.4f} & {eff_total:.4f} \\\\\n"
            )
            if category not in SIGNAL_SEARCH_PARALLEL_CUTFLOW_ROWS:
                previous = value

        out.write(r"\hline" + "\n")
        out.write(r"\end{tabular}" + "\n")
        if include_table_env:
            out.write(r"\end{table}" + "\n")


def write_lepton_pveto_cutflow_latex(
    cutflow: dict[str, Any],
    path: Path,
    *,
    mode: str,
    dataset: str | None = None,
    sample: str | None = None,
    variation: str = "nominal",
    include_table_env: bool = False,
) -> None:
    """Write a compact lepton/tau Pveto diagnostic cutflow table."""
    if mode not in LEPTON_PVETO_CUTFLOW_ROWS:
        raise ValueError(f"unknown lepton Pveto cutflow mode: {mode}")

    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        (
            label,
            _category_count(
                cutflow,
                category,
                dataset=dataset,
                sample=sample,
                variation=variation,
            ),
        )
        for category, label in LEPTON_PVETO_CUTFLOW_ROWS[mode]
    ]

    with path.open("w") as out:
        if include_table_env:
            out.write(r"\begin{table}[htbp]" + "\n")
            out.write(r"\centering" + "\n")
            out.write(r"\caption{Lepton veto tag-and-probe cutflow.}" + "\n")
            out.write(r"\label{tab:lepton_pveto_cutflow}" + "\n")

        out.write(r"\begin{tabular}{lrrr}" + "\n")
        out.write(r"\hline" + "\n")
        out.write(
            r"Cut & Events & $\epsilon_{\mathrm{prev}}$ & "
            r"$\epsilon_{\mathrm{total}}$ \\" + "\n"
        )
        out.write(r"\hline" + "\n")

        first = None
        previous = None
        for label, value in rows:
            if first is None:
                first = value
            eff_prev = value / previous if previous else 1.0
            eff_total = value / first if first else 0.0
            out.write(
                f"{label} & {format_count(value)} & "
                f"{eff_prev:.4f} & {eff_total:.4f} \\\\\n"
            )
            previous = value

        out.write(r"\hline" + "\n")
        out.write(r"\end{tabular}" + "\n")
        if include_table_env:
            out.write(r"\end{table}" + "\n")


def write_muon_pveto_latex(
    cutflow: dict[str, Any],
    path: Path,
    *,
    run_period: str,
    flavor: str = r"$\mu$",
    layers: Sequence[str] = PVETO_TABLE_LAYERS,
    dataset: str | None = None,
    sample: str | None = None,
    variation: str = "nominal",
    os_denominator_name: str = "muon_veto_zwindow",
    os_numerator_name: str = "muon_pveto_zwindow_pass",
    ss_denominator_name: str = "muon_veto_ss_zwindow",
    ss_numerator_name: str = "muon_pveto_ss_zwindow_pass",
    include_table_env: bool = False,
    pair_counts: dict[str, dict[str, float]] | None = None,
) -> dict[str, AsymmetricVetoProbability]:
    """Write AN/Table-24-style muon Pveto rows."""
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(layers, str):
        layers = (layers,)
    summaries = {}

    layer_count_by_name = {}
    rows = []
    for layer in layers:
        suffix = "" if layer == "combinedBins" else f"_{layer}"
        if pair_counts is not None and layer in pair_counts:
            counts = pair_counts[layer]
            den_os_value = counts["den_os"]
            num_os_value = counts["num_os"]
            den_ss_value = counts["den_ss"]
            num_ss_value = counts["num_ss"]
        else:
            den_os_value = _category_count(
                cutflow,
                f"{os_denominator_name}{suffix}",
                dataset=dataset,
                sample=sample,
                variation=variation,
            )
            num_os_value = _category_count(
                cutflow,
                f"{os_numerator_name}{suffix}",
                dataset=dataset,
                sample=sample,
                variation=variation,
            )
            den_ss_value = _category_count(
                cutflow,
                f"{ss_denominator_name}{suffix}",
                dataset=dataset,
                sample=sample,
                variation=variation,
            )
            num_ss_value = _category_count(
                cutflow,
                f"{ss_numerator_name}{suffix}",
                dataset=dataset,
                sample=sample,
                variation=variation,
            )

        layer_count_by_name[layer] = {
            "den_os": den_os_value,
            "num_os": num_os_value,
            "den_ss": den_ss_value,
            "num_ss": num_ss_value,
        }
        if layer == "combinedBins":
            component_counts = [
                layer_count_by_name[name]
                for name in ("NLayers4", "NLayers5", "NLayers6plus")
                if name in layer_count_by_name
            ]
            summary = (
                combined_pveto_from_layer_counts(component_counts)
                if component_counts
                else pveto_with_asymmetric_uncertainty(
                    den_os=CountWithVariance(den_os_value, den_os_value),
                    num_os=CountWithVariance(num_os_value, num_os_value),
                    den_ss=CountWithVariance(den_ss_value, den_ss_value),
                    num_ss=CountWithVariance(num_ss_value, num_ss_value),
                )
            )
        else:
            summary = pveto_with_asymmetric_uncertainty(
                den_os=CountWithVariance(den_os_value, den_os_value),
                num_os=CountWithVariance(num_os_value, num_os_value),
                den_ss=CountWithVariance(den_ss_value, den_ss_value),
                num_ss=CountWithVariance(num_ss_value, num_ss_value),
            )
        summaries[layer] = summary
        rows.append((layer, den_os_value, num_os_value, den_ss_value, num_ss_value, summary))

    with path.open("w") as out:
        if include_table_env:
            out.write(r"\begin{table}[htbp]" + "\n")
            out.write(r"\centering" + "\n")
            out.write(r"\caption{Muon veto probability.}" + "\n")
            out.write(r"\label{tab:muon_pveto}" + "\n")

        out.write(r"\begin{tabular}{llcrrrrc}" + "\n")
        out.write(r"\hline" + "\n")
        out.write(
            r"run period & flavor & $n_{\mathrm{layers}}$ & "
            r"$N_{T\&P}$ & $N^{\mathrm{veto}}_{T\&P}$ & "
            r"$N_{SS,T\&P}$ & $N^{\mathrm{veto}}_{SS,T\&P}$ & "
            r"$P_{\mathrm{veto}}$ \\" + "\n"
        )
        out.write(r"\hline" + "\n")
        for layer, den_os_value, num_os_value, den_ss_value, num_ss_value, summary in rows:
            out.write(
                f"{run_period} & {flavor} & {DISPLAY_LAYER.get(layer, layer)} & "
                f"{format_count(den_os_value)} & {format_count(num_os_value)} & "
                f"{format_count(den_ss_value)} & {format_count(num_ss_value)} & "
                f"{format_pveto_latex(summary)} \\\\\n"
            )
        out.write(r"\hline" + "\n")
        out.write(r"\end{tabular}" + "\n")
        if include_table_env:
            out.write(r"\end{table}" + "\n")

    return summaries
