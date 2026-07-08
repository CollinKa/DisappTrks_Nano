"""Awkward-array selections shared by PocketCoffea and validation tools."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np


MET_TRIGGER_FIELDS = (
    "MET105_IsoTrk50",
    "MET120_IsoTrk50",
    "PFMET105_IsoTrk50",
    "PFMET120_PFMHT120_IDTight",
    "PFMET130_PFMHT130_IDTight",
    "PFMET140_PFMHT140_IDTight",
    "PFMETNoMu120_PFMHTNoMu120_IDTight",
    "PFMETNoMu130_PFMHTNoMu130_IDTight",
    "PFMETNoMu140_PFMHTNoMu140_IDTight",
    "PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60",
    "PFMETNoMu110_PFMHTNoMu110_IDTight_FilterHF",
    "PFMETNoMu120_PFMHTNoMu120_IDTight_FilterHF",
    "PFMETNoMu130_PFMHTNoMu130_IDTight_FilterHF",
    "PFMETNoMu140_PFMHTNoMu140_IDTight_FilterHF",
    "PFMET120_PFMHT120_IDTight_PFHT60",
    "PFMETTypeOne120_PFMHT120_IDTight",
    "PFMETTypeOne130_PFMHT130_IDTight",
    "PFMETTypeOne140_PFMHT140_IDTight",
    "PFMETTypeOne120_PFMHT120_IDTight_PFHT60",
)

SIGNAL_MET_FILTER_FIELDS = (
    "HBHENoiseFilter",
    "HBHENoiseIsoFilter",
    "globalSuperTightHalo2016Filter",
    "HcalStripHaloFilter",
    "EcalDeadCellTriggerPrimitiveFilter",
    "BadPFMuonFilter",
    "BadPFMuonDzFilter",
    "hfNoisyHitsFilter",
    "eeBadScFilter",
)

DEFAULT_FIDUCIAL_MAP_DIR = Path(
    "/uscms_data/d3/czheng/CMSSW_15_0_10/src/OSUT3Analysis/Configuration/data"
)


def _event_bool_like(events, value: bool):
    import awkward as ak

    for field in ("event", "run", "luminosityBlock"):
        if field in events.fields:
            template = events[field]
            return ak.ones_like(template, dtype=bool) if value else ak.zeros_like(template, dtype=bool)
    if "HLT" in events.fields and len(events.HLT.fields) > 0:
        template = events.HLT[events.HLT.fields[0]]
        return ak.ones_like(template, dtype=bool) if value else ak.zeros_like(template, dtype=bool)
    raise ValueError("cannot build an event-shaped boolean mask")


def _branch_mask(events, collection: str, field: str, *, default: bool = True):
    if collection in events.fields and field in events[collection].fields:
        return events[collection][field]
    flat_name = f"{collection}_{field}"
    if flat_name in events.fields:
        return events[flat_name]
    return _event_bool_like(events, default)


def _hlt_or_mask(events, fields=MET_TRIGGER_FIELDS):
    mask = _event_bool_like(events, False)
    for field in fields:
        if "HLT" in events.fields and field in events.HLT.fields:
            mask = mask | events.HLT[field]
        elif f"HLT_{field}" in events.fields:
            mask = mask | events[f"HLT_{field}"]
    return mask


def _signal_met_filters_mask(events):
    mask = _event_bool_like(events, True)
    found = False
    for field in SIGNAL_MET_FILTER_FIELDS:
        in_collection = "Flag" in events.fields and field in events.Flag.fields
        flat_name = f"Flag_{field}"
        if in_collection or flat_name in events.fields:
            mask = mask & _branch_mask(events, "Flag", field)
            found = True
    if found:
        return mask
    return _branch_mask(events, "Flag", "METFilters")


def _ecal_bad_calib_filter_mask(events):
    return _branch_mask(events, "Flag", "ecalBadCalibFilter")


def _good_primary_vertex_mask(events):
    if "Flag" in events.fields and "goodVertices" in events.Flag.fields:
        return events.Flag.goodVertices
    if "Flag_goodVertices" in events.fields:
        return events.Flag_goodVertices
    if "PV" in events.fields and "npvsGood" in events.PV.fields:
        return events.PV.npvsGood > 0
    if "PV_npvsGood" in events.fields:
        return events.PV_npvsGood > 0
    return _event_bool_like(events, True)


def _fiducial_map_path(flavor: str):
    override = os.environ.get(f"DISAPPTRKS_{flavor.upper()}_FIDUCIAL_MAP")
    if override:
        return Path(override)
    base = Path(os.environ.get("DISAPPTRKS_FIDUCIAL_MAP_DIR", DEFAULT_FIDUCIAL_MAP_DIR))
    return base / f"{flavor}FiducialMap_mc.root"


@lru_cache(maxsize=None)
def _fiducial_hot_spots(path: str, before_name="beforeVeto", after_name="afterVeto", threshold=2.0):
    import uproot

    path_obj = Path(path)
    if not path_obj.exists():
        return ()
    with uproot.open(str(path_obj)) as root_file:
        before, x_edges, y_edges = root_file[before_name].to_numpy()
        after, after_x_edges, after_y_edges = root_file[after_name].to_numpy()

    if not np.allclose(x_edges, after_x_edges) or not np.allclose(y_edges, after_y_edges):
        raise ValueError(f"{path}: before/after fiducial histograms have different binning")

    occupied = before > 0.0
    if not np.any(occupied):
        return ()

    inefficiency = np.zeros_like(after, dtype=float)
    inefficiency[occupied] = after[occupied] / before[occupied]
    mean = float(np.sum(after[occupied]) / np.sum(before[occupied]))
    n_occupied = int(np.count_nonzero(occupied))
    if n_occupied < 2:
        return ()
    stddev = float(np.sqrt(np.sum((inefficiency[occupied] - mean) ** 2) / (n_occupied - 1)))
    if stddev == 0.0:
        return ()

    hot_spots = []
    for ix, iy in np.argwhere(occupied):
        excess = inefficiency[ix, iy] - mean
        if excess <= float(threshold) * stddev:
            continue
        eta = float(0.5 * (x_edges[ix] + x_edges[ix + 1]))
        phi = float(0.5 * (y_edges[iy] + y_edges[iy + 1]))
        radius = float(np.hypot(0.5 * (x_edges[ix + 1] - x_edges[ix]), 0.5 * (y_edges[iy + 1] - y_edges[iy])))
        hot_spots.append((eta, phi, radius))
    return tuple(hot_spots)


def _fiducial_map_mask(tracks, path, *, min_delta_r=0.05):
    import awkward as ak

    mask = ak.ones_like(tracks.eta, dtype=bool)
    if path is None:
        return mask
    for eta, phi, radius in _fiducial_hot_spots(str(path)):
        dr = np.sqrt((tracks.eta - eta) ** 2 + delta_phi(tracks.phi, phi) ** 2)
        mask = mask & (dr >= max(float(min_delta_r), radius))
    return mask


def delta_phi(phi1, phi2):
    return np.arctan2(np.sin(phi1 - phi2), np.cos(phi1 - phi2))


def minimum_delta_r(tracks, objects, object_mask=None):
    """Return the minimum ΔR for every track, using -1 when no object exists."""
    import awkward as ak

    if object_mask is not None:
        objects = objects[object_mask]
    track_pairs, object_pairs = ak.unzip(
        ak.cartesian([tracks, objects], nested=True)
    )
    dr = np.sqrt(
        (track_pairs.eta - object_pairs.eta) ** 2
        + delta_phi(track_pairs.phi, object_pairs.phi) ** 2
    )
    return ak.fill_none(ak.min(dr, axis=2, mask_identity=True), -1.0)


def isomu24_trigger_object_mask(
    trigobjs,
    *,
    iso_bit: int = 1 << 1,
    single_muon_bit: int = 1 << 2,
):
    """NanoAOD trigger objects corresponding to the isolated SingleMuon leg.

    The legacy unversioned DisappTrks muon tag-and-probe path configured
    ``EventMuonTPProducer`` to use PAT trigger objects from
    ``hltIterL3MuonCandidates::HLT`` with the
    ``hltL3crIsoL1sSingleMu22L1f0L2f10QL3f24QL3trkIsoFiltered`` filter.  NanoAOD
    stores the trigger-object collection/filter information as compact
    ``TrigObj`` IDs and filter bits.  For ``HLT_IsoMu24``, the closest NanoAOD
    equivalent is a muon trigger object carrying both the isolated-muon and
    SingleMuon filter bits.
    """

    return (
        (trigobjs.id == 13)
        & ((trigobjs.filterBits & iso_bit) != 0)
        & ((trigobjs.filterBits & single_muon_bit) != 0)
    )


def add_muon_derived_fields(events, *, trigger_match_dr: float = 0.3):
    """Attach muon quantities needed for the tag-and-probe selections."""
    import awkward as ak

    muons = events.Muon
    isomu24_objects = isomu24_trigger_object_mask(events.TrigObj)
    d_r_min_isomu24 = minimum_delta_r(muons, events.TrigObj, isomu24_objects)
    matched_isomu24 = (
        events.HLT.IsoMu24
        & (d_r_min_isomu24 >= 0.0)
        & (d_r_min_isomu24 < trigger_match_dr)
    )
    muons = ak.with_field(muons, d_r_min_isomu24, "dRMinIsoMu24TrigObj")
    muons = ak.with_field(muons, matched_isomu24, "matchedIsoMu24")
    return muons


def muon_tag_progression_masks(
    muons, *, pt_min: float = 26.0, eta_max: float = 2.1, iso_max: float = 0.15
):
    """Cumulative muon-tag masks following the Table 16 row order.

    The AN Table 16 labels the object-ID row as "passing tight muon ID".  The
    unversioned DisappTrks ``MuonTagSkim`` also includes tight PF isolation, so
    the final tag mask used for the Nano implementation keeps the isolation and
    trigger-object match as part of the selected tag definition.
    """
    masks = {}
    mask = muons.pt > pt_min
    masks["muon_pt26"] = mask

    mask = mask & (abs(muons.eta) < eta_max)
    masks["muon_eta2p1"] = mask

    mask = mask & muons.tightId
    masks["muon_tight_id"] = mask

    mask = mask & (muons.pfRelIso04_all < iso_max) & muons.matchedIsoMu24
    masks["muon_selected_tag"] = mask
    return masks


def transverse_mass(lepton, met):
    """Transverse mass between a lepton-like object and MET."""
    return np.sqrt(
        2.0
        * lepton.pt
        * met.pt
        * (1.0 - np.cos(delta_phi(lepton.phi, met.phi)))
    )


def electron_tag_mask(
    electrons,
    events,
    *,
    pt_min: float = 35.0,
    eta_max: float = 2.1,
):
    """Run-3 tight electron tag for the electron/tau-to-electron T&P paths."""
    return electron_tag_progression_masks(
        electrons,
        events,
        pt_min=pt_min,
        eta_max=eta_max,
    )["electron_selected_tag"]


def _event_bool_like(events, value: bool):
    import awkward as ak

    if "event" in events.fields:
        template = events.event
    elif "run" in events.fields:
        template = events.run
    elif "HLT" in events.fields and len(events.HLT.fields) > 0:
        template = events.HLT[events.HLT.fields[0]]
    else:
        raise ValueError("cannot build an event-shaped boolean mask")
    return ak.ones_like(template, dtype=bool) if value else ak.zeros_like(template, dtype=bool)


def single_electron_trigger_mask(events):
    """Event-level single-electron trigger decision used for electron T&P.

    ``Ele32_WPTight_Gsf`` is the intended Run-3 path.  A few productions keep
    suffixed or closely related variants, so OR in available compatible names
    rather than failing or silently requiring one exact branch.
    """
    names = (
        "Ele32_WPTight_Gsf",
        "Ele32_WPTight_Gsf_L1DoubleEG",
        "Ele32_WPTight_Gsf_DoubleL1EG",
    )
    mask = None
    for name in names:
        if "HLT" in events.fields and name in events.HLT.fields:
            mask = events.HLT[name] if mask is None else (mask | events.HLT[name])
    return mask if mask is not None else _event_bool_like(events, False)


def electron_tag_progression_masks(
    electrons,
    events,
    *,
    pt_min: float = 35.0,
    eta_max: float = 2.1,
):
    """Cumulative tight-electron tag masks for the electron Pveto cutflow."""
    abs_sc_eta = abs(electrons.eta + electrons.deltaEtaSC)
    barrel = abs_sc_eta <= 1.479
    endcap = abs_sc_eta > 1.479
    dxy_ok = (barrel & (abs(electrons.dxy) < 0.05)) | (
        endcap & (abs(electrons.dxy) < 0.10)
    )
    dz_ok = (barrel & (abs(electrons.dz) < 0.10)) | (
        endcap & (abs(electrons.dz) < 0.20)
    )
    mask = single_electron_trigger_mask(events) & (electrons.pt > pt_min)
    masks = {"electron_pt35": mask}

    mask = mask & (abs(electrons.eta) < eta_max)
    masks["electron_eta2p1"] = mask

    mask = mask & (electrons.cutBased >= 4)
    masks["electron_tight_id"] = mask

    mask = mask & dxy_ok
    masks["electron_dxy"] = mask

    mask = mask & dz_ok
    masks["electron_dz"] = mask
    masks["electron_selected_tag"] = mask
    return masks


def low_mt_mask(leptons, met, *, mt_max: float = 40.0):
    return transverse_mass(leptons, met) < mt_max


def run3_tight_lepton_veto_jet_mask(jets):
    """Run-3 TightLepVeto jet ID for NanoAOD jets.

    Prefer the stored NanoAOD ``jetId`` bitmask when available.  Some OSUNano
    productions do not keep the multiplicity branches needed to reconstruct the
    ID from raw fractions, while ``jetId`` is exactly the compact branch meant
    for this use case.  If ``jetId`` is unavailable, fall back to the explicit
    Run-3 TightLepVeto reconstruction used by the first smoke-test files.
    """
    if "jetId" in jets.fields:
        return (jets.jetId & 6) == 6

    abs_eta = abs(jets.eta)
    multiplicity = jets.chMultiplicity + jets.neMultiplicity
    central = (
        (abs_eta <= 2.6)
        & (jets.neHEF < 0.99)
        & (jets.neEmEF < 0.90)
        & (multiplicity > 1)
        & (jets.muEF < 0.8)
        & (jets.chHEF > 0.01)
        & (jets.chMultiplicity > 0)
        & (jets.chEmEF < 0.80)
    )
    transition = (
        (abs_eta > 2.6)
        & (abs_eta <= 2.7)
        & (jets.neHEF < 0.90)
        & (jets.neEmEF < 0.99)
        & (jets.muEF < 0.8)
        & (jets.chMultiplicity > 0)
        & (jets.chEmEF < 0.80)
    )
    forward = (
        (abs_eta > 2.7)
        & (abs_eta <= 3.0)
        & (jets.neHEF < 0.99)
        & (jets.neEmEF < 0.99)
        & (jets.neMultiplicity > 1)
    )
    very_forward = (
        (abs_eta > 3.0)
        & (abs_eta <= 5.0)
        & (jets.neEmEF < 0.4)
        & (jets.neMultiplicity > 10)
    )
    return central | transition | forward | very_forward


def hadronic_tau_veto_object_mask(
    taus,
    *,
    vsjet_vvvloose: float = 0.4083,
    vse_vvvloose: float = 0.099,
    vsmu_vloose: float = 0.2949,
):
    """Tau objects used for the disappearing-track tau veto.

    The veto follows the tau-expert DeepTau 2018v2p5 loosest working points:
    VVVLoose vs jet, VVVLoose vs electron, and VLoose vs muon.  The decay-mode
    requirement keeps the new-DM decay-mode finding flag and excludes decay
    modes 5 and 6, matching the legacy disappearing-track tau-veto convention.
    """

    return (
        taus.idDecayModeNewDMs
        & (taus.decayMode != 5)
        & (taus.decayMode != 6)
        & (taus.rawDeepTau2018v2p5VSjet > vsjet_vvvloose)
        & (taus.rawDeepTau2018v2p5VSe > vse_vvvloose)
        & (taus.rawDeepTau2018v2p5VSmu > vsmu_vloose)
    )


def layer_mask(tracks, layer: str):
    layers = tracks.hp_nValidTrackerHits
    # Prefer the explicit layer count when supplied by the custom extension.
    if "hp_trackerLayersWithMeasurement" in tracks.fields:
        layers = tracks.hp_trackerLayersWithMeasurement
    if layer == "NLayers4":
        return layers == 4
    if layer == "NLayers5":
        return layers == 5
    if layer == "NLayers6plus":
        return layers >= 6
    if layer == "combinedBins":
        return layers >= 4
    raise ValueError(f"unknown layer bin: {layer}")


def add_isotrack_derived_fields(events):
    """Attach transparent analysis quantities to the ``IsoTrack`` collection."""
    import awkward as ak

    tracks = events.IsoTrack
    abs_eta = abs(tracks.eta)
    tracks = ak.with_field(tracks, (abs_eta >= 1.42) & (abs_eta <= 1.65), "inECALCrack")
    tracks = ak.with_field(tracks, (abs_eta >= 0.15) & (abs_eta <= 0.35), "inDTWheelGap")
    tracks = ak.with_field(tracks, (abs_eta >= 1.55) & (abs_eta <= 1.85), "inCSCTransition")
    tracks = ak.with_field(
        tracks,
        (abs(tracks.dz) > 0.5) | (abs(np.pi / 2.0 - tracks.theta) > 1.0e-3),
        "passesTOBDzOrLambda",
    )
    tracks = ak.with_field(
        tracks,
        ~tracks.passesTOBDzOrLambda,
        "inTOBCrack",
    )
    if "isFiducialElectronTrack" not in tracks.fields:
        tracks = ak.with_field(
            tracks,
            _fiducial_map_mask(tracks, _fiducial_map_path("electron")),
            "isFiducialElectronTrack",
        )
    if "isFiducialMuonTrack" not in tracks.fields:
        tracks = ak.with_field(
            tracks,
            _fiducial_map_mask(tracks, _fiducial_map_path("muon")),
            "isFiducialMuonTrack",
        )
    raw_calo_energy = tracks.caloEm + tracks.caloHad
    if "caloTotNoPU" in tracks.fields:
        calo_energy = tracks.caloTotNoPU
    else:
        rho_central_calo = None
        if "Rho" in events.fields and "fixedGridRhoFastjetCentralCalo" in events.Rho.fields:
            rho_central_calo = events.Rho.fixedGridRhoFastjetCentralCalo
        elif "Rho_fixedGridRhoFastjetCentralCalo" in events.fields:
            rho_central_calo = events.Rho_fixedGridRhoFastjetCentralCalo
        elif "fixedGridRhoFastjetCentralCalo" in events.fields:
            rho_central_calo = events.fixedGridRhoFastjetCentralCalo

        if rho_central_calo is not None:
            raw_calo_energy, rho_central_calo = ak.broadcast_arrays(
                raw_calo_energy, rho_central_calo
            )
            calo_energy = np.maximum(
                0.0,
                raw_calo_energy - rho_central_calo * np.pi * 0.4 * 0.4,
            )
        else:
            calo_energy = raw_calo_energy
    tracks = ak.with_field(tracks, calo_energy, "caloEnergy")

    good_jets = (
        (events.Jet.pt > 30.0)
        & (abs(events.Jet.eta) < 4.5)
        & run3_tight_lepton_veto_jet_mask(events.Jet)
    )
    tracks = ak.with_field(
        tracks, minimum_delta_r(tracks, events.Jet, good_jets), "dRMinJet"
    )
    tracks = ak.with_field(
        tracks, minimum_delta_r(tracks, events.Electron), "dRMinElectron"
    )
    tracks = ak.with_field(
        tracks, minimum_delta_r(tracks, events.Muon), "dRMinMuon"
    )
    good_taus = hadronic_tau_veto_object_mask(events.Tau)
    tracks = ak.with_field(
        tracks, minimum_delta_r(tracks, events.Tau, good_taus), "dRMinTauHad"
    )
    return tracks


def base_probe_track_mask(
    tracks,
    *,
    pt_min: float = 30.0,
    layer: str = "combinedBins",
    apply_jet_cut: bool = True,
    apply_calo_cut: bool = True,
    apply_outer_hits_cut: bool = False,
    apply_electron_fiducial_map: bool = False,
    apply_muon_fiducial_map: bool = False,
):
    mask = (
        (tracks.pt > pt_min)
        & (abs(tracks.eta) < 2.1)
        & ~tracks.inECALCrack
        & ~tracks.inDTWheelGap
        & ~tracks.inCSCTransition
        & ~tracks.inTOBCrack
        & tracks.isFiducialECALTrack
        & (tracks.hp_nValidPixelHits >= 4)
        & (tracks.hp_nValidHits >= 4)
        & (tracks.missingInnerHits == 0)
        & (tracks.missingMiddleHits == 0)
        & (tracks.pfRelIso03_chg < 0.05)
        & (abs(tracks.dxy) < 0.02)
        & (abs(tracks.dz) < 0.5)
        & layer_mask(tracks, layer)
    )
    if apply_electron_fiducial_map:
        mask = mask & tracks.isFiducialElectronTrack
    if apply_muon_fiducial_map:
        mask = mask & tracks.isFiducialMuonTrack
    if apply_jet_cut:
        mask = mask & ((tracks.dRMinJet < 0.0) | (tracks.dRMinJet > 0.5))
    if apply_calo_cut:
        mask = mask & (tracks.caloEnergy < 10.0)
    if apply_outer_hits_cut:
        mask = mask & (tracks.missingOuterHits >= 3)
    return mask


def muon_veto_probe_track_cutflow_masks(tracks, *, layer: str = "combinedBins"):
    """Cumulative probe-track masks in the AN Table 16 order.

    These are used for the displayed muon-Pveto cutflow.  The implementation
    follows the legacy ``ZtoMuProbeTrk`` probe definition: the valid-hit
    requirement from ``isoTrkCuts`` is kept together with the pixel-hit row even
    though Table 16 only prints the pixel-hit label.
    """
    masks = {}
    mask = tracks.pt > 30.0
    masks["track_pt30"] = mask

    mask = mask & (abs(tracks.eta) < 2.1)
    masks["track_eta2p1"] = mask

    mask = mask & ~tracks.inDTWheelGap
    masks["track_noDTWheelGap"] = mask

    mask = mask & ~tracks.inECALCrack
    masks["track_noECALCrack"] = mask

    mask = mask & ~tracks.inCSCTransition
    masks["track_noCSCTransition"] = mask

    mask = mask & tracks.isFiducialECALTrack
    masks["track_fiducialECAL"] = mask

    mask = mask & tracks.passesTOBDzOrLambda
    masks["track_dzOrLambda"] = mask

    mask = mask & (tracks.hp_nValidPixelHits >= 4) & (tracks.hp_nValidHits >= 4)
    masks["track_pixelHits4"] = mask

    mask = mask & (tracks.missingInnerHits == 0)
    masks["track_noMissingInner"] = mask

    mask = mask & (tracks.missingMiddleHits == 0)
    masks["track_noMissingMiddle"] = mask

    mask = mask & (tracks.pfRelIso03_chg < 0.05)
    masks["track_chargedIso0p05"] = mask

    mask = mask & (abs(tracks.dxy) < 0.02)
    masks["track_dxy0p02"] = mask

    mask = mask & (abs(tracks.dz) < 0.5)
    masks["track_dz0p5"] = mask

    mask = mask & ((tracks.dRMinJet < 0.0) | (tracks.dRMinJet > 0.5))
    masks["track_dRJet0p5"] = mask

    mask = mask & (
        ((tracks.dRMinElectron < 0.0) | (tracks.dRMinElectron > 0.15))
    )
    masks["track_electronVeto"] = mask

    mask = mask & ((tracks.dRMinTauHad < 0.0) | (tracks.dRMinTauHad > 0.15))
    masks["track_tauVeto"] = mask

    mask = mask & (tracks.caloEnergy < 10.0)
    masks["track_calo10"] = mask

    mask = mask & layer_mask(tracks, layer)
    masks["track_layers4plus"] = mask

    return masks


def muon_tag_mask(
    muons, *, pt_min: float = 26.0, eta_max: float = 2.1, iso_max: float = 0.15
):
    """Tight, isolated muon tag used for the first tag-and-probe prototype."""
    return muon_tag_progression_masks(
        muons, pt_min=pt_min, eta_max=eta_max, iso_max=iso_max
    )["muon_selected_tag"]


def muon_veto_probe_track_mask(tracks, *, layer: str = "combinedBins"):
    """Probe-track denominator for a first muon-veto tag-and-probe study.

    This intentionally does not apply the muon veto.  The muon-veto pass/fail
    decision is measured on top of this denominator.
    """
    return base_probe_track_mask(
        tracks,
        pt_min=30.0,
        layer=layer,
        apply_calo_cut=True,
        apply_outer_hits_cut=False,
    ) & (
        ((tracks.dRMinElectron < 0.0) | (tracks.dRMinElectron > 0.15))
        & ((tracks.dRMinTauHad < 0.0) | (tracks.dRMinTauHad > 0.15))
    )


def lepton_veto_probe_track_mask(
    tracks,
    *,
    measured_veto: str,
    layer: str = "combinedBins",
):
    """Probe-track denominator with the measured lepton veto intentionally open."""
    mask = base_probe_track_mask(
        tracks,
        pt_min=30.0,
        layer=layer,
        # For the electron Pveto measurement, Table-15-style denominator keeps
        # E_calo open.  The E_calo < 10 GeV requirement belongs to the electron
        # Pveto numerator together with the electron-veto and missing-outer-hit
        # requirements.
        apply_calo_cut=(measured_veto != "electron"),
        apply_outer_hits_cut=False,
    )
    if measured_veto != "electron":
        mask = mask & ((tracks.dRMinElectron < 0.0) | (tracks.dRMinElectron > 0.15))
    if measured_veto != "muon":
        mask = mask & ((tracks.dRMinMuon < 0.0) | (tracks.dRMinMuon > 0.15))
    if measured_veto != "tau":
        mask = mask & ((tracks.dRMinTauHad < 0.0) | (tracks.dRMinTauHad > 0.15))
    return mask


def tau_veto_probe_track_mask(tracks, *, layer: str = "combinedBins"):
    """Tau Pveto tag-and-probe denominator from AN Tables 22/23.

    The tau denominator intentionally leaves the measured tau veto open.  It
    also leaves the jet, calorimeter-energy, and missing-outer-hit requirements
    open, because those are part of the tau Pveto numerator in Table 21.
    """

    return base_probe_track_mask(
        tracks,
        pt_min=30.0,
        layer=layer,
        apply_jet_cut=False,
        apply_calo_cut=False,
        apply_outer_hits_cut=False,
    ) & (
        ((tracks.dRMinElectron < 0.0) | (tracks.dRMinElectron > 0.15))
        & ((tracks.dRMinMuon < 0.0) | (tracks.dRMinMuon > 0.15))
    )


def tau_veto_probe_track_cutflow_masks(tracks, *, layer: str = "combinedBins"):
    """Cumulative tau Pveto probe-track masks in the AN Table 22/23 order."""

    masks = {}
    mask = tracks.pt > 30.0
    masks["track_pt30"] = mask

    mask = mask & (abs(tracks.eta) < 2.1)
    masks["track_eta2p1"] = mask

    mask = mask & ~tracks.inDTWheelGap
    masks["track_noDTWheelGap"] = mask

    mask = mask & ~tracks.inECALCrack
    masks["track_noECALCrack"] = mask

    mask = mask & ~tracks.inCSCTransition
    masks["track_noCSCTransition"] = mask

    mask = mask & ~tracks.inTOBCrack
    masks["track_noTOBCrack"] = mask

    mask = mask & tracks.isFiducialECALTrack
    masks["track_fiducialECAL"] = mask

    mask = mask & (tracks.hp_nValidPixelHits >= 4) & (tracks.hp_nValidHits >= 4)
    masks["track_pixelHits4"] = mask

    mask = mask & (tracks.missingInnerHits == 0)
    masks["track_noMissingInner"] = mask

    mask = mask & (tracks.missingMiddleHits == 0)
    masks["track_noMissingMiddle"] = mask

    mask = mask & (tracks.pfRelIso03_chg < 0.05)
    masks["track_chargedIso0p05"] = mask

    mask = mask & (abs(tracks.dxy) < 0.02)
    masks["track_dxy0p02"] = mask

    mask = mask & (abs(tracks.dz) < 0.5)
    masks["track_dz0p5"] = mask

    mask = mask & ((tracks.dRMinElectron < 0.0) | (tracks.dRMinElectron > 0.15))
    masks["track_electronVeto"] = mask

    mask = mask & ((tracks.dRMinMuon < 0.0) | (tracks.dRMinMuon > 0.15))
    masks["track_muonVeto"] = mask

    mask = mask & layer_mask(tracks, layer)
    masks["track_layers4plus"] = mask

    return masks


def invariant_mass(
    first, second, *, first_mass: float = 0.105658, second_mass: float = 0.105658
):
    """Compute invariant mass from NanoAOD-style pt/eta/phi records."""
    first_px = first.pt * np.cos(first.phi)
    first_py = first.pt * np.sin(first.phi)
    first_pz = first.pt * np.sinh(first.eta)
    first_e = np.sqrt(first_px**2 + first_py**2 + first_pz**2 + first_mass**2)

    second_px = second.pt * np.cos(second.phi)
    second_py = second.pt * np.sin(second.phi)
    second_pz = second.pt * np.sinh(second.eta)
    second_e = np.sqrt(second_px**2 + second_py**2 + second_pz**2 + second_mass**2)

    mass2 = (
        (first_e + second_e) ** 2
        - (first_px + second_px) ** 2
        - (first_py + second_py) ** 2
        - (first_pz + second_pz) ** 2
    )
    return np.sqrt(np.maximum(mass2, 0.0))


def build_muon_veto_tag_probe_pairs(tags, probes):
    """Build flat per-event muon-tag/probe-track pairs for P(muon veto)."""
    import awkward as ak

    tag, probe = ak.unzip(ak.cartesian([tags, probes], axis=1))
    mass = invariant_mass(tag, probe)
    return ak.zip(
        {
            "mass": mass,
            "os": tag.charge * probe.charge < 0,
            "ss": tag.charge * probe.charge > 0,
            "probe_pt": probe.pt,
            "probe_eta": probe.eta,
            "probe_phi": probe.phi,
            "probe_dRMinMuon": probe.dRMinMuon,
            "probe_missingOuterHits": probe.missingOuterHits,
            "probe_caloEnergy": probe.caloEnergy,
            "probe_nLayers": (
                probe.hp_trackerLayersWithMeasurement
                if "hp_trackerLayersWithMeasurement" in probe.fields
                else probe.hp_nValidTrackerHits
            ),
            "probe_passMuonVeto": (probe.dRMinMuon < 0.0) | (probe.dRMinMuon > 0.15),
            "probe_passMuonPVetoNoFiducial": (
                ((probe.dRMinMuon < 0.0) | (probe.dRMinMuon > 0.15))
                & (probe.missingOuterHits >= 3)
            ),
        }
    )


def build_lepton_veto_tag_probe_pairs(
    tags,
    probes,
    *,
    tag_mass: float,
    probe_mass: float,
):
    """Build generic lepton-tag/probe-track pairs for electron/tau Pveto."""
    import awkward as ak

    tag, probe = ak.unzip(ak.cartesian([tags, probes], axis=1))
    mass = invariant_mass(tag, probe, first_mass=tag_mass, second_mass=probe_mass)
    return ak.zip(
        {
            "mass": mass,
            "os": tag.charge * probe.charge < 0,
            "ss": tag.charge * probe.charge > 0,
            "probe_dRMinElectron": probe.dRMinElectron,
            "probe_dRMinMuon": probe.dRMinMuon,
            "probe_dRMinTauHad": probe.dRMinTauHad,
            "probe_dRMinJet": probe.dRMinJet,
            "probe_caloEnergy": probe.caloEnergy,
            "probe_missingOuterHits": probe.missingOuterHits,
            "probe_nLayers": (
                probe.hp_trackerLayersWithMeasurement
                if "hp_trackerLayersWithMeasurement" in probe.fields
                else probe.hp_nValidTrackerHits
            ),
            "probe_passElectronVeto": (
                (probe.dRMinElectron < 0.0) | (probe.dRMinElectron > 0.15)
            ),
            "probe_passTauVeto": (
                (probe.dRMinTauHad < 0.0) | (probe.dRMinTauHad > 0.15)
            ),
            "probe_passElectronPVetoNoFiducial": (
                ((probe.dRMinElectron < 0.0) | (probe.dRMinElectron > 0.15))
                & (probe.caloEnergy < 10.0)
                & (probe.missingOuterHits >= 3)
            ),
            "probe_passTauPVetoNoFiducial": (
                ((probe.dRMinTauHad < 0.0) | (probe.dRMinTauHad > 0.15))
                & ((probe.dRMinJet < 0.0) | (probe.dRMinJet > 0.5))
                & (probe.caloEnergy < 10.0)
                & (probe.missingOuterHits >= 3)
            ),
        }
    )


def os_muon_probe_pair_mask(pairs):
    return pairs.os


def ss_muon_probe_pair_mask(pairs):
    return pairs.ss


def mass10_muon_probe_pair_mask(pairs):
    return pairs.mass > 10.0


def os_mass10_muon_probe_pair_mask(pairs):
    return pairs.os & mass10_muon_probe_pair_mask(pairs)


def ss_mass10_muon_probe_pair_mask(pairs):
    return pairs.ss & (pairs.mass > 10.0)


def z_window_muon_probe_pair_mask(
    pairs, *, z_mass: float = 91.1876, window: float = 10.0
):
    return (pairs.mass > z_mass - window) & (pairs.mass < z_mass + window)


def os_z_window_muon_probe_pair_mask(
    pairs, *, z_mass: float = 91.1876, window: float = 10.0
):
    return pairs.os & z_window_muon_probe_pair_mask(
        pairs, z_mass=z_mass, window=window
    )


def ss_z_window_muon_probe_pair_mask(
    pairs, *, z_mass: float = 91.1876, window: float = 10.0
):
    return pairs.ss & (pairs.mass > z_mass - window) & (pairs.mass < z_mass + window)


def os_pair_mask(pairs):
    return pairs.os


def ss_pair_mask(pairs):
    return pairs.ss


def mass_window_pair_mask(pairs, low: float, high: float):
    return (pairs.mass > low) & (pairs.mass < high)


def os_mass_window_pair_mask(pairs, low: float, high: float):
    return pairs.os & mass_window_pair_mask(pairs, low, high)


def ss_mass_window_pair_mask(pairs, low: float, high: float):
    return pairs.ss & mass_window_pair_mask(pairs, low, high)


def electron_pveto_pair_pass_mask(pairs):
    return pairs.probe_passElectronPVetoNoFiducial


def tau_pveto_pair_pass_mask(pairs):
    return pairs.probe_passTauPVetoNoFiducial


def generic_probe_pair_layer_mask(pairs, layer: str):
    if layer == "NLayers4":
        return pairs.probe_nLayers == 4
    if layer == "NLayers5":
        return pairs.probe_nLayers == 5
    if layer == "NLayers6plus":
        return pairs.probe_nLayers >= 6
    if layer == "combinedBins":
        return pairs.probe_nLayers >= 4
    raise ValueError(f"unknown layer bin: {layer}")


def muon_veto_pair_pass_mask(pairs):
    return pairs.probe_passMuonVeto


def muon_veto_pair_fail_mask(pairs):
    return ~pairs.probe_passMuonVeto


def muon_pveto_pair_pass_mask(pairs):
    """Muon-Pveto numerator before applying lepton fiducial-map vetoes.

    The AN numerator counts tag-and-probe denominator pairs for which the probe
    track both passes the muon veto and satisfies the disappearing-track
    missing-outer-hit requirement.  Fiducial maps are applied as an additional
    veto in the legacy ntuple workflow; Nano-side fiducial-map application will
    be layered on top of this once the maps are available in this repository.
    """
    return pairs.probe_passMuonPVetoNoFiducial


def muon_probe_pair_layer_mask(pairs, layer: str):
    if layer == "NLayers4":
        return pairs.probe_nLayers == 4
    if layer == "NLayers5":
        return pairs.probe_nLayers == 5
    if layer == "NLayers6plus":
        return pairs.probe_nLayers >= 6
    if layer == "combinedBins":
        return pairs.probe_nLayers >= 4
    raise ValueError(f"unknown layer bin: {layer}")


def search_track_mask(tracks, *, layer: str = "combinedBins"):
    return base_probe_track_mask(
        tracks,
        pt_min=55.0,
        layer=layer,
        apply_calo_cut=True,
        apply_outer_hits_cut=True,
        apply_electron_fiducial_map=True,
        apply_muon_fiducial_map=True,
    ) & (
        ((tracks.dRMinElectron < 0.0) | (tracks.dRMinElectron > 0.15))
        & ((tracks.dRMinMuon < 0.0) | (tracks.dRMinMuon > 0.15))
        & ((tracks.dRMinTauHad < 0.0) | (tracks.dRMinTauHad > 0.15))
    )


def fake_track_no_d0_mask(
    tracks,
    *,
    layer: str = "combinedBins",
    d0_region: str = "sideband",
    pt_min: float = 55.0,
    sideband_min: float = 0.05,
    sideband_max: float = 0.50,
):
    """Fake-track control selection with the d0 requirement replaced.

    The fake-track estimate uses disappearing-track-like candidates with the
    nominal d0 requirement removed.  The transfer factor uses the ratio of the
    signal d0 window to the sideband, while the target-layer control yield is
    counted in the sideband.
    """

    abs_dxy = abs(tracks.dxy)
    if d0_region == "signal":
        d0_mask = abs_dxy < 0.02
    elif d0_region == "sideband":
        d0_mask = (abs_dxy >= sideband_min) & (abs_dxy < sideband_max)
    else:
        raise ValueError(f"unknown fake-track d0 region: {d0_region}")

    return (
        (tracks.pt > pt_min)
        & (abs(tracks.eta) < 2.1)
        & ~tracks.inECALCrack
        & ~tracks.inDTWheelGap
        & ~tracks.inCSCTransition
        & ~tracks.inTOBCrack
        & tracks.isFiducialECALTrack
        & (tracks.hp_nValidPixelHits >= 4)
        & (tracks.hp_nValidHits >= 4)
        & (tracks.missingInnerHits == 0)
        & (tracks.missingMiddleHits == 0)
        & (tracks.pfRelIso03_chg < 0.05)
        & (abs(tracks.dz) < 0.5)
        & ((tracks.dRMinJet < 0.0) | (tracks.dRMinJet > 0.5))
        & layer_mask(tracks, layer)
        & (tracks.caloEnergy < 10.0)
        & (tracks.missingOuterHits >= 3)
        & ((tracks.dRMinElectron < 0.0) | (tracks.dRMinElectron > 0.15))
        & ((tracks.dRMinMuon < 0.0) | (tracks.dRMinMuon > 0.15))
        & ((tracks.dRMinTauHad < 0.0) | (tracks.dRMinTauHad > 0.15))
        & d0_mask
    )



def search_figure17_track_cutflow_masks(tracks, *, layer: str = "NLayers6plus"):
    """Return cumulative track masks in the exact Figure-17 signal order."""
    masks = {}
    mask = abs(tracks.eta) < 2.1
    masks["track_eta2p1"] = mask

    mask = mask & (tracks.pt > 55.0)
    masks["track_pt55"] = mask

    mask = mask & ~tracks.inECALCrack
    masks["track_noECALCrack"] = mask

    mask = mask & ~tracks.inDTWheelGap
    masks["track_noDTWheelGap"] = mask

    mask = mask & ~tracks.inCSCTransition
    masks["track_noCSCTransition"] = mask

    mask = mask & ~tracks.inTOBCrack
    masks["track_noTOBCrack"] = mask

    mask = mask & tracks.isFiducialElectronTrack
    masks["track_fiducialElectron"] = mask

    mask = mask & tracks.isFiducialMuonTrack
    masks["track_fiducialMuon"] = mask

    mask = mask & tracks.isFiducialECALTrack
    masks["track_fiducialECAL"] = mask

    mask = mask & (tracks.hp_nValidPixelHits >= 4)
    masks["track_pixelHits4"] = mask

    mask = mask & (tracks.hp_nValidHits >= 4)
    masks["track_validHits4"] = mask

    mask = mask & (tracks.missingInnerHits == 0)
    masks["track_noMissingInner"] = mask

    mask = mask & (tracks.missingMiddleHits == 0)
    masks["track_noMissingMiddle"] = mask

    mask = mask & (tracks.pfRelIso03_chg < 0.05)
    masks["track_chargedIso0p05"] = mask

    mask = mask & (abs(tracks.dxy) < 0.02)
    masks["track_dxy0p02"] = mask

    mask = mask & (abs(tracks.dz) < 0.5)
    masks["track_dz0p5"] = mask

    mask = mask & ((tracks.dRMinJet < 0.0) | (tracks.dRMinJet > 0.5))
    masks["track_dRJet0p5"] = mask

    mask = mask & ((tracks.dRMinElectron < 0.0) | (tracks.dRMinElectron > 0.15))
    masks["track_electronVeto"] = mask

    mask = mask & ((tracks.dRMinMuon < 0.0) | (tracks.dRMinMuon > 0.15))
    masks["track_muonVeto"] = mask

    mask = mask & ((tracks.dRMinTauHad < 0.0) | (tracks.dRMinTauHad > 0.15))
    masks["track_tauVeto"] = mask

    mask = mask & (tracks.caloEnergy < 10.0)
    masks["track_calo10"] = mask

    mask = mask & (tracks.missingOuterHits >= 3)
    masks["track_missingOuter3"] = mask

    mask = mask & layer_mask(tracks, layer)
    masks["track_layers6plus"] = mask

    return masks


def search_track_cutflow_masks(tracks, *, layer: str = "combinedBins"):
    """Return cumulative track masks for debugging the search-track selection."""
    masks = {}
    mask = tracks.pt > 55.0
    masks["track_pt55"] = mask

    mask = mask & (abs(tracks.eta) < 2.1)
    masks["track_eta2p1"] = mask

    mask = mask & ~tracks.inECALCrack
    masks["track_noECALCrack"] = mask

    mask = mask & ~tracks.inDTWheelGap
    masks["track_noDTWheelGap"] = mask

    mask = mask & ~tracks.inCSCTransition
    masks["track_noCSCTransition"] = mask

    mask = mask & ~tracks.inTOBCrack
    masks["track_noTOBCrack"] = mask

    mask = mask & tracks.isFiducialECALTrack
    masks["track_fiducialECAL"] = mask

    mask = mask & tracks.isFiducialElectronTrack
    masks["track_fiducialElectron"] = mask

    mask = mask & tracks.isFiducialMuonTrack
    masks["track_fiducialMuon"] = mask

    mask = mask & (tracks.hp_nValidPixelHits >= 4)
    masks["track_pixelHits4"] = mask

    mask = mask & (tracks.hp_nValidHits >= 4)
    masks["track_validHits4"] = mask

    mask = mask & (tracks.missingInnerHits == 0)
    masks["track_noMissingInner"] = mask

    mask = mask & (tracks.missingMiddleHits == 0)
    masks["track_noMissingMiddle"] = mask

    mask = mask & (tracks.pfRelIso03_chg < 0.05)
    masks["track_chargedIso0p05"] = mask

    mask = mask & (abs(tracks.dxy) < 0.02)
    masks["track_dxy0p02"] = mask

    mask = mask & (abs(tracks.dz) < 0.5)
    masks["track_dz0p5"] = mask

    mask = mask & ((tracks.dRMinJet < 0.0) | (tracks.dRMinJet > 0.5))
    masks["track_dRJet0p5"] = mask

    pre_layer_mask = mask
    masks["track_layers4"] = pre_layer_mask & layer_mask(tracks, "NLayers4")
    masks["track_layers5"] = pre_layer_mask & layer_mask(tracks, "NLayers5")
    masks["track_layers6plus"] = pre_layer_mask & layer_mask(tracks, "NLayers6plus")

    mask = pre_layer_mask & layer_mask(tracks, layer)
    masks["track_layers4plus"] = mask

    mask = mask & (tracks.caloEnergy < 10.0)
    masks["track_calo10"] = mask

    mask = mask & (tracks.missingOuterHits >= 3)
    masks["track_missingOuter3"] = mask

    mask = mask & ((tracks.dRMinElectron < 0.0) | (tracks.dRMinElectron > 0.15))
    masks["track_electronVeto"] = mask

    mask = mask & ((tracks.dRMinMuon < 0.0) | (tracks.dRMinMuon > 0.15))
    masks["track_muonVeto"] = mask

    mask = mask & ((tracks.dRMinTauHad < 0.0) | (tracks.dRMinTauHad > 0.15))
    masks["track_tauVeto"] = mask

    return masks


def search_event_cutflow_masks(
    analysis_event,
    *,
    met_min: float = 120.0,
    jet_pt_min: float = 110.0,
    jet_met_dphi_min: float = 0.5,
    dijet_dphi_max: float = 2.5,
):
    """Return cumulative event masks for debugging the search event selection."""
    masks = {}

    mask = analysis_event.passesMETTrigger
    masks["event_metTrigger"] = mask

    mask = mask & analysis_event.passesSignalMETFilters
    masks["event_metFilters"] = mask

    mask = mask & analysis_event.passEcalBadCalibFilterUpdate
    masks["event_passEcalBadCalibFilterUpdate"] = mask

    mask = mask & analysis_event.hasGoodPV
    masks["event_goodPV"] = mask

    mask = mask & (analysis_event.METNoMu_pt >= met_min)
    masks["event_metNoMu120"] = mask

    mask = mask & analysis_event.hasJetPt110
    masks["event_leadingJet110"] = mask

    mask = mask & analysis_event.hasJetPt110Eta2p4
    masks["event_leadingJetEta2p4"] = mask

    mask = mask & analysis_event.hasJetPt110Eta2p4TightLepVeto
    masks["event_leadingJetTightLepVeto"] = mask

    mask = mask & (
        (analysis_event.dijetMaxDeltaPhi < 0.0)
        | (analysis_event.dijetMaxDeltaPhi < dijet_dphi_max)
    )
    masks["event_dijetDphi2p5"] = mask

    mask = mask & (analysis_event.leadingJetMETNoMuDeltaPhi >= jet_met_dphi_min)
    masks["event_jetMetDphi0p5"] = mask

    return masks


def add_event_derived_fields(events):
    """Build no-muon-MET/jet angular quantities without a custom event table."""
    import awkward as ak

    jet_tight_lep_veto = run3_tight_lepton_veto_jet_mask(events.Jet)
    has_jet_pt110 = ak.any(events.Jet.pt > 110.0, axis=1)
    has_jet_pt110_eta2p4 = ak.any(
        (events.Jet.pt > 110.0) & (abs(events.Jet.eta) < 2.4), axis=1
    )
    has_jet_pt110_eta2p4_tight = ak.any(
        (events.Jet.pt > 110.0) & (abs(events.Jet.eta) < 2.4) & jet_tight_lep_veto,
        axis=1,
    )

    good = (events.Jet.pt > 30.0) & (abs(events.Jet.eta) < 2.4) & jet_tight_lep_veto
    jets = events.Jet[good]
    order = ak.argsort(jets.pt, ascending=False)
    jets = jets[order]
    leading_pt = ak.fill_none(ak.firsts(jets.pt), -1.0)
    leading_phi = ak.fill_none(ak.firsts(jets.phi), 0.0)

    first, second = ak.unzip(ak.combinations(jets.phi, 2, axis=1))
    dijet_max = ak.fill_none(
        ak.max(abs(delta_phi(first, second)), axis=1, mask_identity=True), -1.0
    )
    return ak.zip(
        {
            "METNoMu_pt": events.MetNoMu.pt,
            "METNoMu_phi": events.MetNoMu.phi,
            "passesMETTrigger": _hlt_or_mask(events),
            "passesSignalMETFilters": _signal_met_filters_mask(events),
            "passEcalBadCalibFilterUpdate": _ecal_bad_calib_filter_mask(events),
            "hasGoodPV": _good_primary_vertex_mask(events),
            "hasJetPt110": has_jet_pt110,
            "hasJetPt110Eta2p4": has_jet_pt110_eta2p4,
            "hasJetPt110Eta2p4TightLepVeto": has_jet_pt110_eta2p4_tight,
            "leadingJet_pt": leading_pt,
            "leadingJet_phi": leading_phi,
            "dijetMaxDeltaPhi": dijet_max,
            "leadingJetMETNoMuDeltaPhi": abs(
                delta_phi(leading_phi, events.MetNoMu.phi)
            ),
        }
    )
