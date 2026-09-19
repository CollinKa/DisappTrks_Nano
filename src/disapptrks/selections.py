"""Awkward-array selections shared by PocketCoffea and validation tools."""

from __future__ import annotations

import numpy as np


def delta_phi(phi1, phi2):
    return np.arctan2(np.sin(phi1 - phi2), np.cos(phi1 - phi2))


def met_no_mu_minus_lepton(events, leptons, *, flavor: str):
    """Return legacy MET-no-muon-minus-one for a selected lepton tag.

    ``MetNoMu`` already treats every muon as invisible, so a muon tag must not
    be added a second time. Visible electron and tau tags still need to be
    added to the stored no-muon MET vector.
    """
    import awkward as ak

    if flavor == "muon":
        return events.MetNoMu.pt, events.MetNoMu.phi
    if flavor not in {"electron", "tau"}:
        raise ValueError(f"unknown lepton flavor: {flavor}")

    tag = ak.firsts(leptons)
    tag_pt = ak.fill_none(tag.pt, 0.0)
    tag_phi = ak.fill_none(tag.phi, 0.0)
    met_x = events.MetNoMu.pt * np.cos(events.MetNoMu.phi)
    met_y = events.MetNoMu.pt * np.sin(events.MetNoMu.phi)
    met_x = met_x + tag_pt * np.cos(tag_phi)
    met_y = met_y + tag_pt * np.sin(tag_phi)
    return np.sqrt(met_x * met_x + met_y * met_y), np.arctan2(met_y, met_x)


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
    single_muon_bit: int = 1 << 3,
):
    """NanoAOD trigger objects corresponding to the isolated SingleMuon leg.

    The legacy unversioned DisappTrks muon tag-and-probe path configured
    ``EventMuonTPProducer`` to use PAT trigger objects from
    ``hltIterL3MuonCandidates::HLT`` with the
    ``hltL3crIsoL1sSingleMu22L1f0L2f10QL3f24QL3trkIsoFiltered`` filter.  For
    2022 C/D the legacy config uses the ``...Filtered0p08`` variant.  NanoAOD
    stores this trigger-object collection/filter information as compact
    ``TrigObj`` IDs and filter bits: bit 1 is the isolated-muon bit, and bit 3
    is the SingleMuon-path bit.  Bit 2 is the muon-tau overlap bit, so it should
    not be accepted for the SingleMuon tag-and-probe trigger match.
    """

    return (
        (trigobjs.id == 13)
        & ((trigobjs.filterBits & iso_bit) != 0)
        & ((trigobjs.filterBits & single_muon_bit) != 0)
    )


def single_electron_trigger_object_mask(
    trigobjs,
    *,
    wptight_bit: int = 1 << 1,
):
    """NanoAOD trigger objects corresponding to the tight SingleElectron leg.

    In the NanoAOD ``TrigObj`` electron quality-bit ordering, bit 1 corresponds
    to ``hltEle*WPTight*TrackIsoFilter*``.  That is the compact-Nano analogue of
    the legacy ``hltEle32WPTightGsfTrackIsoFilter`` trigger-object filter used
    for the electron tag-and-probe trigger matching.
    """

    return (trigobjs.id == 11) & ((trigobjs.filterBits & wptight_bit) != 0)


def add_electron_derived_fields(events, *, trigger_match_dr: float = 0.3):
    """Attach electron quantities needed for legacy-style tag selections."""
    import awkward as ak

    electrons = events.Electron
    single_ele_objects = single_electron_trigger_object_mask(events.TrigObj)
    d_r_min_single_ele = minimum_delta_r(
        electrons,
        events.TrigObj,
        single_ele_objects,
    )
    matched_single_ele = (
        single_electron_trigger_mask(events)
        & (d_r_min_single_ele >= 0.0)
        & (d_r_min_single_ele < trigger_match_dr)
    )
    electrons = ak.with_field(
        electrons,
        d_r_min_single_ele,
        "dRMinSingleElectronTrigObj",
    )
    electrons = ak.with_field(electrons, matched_single_ele, "matchedSingleElectron")
    return electrons


def add_muon_derived_fields(events, *, trigger_match_dr: float = 0.3):
    """Attach muon quantities needed for the tag-and-probe selections."""
    import awkward as ak

    muons = events.Muon
    isomu24_objects = isomu24_trigger_object_mask(events.TrigObj)
    d_r_min_isomu24 = minimum_delta_r(muons, events.TrigObj, isomu24_objects)
    hlt_isomu24 = (
        events.HLT.IsoMu24
        if "HLT" in events.fields and "IsoMu24" in events.HLT.fields
        else _event_bool_like(events, False)
    )
    matched_isomu24 = (
        hlt_isomu24
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


def trigger_matched_track_mask(
    events,
    tracks,
    *,
    flavor: str,
    trigger_match_dr: float = 0.3,
):
    """Return probe tracks matched to the lepton trigger object."""

    import awkward as ak

    if flavor in ("muon", "tau_mu"):
        trigger_objects = events.TrigObj[isomu24_trigger_object_mask(events.TrigObj)]
        event_trigger = (
            events.HLT.IsoMu24
            if "HLT" in events.fields and "IsoMu24" in events.HLT.fields
            else _event_bool_like(events, False)
        )
    elif flavor in ("electron", "tau_ele"):
        trigger_objects = events.TrigObj[
            single_electron_trigger_object_mask(events.TrigObj)
        ]
        event_trigger = single_electron_trigger_mask(events)
    else:
        raise ValueError(f"unknown lepton trigger flavor: {flavor}")

    d_r_min = minimum_delta_r(tracks, trigger_objects)
    return event_trigger & (d_r_min >= 0.0) & (d_r_min < trigger_match_dr)


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
    tag_trigger = single_electron_trigger_mask(events)
    if "matchedSingleElectron" in electrons.fields:
        tag_trigger = tag_trigger & electrons.matchedSingleElectron
    mask = tag_trigger & (electrons.pt > pt_min)
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


def hadronic_tau_control_object_mask(
    taus,
    *,
    vsjet_tight: float = 0.8841,
    vse_vvvloose: float = 0.099,
    vsmu_vloose: float = 0.2949,
):
    """Hadronic taus for the AN Table-27 single-tau control sample.

    Table 14 requests decay-mode finding, tight isolation, VVVLoose rejection
    against electrons, and VLoose rejection against muons.  Run-3 NanoAOD
    stores the DeepTau 2018v2p5 discriminators rather than the legacy
    ``byTightCombinedIsolationDeltaBetaCorr3Hits`` flag, so Tight VSjet is the
    NanoAOD representation of the isolation/light-jet-rejection requirement.

    Prefer the integer working-point fields when present.  Custom NanoAOD
    productions that only retain raw scores use the corresponding Run-3
    working-point thresholds.
    """

    decay_mode = taus.idDecayModeNewDMs
    if "idDeepTau2018v2p5VSjet" in taus.fields:
        # OSUNano stores the highest passed working-point ordinal, not a
        # bitmap.  For VSjet/VSe: 1=VVVLoose, ..., 6=Tight; for VSmu:
        # 1=VLoose, ..., 4=Tight.
        pass_vsjet = taus.idDeepTau2018v2p5VSjet >= 6
        pass_vse = taus.idDeepTau2018v2p5VSe >= 1
        pass_vsmu = taus.idDeepTau2018v2p5VSmu >= 1
    else:
        required = {
            "rawDeepTau2018v2p5VSjet",
            "rawDeepTau2018v2p5VSe",
            "rawDeepTau2018v2p5VSmu",
        }
        missing = required.difference(taus.fields)
        if missing:
            raise AttributeError(
                "Table-27 tau ID requires DeepTau 2018v2p5 fields; missing "
                + ", ".join(sorted(missing))
            )
        pass_vsjet = taus.rawDeepTau2018v2p5VSjet > vsjet_tight
        pass_vse = taus.rawDeepTau2018v2p5VSe > vse_vvvloose
        pass_vsmu = taus.rawDeepTau2018v2p5VSmu > vsmu_vloose

    return (
        (taus.pt > 50.0)
        & (abs(taus.eta) < 2.1)
        & decay_mode
        & pass_vsjet
        & pass_vse
        & pass_vsmu
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


def analysis_layer_mask(tracks, layer: str):
    """Layer-bin selection including the nominal high-purity requirement.

    Every analysis layer bin requires the CMS high-purity track-quality bit.
    Keep this separate from :func:`layer_mask` so diagnostic studies can still
    classify rejected tracks by their measured layer count.
    """

    return layer_mask(tracks, layer) & tracks.isHighPurityTrack


# Maximum allowed ratio of a track's largest per-hit dE/dx measurement to its
# median per-hit dE/dx.  Originally derived for the fake-track background
# (see :func:`fake_track_layer_cut`) and shared with the lepton-background
# probe-track selections below.  The working point is layer-bin dependent;
# only NLayers4 and NLayers5 have derived values so far -- other layer bins
# get no dE/dx cut until a value is derived for them.
PROBE_TRACK_DEDX_MAX_OVER_MEDIAN = {
    "NLayers4": 2.5,
    "NLayers5": 2.5,
}


def dedx_max_over_median_mask(tracks, layer: str):
    """dE/dx max-over-median term shared by the fake-track and probe-track cuts.

    Returns ``True`` (a no-op AND term) if ``layer`` has no configured
    working point in :data:`PROBE_TRACK_DEDX_MAX_OVER_MEDIAN`, or if
    ``tracks`` does not carry the ``dEdxMaximumOverMedian`` summary field
    (callers must attach it, e.g. via ``_dedx_track_summaries``, before this
    cut can take effect).
    """

    dedx_max_over_median = PROBE_TRACK_DEDX_MAX_OVER_MEDIAN.get(layer)
    if dedx_max_over_median is None or "dEdxMaximumOverMedian" not in tracks.fields:
        return True

    import awkward as ak

    return ak.fill_none(tracks.dEdxMaximumOverMedian <= dedx_max_over_median, False)


def probe_track_dedx_mask(tracks):
    """Per-track dE/dx max-over-median requirement, looked up by each track's
    own measured layer count.

    Used by the lepton-background Pveto probe-track selections
    (:func:`muon_veto_probe_track_mask`, :func:`lepton_veto_probe_track_mask`,
    :func:`tau_veto_probe_track_mask`, and their AN Table-16/22/23 cutflow
    variants), which build one mixed-NLayers probe-track collection at
    ``layer="combinedBins"`` rather than a separate collection per layer bin.
    Contrast :func:`dedx_max_over_median_mask`, which looks up a single
    working point for a caller-chosen layer bin and is correct only when the
    caller has already restricted ``tracks`` to that one bin (e.g. the
    fake-track background, which does loop per layer bin) -- calling it with
    ``"combinedBins"`` against a mixed population is a no-op, since
    ``PROBE_TRACK_DEDX_MAX_OVER_MEDIAN`` has no ``"combinedBins"`` entry.

    Here, each track is instead checked against whichever entry of
    :data:`PROBE_TRACK_DEDX_MAX_OVER_MEDIAN` matches its own layer count, so
    NLayers4/NLayers5 tracks mixed into a combinedBins collection still get
    their bin's working point; tracks whose own layer count has no configured
    working point (NLayers6plus and up) pass unconditionally.  Returns
    ``True`` (a no-op) if ``tracks`` doesn't carry the
    ``dEdxMaximumOverMedian`` summary field.
    """

    if "dEdxMaximumOverMedian" not in tracks.fields:
        return True

    import awkward as ak

    mask = None
    for configured_layer, dedx_max_over_median in PROBE_TRACK_DEDX_MAX_OVER_MEDIAN.items():
        applies = layer_mask(tracks, configured_layer)
        passes = ak.fill_none(
            tracks.dEdxMaximumOverMedian <= dedx_max_over_median, False
        )
        term = ~applies | passes
        mask = term if mask is None else (mask & term)
    return mask if mask is not None else True


ISOLATED_TRACK_SELECTION_FIELDS = (
    "track_pt55",
    "track_eta2p1",
    "track_noECALCrack",
    "track_noDTWheelGap",
    "track_noCSCTransition",
    "track_noTOBCrack",
    "track_fiducialECAL",
    "track_pixelHits4",
    "track_validHits4",
    "track_noMissingInner",
    "track_noMissingMiddle",
    "track_chargedIso0p05",
    "track_dxy0p02",
    "track_dz0p5",
    "track_dRJet0p5",
    "track_layers4plus",
    "track_highPurity",
)


CANDIDATE_TRACK_SELECTION_FIELDS = (
    *ISOLATED_TRACK_SELECTION_FIELDS,
    "track_electronVeto",
    "track_muonVeto",
    "track_tauVeto",
)


DISAPPEARING_TRACK_SELECTION_FIELDS = (
    *CANDIDATE_TRACK_SELECTION_FIELDS,
    "track_calo10",
    "track_missingOuter3",
)


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
    veto_electrons = events.Electron.cutBased >= 1
    tracks = ak.with_field(
        tracks,
        minimum_delta_r(tracks, events.Electron, veto_electrons),
        "dRMinVetoElectron",
    )
    tracks = ak.with_field(
        tracks, minimum_delta_r(tracks, events.Muon), "dRMinMuon"
    )
    loose_muons = events.Muon.looseId
    tracks = ak.with_field(
        tracks,
        minimum_delta_r(tracks, events.Muon, loose_muons),
        "dRMinLooseMuon",
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
    require_high_purity: bool = True,
    # Defaults to off: this helper is also shared by the signal-region
    # `search_track_mask`, which must not silently pick up a dE/dx cut just
    # because some other caller (e.g. a lepton-background probe track)
    # attached `dEdxMaximumOverMedian` onto the same track collection earlier
    # in the same event pass.  Callers that want the cut (the lepton-
    # background probe-track wrappers below) pass it explicitly.
    require_dedx_max_over_median: bool = False,
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
        & (
            analysis_layer_mask(tracks, layer)
            if require_high_purity
            else layer_mask(tracks, layer)
        )
    )
    if require_dedx_max_over_median:
        mask = mask & probe_track_dedx_mask(tracks)
    if apply_jet_cut:
        mask = mask & ((tracks.dRMinJet < 0.0) | (tracks.dRMinJet > 0.5))
    if apply_calo_cut:
        mask = mask & (tracks.caloEnergy < 10.0)
    if apply_outer_hits_cut:
        mask = mask & (tracks.missingOuterHits >= 3)
    return mask


def isolated_track_selection_mask(
    tracks,
    *,
    pt_min: float = 55.0,
    layer: str = "combinedBins",
):
    """AN Table-18 isolated-track selection before later candidate-track cuts.

    This wrapper gives the legacy ``isoTrkWithPt55Cuts`` requirements a name
    matching the AN: pT, eta/crack/fiducial regions, hit and missing-hit
    quality, track isolation, impact parameters, track-jet separation, and the
    requested layer bin.  Calorimeter energy, missing outer hits, and lepton
    vetoes are intentionally left for the disappearing-track candidate stage.
    """

    if pt_min == 55.0:
        return isolated_track_selection_cutflow_masks(tracks, layer=layer)[
            "track_highPurity"
        ]

    return base_probe_track_mask(
        tracks,
        pt_min=pt_min,
        layer=layer,
        apply_jet_cut=True,
        apply_calo_cut=False,
        apply_outer_hits_cut=False,
    )


def isolated_track_selection_cutflow_masks(
    tracks,
    *,
    layer: str = "combinedBins",
    require_high_purity: bool = True,
):
    """Cumulative masks through the AN Table-18 isolated-track endpoint."""

    search_masks = search_track_cutflow_masks(
        tracks,
        layer=layer,
        require_high_purity=require_high_purity,
    )
    return {
        field: search_masks[field]
        for field in ISOLATED_TRACK_SELECTION_FIELDS
    }


def candidate_track_selection_cutflow_masks(
    tracks,
    *,
    layer: str = "combinedBins",
    require_high_purity: bool = True,
):
    """Cumulative masks through the AN Table-19 candidate-track endpoint."""

    masks = dict(
        isolated_track_selection_cutflow_masks(
            tracks,
            layer=layer,
            require_high_purity=require_high_purity,
        )
    )
    mask = masks["track_highPurity"]

    mask = mask & ((tracks.dRMinElectron < 0.0) | (tracks.dRMinElectron > 0.15))
    masks["track_electronVeto"] = mask

    mask = mask & ((tracks.dRMinMuon < 0.0) | (tracks.dRMinMuon > 0.15))
    masks["track_muonVeto"] = mask

    mask = mask & ((tracks.dRMinTauHad < 0.0) | (tracks.dRMinTauHad > 0.15))
    masks["track_tauVeto"] = mask

    return masks


def candidate_track_selection_mask(tracks, *, layer: str = "combinedBins"):
    """AN Table-19 candidate-track selection."""

    return candidate_track_selection_cutflow_masks(tracks, layer=layer)["track_tauVeto"]


def disappearing_track_selection_cutflow_masks(
    tracks,
    *,
    layer: str = "combinedBins",
    require_high_purity: bool = True,
):
    """Cumulative masks through the AN Table-20 disappearing-track endpoint."""

    masks = dict(
        candidate_track_selection_cutflow_masks(
            tracks,
            layer=layer,
            require_high_purity=require_high_purity,
        )
    )
    mask = masks["track_tauVeto"]

    mask = mask & (tracks.caloEnergy < 10.0)
    masks["track_calo10"] = mask

    mask = mask & (tracks.missingOuterHits >= 3)
    masks["track_missingOuter3"] = mask

    return masks


def fiducial_map_probe_track_mask(
    tracks,
    *,
    flavor: str,
    layer: str = "combinedBins",
):
    """Legacy ``*FiducialCalc*OldCuts`` probe-track mask.

    The Run-3 DisappTrks fiducial maps are built from the
    ``ElectronFiducialCalcBeforeOldCuts/AfterOldCuts`` and
    ``MuonFiducialCalcBeforeOldCuts/AfterOldCuts`` channels.  These are based
    on the Z tag-and-probe selections with the electron/muon fiducial-map vetoes
    removed, but with the "old" hit requirements restored:

    * number of valid pixel hits >= 3
    * number of valid hits >= 7

    The measured lepton veto is intentionally left open here; the corresponding
    ``After`` collection is formed by applying the loose/veto lepton veto on top
    of the Z-window pairs.
    """

    if flavor not in ("electron", "muon"):
        raise ValueError(f"unknown fiducial-map flavor: {flavor}")

    mask = (
        (tracks.pt > 30.0)
        & (abs(tracks.eta) < 2.1)
        & ~tracks.inECALCrack
        & ~tracks.inDTWheelGap
        & ~tracks.inCSCTransition
        & ~tracks.inTOBCrack
        # The legacy fiducial-map channels remove only the electron/muon
        # fiducial-map cuts.  The ECAL fiducial cut remains in ``isoTrkCuts``.
        & tracks.isFiducialECALTrack
        & (tracks.hp_nValidPixelHits >= 3)
        & (tracks.hp_nValidHits >= 7)
        & (tracks.missingInnerHits == 0)
        & (tracks.missingMiddleHits == 0)
        & (tracks.pfRelIso03_chg < 0.05)
        & (abs(tracks.dxy) < 0.02)
        & (abs(tracks.dz) < 0.5)
        # Fiducial maps measure detector hot spots and are kept independent of
        # the layer-dependent high-purity requirement in the analysis bins.
        & layer_mask(tracks, layer)
        & ((tracks.dRMinJet < 0.0) | (tracks.dRMinJet > 0.5))
    )

    if flavor == "muon":
        # ``ZtoMuProbeTrkWithZCuts`` includes the electron veto, tau-had veto,
        # and E_calo requirement.  The measured loose-muon veto is applied only
        # when forming ``MuonFiducialAfter``.
        mask = (
            mask
            & ((tracks.dRMinElectron < 0.0) | (tracks.dRMinElectron > 0.15))
            & ((tracks.dRMinTauHad < 0.0) | (tracks.dRMinTauHad > 0.15))
            & (tracks.caloEnergy < 10.0)
        )
    else:
        # ``ZtoEleProbeTrkWithZCuts`` includes the muon veto and tau-had veto,
        # but not the E_calo requirement.  The measured veto-electron veto is
        # applied only when forming ``ElectronFiducialAfter``.
        mask = (
            mask
            & ((tracks.dRMinMuon < 0.0) | (tracks.dRMinMuon > 0.15))
            & ((tracks.dRMinTauHad < 0.0) | (tracks.dRMinTauHad > 0.15))
        )

    return mask


def muon_veto_probe_track_cutflow_masks(
    tracks,
    *,
    layer: str = "combinedBins",
    require_dedx_max_over_median: bool = True,
):
    """Cumulative probe-track masks in the AN Table 16 order.

    These are used for the displayed muon-Pveto cutflow.  The implementation
    follows the legacy ``ZtoMuProbeTrk`` probe definition: the valid-hit
    requirement from ``isoTrkCuts`` is kept together with the pixel-hit row even
    though Table 16 only prints the pixel-hit label.  The high-purity and
    dE/dx requirements are applied right after the ``dz`` cut, matching
    :func:`base_probe_track_mask`; the trailing ``track_layers4plus`` row is a
    pure layer-count cut since purity was already applied earlier.
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

    mask = mask & tracks.isHighPurityTrack
    masks["track_highPurity"] = mask

    if require_dedx_max_over_median:
        mask = mask & probe_track_dedx_mask(tracks)
    masks["track_dedx"] = mask

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


def muon_veto_probe_track_mask(
    tracks,
    *,
    layer: str = "combinedBins",
    require_dedx_max_over_median: bool = True,
):
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
        require_dedx_max_over_median=require_dedx_max_over_median,
    ) & (
        ((tracks.dRMinElectron < 0.0) | (tracks.dRMinElectron > 0.15))
        & ((tracks.dRMinTauHad < 0.0) | (tracks.dRMinTauHad > 0.15))
    )


def lepton_veto_probe_track_mask(
    tracks,
    *,
    measured_veto: str,
    layer: str = "combinedBins",
    require_dedx_max_over_median: bool = True,
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
        require_dedx_max_over_median=require_dedx_max_over_median,
    )
    if measured_veto != "electron":
        mask = mask & ((tracks.dRMinElectron < 0.0) | (tracks.dRMinElectron > 0.15))
    if measured_veto != "muon":
        mask = mask & ((tracks.dRMinMuon < 0.0) | (tracks.dRMinMuon > 0.15))
    if measured_veto != "tau":
        mask = mask & ((tracks.dRMinTauHad < 0.0) | (tracks.dRMinTauHad > 0.15))
    return mask


def tau_veto_probe_track_mask(
    tracks,
    *,
    layer: str = "combinedBins",
    require_dedx_max_over_median: bool = True,
):
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
        require_dedx_max_over_median=require_dedx_max_over_median,
    ) & (
        ((tracks.dRMinElectron < 0.0) | (tracks.dRMinElectron > 0.15))
        & ((tracks.dRMinMuon < 0.0) | (tracks.dRMinMuon > 0.15))
    )


def tau_veto_probe_track_cutflow_masks(
    tracks,
    *,
    layer: str = "combinedBins",
    require_dedx_max_over_median: bool = True,
):
    """Cumulative tau Pveto probe-track masks in the AN Table 22/23 order.

    The high-purity and dE/dx requirements are applied right after the ``dz``
    cut, matching :func:`base_probe_track_mask`; the trailing
    ``track_layers4plus`` row is a pure layer-count cut since purity was
    already applied earlier.
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

    mask = mask & tracks.isHighPurityTrack
    masks["track_highPurity"] = mask

    if require_dedx_max_over_median:
        mask = mask & probe_track_dedx_mask(tracks)
    masks["track_dedx"] = mask

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
    d_r_min_loose_muon = (
        probe.dRMinLooseMuon if "dRMinLooseMuon" in probe.fields else probe.dRMinMuon
    )
    probe_fires_trigger = (
        probe.firesTrigger if "firesTrigger" in probe.fields else probe.pt > 1.0e12
    )
    return ak.zip(
        {
            "mass": mass,
            "os": tag.charge * probe.charge < 0,
            "ss": tag.charge * probe.charge > 0,
            "probe_pt": probe.pt,
            "probe_eta": probe.eta,
            "probe_phi": probe.phi,
            "probe_dRMinMuon": probe.dRMinMuon,
            "probe_dRMinLooseMuon": d_r_min_loose_muon,
            "probe_missingOuterHits": probe.missingOuterHits,
            "probe_caloEnergy": probe.caloEnergy,
            "probe_nLayers": (
                probe.hp_trackerLayersWithMeasurement
                if "hp_trackerLayersWithMeasurement" in probe.fields
                else probe.hp_nValidTrackerHits
            ),
            "probe_passMuonVeto": (probe.dRMinMuon < 0.0) | (probe.dRMinMuon > 0.15),
            "probe_passLooseMuonVeto": (d_r_min_loose_muon < 0.0)
            | (d_r_min_loose_muon > 0.15),
            "probe_passMuonPVetoNoFiducial": (
                ((probe.dRMinMuon < 0.0) | (probe.dRMinMuon > 0.15))
                & (probe.missingOuterHits >= 3)
            ),
            "probe_firesTrigger": probe_fires_trigger,
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
    d_r_min_veto_electron = (
        probe.dRMinVetoElectron
        if "dRMinVetoElectron" in probe.fields
        else probe.dRMinElectron
    )
    probe_fires_trigger = (
        probe.firesTrigger if "firesTrigger" in probe.fields else probe.pt > 1.0e12
    )
    return ak.zip(
        {
            "mass": mass,
            "os": tag.charge * probe.charge < 0,
            "ss": tag.charge * probe.charge > 0,
            "probe_dRMinElectron": probe.dRMinElectron,
            "probe_dRMinVetoElectron": d_r_min_veto_electron,
            "probe_dRMinMuon": probe.dRMinMuon,
            "probe_dRMinTauHad": probe.dRMinTauHad,
            "probe_dRMinJet": probe.dRMinJet,
            "probe_pt": probe.pt,
            "probe_eta": probe.eta,
            "probe_phi": probe.phi,
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
            "probe_passVetoElectronVeto": (d_r_min_veto_electron < 0.0)
            | (d_r_min_veto_electron > 0.15),
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
            "probe_firesTrigger": probe_fires_trigger,
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


def search_track_mask(
    tracks,
    *,
    layer: str = "combinedBins",
    require_high_purity: bool = True,
):
    return base_probe_track_mask(
        tracks,
        pt_min=55.0,
        layer=layer,
        apply_calo_cut=True,
        apply_outer_hits_cut=True,
        require_high_purity=require_high_purity,
    ) & (
        ((tracks.dRMinElectron < 0.0) | (tracks.dRMinElectron > 0.15))
        & ((tracks.dRMinMuon < 0.0) | (tracks.dRMinMuon > 0.15))
        & ((tracks.dRMinTauHad < 0.0) | (tracks.dRMinTauHad > 0.15))
    )


def disappearing_track_selection_mask(tracks, *, layer: str = "combinedBins"):
    """AN Table-20 disappearing-track selection for the search region."""

    return disappearing_track_selection_cutflow_masks(tracks, layer=layer)[
        "track_missingOuter3"
    ]


def fake_track_layer_cut(
    tracks,
    *,
    layer: str = "combinedBins",
    require_high_purity: bool = True,
    require_dedx_max_over_median: bool = True,
):
    """Layer/high-purity/dE/dx term of the fake-track selection.

    Split out of :func:`fake_track_no_d0_mask` so a caller looping over
    several layer bins for the same tracks/d0 region can compute
    :func:`fake_track_base_mask` once and combine it with this cheaper
    per-layer term, instead of re-evaluating the whole selection per bin.

    For layer bins listed in :data:`PROBE_TRACK_DEDX_MAX_OVER_MEDIAN`, an
    additional cut requires ``tracks.dEdxMaximumOverMedian`` (the track's
    largest per-hit dE/dx divided by its median per-hit dE/dx) to be at or
    below the configured working point.  This is skipped if ``tracks`` does
    not carry that field, so callers must attach it (e.g. via
    ``_dedx_track_summaries``) before requesting a layer bin with a
    configured threshold.
    """

    layer_cut = (
        analysis_layer_mask(tracks, layer)
        if require_high_purity
        else layer_mask(tracks, layer)
    )

    if require_dedx_max_over_median:
        layer_cut = layer_cut & dedx_max_over_median_mask(tracks, layer)

    return layer_cut


def fake_track_base_mask(
    tracks,
    *,
    d0_region: str = "sideband",
    pt_min: float = 55.0,
    sideband_min: float = 0.05,
    sideband_max: float = 0.50,
):
    """Fake-track control selection minus the layer/high-purity/dE/dx term.

    Every condition here is independent of the layer bin, so a caller
    looping over layer bins for the same tracks/d0 region can compute this
    once and AND it with :func:`fake_track_layer_cut` per bin.
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
        & (tracks.caloEnergy < 10.0)
        & (tracks.missingOuterHits >= 3)
        & ((tracks.dRMinElectron < 0.0) | (tracks.dRMinElectron > 0.15))
        & ((tracks.dRMinMuon < 0.0) | (tracks.dRMinMuon > 0.15))
        & ((tracks.dRMinTauHad < 0.0) | (tracks.dRMinTauHad > 0.15))
        & d0_mask
    )


def fake_track_no_d0_mask(
    tracks,
    *,
    layer: str = "combinedBins",
    d0_region: str = "sideband",
    pt_min: float = 55.0,
    sideband_min: float = 0.05,
    sideband_max: float = 0.50,
    require_high_purity: bool = True,
    require_dedx_max_over_median: bool = True,
):
    """Fake-track control selection with the d0 requirement replaced.

    The fake-track estimate uses disappearing-track-like candidates with the
    nominal d0 requirement removed.  The transfer factor uses the ratio of the
    signal d0 window to the sideband, while the target-layer control yield is
    counted in the sideband.

    A thin wrapper combining :func:`fake_track_base_mask` and
    :func:`fake_track_layer_cut` -- see those for what each term covers.  A
    caller evaluating several layer bins for the same tracks/d0 region should
    call them separately instead, to avoid recomputing the layer-independent
    terms once per bin.
    """

    return fake_track_base_mask(
        tracks,
        d0_region=d0_region,
        pt_min=pt_min,
        sideband_min=sideband_min,
        sideband_max=sideband_max,
    ) & fake_track_layer_cut(
        tracks,
        layer=layer,
        require_high_purity=require_high_purity,
        require_dedx_max_over_median=require_dedx_max_over_median,
    )


def fake_track_sideband_cutflow_masks(
    tracks,
    *,
    sideband_min: float = 0.05,
    sideband_max: float = 0.50,
):
    """Return cumulative fake-track sideband candidate masks.

    This is a diagnostic view of the fake-track sideband branch used for the
    JetMET fake-track normalization.  It follows the disappearing-track-like
    requirements with the nominal signal d0 requirement replaced by the
    sideband ``sideband_min <= |d0| < sideband_max`` requirement.
    """

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

    mask = mask & (abs(tracks.dz) < 0.5)
    masks["track_dz0p5"] = mask

    mask = mask & ((tracks.dRMinJet < 0.0) | (tracks.dRMinJet > 0.5))
    masks["track_dRJet0p5"] = mask

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

    abs_dxy = abs(tracks.dxy)
    mask = mask & (abs_dxy >= sideband_min) & (abs_dxy < sideband_max)
    masks["track_d0Sideband"] = mask

    masks["track_NLayers4"] = mask & analysis_layer_mask(tracks, "NLayers4")
    masks["track_NLayers5"] = mask & analysis_layer_mask(tracks, "NLayers5")
    masks["track_NLayers6plus"] = mask & analysis_layer_mask(tracks, "NLayers6plus")
    masks["track_combinedBins"] = mask & analysis_layer_mask(tracks, "combinedBins")

    return masks


def search_track_cutflow_masks(
    tracks,
    *,
    layer: str = "combinedBins",
    require_high_purity: bool = True,
):
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

    mask = mask & layer_mask(tracks, layer)
    masks["track_layers4plus"] = mask

    if require_high_purity:
        mask = mask & tracks.isHighPurityTrack
    masks["track_highPurity"] = mask

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

    mask = analysis_event.METNoMu_pt >= met_min
    masks["event_metNoMu120"] = mask

    mask = mask & (analysis_event.leadingJet_pt > jet_pt_min)
    masks["event_leadingJet110"] = mask

    mask = mask & (abs(analysis_event.leadingJet_eta) < 2.4)
    masks["event_leadingJetEta2p4"] = mask

    mask = mask & analysis_event.leadingJet_tightLepVeto
    masks["event_leadingJetTightLepVeto"] = mask

    mask = mask & (
        (analysis_event.dijetMaxDeltaPhi < 0.0)
        | (analysis_event.dijetMaxDeltaPhi < dijet_dphi_max)
    )
    masks["event_dijetDphi2p5"] = mask

    mask = mask & (analysis_event.leadingJetMETNoMuDeltaPhi >= jet_met_dphi_min)
    masks["event_jetMetDphi0p5"] = mask

    return masks


def basic_event_selection_mask(
    analysis_event,
    *,
    met_min: float = 120.0,
    jet_pt_min: float = 110.0,
    jet_met_dphi_min: float = 0.5,
    dijet_dphi_max: float = 2.5,
):
    """AN-style BasicSelection event mask."""

    return search_event_cutflow_masks(
        analysis_event,
        met_min=met_min,
        jet_pt_min=jet_pt_min,
        jet_met_dphi_min=jet_met_dphi_min,
        dijet_dphi_max=dijet_dphi_max,
    )["event_jetMetDphi0p5"]


def add_event_derived_fields(events):
    """Build no-muon-MET/jet angular quantities without a custom event table."""
    import awkward as ak

    good = (
        (events.Jet.pt > 30.0)
        & (abs(events.Jet.eta) < 4.5)
        & run3_tight_lepton_veto_jet_mask(events.Jet)
    )
    jets = events.Jet[good]
    order = ak.argsort(jets.pt, ascending=False)
    jets = jets[order]
    leading_pt = ak.fill_none(ak.firsts(jets.pt), -1.0)
    leading_eta = ak.fill_none(ak.firsts(jets.eta), 999.0)
    leading_tight_lep_veto = leading_pt > 0.0
    leading_phi = ak.fill_none(ak.firsts(jets.phi), 0.0)

    first, second = ak.unzip(ak.combinations(jets.phi, 2, axis=1))
    dijet_max = ak.fill_none(
        ak.max(abs(delta_phi(first, second)), axis=1, mask_identity=True), -1.0
    )
    return ak.zip(
        {
            "METNoMu_pt": events.MetNoMu.pt,
            "METNoMu_phi": events.MetNoMu.phi,
            "leadingJet_pt": leading_pt,
            "leadingJet_eta": leading_eta,
            "leadingJet_tightLepVeto": leading_tight_lep_veto,
            "leadingJet_phi": leading_phi,
            "dijetMaxDeltaPhi": dijet_max,
            "leadingJetMETNoMuDeltaPhi": abs(
                delta_phi(leading_phi, events.MetNoMu.phi)
            ),
        }
    )
