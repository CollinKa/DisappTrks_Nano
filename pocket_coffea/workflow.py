"""PocketCoffea workflow for the custom disappearing-track NanoAOD."""

from __future__ import annotations

import json
import os
from pathlib import Path

import awkward as ak
import numpy as np
from coffea.lumi_tools import LumiMask

from pocket_coffea.workflows.base import BaseProcessorABC

from disapptrks.selections import (
    add_event_derived_fields,
    add_isotrack_derived_fields,
    add_muon_derived_fields,
    base_probe_track_mask,
    build_lepton_veto_tag_probe_pairs,
    build_muon_veto_tag_probe_pairs,
    candidate_track_selection_mask,
    electron_pveto_pair_pass_mask,
    electron_tag_progression_masks,
    electron_tag_mask,
    delta_phi,
    fake_track_no_d0_mask,
    fiducial_map_probe_track_mask,
    generic_probe_pair_layer_mask,
    invariant_mass,
    isolated_track_selection_mask,
    lepton_veto_probe_track_mask,
    low_mt_mask,
    mass10_muon_probe_pair_mask,
    mass_window_pair_mask,
    muon_tag_progression_masks,
    muon_tag_mask,
    muon_pveto_pair_pass_mask,
    muon_probe_pair_layer_mask,
    muon_veto_probe_track_mask,
    muon_veto_probe_track_cutflow_masks,
    muon_veto_pair_fail_mask,
    muon_veto_pair_pass_mask,
    os_mass10_muon_probe_pair_mask,
    os_muon_probe_pair_mask,
    os_mass_window_pair_mask,
    os_z_window_muon_probe_pair_mask,
    search_event_cutflow_masks,
    search_track_cutflow_masks,
    search_track_mask,
    single_electron_trigger_mask,
    run3_tight_lepton_veto_jet_mask,
    ss_mass10_muon_probe_pair_mask,
    ss_muon_probe_pair_mask,
    ss_mass_window_pair_mask,
    ss_z_window_muon_probe_pair_mask,
    tau_pveto_pair_pass_mask,
    tau_veto_probe_track_cutflow_masks,
    tau_veto_probe_track_mask,
    z_window_muon_probe_pair_mask,
)

try:
    from disapptrks.selections import fake_track_sideband_cutflow_masks
except ImportError:

    def _fake_track_layer_mask(tracks, layer: str):
        layers = tracks.hp_nValidTrackerHits
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

    def fake_track_sideband_cutflow_masks(
        tracks,
        *,
        sideband_min: float = 0.05,
        sideband_max: float = 0.50,
    ):
        """Compatibility fallback for older installed disapptrks.selections."""

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
        masks["track_NLayers4"] = mask & _fake_track_layer_mask(tracks, "NLayers4")
        masks["track_NLayers5"] = mask & _fake_track_layer_mask(tracks, "NLayers5")
        masks["track_NLayers6plus"] = mask & _fake_track_layer_mask(tracks, "NLayers6plus")
        masks["track_combinedBins"] = mask & _fake_track_layer_mask(tracks, "combinedBins")
        return masks

PVETO_LAYERS = ("NLayers4", "NLayers5", "NLayers6plus")
ELECTRON_MASS = 0.000511
MUON_MASS = 0.105658


def _fiducial_map_path(flavor: str) -> Path | None:
    env_name = f"DISAPPTRKS_{flavor.upper()}_FIDUCIAL_MAP_JSON"
    env_path = os.environ.get(env_name)
    if env_path:
        return Path(env_path)

    env_dir = os.environ.get("DISAPPTRKS_FIDUCIAL_MAP_DIR")
    if env_dir:
        return Path(env_dir) / f"{flavor}_fiducial_map.json"
    return None


def _load_fiducial_hot_spots(flavor: str) -> tuple[dict[str, float], ...]:
    path = _fiducial_map_path(flavor)
    if path is None:
        return ()
    with path.open() as handle:
        payload = json.load(handle)
    return tuple(payload.get("hot_spots", ()))


def _outside_fiducial_hot_spots(pairs, hot_spots):
    eta = pairs.probe_eta if "probe_eta" in pairs.fields else pairs.eta
    phi = pairs.probe_phi if "probe_phi" in pairs.fields else pairs.phi
    mask = ak.ones_like(eta, dtype=bool)
    for hot_spot in hot_spots:
        deta = eta - float(hot_spot["eta"])
        dphi = np.arctan2(
            np.sin(phi - float(hot_spot["phi"])),
            np.cos(phi - float(hot_spot["phi"])),
        )
        radius = float(hot_spot["radius"])
        mask = mask & (np.sqrt(deta * deta + dphi * dphi) > radius)
    return mask
Z_MASS = 91.1876

JET_VETO_MAP_FILES = {
    "2022_preEE": "Run3-22CDSep23-Summer22-NanoAODv12_jetvetomaps.json.gz",
    "2022_postEE": "Run3-22EFGSep23-Summer22EE-NanoAODv12_jetvetomaps.json.gz",
    "2023_preBPix": "Run3-23CSep23-Summer23-NanoAODv12_jetvetomaps.json.gz",
    "2023_postBPix": "Run3-23DSep23-Summer23BPix-NanoAODv12_jetvetomaps.json.gz",
    "2024": "Run3-24CDEReprocessingFGHIPrompt-Summer24-NanoAODv15_jetvetomaps.json.gz",
    "2025": "Run3-25Prompt-Winter25-NanoAODv15_jetvetomaps.json.gz",
}

JET_VETO_MAP_FALLBACK_YEARS = {
    "2026": ("2025",),
}

GOLDEN_JSON_FILES = {
    "2022_preEE": "Cert_Collisions2022_355100_362760_Golden.json",
    "2022_postEE": "Cert_Collisions2022_355100_362760_Golden.json",
    "2023_preBPix": "Cert_Collisions2023_366442_370790_Golden.json",
    "2023_postBPix": "Cert_Collisions2023_366442_370790_Golden.json",
    "2024": "Cert_Collisions2024_378981_386951_Golden.json",
    "2025": "Cert_Collisions2025_391658_398903_Golden.json",
    "2026": "Collisions26_MLEnhancedGolden_Latest.json",
}

GOLDEN_JSON_PAYLOADS = {}


def _all_true_like(events):
    return ak.ones_like(events.HLT.IsoMu24, dtype=bool)


def _event_flag(events, name: str, *, default: bool = True):
    if "Flag" in events.fields and name in events.Flag.fields:
        return events.Flag[name]
    if name in events.fields:
        return events[name]
    return _all_true_like(events) if default else ~_all_true_like(events)


def _met_filters_mask(events):
    if "Flag" in events.fields and "METFilters" in events.Flag.fields:
        return events.Flag.METFilters
    if "METFilters" in events.fields:
        return events.METFilters

    filters = [
        "goodVertices",
        "globalSuperTightHalo2016Filter",
        "EcalDeadCellTriggerPrimitiveFilter",
        "BadPFMuonFilter",
        "BadPFMuonDzFilter",
        "hfNoisyHitsFilter",
        "eeBadScFilter",
        "ecalBadCalibFilter",
    ]
    mask = _all_true_like(events)
    for name in filters:
        mask = mask & _event_flag(events, name, default=True)
    return mask


def _met_for_transverse_mass(events):
    """Return the event-level MET collection used for lepton MT cuts.

    Some of the newer/custom NanoAOD files used in the 2024/2025 productions do
    not store the classic ``MET`` collection, but do store ``PuppiMET``.  The
    low-MT tag selection only needs an event-level ``pt`` and ``phi``, so use
    ``MET`` when available and fall back to ``PuppiMET`` otherwise.
    """
    for name in ("MET", "PuppiMET"):
        if name in events.fields:
            return events[name]
    raise AttributeError(
        "No MET-like collection found for lepton transverse-mass cuts; "
        "expected either MET or PuppiMET."
    )


def _single_muon_trigger_mask(events):
    if "HLT" in events.fields and "IsoMu24" in events.HLT.fields:
        return events.HLT.IsoMu24
    return _all_true_like(events) & False


def _met_trigger_mask(events):
    names = (
        "MET105_IsoTrk50",
        "MET120_IsoTrk50",
        "PFMET105_IsoTrk50",
        "PFMET120_PFMHT120_IDTight",
        "PFMET130_PFMHT130_IDTight",
        "PFMET140_PFMHT140_IDTight",
        "PFMET120_PFMHT120_IDTight_PFHT60",
        "PFMETNoMu110_PFMHTNoMu110_IDTight_FilterHF",
        "PFMETNoMu120_PFMHTNoMu120_IDTight_FilterHF",
        "PFMETNoMu130_PFMHTNoMu130_IDTight_FilterHF",
        "PFMETNoMu140_PFMHTNoMu140_IDTight_FilterHF",
        "PFMETNoMu120_PFMHTNoMu120_IDTight_PFHT60",
        "PFMETNoMu120_PFMHTNoMu120_IDTight",
        "PFMETNoMu130_PFMHTNoMu130_IDTight",
        "PFMETNoMu140_PFMHTNoMu140_IDTight",
        "PFMET250_HBHECleaned",
        "PFMET300_HBHECleaned",
    )
    mask = None
    for name in names:
        if "HLT" in events.fields and name in events.HLT.fields:
            mask = events.HLT[name] if mask is None else (mask | events.HLT[name])
    return mask if mask is not None else _all_true_like(events) & False


def _met_no_mu_minus_lepton(events, leptons):
    """Legacy MET-minus-one proxy: remove the selected lepton from MET-no-mu."""
    import awkward as ak

    tag = ak.firsts(leptons)
    tag_pt = ak.fill_none(tag.pt, 0.0)
    tag_phi = ak.fill_none(tag.phi, 0.0)
    met_x = events.MetNoMu.pt * np.cos(events.MetNoMu.phi) + tag_pt * np.cos(tag_phi)
    met_y = events.MetNoMu.pt * np.sin(events.MetNoMu.phi) + tag_pt * np.sin(tag_phi)
    return np.sqrt(met_x * met_x + met_y * met_y), np.arctan2(met_y, met_x)


def _leading_jet_delta_phi(events, phi):
    import awkward as ak

    good_jets = events.Jet[
        (events.Jet.pt > 30.0)
        & (abs(events.Jet.eta) < 4.5)
        & run3_tight_lepton_veto_jet_mask(events.Jet)
    ]
    leading_phi = ak.fill_none(ak.firsts(good_jets.phi), 0.0)
    return abs(delta_phi(leading_phi, phi))


def _lepton_background_track_mask(tracks, *, flavor: str, layer: str):
    mask = base_probe_track_mask(
        tracks,
        pt_min=55.0,
        layer=layer,
        apply_jet_cut=False,
        apply_calo_cut=(flavor == "muon"),
        apply_outer_hits_cut=False,
    )
    if flavor == "electron":
        mask = mask & ((tracks.dRMinElectron >= 0.0) & (tracks.dRMinElectron < 0.1))
    elif flavor == "muon":
        mask = mask & ((tracks.dRMinMuon >= 0.0) & (tracks.dRMinMuon < 0.1))
    elif flavor == "tau":
        mask = mask & ((tracks.dRMinTauHad >= 0.0) & (tracks.dRMinTauHad < 0.1))
    else:
        raise ValueError(f"unknown lepton background flavor: {flavor}")
    return mask


def _z_to_mumu_control_mask(events, muons):
    selected = muons[muon_tag_mask(muons)]
    first, second = ak.unzip(ak.combinations(selected, 2, axis=1))
    pair_mass = invariant_mass(first, second, first_mass=MUON_MASS, second_mass=MUON_MASS)
    os_z = (first.charge * second.charge < 0) & (abs(pair_mass - Z_MASS) < 10.0)
    return _single_muon_trigger_mask(events) & (ak.num(selected) == 2) & ak.any(os_z, axis=1)


def _z_to_mumu_control_diagnostics(events, muons):
    trigger = _single_muon_trigger_mask(events)
    muon_masks = muon_tag_progression_masks(muons)
    selected = muons[muon_masks["muon_selected_tag"]]
    first, second = ak.unzip(ak.combinations(selected, 2, axis=1))
    pair_mass = invariant_mass(first, second, first_mass=MUON_MASS, second_mass=MUON_MASS)
    os_z = (first.charge * second.charge < 0) & (abs(pair_mass - Z_MASS) < 10.0)
    z_os_window = trigger & (ak.num(selected) == 2) & ak.any(os_z, axis=1)
    diagnostics = {
        "event_trigger": trigger,
        "muon_pt26": trigger & (ak.num(muons[muon_masks["muon_pt26"]]) >= 2),
        "muon_eta2p1": trigger & (ak.num(muons[muon_masks["muon_eta2p1"]]) >= 2),
        "muon_tight_id": trigger & (ak.num(muons[muon_masks["muon_tight_id"]]) >= 2),
        "muon_selected_tag": trigger & (ak.num(selected) == 2),
        "z_os_window": z_os_window,
    }
    for layer in (*PVETO_LAYERS, "combinedBins"):
        diagnostics[f"sideband_{layer}"] = (
            _fake_track_count_for_control(
                events,
                z_os_window,
                layer=layer,
                d0_region="sideband",
            )
            >= 1
        )
    return diagnostics


def _z_electron_tag_mask(electrons, *, pt_min=25.0, eta_max=2.1):
    abs_sc_eta = abs(electrons.eta + electrons.deltaEtaSC)
    barrel = abs_sc_eta <= 1.479
    endcap = abs_sc_eta > 1.479
    dxy_ok = (barrel & (abs(electrons.dxy) < 0.05)) | (
        endcap & (abs(electrons.dxy) < 0.10)
    )
    dz_ok = (barrel & (abs(electrons.dz) < 0.10)) | (
        endcap & (abs(electrons.dz) < 0.20)
    )
    return (
        (electrons.pt > pt_min)
        & (abs(electrons.eta) < eta_max)
        & (electrons.cutBased >= 4)
        & dxy_ok
        & dz_ok
    )


def _z_to_ee_control_mask(events, electrons):
    selected = electrons[_z_electron_tag_mask(electrons, pt_min=25.0)]
    first, second = ak.unzip(ak.combinations(selected, 2, axis=1))
    pair_mass = invariant_mass(
        first,
        second,
        first_mass=ELECTRON_MASS,
        second_mass=ELECTRON_MASS,
    )
    os_z = (first.charge * second.charge < 0) & (abs(pair_mass - Z_MASS) < 10.0)
    return (
        single_electron_trigger_mask(events)
        & (ak.num(selected) == 2)
        & ak.any(selected.pt > 32.0, axis=1)
        & ak.any(os_z, axis=1)
    )


def _z_to_ee_control_diagnostics(events, electrons):
    trigger = single_electron_trigger_mask(events)
    abs_sc_eta = abs(electrons.eta + electrons.deltaEtaSC)
    barrel = abs_sc_eta <= 1.479
    endcap = abs_sc_eta > 1.479
    dxy_ok = (barrel & (abs(electrons.dxy) < 0.05)) | (
        endcap & (abs(electrons.dxy) < 0.10)
    )
    dz_ok = (barrel & (abs(electrons.dz) < 0.10)) | (
        endcap & (abs(electrons.dz) < 0.20)
    )
    mask = electrons.pt > 25.0
    electron_pt25 = trigger & (ak.num(electrons[mask]) >= 2)
    mask = mask & (abs(electrons.eta) < 2.1)
    electron_eta2p1 = trigger & (ak.num(electrons[mask]) >= 2)
    mask = mask & (electrons.cutBased >= 4)
    electron_tight_id = trigger & (ak.num(electrons[mask]) >= 2)
    mask = mask & dxy_ok
    electron_dxy = trigger & (ak.num(electrons[mask]) >= 2)
    mask = mask & dz_ok
    selected = electrons[mask]
    electron_dz = trigger & (ak.num(selected) >= 2)
    electron_pt32 = electron_dz & ak.any(selected.pt > 32.0, axis=1)
    first, second = ak.unzip(ak.combinations(selected, 2, axis=1))
    pair_mass = invariant_mass(
        first,
        second,
        first_mass=ELECTRON_MASS,
        second_mass=ELECTRON_MASS,
    )
    os_z = (first.charge * second.charge < 0) & (abs(pair_mass - Z_MASS) < 10.0)
    z_os_window = electron_pt32 & (ak.num(selected) == 2) & ak.any(os_z, axis=1)
    diagnostics = {
        "event_trigger": trigger,
        "electron_pt25": electron_pt25,
        "electron_eta2p1": electron_eta2p1,
        "electron_tight_id": electron_tight_id,
        "electron_dxy": electron_dxy,
        "electron_dz": electron_dz,
        "electron_pt32": electron_pt32,
        "z_os_window": z_os_window,
    }
    for layer in (*PVETO_LAYERS, "combinedBins"):
        diagnostics[f"sideband_{layer}"] = (
            _fake_track_count_for_control(
                events,
                z_os_window,
                layer=layer,
                d0_region="sideband",
            )
            >= 1
        )
    return diagnostics


def _fake_track_count_for_control(events, control_mask, *, layer, d0_region):
    count = ak.num(
        events.IsoTrack[
            fake_track_no_d0_mask(events.IsoTrack, layer=layer, d0_region=d0_region)
        ]
    )
    return ak.where(control_mask, count, 0)


def _fake_fit_tracks_for_control(events, control_mask):
    tracks = events.IsoTrack[
        fake_track_no_d0_mask(
            events.IsoTrack,
            layer="NLayers4",
            d0_region="sideband",
            sideband_min=0.0,
            sideband_max=0.5,
        )
    ]
    tracks = ak.with_field(tracks, abs(tracks.dxy), "absDxy")
    control_track_mask, _ = ak.broadcast_arrays(control_mask, tracks.pt)
    return tracks[control_track_mask]


def _jet_veto_map_parameter_year(year, era, processor_params):
    year = str(year)
    for container in (
        processor_params.jet_scale_factors.vetomaps,
        processor_params.lumi.goldenJSON,
        processor_params.event_flags,
    ):
        if year in container:
            return year

    if year == "2022":
        return "2022_preEE" if era in ("C", "D") else "2022_postEE"
    if year == "2023":
        return "2023_preBPix" if era == "C" else "2023_postBPix"
    return year


def _local_jet_veto_map_path(mapped_year):
    search_years = (str(mapped_year), *JET_VETO_MAP_FALLBACK_YEARS.get(str(mapped_year), ()))

    search_dirs = []
    env_dir = os.environ.get("DISAPPTRKS_JET_VETO_MAP_DIR")
    if env_dir:
        search_dirs.append(Path(env_dir))
    search_dirs.append(Path(__file__).resolve().parent / "data" / "jet_veto_maps")
    search_dirs.append(Path.cwd() / "data" / "jet_veto_maps")
    search_dirs.append(Path.cwd() / "jet_veto_maps")

    for directory in search_dirs:
        for search_year in search_years:
            filename = JET_VETO_MAP_FILES.get(search_year)
            candidates = [directory / search_year / "jetvetomaps.json.gz"]
            if filename is not None:
                candidates.extend(
                    [
                        directory / filename,
                        directory
                        / filename.removesuffix("_jetvetomaps.json.gz")
                        / "jetvetomaps.json.gz",
                    ]
                )
            candidates.extend(sorted(directory.glob(f"*{search_year}*jetvetomaps*.json.gz")))
            candidates.extend(sorted(directory.glob(f"*{search_year[-2:]}*jetvetomaps*.json.gz")))
            for candidate in candidates:
                if candidate.exists():
                    return candidate
    return None


def _configured_jet_veto_map_path(processor_params, mapped_year):
    for search_year in (
        str(mapped_year),
        *JET_VETO_MAP_FALLBACK_YEARS.get(str(mapped_year), ()),
    ):
        try:
            payload = processor_params.jet_scale_factors.vetomaps[search_year]["file"]
        except Exception:
            continue
        return Path(str(payload))

    return None


def _cvmfs_jet_veto_map_path(mapped_year):
    for search_year in (
        str(mapped_year),
        *JET_VETO_MAP_FALLBACK_YEARS.get(str(mapped_year), ()),
    ):
        filename = JET_VETO_MAP_FILES.get(search_year)
        if filename is None:
            continue

        period = filename.removesuffix("_jetvetomaps.json.gz")
        return (
            Path("/cvmfs/cms-griddata.cern.ch/cat/metadata")
            / "JME"
            / period
            / "latest"
            / "jetvetomaps.json.gz"
        )

    return None


def _local_golden_json_path(mapped_year):
    filename = GOLDEN_JSON_FILES.get(str(mapped_year))

    search_dirs = []
    env_dir = os.environ.get("DISAPPTRKS_GOLDEN_JSON_DIR")
    if env_dir:
        search_dirs.append(Path(env_dir))
    search_dirs.append(Path(__file__).resolve().parent / "data" / "golden_jsons")
    search_dirs.append(Path.cwd() / "data" / "golden_jsons")
    search_dirs.append(Path.cwd() / "golden_jsons")

    for directory in search_dirs:
        candidates = []
        if filename is not None:
            candidates.append(directory / filename)
        if str(mapped_year) == "2026":
            candidates.append(directory / "Collisions26_MLEnhancedGolden_Latest.json")
            candidates.extend(sorted(directory.glob("Collisions26*Golden*.json")))
        candidates.extend(sorted(directory.glob(f"Cert_Collisions{mapped_year}*_Golden.json")))
        for candidate in candidates:
            if candidate.exists():
                return candidate
    return None


def _payload_golden_json_mask(events, mapped_year):
    payload = GOLDEN_JSON_PAYLOADS.get(str(mapped_year))
    if payload is None:
        return None

    runs = ak.to_numpy(events.run)
    lumis = ak.to_numpy(events.luminosityBlock)
    mask = np.zeros(len(runs), dtype=bool)
    for run, ranges in payload.items():
        run_mask = runs == int(run)
        if not np.any(run_mask):
            continue
        run_lumis = lumis[run_mask]
        run_pass = np.zeros(len(run_lumis), dtype=bool)
        for first_lumi, last_lumi in ranges:
            run_pass |= (run_lumis >= int(first_lumi)) & (run_lumis <= int(last_lumi))
        mask[run_mask] = run_pass
    return ak.Array(mask)


def _jet_veto_map_correction_name(cset, processor_params, mapped_year):
    try:
        return processor_params.jet_scale_factors.vetomaps[str(mapped_year)]["name"]
    except Exception:
        pass

    keys = list(cset.keys())
    for key in keys:
        if "jetvetomap" in str(key).lower():
            return key
    if len(keys) == 1:
        return keys[0]
    raise KeyError(
        f"Could not determine jet-veto-map correction name for {mapped_year}; "
        f"available corrections are {keys}"
    )


def _evaluate_jet_veto_map(events, processor_params, mapped_year, payload_file):
    import correctionlib

    jets = events["Jet"]
    if "jetId" in jets.fields:
        jet_id_mask = (jets.jetId & 6) == 6
    else:
        # Only recompute if the stored NanoAOD jetId branch is absent.  Use the
        # local Run-3 TightLepVeto reconstruction rather than PocketCoffea's
        # correctionlib-backed compute_jetId helper because our Awkward-1 event
        # loop cannot call correctionlib's awkward.transform path.
        jet_id_mask = run3_tight_lepton_veto_jet_mask(jets)
    jets = ak.with_field(jets, jet_id_mask, "jetId_corrected")
    mask_for_veto_map = (
        jets["jetId_corrected"]
        & (abs(jets.eta) < 5.19)
        & (jets.pt > 15.0)
        & ((jets["neEmEF"] + jets["chEmEF"]) < 0.9)
    )
    jets = jets[mask_for_veto_map]

    cset = correctionlib.CorrectionSet.from_file(str(payload_file))
    corr = cset[_jet_veto_map_correction_name(cset, processor_params, mapped_year)]
    eta_flat = ak.to_numpy(ak.flatten(jets.eta))
    phi_flat = np.clip(ak.to_numpy(ak.flatten(jets.phi)), -3.14159, 3.14159)
    eta_counts = ak.num(jets.eta)
    weight = ak.unflatten(
        corr.evaluate("jetvetomap", eta_flat, phi_flat),
        counts=eta_counts,
    )
    event_mask = ak.sum(weight, axis=-1) == 0
    return ak.where(ak.is_none(event_mask), False, event_mask)


def _golden_json_mask(
    events,
    *,
    processor_params=None,
    year=None,
    era=None,
    sample=None,
    is_mc=False,
):
    if is_mc or processor_params is None or year is None:
        return _all_true_like(events)

    if "run" not in events.fields or "luminosityBlock" not in events.fields:
        return _all_true_like(events)

    from pocket_coffea.lib.cut_functions import apply_golden_json

    golden_json_year = _jet_veto_map_parameter_year(year, era, processor_params)
    local_json = _local_golden_json_path(golden_json_year)
    if local_json is not None:
        return LumiMask(str(local_json))(events.run, events.luminosityBlock)
    payload_mask = _payload_golden_json_mask(events, golden_json_year)
    if payload_mask is not None:
        return payload_mask

    return apply_golden_json(
        events,
        params={},
        year=golden_json_year,
        processor_params=processor_params,
        sample=sample,
        isMC=is_mc,
    )


def _jet_veto_map_mask(
    events,
    *,
    processor_params=None,
    year=None,
    era=None,
    sample=None,
    is_mc=False,
):
    # The JME jet-veto-map decision is not guaranteed to exist in central
    # NanoAOD.  Legacy DisappTrks computed and saved it as ``jetVeto2022``; some
    # custom NanoAOD productions may store the same decision under Flag.  When
    # it is absent, use PocketCoffea's correctionlib-based Run-3 jet-veto-map
    # evaluator, preferring local payloads in environments without CMS CVMFS.
    if "Flag" in events.fields and "jetVeto2022" in events.Flag.fields:
        return events.Flag.jetVeto2022
    if "jetVeto2022" in events.fields:
        return events.jetVeto2022

    if processor_params is None or year is None:
        if os.environ.get("DISAPPTRKS_ALLOW_MISSING_JET_VETO_MAP", "").lower() in (
            "1",
            "true",
            "yes",
            "on",
        ):
            return _all_true_like(events)
        raise RuntimeError("Cannot apply the JME jet-veto map without processor parameters and year.")

    veto_map_year = _jet_veto_map_parameter_year(year, era, processor_params)
    payload = _local_jet_veto_map_path(veto_map_year)
    if payload is None:
        payload = _configured_jet_veto_map_path(processor_params, veto_map_year)
    if payload is None:
        payload = _cvmfs_jet_veto_map_path(veto_map_year)
    if payload is not None:
        return _evaluate_jet_veto_map(events, processor_params, veto_map_year, payload)

    try:
        from pocket_coffea.lib.cut_functions import get_JetVetoMap_Mask

        return get_JetVetoMap_Mask(
            events,
            params={},
            year=veto_map_year,
            processor_params=processor_params,
            sample=sample,
            isMC=is_mc,
        )
    except Exception as exc:
        if os.environ.get("DISAPPTRKS_ALLOW_MISSING_JET_VETO_MAP", "").lower() in (
            "1",
            "true",
            "yes",
            "on",
        ):
            return _all_true_like(events)
        raise RuntimeError(
            "Could not apply the JME jet-veto map. Provide local payloads in "
            "pocket_coffea/data/jet_veto_maps, set DISAPPTRKS_JET_VETO_MAP_DIR, "
            "or explicitly set DISAPPTRKS_ALLOW_MISSING_JET_VETO_MAP=1 for a "
            "non-production diagnostic run."
        ) from exc


class DisappTrksProcessor(BaseProcessorABC):
    def _category_mode(self):
        try:
            return self.params.disapptrks.category_mode
        except Exception:
            return os.environ.get("DISAPPTRKS_CATEGORY_MODE", "muon_pveto")

    def _full_workflow_enabled(self):
        try:
            return bool(self.params.disapptrks.full_workflow) or bool(
                self.params.disapptrks.full_variables
            )
        except Exception:
            pass
        return (
            os.environ.get("DISAPPTRKS_FULL_WORKFLOW", "").lower()
            in ("1", "true", "yes", "on")
            or os.environ.get("DISAPPTRKS_FULL_VARIABLES", "").lower()
            in ("1", "true", "yes", "on")
        )

    def _mode_enabled(self, *modes):
        mode = self._category_mode()
        expanded_modes = {
            "muon_backgrounds": ("muon_pveto", "tau_mu_pveto", "fake_zmumu"),
            "egamma_backgrounds": ("electron_pveto", "tau_ele_pveto", "fake_zee"),
        }.get(mode, (mode,))
        return (
            self._full_workflow_enabled()
            or mode == "all"
            or any(requested in expanded_modes for requested in modes)
        )

    def _fiducial_hot_spots(self, flavor):
        if not hasattr(self, "_disapptrks_fiducial_hot_spots"):
            self._disapptrks_fiducial_hot_spots = {}
        if flavor not in self._disapptrks_fiducial_hot_spots:
            self._disapptrks_fiducial_hot_spots[flavor] = _load_fiducial_hot_spots(
                flavor
            )
        return self._disapptrks_fiducial_hot_spots[flavor]

    def _lepton_fiducial_hot_spots(self):
        return self._fiducial_hot_spots("electron") + self._fiducial_hot_spots("muon")

    def _apply_lepton_fiducial_maps_to_track_cutflow(self, masks):
        fiducial_hot_spots = self._lepton_fiducial_hot_spots()
        fiducial_map_mask = None
        if fiducial_hot_spots:
            fiducial_map_mask = _outside_fiducial_hot_spots(
                self.events.IsoTrack,
                fiducial_hot_spots,
            )
        updated = {}
        after_fiducial_row = False
        for name, mask in masks.items():
            if name == "track_fiducialECAL":
                after_fiducial_row = True
            output_name = (
                "track_fiducialSelections"
                if name == "track_fiducialECAL"
                else name
            )
            updated[output_name] = (
                mask & fiducial_map_mask
                if after_fiducial_row and fiducial_map_mask is not None
                else mask
            )
        return updated

    def _search_diagnostics_enabled(self):
        try:
            return bool(self.params.disapptrks.search_diagnostics)
        except Exception:
            pass
        return os.environ.get("DISAPPTRKS_ENABLE_SEARCH_DIAGNOSTICS", "").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

    def _fake_track_controls(self):
        try:
            mode = str(self.params.disapptrks.fake_track_control).lower()
        except Exception:
            mode = os.environ.get("DISAPPTRKS_FAKE_TRACK_CONTROL", "basic").lower()
        if mode in ("jetmet", "basic_selection"):
            mode = "basic"
        return {
            "basic": ("basic",),
            "zmumu": ("zmumu",),
            "zee": ("zee",),
        }.get(mode, ("basic",))

    def _store_lepton_pveto_pairs(
        self,
        *,
        prefix,
        tags,
        probes,
        tag_mass,
        probe_mass,
        window_low,
        window_high,
        pass_mask_function,
        fiducial_hot_spots=(),
    ):
        pairs = build_lepton_veto_tag_probe_pairs(
            tags,
            probes,
            tag_mass=tag_mass,
            probe_mass=probe_mass,
        )
        pveto_pass_mask = pass_mask_function(pairs) & _outside_fiducial_hot_spots(
            pairs, fiducial_hot_spots
        )
        self.events[f"{prefix}TagProbePair"] = pairs
        self.events[f"{prefix}TagProbePairMassWindow"] = pairs[
            mass_window_pair_mask(pairs, window_low, window_high)
        ]
        self.events[f"{prefix}TagProbePairOSMassWindow"] = pairs[
            os_mass_window_pair_mask(pairs, window_low, window_high)
        ]
        self.events[f"{prefix}TagProbePairSSMassWindow"] = pairs[
            ss_mass_window_pair_mask(pairs, window_low, window_high)
        ]
        self.events[f"{prefix}PVetoTagProbePairMassWindowPass"] = pairs[
            os_mass_window_pair_mask(pairs, window_low, window_high)
            & pveto_pass_mask
        ]
        self.events[f"{prefix}PVetoTagProbePairSSMassWindowPass"] = pairs[
            ss_mass_window_pair_mask(pairs, window_low, window_high)
            & pveto_pass_mask
        ]
        for layer in PVETO_LAYERS:
            layer_mask = generic_probe_pair_layer_mask(pairs, layer)
            self.events[f"{prefix}TagProbePairMassWindow_{layer}"] = pairs[
                os_mass_window_pair_mask(pairs, window_low, window_high) & layer_mask
            ]
            self.events[f"{prefix}PVetoTagProbePairMassWindowPass_{layer}"] = pairs[
                os_mass_window_pair_mask(pairs, window_low, window_high)
                & layer_mask
                & pveto_pass_mask
            ]
            self.events[f"{prefix}TagProbePairSSMassWindow_{layer}"] = pairs[
                ss_mass_window_pair_mask(pairs, window_low, window_high) & layer_mask
            ]
            self.events[f"{prefix}PVetoTagProbePairSSMassWindowPass_{layer}"] = pairs[
                ss_mass_window_pair_mask(pairs, window_low, window_high)
                & layer_mask
                & pveto_pass_mask
            ]

    def _store_lepton_background_controls(
        self,
        *,
        prefix,
        flavor,
        tags,
        event_quality,
        met_cut=120.0,
        phi_cut=0.5,
    ):
        met_pt, met_phi = _met_no_mu_minus_lepton(self.events, tags)
        offline_event = (
            event_quality
            & (ak.num(tags) >= 1)
            & (met_pt >= met_cut)
            & (_leading_jet_delta_phi(self.events, met_phi) >= phi_cut)
        )
        trigger_event = offline_event & _met_trigger_mask(self.events)
        for layer in (*PVETO_LAYERS, "combinedBins"):
            track_mask = _lepton_background_track_mask(
                self.events.IsoTrack,
                flavor=flavor,
                layer=layer,
            )
            has_track = ak.num(self.events.IsoTrack[track_mask]) >= 1
            self.events[f"n{prefix}BackgroundControl_{layer}"] = ak.values_astype(
                event_quality & (ak.num(tags) >= 1) & has_track,
                np.int64,
            )
            self.events[f"n{prefix}BackgroundOffline_{layer}"] = ak.values_astype(
                offline_event & has_track,
                np.int64,
            )
            self.events[f"n{prefix}BackgroundTrigger_{layer}"] = ak.values_astype(
                trigger_event & has_track,
                np.int64,
            )

    def _store_fiducial_map_pairs(self):
        """Store before/after probe pairs for electron and muon fiducial maps.

        Legacy fiducial maps used Z tag-and-probe selections with the
        electron/muon fiducial-map vetoes removed.  The "after" channels then
        added the measured loose lepton veto.  In Nano we keep the same
        before/after pair topology and histogram the probe eta-phi coordinates.
        """

        if "Muon" in self.events.fields:
            self.events["MuonFiducialTag"] = self.events.Muon[
                muon_tag_mask(self.events.Muon)
            ]
            muon_probes = self.events.IsoTrack[
                fiducial_map_probe_track_mask(self.events.IsoTrack, flavor="muon")
            ]
            muon_pairs = build_muon_veto_tag_probe_pairs(
                self.events.MuonFiducialTag,
                muon_probes,
            )
            muon_z = os_z_window_muon_probe_pair_mask(muon_pairs)
            self.events["MuonFiducialBefore"] = muon_pairs[muon_z]
            self.events["MuonFiducialAfter"] = muon_pairs[
                muon_z & muon_pairs.probe_passLooseMuonVeto
            ]

        if "Electron" in self.events.fields:
            self.events["ElectronFiducialTag"] = self.events.Electron[
                electron_tag_mask(self.events.Electron, self.events)
            ]
            electron_probes = self.events.IsoTrack[
                fiducial_map_probe_track_mask(self.events.IsoTrack, flavor="electron")
            ]
            electron_pairs = build_lepton_veto_tag_probe_pairs(
                self.events.ElectronFiducialTag,
                electron_probes,
                tag_mass=ELECTRON_MASS,
                probe_mass=ELECTRON_MASS,
            )
            electron_z = os_mass_window_pair_mask(
                electron_pairs,
                91.1876 - 10.0,
                91.1876 + 10.0,
            )
            self.events["ElectronFiducialBefore"] = electron_pairs[electron_z]
            self.events["ElectronFiducialAfter"] = electron_pairs[
                electron_z & electron_pairs.probe_passVetoElectronVeto
            ]

    def apply_object_preselection(self, variation):
        if (
            self._mode_enabled("muon_pveto", "tau_mu_pveto", "fake_zmumu", "fiducial_maps")
            or self._category_mode() == "fake_tracks"
        ):
            self.events["Muon"] = add_muon_derived_fields(self.events)
        self.events["IsoTrack"] = add_isotrack_derived_fields(self.events)
        self.events["AnalysisEvent"] = add_event_derived_fields(self.events)
        self.events["IsoTrackProbe"] = self.events.IsoTrack[
            base_probe_track_mask(self.events.IsoTrack)
        ]
        self.events["IsoTrackIsolated"] = self.events.IsoTrack[
            isolated_track_selection_mask(self.events.IsoTrack)
        ]
        self.events["IsoTrackCandidate"] = self.events.IsoTrack[
            candidate_track_selection_mask(self.events.IsoTrack)
        ]

        tag_met = None
        if self._mode_enabled("muon_pveto"):
            self.events["MuonTag"] = self.events.Muon[muon_tag_mask(self.events.Muon)]
            self.events["MuonVetoProbeTrack"] = self.events.IsoTrack[
                muon_veto_probe_track_mask(self.events.IsoTrack)
            ]
            muon_veto_pairs = build_muon_veto_tag_probe_pairs(
                self.events.MuonTag, self.events.MuonVetoProbeTrack
            )
            muon_pveto_pass_mask = muon_pveto_pair_pass_mask(
                muon_veto_pairs
            ) & _outside_fiducial_hot_spots(
                muon_veto_pairs,
                self._lepton_fiducial_hot_spots(),
            )
            self.events["MuonVetoTagProbePair"] = muon_veto_pairs
            self.events["MuonVetoTagProbePairOS"] = muon_veto_pairs[
                os_muon_probe_pair_mask(muon_veto_pairs)
            ]
            self.events["MuonVetoTagProbePairMass10"] = muon_veto_pairs[
                mass10_muon_probe_pair_mask(muon_veto_pairs)
            ]
            self.events["MuonVetoTagProbePairOSMass10"] = muon_veto_pairs[
                os_mass10_muon_probe_pair_mask(muon_veto_pairs)
            ]
            self.events["MuonVetoTagProbePairSS"] = muon_veto_pairs[
                ss_muon_probe_pair_mask(muon_veto_pairs)
            ]
            self.events["MuonVetoTagProbePairSSMass10"] = muon_veto_pairs[
                ss_mass10_muon_probe_pair_mask(muon_veto_pairs)
            ]
            self.events["MuonVetoTagProbePairZWindow"] = muon_veto_pairs[
                z_window_muon_probe_pair_mask(muon_veto_pairs)
            ]
            self.events["MuonVetoTagProbePairOSZWindow"] = muon_veto_pairs[
                os_z_window_muon_probe_pair_mask(muon_veto_pairs)
            ]
            self.events["MuonVetoTagProbePairZWindowPass"] = muon_veto_pairs[
                os_z_window_muon_probe_pair_mask(muon_veto_pairs)
                & muon_veto_pair_pass_mask(muon_veto_pairs)
            ]
            self.events["MuonVetoTagProbePairZWindowFail"] = muon_veto_pairs[
                os_z_window_muon_probe_pair_mask(muon_veto_pairs)
                & muon_veto_pair_fail_mask(muon_veto_pairs)
            ]
            self.events["MuonPVetoTagProbePairZWindowPass"] = muon_veto_pairs[
                os_z_window_muon_probe_pair_mask(muon_veto_pairs)
                & muon_pveto_pass_mask
            ]
            self.events["MuonVetoTagProbePairSSZWindow"] = muon_veto_pairs[
                ss_z_window_muon_probe_pair_mask(muon_veto_pairs)
            ]
            self.events["MuonVetoTagProbePairSSZWindowPass"] = muon_veto_pairs[
                ss_z_window_muon_probe_pair_mask(muon_veto_pairs)
                & muon_veto_pair_pass_mask(muon_veto_pairs)
            ]
            self.events["MuonVetoTagProbePairSSZWindowFail"] = muon_veto_pairs[
                ss_z_window_muon_probe_pair_mask(muon_veto_pairs)
                & muon_veto_pair_fail_mask(muon_veto_pairs)
            ]
            self.events["MuonPVetoTagProbePairSSZWindowPass"] = muon_veto_pairs[
                ss_z_window_muon_probe_pair_mask(muon_veto_pairs)
                & muon_pveto_pass_mask
            ]
            for layer in PVETO_LAYERS:
                layer_mask = muon_probe_pair_layer_mask(muon_veto_pairs, layer)
                self.events[f"MuonVetoTagProbePairZWindow_{layer}"] = muon_veto_pairs[
                    os_z_window_muon_probe_pair_mask(muon_veto_pairs) & layer_mask
                ]
                self.events[f"MuonPVetoTagProbePairZWindowPass_{layer}"] = muon_veto_pairs[
                    os_z_window_muon_probe_pair_mask(muon_veto_pairs)
                    & layer_mask
                    & muon_pveto_pass_mask
                ]
                self.events[f"MuonVetoTagProbePairSSZWindow_{layer}"] = muon_veto_pairs[
                    ss_z_window_muon_probe_pair_mask(muon_veto_pairs) & layer_mask
                ]
                self.events[f"MuonPVetoTagProbePairSSZWindowPass_{layer}"] = muon_veto_pairs[
                    ss_z_window_muon_probe_pair_mask(muon_veto_pairs)
                    & layer_mask
                    & muon_pveto_pass_mask
                ]

        if self._mode_enabled("electron_pveto"):
            self.events["ElectronTag"] = self.events.Electron[
                electron_tag_mask(self.events.Electron, self.events)
            ]
            self.events["ElectronVetoProbeTrack"] = self.events.IsoTrack[
                lepton_veto_probe_track_mask(self.events.IsoTrack, measured_veto="electron")
            ]
            self._store_lepton_pveto_pairs(
                prefix="Electron",
                tags=self.events.ElectronTag,
                probes=self.events.ElectronVetoProbeTrack,
                tag_mass=ELECTRON_MASS,
                probe_mass=ELECTRON_MASS,
                window_low=91.1876 - 10.0,
                window_high=91.1876 + 10.0,
                pass_mask_function=electron_pveto_pair_pass_mask,
                fiducial_hot_spots=self._lepton_fiducial_hot_spots(),
            )

        if self._mode_enabled("tau_mu_pveto", "tau_ele_pveto"):
            tag_met = _met_for_transverse_mass(self.events)
            self.events["TauVetoProbeTrack"] = self.events.IsoTrack[
                tau_veto_probe_track_mask(self.events.IsoTrack)
            ]

        if self._mode_enabled("tau_mu_pveto"):
            tau_mu_tag_mask = muon_tag_progression_masks(self.events.Muon)[
                "muon_selected_tag"
            ]
            self.events["MuonLowMTTag"] = self.events.Muon[tau_mu_tag_mask][
                low_mt_mask(self.events.Muon[tau_mu_tag_mask], tag_met)
            ]
            self._store_lepton_pveto_pairs(
                prefix="TauMu",
                tags=self.events.MuonLowMTTag,
                probes=self.events.TauVetoProbeTrack,
                tag_mass=MUON_MASS,
                probe_mass=MUON_MASS,
                window_low=91.1876 - 50.0,
                window_high=91.1876 - 15.0,
                pass_mask_function=tau_pveto_pair_pass_mask,
            )

        if self._mode_enabled("tau_ele_pveto"):
            tau_ele_tag_mask = _z_electron_tag_mask(self.events.Electron, pt_min=32.0)
            self.events["ElectronLowMTTag"] = self.events.Electron[tau_ele_tag_mask][
                low_mt_mask(self.events.Electron[tau_ele_tag_mask], tag_met)
            ]
            self._store_lepton_pveto_pairs(
                prefix="TauEle",
                tags=self.events.ElectronLowMTTag,
                probes=self.events.TauVetoProbeTrack,
                tag_mass=ELECTRON_MASS,
                probe_mass=ELECTRON_MASS,
                window_low=91.1876 - 50.0,
                window_high=91.1876 - 15.0,
                pass_mask_function=tau_pveto_pair_pass_mask,
            )

        if self._mode_enabled("muon_pveto", "electron_pveto", "tau_mu_pveto", "tau_ele_pveto"):
            event_quality = (
                _golden_json_mask(
                    self.events,
                    processor_params=self.params,
                    year=self._year,
                    era=self._era,
                    sample=self._sample,
                    is_mc=self._isMC,
                )
                & _met_filters_mask(self.events)
                & _jet_veto_map_mask(
                    self.events,
                    processor_params=self.params,
                    year=self._year,
                    era=self._era,
                    sample=self._sample,
                    is_mc=self._isMC,
                )
            )
            if self._mode_enabled("muon_pveto") and "MuonTag" in self.events.fields:
                self._store_lepton_background_controls(
                    prefix="Muon",
                    flavor="muon",
                    tags=self.events.MuonTag,
                    event_quality=event_quality,
                )
            if self._mode_enabled("electron_pveto") and "ElectronTag" in self.events.fields:
                self._store_lepton_background_controls(
                    prefix="Electron",
                    flavor="electron",
                    tags=self.events.ElectronTag,
                    event_quality=event_quality,
                )
            # The Run-3 tau estimate is measured with muon/electron tau-control
            # legs in this Nano workflow, matching the existing tau Pveto split.
            if self._mode_enabled("tau_mu_pveto") and "MuonLowMTTag" in self.events.fields:
                self._store_lepton_background_controls(
                    prefix="TauMu",
                    flavor="muon",
                    tags=self.events.MuonLowMTTag,
                    event_quality=event_quality,
                )
            if self._mode_enabled("tau_ele_pveto") and "ElectronLowMTTag" in self.events.fields:
                self._store_lepton_background_controls(
                    prefix="TauEle",
                    flavor="electron",
                    tags=self.events.ElectronLowMTTag,
                    event_quality=event_quality,
                )

        if self._mode_enabled("fiducial_maps"):
            self._store_fiducial_map_pairs()

        search_diagnostic_masks = search_track_cutflow_masks(self.events.IsoTrack)
        self.events["IsoTrackSearchPreMissingOuter"] = self.events.IsoTrack[
            search_diagnostic_masks["track_calo10"]
        ]
        self.events["IsoTrackSearchPreLeptonVeto"] = self.events.IsoTrack[
            search_diagnostic_masks["track_missingOuter3"]
        ]
        self.events["IsoTrackSearch"] = self.events.IsoTrack[
            search_track_mask(self.events.IsoTrack)
        ]

    def _count_lepton_pair_fields(self, prefix):
        self.events[f"n{prefix}TagProbePair"] = ak.num(
            self.events[f"{prefix}TagProbePair"]
        )
        self.events[f"n{prefix}TagProbePairMassWindow"] = ak.num(
            self.events[f"{prefix}TagProbePairMassWindow"]
        )
        self.events[f"n{prefix}TagProbePairOSMassWindow"] = ak.num(
            self.events[f"{prefix}TagProbePairOSMassWindow"]
        )
        self.events[f"n{prefix}TagProbePairSSMassWindow"] = ak.num(
            self.events[f"{prefix}TagProbePairSSMassWindow"]
        )
        self.events[f"n{prefix}PVetoTagProbePairMassWindowPass"] = ak.num(
            self.events[f"{prefix}PVetoTagProbePairMassWindowPass"]
        )
        self.events[f"n{prefix}PVetoTagProbePairSSMassWindowPass"] = ak.num(
            self.events[f"{prefix}PVetoTagProbePairSSMassWindowPass"]
        )
        for layer in PVETO_LAYERS:
            self.events[f"n{prefix}TagProbePairMassWindow_{layer}"] = ak.num(
                self.events[f"{prefix}TagProbePairMassWindow_{layer}"]
            )
            self.events[f"n{prefix}PVetoTagProbePairMassWindowPass_{layer}"] = ak.num(
                self.events[f"{prefix}PVetoTagProbePairMassWindowPass_{layer}"]
            )
            self.events[f"n{prefix}TagProbePairSSMassWindow_{layer}"] = ak.num(
                self.events[f"{prefix}TagProbePairSSMassWindow_{layer}"]
            )
            self.events[f"n{prefix}PVetoTagProbePairSSMassWindowPass_{layer}"] = ak.num(
                self.events[f"{prefix}PVetoTagProbePairSSMassWindowPass_{layer}"]
            )

    def _count_common_search_fields(self):
        self.events["nIsoTrack"] = ak.num(self.events.IsoTrack)
        self.events["nIsoTrackProbe"] = ak.num(self.events.IsoTrackProbe)
        self.events["nIsoTrackIsolated"] = ak.num(self.events.IsoTrackIsolated)
        self.events["nIsoTrackCandidate"] = ak.num(self.events.IsoTrackCandidate)
        self.events["nIsoTrackSearchPreMissingOuter"] = ak.num(
            self.events.IsoTrackSearchPreMissingOuter
        )
        self.events["nIsoTrackSearchPreLeptonVeto"] = ak.num(
            self.events.IsoTrackSearchPreLeptonVeto
        )
        self.events["nIsoTrackSearch"] = ak.num(self.events.IsoTrackSearch)

    def _count_muon_pveto_fields(self):
        self.events["nMuonTag"] = ak.num(self.events.MuonTag)
        self.events["nMuonVetoProbeTrack"] = ak.num(self.events.MuonVetoProbeTrack)
        self.events["nMuonVetoTagProbePair"] = ak.num(
            self.events.MuonVetoTagProbePair
        )
        self.events["nMuonVetoTagProbePairOS"] = ak.num(
            self.events.MuonVetoTagProbePairOS
        )
        self.events["nMuonVetoTagProbePairMass10"] = ak.num(
            self.events.MuonVetoTagProbePairMass10
        )
        self.events["nMuonVetoTagProbePairOSMass10"] = ak.num(
            self.events.MuonVetoTagProbePairOSMass10
        )
        self.events["nMuonVetoTagProbePairSS"] = ak.num(
            self.events.MuonVetoTagProbePairSS
        )
        self.events["nMuonVetoTagProbePairSSMass10"] = ak.num(
            self.events.MuonVetoTagProbePairSSMass10
        )
        self.events["nMuonVetoTagProbePairZWindow"] = ak.num(
            self.events.MuonVetoTagProbePairZWindow
        )
        self.events["nMuonVetoTagProbePairOSZWindow"] = ak.num(
            self.events.MuonVetoTagProbePairOSZWindow
        )
        self.events["nMuonVetoTagProbePairZWindowPass"] = ak.num(
            self.events.MuonVetoTagProbePairZWindowPass
        )
        self.events["nMuonVetoTagProbePairZWindowFail"] = ak.num(
            self.events.MuonVetoTagProbePairZWindowFail
        )
        self.events["nMuonPVetoTagProbePairZWindowPass"] = ak.num(
            self.events.MuonPVetoTagProbePairZWindowPass
        )
        self.events["nMuonVetoTagProbePairSSZWindow"] = ak.num(
            self.events.MuonVetoTagProbePairSSZWindow
        )
        self.events["nMuonVetoTagProbePairSSZWindowPass"] = ak.num(
            self.events.MuonVetoTagProbePairSSZWindowPass
        )
        self.events["nMuonVetoTagProbePairSSZWindowFail"] = ak.num(
            self.events.MuonVetoTagProbePairSSZWindowFail
        )
        self.events["nMuonPVetoTagProbePairSSZWindowPass"] = ak.num(
            self.events.MuonPVetoTagProbePairSSZWindowPass
        )
        for layer in PVETO_LAYERS:
            self.events[f"nMuonVetoTagProbePairZWindow_{layer}"] = ak.num(
                self.events[f"MuonVetoTagProbePairZWindow_{layer}"]
            )
            self.events[f"nMuonPVetoTagProbePairZWindowPass_{layer}"] = ak.num(
                self.events[f"MuonPVetoTagProbePairZWindowPass_{layer}"]
            )
            self.events[f"nMuonVetoTagProbePairSSZWindow_{layer}"] = ak.num(
                self.events[f"MuonVetoTagProbePairSSZWindow_{layer}"]
            )
            self.events[f"nMuonPVetoTagProbePairSSZWindowPass_{layer}"] = ak.num(
                self.events[f"MuonPVetoTagProbePairSSZWindowPass_{layer}"]
            )

    def _count_fake_track_fields(self, *, controls=("basic", "zmumu", "zee")):
        fake_basic3hits_d0_signal = self.events.IsoTrack[
            fake_track_no_d0_mask(
                self.events.IsoTrack,
                layer="NLayers4",
                d0_region="signal",
            )
        ]
        fake_basic3hits_d0_sideband = self.events.IsoTrack[
            fake_track_no_d0_mask(
                self.events.IsoTrack,
                layer="NLayers4",
                d0_region="sideband",
            )
        ]

        if "basic" in controls:
            self.events["nFakeBasic3HitsD0Signal"] = ak.num(fake_basic3hits_d0_signal)
            self.events["nFakeBasic3HitsD0Sideband"] = ak.num(fake_basic3hits_d0_sideband)
            for layer in (*PVETO_LAYERS, "combinedBins"):
                self.events[f"nFakeControl_{layer}"] = ak.num(
                    self.events.IsoTrack[
                        fake_track_no_d0_mask(
                            self.events.IsoTrack,
                            layer=layer,
                            d0_region="sideband",
                        )
                    ]
                )

        if "zmumu" in controls:
            fake_zmumu_control = _z_to_mumu_control_mask(self.events, self.events.Muon)
            self.events["nFakeZMuMuControl"] = ak.values_astype(
                fake_zmumu_control, np.int64
            )
            self.events["FakeZMuMuDiag"] = ak.zip(
                _z_to_mumu_control_diagnostics(self.events, self.events.Muon)
            )
            self.events["FakeZMuMuFitTrack"] = _fake_fit_tracks_for_control(
                self.events,
                fake_zmumu_control,
            )
            for layer in (*PVETO_LAYERS, "combinedBins"):
                self.events[f"nFakeZMuMuSideband_{layer}"] = _fake_track_count_for_control(
                    self.events,
                    fake_zmumu_control,
                    layer=layer,
                    d0_region="sideband",
                )

        if "zee" in controls:
            fake_zee_control = _z_to_ee_control_mask(self.events, self.events.Electron)
            self.events["nFakeZeeControl"] = ak.values_astype(fake_zee_control, np.int64)
            self.events["FakeZeeDiag"] = ak.zip(
                _z_to_ee_control_diagnostics(self.events, self.events.Electron)
            )
            self.events["FakeZeeFitTrack"] = _fake_fit_tracks_for_control(
                self.events,
                fake_zee_control,
            )
            for layer in (*PVETO_LAYERS, "combinedBins"):
                self.events[f"nFakeZeeSideband_{layer}"] = _fake_track_count_for_control(
                    self.events,
                    fake_zee_control,
                    layer=layer,
                    d0_region="sideband",
                )

    def _event_quality_masks(self):
        event_golden_json = _golden_json_mask(
            self.events,
            processor_params=self.params,
            year=self._year,
            era=self._era,
            sample=self._sample,
            is_mc=self._isMC,
        )
        event_met_filters = _met_filters_mask(self.events)
        event_jet_veto_map = _jet_veto_map_mask(
            self.events,
            processor_params=self.params,
            year=self._year,
            era=self._era,
            sample=self._sample,
            is_mc=self._isMC,
        )
        return event_golden_json, event_met_filters, event_jet_veto_map

    def _store_muon_pveto_diagnostics(self):
        event_golden_json, met_filters, jet_veto_map_mask = self._event_quality_masks()
        event_singlemu_trigger = event_golden_json & self.events.HLT.IsoMu24
        event_met_filters = event_singlemu_trigger & met_filters
        event_jet_veto_map = event_met_filters & jet_veto_map_mask
        muon_table16_diagnostics = {
            "event_singlemu_trigger": event_singlemu_trigger,
            "event_met_filters": event_met_filters,
            "event_jet_veto_map": event_jet_veto_map,
        }
        muon_tag_masks = muon_tag_progression_masks(self.events.Muon)
        for name, mask in muon_tag_masks.items():
            self.events[f"n{name[0].upper()}{name[1:]}"] = ak.num(
                self.events.Muon[mask]
            )
            muon_table16_diagnostics[name] = (
                event_jet_veto_map
                & (self.events[f"n{name[0].upper()}{name[1:]}"] >= 1)
            )

        has_selected_muon_tag = muon_table16_diagnostics["muon_selected_tag"]
        table16_track_masks = self._apply_lepton_fiducial_maps_to_track_cutflow(
            muon_veto_probe_track_cutflow_masks(self.events.IsoTrack)
        )
        pre_pair_track_fields = {
            "track_pt30",
            "track_eta2p1",
            "track_noDTWheelGap",
            "track_noECALCrack",
            "track_noCSCTransition",
            "track_fiducialSelections",
            "track_dzOrLambda",
            "track_pixelHits4",
            "track_noMissingInner",
            "track_noMissingMiddle",
            "track_chargedIso0p05",
            "track_dxy0p02",
            "track_dz0p5",
            "track_dRJet0p5",
        }
        for name, mask in table16_track_masks.items():
            self.events[f"n{name[0].upper()}{name[1:]}Table16"] = ak.num(
                self.events.IsoTrack[mask]
            )
            if name in pre_pair_track_fields:
                muon_table16_diagnostics[name] = (
                    has_selected_muon_tag
                    & (self.events[f"n{name[0].upper()}{name[1:]}Table16"] >= 1)
                )

        table16_mass_probe_tracks = self.events.IsoTrack[
            table16_track_masks["track_dRJet0p5"]
        ]
        table16_mass_pairs = build_muon_veto_tag_probe_pairs(
            self.events.MuonTag, table16_mass_probe_tracks
        )
        table16_electron_probe_tracks = self.events.IsoTrack[
            table16_track_masks["track_electronVeto"]
        ]
        table16_electron_pairs = build_muon_veto_tag_probe_pairs(
            self.events.MuonTag, table16_electron_probe_tracks
        )
        table16_tau_probe_tracks = self.events.IsoTrack[
            table16_track_masks["track_tauVeto"]
        ]
        table16_tau_pairs = build_muon_veto_tag_probe_pairs(
            self.events.MuonTag, table16_tau_probe_tracks
        )
        table16_probe_tracks = self.events.IsoTrack[table16_track_masks["track_calo10"]]
        table16_pairs = build_muon_veto_tag_probe_pairs(
            self.events.MuonTag, table16_probe_tracks
        )
        muon_table16_diagnostics.update(
            {
                "pair_mass10": ak.num(
                    table16_mass_pairs[
                        mass10_muon_probe_pair_mask(table16_mass_pairs)
                    ]
                )
                >= 1,
                "track_electronVeto": ak.num(
                    table16_electron_pairs[
                        mass10_muon_probe_pair_mask(table16_electron_pairs)
                    ]
                )
                >= 1,
                "track_tauVeto": ak.num(
                    table16_tau_pairs[
                        mass10_muon_probe_pair_mask(table16_tau_pairs)
                    ]
                )
                >= 1,
                "track_calo10": ak.num(
                    table16_pairs[
                        mass10_muon_probe_pair_mask(table16_pairs)
                    ]
                )
                >= 1,
                "pair_zwindow": ak.num(
                    table16_pairs[z_window_muon_probe_pair_mask(table16_pairs)]
                )
                >= 1,
                "pair_os": ak.num(
                    table16_pairs[os_z_window_muon_probe_pair_mask(table16_pairs)]
                )
                >= 1,
            }
        )
        muon_table16_diagnostics["track_probe_before_layer"] = (
            muon_table16_diagnostics["track_calo10"]
        )
        muon_table16_diagnostics["layer_combinedBins"] = ak.num(
            table16_pairs[
                os_z_window_muon_probe_pair_mask(table16_pairs)
                & muon_probe_pair_layer_mask(table16_pairs, "combinedBins")
            ]
        ) >= 1
        self.events["MuonTable16Diag"] = ak.zip(muon_table16_diagnostics)

    def _store_electron_pveto_diagnostics(self):
        event_golden_json, met_filters, jet_veto_map_mask = self._event_quality_masks()
        event_singleele_trigger = event_golden_json & single_electron_trigger_mask(
            self.events
        )
        event_ele_met_filters = event_singleele_trigger & met_filters
        event_ele_jet_veto_map = event_ele_met_filters & jet_veto_map_mask
        electron_pveto_diagnostics = {
            "event_singleele_trigger": event_singleele_trigger,
            "event_met_filters": event_ele_met_filters,
            "event_jet_veto_map": event_ele_jet_veto_map,
        }
        electron_tag_masks = electron_tag_progression_masks(
            self.events.Electron, self.events
        )
        for name, mask in electron_tag_masks.items():
            self.events[f"n{name[0].upper()}{name[1:]}"] = ak.num(
                self.events.Electron[mask]
            )
            electron_pveto_diagnostics[name] = (
                event_ele_jet_veto_map
                & (self.events[f"n{name[0].upper()}{name[1:]}"] >= 1)
            )

        has_selected_electron_tag = electron_pveto_diagnostics[
            "electron_selected_tag"
        ]
        electron_track_masks = self._apply_lepton_fiducial_maps_to_track_cutflow(
            muon_veto_probe_track_cutflow_masks(self.events.IsoTrack)
        )
        for name in (
            "track_pt30",
            "track_eta2p1",
            "track_noDTWheelGap",
            "track_noECALCrack",
            "track_noCSCTransition",
            "track_fiducialSelections",
            "track_dzOrLambda",
            "track_pixelHits4",
            "track_noMissingInner",
            "track_noMissingMiddle",
            "track_chargedIso0p05",
            "track_dxy0p02",
            "track_dz0p5",
            "track_dRJet0p5",
        ):
            n_name = f"n{name[0].upper()}{name[1:]}ElectronPVeto"
            self.events[n_name] = ak.num(self.events.IsoTrack[electron_track_masks[name]])
            electron_pveto_diagnostics[name] = (
                has_selected_electron_tag & (self.events[n_name] >= 1)
            )

        electron_mass_probe_tracks = self.events.IsoTrack[
            electron_track_masks["track_dRJet0p5"]
        ]
        electron_mass_pairs = build_lepton_veto_tag_probe_pairs(
            self.events.ElectronTag,
            electron_mass_probe_tracks,
            tag_mass=ELECTRON_MASS,
            probe_mass=ELECTRON_MASS,
        )
        electron_muon_veto_mask = electron_track_masks["track_dRJet0p5"] & (
            (self.events.IsoTrack.dRMinMuon < 0.0)
            | (self.events.IsoTrack.dRMinMuon > 0.15)
        )
        electron_tau_veto_mask = electron_muon_veto_mask & (
            (self.events.IsoTrack.dRMinTauHad < 0.0)
            | (self.events.IsoTrack.dRMinTauHad > 0.15)
        )
        electron_calo_mask = electron_tau_veto_mask & (
            self.events.IsoTrack.caloEnergy < 10.0
        )
        electron_muon_veto_pairs = build_lepton_veto_tag_probe_pairs(
            self.events.ElectronTag,
            self.events.IsoTrack[electron_muon_veto_mask],
            tag_mass=ELECTRON_MASS,
            probe_mass=ELECTRON_MASS,
        )
        electron_tau_veto_pairs = build_lepton_veto_tag_probe_pairs(
            self.events.ElectronTag,
            self.events.IsoTrack[electron_tau_veto_mask],
            tag_mass=ELECTRON_MASS,
            probe_mass=ELECTRON_MASS,
        )
        electron_pairs = build_lepton_veto_tag_probe_pairs(
            self.events.ElectronTag,
            self.events.IsoTrack[electron_tau_veto_mask],
            tag_mass=ELECTRON_MASS,
            probe_mass=ELECTRON_MASS,
        )
        electron_calo_pairs = build_lepton_veto_tag_probe_pairs(
            self.events.ElectronTag,
            self.events.IsoTrack[electron_calo_mask],
            tag_mass=ELECTRON_MASS,
            probe_mass=ELECTRON_MASS,
        )
        electron_z_window = mass_window_pair_mask(
            electron_pairs, 91.1876 - 10.0, 91.1876 + 10.0
        )
        electron_os_z_window = os_mass_window_pair_mask(
            electron_pairs, 91.1876 - 10.0, 91.1876 + 10.0
        )
        electron_pveto_diagnostics.update(
            {
                "pair_mass10": ak.num(electron_mass_pairs[electron_mass_pairs.mass > 10.0])
                >= 1,
                "track_muonVeto": ak.num(
                    electron_muon_veto_pairs[electron_muon_veto_pairs.mass > 10.0]
                )
                >= 1,
                "track_tauVeto": ak.num(
                    electron_tau_veto_pairs[electron_tau_veto_pairs.mass > 10.0]
                )
                >= 1,
                "track_calo10": ak.num(
                    electron_calo_pairs[electron_calo_pairs.mass > 10.0]
                )
                >= 1,
                "track_probe_before_layer": ak.num(
                    electron_pairs[electron_pairs.mass > 10.0]
                )
                >= 1,
                "pair_zwindow": ak.num(electron_pairs[electron_z_window]) >= 1,
                "pair_os": ak.num(electron_pairs[electron_os_z_window]) >= 1,
                "layer_combinedBins": ak.num(
                    electron_pairs[
                        electron_os_z_window
                        & generic_probe_pair_layer_mask(electron_pairs, "combinedBins")
                    ]
                )
                >= 1,
                "pair_pass_electron_pveto": ak.num(
                    electron_pairs[
                        electron_os_z_window
                        & generic_probe_pair_layer_mask(electron_pairs, "combinedBins")
                        & electron_pveto_pair_pass_mask(electron_pairs)
                    ]
                )
                >= 1,
            }
        )
        self.events["ElectronPVetoDiag"] = ak.zip(electron_pveto_diagnostics)

    def _store_tau_pveto_diagnostics(self, mode):
        event_golden_json, met_filters, jet_veto_map_mask = self._event_quality_masks()
        tau_track_masks = tau_veto_probe_track_cutflow_masks(self.events.IsoTrack)

        def store(
            *,
            collection,
            tag_source,
            event_trigger,
            tag_masks,
            low_mt_tags,
            tag_mass,
            probe_mass,
        ):
            event_met_filters = event_trigger & met_filters
            event_jet_veto_map = event_met_filters & jet_veto_map_mask
            diagnostics = {
                "event_trigger": event_trigger,
                "event_met_filters": event_met_filters,
                "event_jet_veto_map": event_jet_veto_map,
            }
            for name, mask in tag_masks.items():
                diagnostics[name] = event_jet_veto_map & (
                    ak.num(tag_source[mask]) >= 1
                )
            diagnostics["tag_low_mt"] = event_jet_veto_map & (ak.num(low_mt_tags) >= 1)
            for name, mask in tau_track_masks.items():
                diagnostics[name] = diagnostics["tag_low_mt"] & (
                    ak.num(self.events.IsoTrack[mask]) >= 1
                )

            mass_probe_tracks = self.events.IsoTrack[tau_track_masks["track_muonVeto"]]
            pairs = build_lepton_veto_tag_probe_pairs(
                low_mt_tags,
                mass_probe_tracks,
                tag_mass=tag_mass,
                probe_mass=probe_mass,
            )
            mass_window = mass_window_pair_mask(
                pairs, 91.1876 - 50.0, 91.1876 - 15.0
            )
            os_mass_window = os_mass_window_pair_mask(
                pairs, 91.1876 - 50.0, 91.1876 - 15.0
            )
            ss_mass_window = ss_mass_window_pair_mask(
                pairs, 91.1876 - 50.0, 91.1876 - 15.0
            )
            layer_mask = generic_probe_pair_layer_mask(pairs, "combinedBins")
            diagnostics.update(
                {
                    "pair_masswindow": ak.num(pairs[mass_window]) >= 1,
                    "pair_os": ak.num(pairs[os_mass_window]) >= 1,
                    "layer_combinedBins": ak.num(pairs[os_mass_window & layer_mask])
                    >= 1,
                    "pair_pass_tau_pveto": ak.num(
                        pairs[
                            os_mass_window
                            & layer_mask
                            & tau_pveto_pair_pass_mask(pairs)
                        ]
                    )
                    >= 1,
                    "pair_ss_masswindow": ak.num(pairs[ss_mass_window]) >= 1,
                    "pair_ss_pass_tau_pveto": ak.num(
                        pairs[
                            ss_mass_window
                            & layer_mask
                            & tau_pveto_pair_pass_mask(pairs)
                        ]
                    )
                    >= 1,
                }
            )
            self.events[collection] = ak.zip(diagnostics)

        if mode == "tau_mu_pveto":
            event_trigger = event_golden_json & self.events.HLT.IsoMu24
            muon_tag_masks = muon_tag_progression_masks(self.events.Muon)
            tau_mu_tag_masks = {
                "tag_pt": muon_tag_masks["muon_pt26"],
                "tag_eta2p1": muon_tag_masks["muon_eta2p1"],
                "tag_tight_id": muon_tag_masks["muon_tight_id"],
                "tag_selected": muon_tag_masks["muon_selected_tag"],
            }
            store(
                collection="TauMuPVetoDiag",
                tag_source=self.events.Muon,
                event_trigger=event_trigger,
                tag_masks=tau_mu_tag_masks,
                low_mt_tags=self.events.MuonLowMTTag,
                tag_mass=MUON_MASS,
                probe_mass=MUON_MASS,
            )
        elif mode == "tau_ele_pveto":
            event_trigger = event_golden_json & single_electron_trigger_mask(
                self.events
            )
            tau_ele_tag_masks = {"tag_pt": self.events.Electron.pt > 32.0}
            tau_ele_tag_masks["tag_eta2p1"] = tau_ele_tag_masks["tag_pt"] & (
                abs(self.events.Electron.eta) < 2.1
            )
            tau_ele_tag_masks["tag_tight_id"] = (
                tau_ele_tag_masks["tag_eta2p1"] & (self.events.Electron.cutBased >= 4)
            )
            tau_ele_tag_masks["tag_selected"] = _z_electron_tag_mask(
                self.events.Electron,
                pt_min=32.0,
            )
            store(
                collection="TauElePVetoDiag",
                tag_source=self.events.Electron,
                event_trigger=event_trigger,
                tag_masks=tau_ele_tag_masks,
                low_mt_tags=self.events.ElectronLowMTTag,
                tag_mass=ELECTRON_MASS,
                probe_mass=ELECTRON_MASS,
            )

    def _store_search_diagnostics(self):
        track_diagnostics = {}
        diagnostics = {}
        for name, mask in search_track_cutflow_masks(self.events.IsoTrack).items():
            n_name = f"n{name[0].upper()}{name[1:]}"
            self.events[n_name] = ak.num(self.events.IsoTrack[mask])
            track_diagnostics[name] = self.events[n_name] >= 1

        diagnostics.update(track_diagnostics)
        event_diagnostics = search_event_cutflow_masks(self.events.AnalysisEvent)
        diagnostics.update(event_diagnostics)
        event_search_kinematics = event_diagnostics["event_dijetDphi2p5"]
        for name, mask in track_diagnostics.items():
            diagnostics[f"eventKinematics_{name}"] = event_search_kinematics & mask
        self.events["SearchDiag"] = ak.zip(diagnostics)

    def _store_fake_track_diagnostics(self):
        event_search_kinematics = search_event_cutflow_masks(
            self.events.AnalysisEvent
        )["event_dijetDphi2p5"]
        diagnostics = {}
        for name, mask in fake_track_sideband_cutflow_masks(self.events.IsoTrack).items():
            n_name = f"nFakeDiag{name[0].upper()}{name[1:]}"
            self.events[n_name] = ak.num(self.events.IsoTrack[mask])
            diagnostics[name] = event_search_kinematics & (self.events[n_name] >= 1)
        self.events["FakeTrackDiag"] = ak.zip(diagnostics)

    def _count_objects_mode_aware(self, variation):
        mode = self._category_mode()
        self._count_common_search_fields()

        if mode == "muon_pveto":
            self._count_muon_pveto_fields()
            self._store_muon_pveto_diagnostics()
        elif mode == "electron_pveto":
            self.events["nElectronTag"] = ak.num(self.events.ElectronTag)
            self.events["nElectronVetoProbeTrack"] = ak.num(
                self.events.ElectronVetoProbeTrack
            )
            self._count_lepton_pair_fields("Electron")
            self._store_electron_pveto_diagnostics()
        elif mode == "tau_mu_pveto":
            self.events["nMuonLowMTTag"] = ak.num(self.events.MuonLowMTTag)
            self.events["nTauVetoProbeTrack"] = ak.num(self.events.TauVetoProbeTrack)
            self._count_lepton_pair_fields("TauMu")
            self._store_tau_pveto_diagnostics("tau_mu_pveto")
        elif mode == "tau_ele_pveto":
            self.events["nElectronLowMTTag"] = ak.num(self.events.ElectronLowMTTag)
            self.events["nTauVetoProbeTrack"] = ak.num(self.events.TauVetoProbeTrack)
            self._count_lepton_pair_fields("TauEle")
            self._store_tau_pveto_diagnostics("tau_ele_pveto")
        elif mode == "fake_tracks":
            self._count_fake_track_fields(controls=self._fake_track_controls())
        elif mode == "muon_backgrounds":
            self._count_muon_pveto_fields()
            self._store_muon_pveto_diagnostics()
            self.events["nMuonLowMTTag"] = ak.num(self.events.MuonLowMTTag)
            self.events["nTauVetoProbeTrack"] = ak.num(self.events.TauVetoProbeTrack)
            self._count_lepton_pair_fields("TauMu")
            self._store_tau_pveto_diagnostics("tau_mu_pveto")
            self._count_fake_track_fields(controls=("zmumu",))
        elif mode == "egamma_backgrounds":
            self.events["nElectronTag"] = ak.num(self.events.ElectronTag)
            self.events["nElectronVetoProbeTrack"] = ak.num(
                self.events.ElectronVetoProbeTrack
            )
            self._count_lepton_pair_fields("Electron")
            self._store_electron_pveto_diagnostics()
            self.events["nElectronLowMTTag"] = ak.num(self.events.ElectronLowMTTag)
            self.events["nTauVetoProbeTrack"] = ak.num(self.events.TauVetoProbeTrack)
            self._count_lepton_pair_fields("TauEle")
            self._store_tau_pveto_diagnostics("tau_ele_pveto")
            self._count_fake_track_fields(controls=("zee",))

        if self._search_diagnostics_enabled():
            if mode != "fake_tracks" or self._fake_track_controls() == ("basic",):
                self._store_search_diagnostics()
            if mode == "fake_tracks" and self._fake_track_controls() == ("basic",):
                self._store_fake_track_diagnostics()

    def count_objects(self, variation):
        if self._category_mode() != "all" and not self._full_workflow_enabled():
            self._count_objects_mode_aware(variation)
            return

        self.events["nIsoTrack"] = ak.num(self.events.IsoTrack)
        self.events["nMuonTag"] = ak.num(self.events.MuonTag)
        self.events["nIsoTrackProbe"] = ak.num(self.events.IsoTrackProbe)
        self.events["nElectronTag"] = ak.num(self.events.ElectronTag)
        self.events["nMuonLowMTTag"] = ak.num(self.events.MuonLowMTTag)
        self.events["nElectronLowMTTag"] = ak.num(self.events.ElectronLowMTTag)
        self.events["nMuonVetoProbeTrack"] = ak.num(self.events.MuonVetoProbeTrack)
        self.events["nElectronVetoProbeTrack"] = ak.num(self.events.ElectronVetoProbeTrack)
        self.events["nTauVetoProbeTrack"] = ak.num(self.events.TauVetoProbeTrack)
        self.events["nMuonVetoTagProbePair"] = ak.num(
            self.events.MuonVetoTagProbePair
        )
        self.events["nMuonVetoTagProbePairOS"] = ak.num(
            self.events.MuonVetoTagProbePairOS
        )
        self.events["nMuonVetoTagProbePairMass10"] = ak.num(
            self.events.MuonVetoTagProbePairMass10
        )
        self.events["nMuonVetoTagProbePairOSMass10"] = ak.num(
            self.events.MuonVetoTagProbePairOSMass10
        )
        self.events["nMuonVetoTagProbePairSS"] = ak.num(
            self.events.MuonVetoTagProbePairSS
        )
        self.events["nMuonVetoTagProbePairSSMass10"] = ak.num(
            self.events.MuonVetoTagProbePairSSMass10
        )
        self.events["nMuonVetoTagProbePairZWindow"] = ak.num(
            self.events.MuonVetoTagProbePairZWindow
        )
        self.events["nMuonVetoTagProbePairOSZWindow"] = ak.num(
            self.events.MuonVetoTagProbePairOSZWindow
        )
        self.events["nMuonVetoTagProbePairZWindowPass"] = ak.num(
            self.events.MuonVetoTagProbePairZWindowPass
        )
        self.events["nMuonVetoTagProbePairZWindowFail"] = ak.num(
            self.events.MuonVetoTagProbePairZWindowFail
        )
        self.events["nMuonPVetoTagProbePairZWindowPass"] = ak.num(
            self.events.MuonPVetoTagProbePairZWindowPass
        )
        self.events["nMuonVetoTagProbePairSSZWindow"] = ak.num(
            self.events.MuonVetoTagProbePairSSZWindow
        )
        self.events["nMuonVetoTagProbePairSSZWindowPass"] = ak.num(
            self.events.MuonVetoTagProbePairSSZWindowPass
        )
        self.events["nMuonVetoTagProbePairSSZWindowFail"] = ak.num(
            self.events.MuonVetoTagProbePairSSZWindowFail
        )
        self.events["nMuonPVetoTagProbePairSSZWindowPass"] = ak.num(
            self.events.MuonPVetoTagProbePairSSZWindowPass
        )
        for layer in PVETO_LAYERS:
            self.events[f"nMuonVetoTagProbePairZWindow_{layer}"] = ak.num(
                self.events[f"MuonVetoTagProbePairZWindow_{layer}"]
            )
            self.events[f"nMuonPVetoTagProbePairZWindowPass_{layer}"] = ak.num(
                self.events[f"MuonPVetoTagProbePairZWindowPass_{layer}"]
            )
            self.events[f"nMuonVetoTagProbePairSSZWindow_{layer}"] = ak.num(
                self.events[f"MuonVetoTagProbePairSSZWindow_{layer}"]
            )
            self.events[f"nMuonPVetoTagProbePairSSZWindowPass_{layer}"] = ak.num(
                self.events[f"MuonPVetoTagProbePairSSZWindowPass_{layer}"]
            )
        for prefix in ("Electron", "TauMu", "TauEle"):
            self.events[f"n{prefix}TagProbePair"] = ak.num(
                self.events[f"{prefix}TagProbePair"]
            )
            self.events[f"n{prefix}TagProbePairMassWindow"] = ak.num(
                self.events[f"{prefix}TagProbePairMassWindow"]
            )
            self.events[f"n{prefix}TagProbePairOSMassWindow"] = ak.num(
                self.events[f"{prefix}TagProbePairOSMassWindow"]
            )
            self.events[f"n{prefix}TagProbePairSSMassWindow"] = ak.num(
                self.events[f"{prefix}TagProbePairSSMassWindow"]
            )
            self.events[f"n{prefix}PVetoTagProbePairMassWindowPass"] = ak.num(
                self.events[f"{prefix}PVetoTagProbePairMassWindowPass"]
            )
            self.events[f"n{prefix}PVetoTagProbePairSSMassWindowPass"] = ak.num(
                self.events[f"{prefix}PVetoTagProbePairSSMassWindowPass"]
            )
            for layer in PVETO_LAYERS:
                self.events[f"n{prefix}TagProbePairMassWindow_{layer}"] = ak.num(
                    self.events[f"{prefix}TagProbePairMassWindow_{layer}"]
                )
                self.events[f"n{prefix}PVetoTagProbePairMassWindowPass_{layer}"] = ak.num(
                    self.events[f"{prefix}PVetoTagProbePairMassWindowPass_{layer}"]
                )
                self.events[f"n{prefix}TagProbePairSSMassWindow_{layer}"] = ak.num(
                    self.events[f"{prefix}TagProbePairSSMassWindow_{layer}"]
                )
                self.events[f"n{prefix}PVetoTagProbePairSSMassWindowPass_{layer}"] = ak.num(
                    self.events[f"{prefix}PVetoTagProbePairSSMassWindowPass_{layer}"]
                )
        self.events["nIsoTrackSearchPreMissingOuter"] = ak.num(
            self.events.IsoTrackSearchPreMissingOuter
        )
        self.events["nIsoTrackSearchPreLeptonVeto"] = ak.num(
            self.events.IsoTrackSearchPreLeptonVeto
        )
        self.events["nIsoTrackSearch"] = ak.num(self.events.IsoTrackSearch)

        fake_basic3hits_d0_signal = self.events.IsoTrack[
            fake_track_no_d0_mask(
                self.events.IsoTrack,
                layer="NLayers4",
                d0_region="signal",
            )
        ]
        fake_basic3hits_d0_sideband = self.events.IsoTrack[
            fake_track_no_d0_mask(
                self.events.IsoTrack,
                layer="NLayers4",
                d0_region="sideband",
            )
        ]
        self.events["nFakeBasic3HitsD0Signal"] = ak.num(fake_basic3hits_d0_signal)
        self.events["nFakeBasic3HitsD0Sideband"] = ak.num(fake_basic3hits_d0_sideband)
        for layer in (*PVETO_LAYERS, "combinedBins"):
            self.events[f"nFakeControl_{layer}"] = ak.num(
                self.events.IsoTrack[
                    fake_track_no_d0_mask(
                        self.events.IsoTrack,
                        layer=layer,
                        d0_region="sideband",
                    )
                ]
            )

        fake_zmumu_control = _z_to_mumu_control_mask(self.events, self.events.Muon)
        fake_zee_control = _z_to_ee_control_mask(self.events, self.events.Electron)
        self.events["nFakeZMuMuControl"] = ak.values_astype(fake_zmumu_control, np.int64)
        self.events["nFakeZeeControl"] = ak.values_astype(fake_zee_control, np.int64)
        self.events["FakeZMuMuDiag"] = ak.zip(
            _z_to_mumu_control_diagnostics(self.events, self.events.Muon)
        )
        self.events["FakeZeeDiag"] = ak.zip(
            _z_to_ee_control_diagnostics(self.events, self.events.Electron)
        )
        self.events["FakeZMuMuFitTrack"] = _fake_fit_tracks_for_control(
            self.events,
            fake_zmumu_control,
        )
        self.events["FakeZeeFitTrack"] = _fake_fit_tracks_for_control(
            self.events,
            fake_zee_control,
        )
        for layer in (*PVETO_LAYERS, "combinedBins"):
            self.events[f"nFakeZMuMuSideband_{layer}"] = _fake_track_count_for_control(
                self.events,
                fake_zmumu_control,
                layer=layer,
                d0_region="sideband",
            )
            self.events[f"nFakeZeeSideband_{layer}"] = _fake_track_count_for_control(
                self.events,
                fake_zee_control,
                layer=layer,
                d0_region="sideband",
            )

        event_golden_json = _golden_json_mask(
            self.events,
            processor_params=self.params,
            year=self._year,
            era=self._era,
            sample=self._sample,
            is_mc=self._isMC,
        )
        event_singlemu_trigger = event_golden_json & self.events.HLT.IsoMu24
        event_met_filters = event_singlemu_trigger & _met_filters_mask(self.events)
        event_jet_veto_map = event_met_filters & _jet_veto_map_mask(
            self.events,
            processor_params=self.params,
            year=self._year,
            era=self._era,
            sample=self._sample,
            is_mc=self._isMC,
        )
        muon_table16_diagnostics = {
            "event_singlemu_trigger": event_singlemu_trigger,
            "event_met_filters": event_met_filters,
            "event_jet_veto_map": event_jet_veto_map,
        }
        muon_tag_masks = muon_tag_progression_masks(self.events.Muon)
        for name, mask in muon_tag_masks.items():
            self.events[f"n{name[0].upper()}{name[1:]}"] = ak.num(
                self.events.Muon[mask]
            )
            muon_table16_diagnostics[name] = (
                event_jet_veto_map
                & (self.events[f"n{name[0].upper()}{name[1:]}"] >= 1)
            )

        has_selected_muon_tag = muon_table16_diagnostics["muon_selected_tag"]
        table16_track_masks = self._apply_lepton_fiducial_maps_to_track_cutflow(
            muon_veto_probe_track_cutflow_masks(self.events.IsoTrack)
        )
        pre_pair_track_fields = {
            "track_pt30",
            "track_eta2p1",
            "track_noDTWheelGap",
            "track_noECALCrack",
            "track_noCSCTransition",
            "track_fiducialSelections",
            "track_dzOrLambda",
            "track_pixelHits4",
            "track_noMissingInner",
            "track_noMissingMiddle",
            "track_chargedIso0p05",
            "track_dxy0p02",
            "track_dz0p5",
            "track_dRJet0p5",
        }
        for name, mask in table16_track_masks.items():
            self.events[f"n{name[0].upper()}{name[1:]}Table16"] = ak.num(
                self.events.IsoTrack[mask]
            )
            if name in pre_pair_track_fields:
                muon_table16_diagnostics[name] = (
                    has_selected_muon_tag
                    & (self.events[f"n{name[0].upper()}{name[1:]}Table16"] >= 1)
                )

        table16_mass_probe_tracks = self.events.IsoTrack[
            table16_track_masks["track_dRJet0p5"]
        ]
        table16_mass_pairs = build_muon_veto_tag_probe_pairs(
            self.events.MuonTag, table16_mass_probe_tracks
        )
        table16_electron_probe_tracks = self.events.IsoTrack[
            table16_track_masks["track_electronVeto"]
        ]
        table16_electron_pairs = build_muon_veto_tag_probe_pairs(
            self.events.MuonTag, table16_electron_probe_tracks
        )
        table16_tau_probe_tracks = self.events.IsoTrack[
            table16_track_masks["track_tauVeto"]
        ]
        table16_tau_pairs = build_muon_veto_tag_probe_pairs(
            self.events.MuonTag, table16_tau_probe_tracks
        )
        table16_probe_tracks = self.events.IsoTrack[table16_track_masks["track_calo10"]]
        table16_pairs = build_muon_veto_tag_probe_pairs(
            self.events.MuonTag, table16_probe_tracks
        )
        muon_table16_diagnostics.update(
            {
                "pair_mass10": ak.num(
                    table16_mass_pairs[
                        mass10_muon_probe_pair_mask(table16_mass_pairs)
                    ]
                )
                >= 1,
                "track_electronVeto": ak.num(
                    table16_electron_pairs[
                        mass10_muon_probe_pair_mask(table16_electron_pairs)
                    ]
                )
                >= 1,
                "track_tauVeto": ak.num(
                    table16_tau_pairs[
                        mass10_muon_probe_pair_mask(table16_tau_pairs)
                    ]
                )
                >= 1,
                "track_calo10": ak.num(
                    table16_pairs[
                        mass10_muon_probe_pair_mask(table16_pairs)
                    ]
                )
                >= 1,
                "pair_zwindow": ak.num(
                    table16_pairs[z_window_muon_probe_pair_mask(table16_pairs)]
                )
                >= 1,
                "pair_os": ak.num(
                    table16_pairs[os_z_window_muon_probe_pair_mask(table16_pairs)]
                )
                >= 1,
            }
        )
        muon_table16_diagnostics["track_probe_before_layer"] = (
            muon_table16_diagnostics["track_calo10"]
        )
        muon_table16_diagnostics["layer_combinedBins"] = ak.num(
            table16_pairs[
                os_z_window_muon_probe_pair_mask(table16_pairs)
                & muon_probe_pair_layer_mask(table16_pairs, "combinedBins")
            ]
        ) >= 1
        self.events["MuonTable16Diag"] = ak.zip(muon_table16_diagnostics)

        event_singleele_trigger = event_golden_json & single_electron_trigger_mask(
            self.events
        )
        event_ele_met_filters = event_singleele_trigger & _met_filters_mask(
            self.events
        )
        event_ele_jet_veto_map = event_ele_met_filters & _jet_veto_map_mask(
            self.events,
            processor_params=self.params,
            year=self._year,
            era=self._era,
            sample=self._sample,
            is_mc=self._isMC,
        )
        electron_pveto_diagnostics = {
            "event_singleele_trigger": event_singleele_trigger,
            "event_met_filters": event_ele_met_filters,
            "event_jet_veto_map": event_ele_jet_veto_map,
        }
        electron_tag_masks = electron_tag_progression_masks(
            self.events.Electron, self.events
        )
        for name, mask in electron_tag_masks.items():
            self.events[f"n{name[0].upper()}{name[1:]}"] = ak.num(
                self.events.Electron[mask]
            )
            electron_pveto_diagnostics[name] = (
                event_ele_jet_veto_map
                & (self.events[f"n{name[0].upper()}{name[1:]}"] >= 1)
            )

        has_selected_electron_tag = electron_pveto_diagnostics[
            "electron_selected_tag"
        ]
        electron_track_masks = self._apply_lepton_fiducial_maps_to_track_cutflow(
            muon_veto_probe_track_cutflow_masks(self.events.IsoTrack)
        )
        for name in (
            "track_pt30",
            "track_eta2p1",
            "track_noDTWheelGap",
            "track_noECALCrack",
            "track_noCSCTransition",
            "track_fiducialSelections",
            "track_dzOrLambda",
            "track_pixelHits4",
            "track_noMissingInner",
            "track_noMissingMiddle",
            "track_chargedIso0p05",
            "track_dxy0p02",
            "track_dz0p5",
            "track_dRJet0p5",
        ):
            n_name = f"n{name[0].upper()}{name[1:]}ElectronPVeto"
            self.events[n_name] = ak.num(self.events.IsoTrack[electron_track_masks[name]])
            electron_pveto_diagnostics[name] = (
                has_selected_electron_tag & (self.events[n_name] >= 1)
            )

        electron_mass_probe_tracks = self.events.IsoTrack[
            electron_track_masks["track_dRJet0p5"]
        ]
        electron_mass_pairs = build_lepton_veto_tag_probe_pairs(
            self.events.ElectronTag,
            electron_mass_probe_tracks,
            tag_mass=ELECTRON_MASS,
            probe_mass=ELECTRON_MASS,
        )
        electron_muon_veto_mask = electron_track_masks["track_dRJet0p5"] & (
            (self.events.IsoTrack.dRMinMuon < 0.0)
            | (self.events.IsoTrack.dRMinMuon > 0.15)
        )
        electron_tau_veto_mask = electron_muon_veto_mask & (
            (self.events.IsoTrack.dRMinTauHad < 0.0)
            | (self.events.IsoTrack.dRMinTauHad > 0.15)
        )
        electron_calo_mask = electron_tau_veto_mask & (
            self.events.IsoTrack.caloEnergy < 10.0
        )
        electron_muon_veto_pairs = build_lepton_veto_tag_probe_pairs(
            self.events.ElectronTag,
            self.events.IsoTrack[electron_muon_veto_mask],
            tag_mass=ELECTRON_MASS,
            probe_mass=ELECTRON_MASS,
        )
        electron_tau_veto_pairs = build_lepton_veto_tag_probe_pairs(
            self.events.ElectronTag,
            self.events.IsoTrack[electron_tau_veto_mask],
            tag_mass=ELECTRON_MASS,
            probe_mass=ELECTRON_MASS,
        )
        electron_pairs = build_lepton_veto_tag_probe_pairs(
            self.events.ElectronTag,
            self.events.IsoTrack[electron_tau_veto_mask],
            tag_mass=ELECTRON_MASS,
            probe_mass=ELECTRON_MASS,
        )
        electron_calo_pairs = build_lepton_veto_tag_probe_pairs(
            self.events.ElectronTag,
            self.events.IsoTrack[electron_calo_mask],
            tag_mass=ELECTRON_MASS,
            probe_mass=ELECTRON_MASS,
        )
        electron_z_window = mass_window_pair_mask(
            electron_pairs, 91.1876 - 10.0, 91.1876 + 10.0
        )
        electron_os_z_window = os_mass_window_pair_mask(
            electron_pairs, 91.1876 - 10.0, 91.1876 + 10.0
        )
        electron_pveto_diagnostics.update(
            {
                "pair_mass10": ak.num(electron_mass_pairs[electron_mass_pairs.mass > 10.0])
                >= 1,
                "track_muonVeto": ak.num(
                    electron_muon_veto_pairs[electron_muon_veto_pairs.mass > 10.0]
                )
                >= 1,
                "track_tauVeto": ak.num(
                    electron_tau_veto_pairs[electron_tau_veto_pairs.mass > 10.0]
                )
                >= 1,
                "track_calo10": ak.num(
                    electron_calo_pairs[electron_calo_pairs.mass > 10.0]
                )
                >= 1,
                "track_probe_before_layer": ak.num(
                    electron_pairs[electron_pairs.mass > 10.0]
                )
                >= 1,
                "pair_zwindow": ak.num(electron_pairs[electron_z_window]) >= 1,
                "pair_os": ak.num(electron_pairs[electron_os_z_window]) >= 1,
                "layer_combinedBins": ak.num(
                    electron_pairs[
                        electron_os_z_window
                        & generic_probe_pair_layer_mask(electron_pairs, "combinedBins")
                    ]
                )
                >= 1,
                "pair_pass_electron_pveto": ak.num(
                    electron_pairs[
                        electron_os_z_window
                        & generic_probe_pair_layer_mask(electron_pairs, "combinedBins")
                        & electron_pveto_pair_pass_mask(electron_pairs)
                    ]
                )
                >= 1,
            }
        )
        self.events["ElectronPVetoDiag"] = ak.zip(electron_pveto_diagnostics)

        tau_track_masks = tau_veto_probe_track_cutflow_masks(self.events.IsoTrack)

        def _store_tau_pveto_diagnostics(
            *,
            collection,
            tag_source,
            event_trigger,
            event_met_filters,
            event_jet_veto_map,
            tag_masks,
            low_mt_tags,
            tag_mass,
            probe_mass,
        ):
            diagnostics = {
                "event_trigger": event_trigger,
                "event_met_filters": event_met_filters,
                "event_jet_veto_map": event_jet_veto_map,
            }

            for name, mask in tag_masks.items():
                diagnostics[name] = event_jet_veto_map & (
                    ak.num(tag_source[mask]) >= 1
                )

            diagnostics["tag_low_mt"] = event_jet_veto_map & (ak.num(low_mt_tags) >= 1)

            for name, mask in tau_track_masks.items():
                diagnostics[name] = diagnostics["tag_low_mt"] & (
                    ak.num(self.events.IsoTrack[mask]) >= 1
                )

            mass_probe_tracks = self.events.IsoTrack[tau_track_masks["track_muonVeto"]]
            pairs = build_lepton_veto_tag_probe_pairs(
                low_mt_tags,
                mass_probe_tracks,
                tag_mass=tag_mass,
                probe_mass=probe_mass,
            )
            mass_window = mass_window_pair_mask(
                pairs, 91.1876 - 50.0, 91.1876 - 15.0
            )
            os_mass_window = os_mass_window_pair_mask(
                pairs, 91.1876 - 50.0, 91.1876 - 15.0
            )
            ss_mass_window = ss_mass_window_pair_mask(
                pairs, 91.1876 - 50.0, 91.1876 - 15.0
            )
            layer_mask = generic_probe_pair_layer_mask(pairs, "combinedBins")
            diagnostics.update(
                {
                    "pair_masswindow": ak.num(pairs[mass_window]) >= 1,
                    "pair_os": ak.num(pairs[os_mass_window]) >= 1,
                    "layer_combinedBins": ak.num(pairs[os_mass_window & layer_mask])
                    >= 1,
                    "pair_pass_tau_pveto": ak.num(
                        pairs[
                            os_mass_window
                            & layer_mask
                            & tau_pveto_pair_pass_mask(pairs)
                        ]
                    )
                    >= 1,
                    "pair_ss_masswindow": ak.num(pairs[ss_mass_window]) >= 1,
                    "pair_ss_pass_tau_pveto": ak.num(
                        pairs[
                            ss_mass_window
                            & layer_mask
                            & tau_pveto_pair_pass_mask(pairs)
                        ]
                    )
                    >= 1,
                }
            )
            self.events[collection] = ak.zip(diagnostics)

        tau_mu_tag_masks = {
            "tag_pt": self.events.Muon.pt > 26.0,
        }
        tau_mu_tag_masks["tag_eta2p1"] = tau_mu_tag_masks["tag_pt"] & (
            abs(self.events.Muon.eta) < 2.1
        )
        tau_mu_tag_masks["tag_tight_id"] = (
            tau_mu_tag_masks["tag_eta2p1"] & self.events.Muon.tightId
        )
        _store_tau_pveto_diagnostics(
            collection="TauMuPVetoDiag",
            tag_source=self.events.Muon,
            event_trigger=event_singlemu_trigger,
            event_met_filters=event_met_filters,
            event_jet_veto_map=event_jet_veto_map,
            tag_masks=tau_mu_tag_masks,
            low_mt_tags=self.events.MuonLowMTTag,
            tag_mass=MUON_MASS,
            probe_mass=MUON_MASS,
        )

        tau_ele_tag_masks = {
            "tag_pt": self.events.Electron.pt > 32.0,
        }
        tau_ele_tag_masks["tag_eta2p1"] = tau_ele_tag_masks["tag_pt"] & (
            abs(self.events.Electron.eta) < 2.1
        )
        tau_ele_tag_masks["tag_tight_id"] = (
            tau_ele_tag_masks["tag_eta2p1"] & (self.events.Electron.cutBased >= 4)
        )
        tau_ele_tag_masks["tag_selected"] = _z_electron_tag_mask(
            self.events.Electron,
            pt_min=32.0,
        )
        _store_tau_pveto_diagnostics(
            collection="TauElePVetoDiag",
            tag_source=self.events.Electron,
            event_trigger=event_singleele_trigger,
            event_met_filters=event_ele_met_filters,
            event_jet_veto_map=event_ele_jet_veto_map,
            tag_masks=tau_ele_tag_masks,
            low_mt_tags=self.events.ElectronLowMTTag,
            tag_mass=ELECTRON_MASS,
            probe_mass=ELECTRON_MASS,
        )

        track_diagnostics = {}
        diagnostics = {}
        for name, mask in search_track_cutflow_masks(self.events.IsoTrack).items():
            n_name = f"n{name[0].upper()}{name[1:]}"
            self.events[n_name] = ak.num(self.events.IsoTrack[mask])
            track_diagnostics[name] = self.events[n_name] >= 1

        diagnostics.update(track_diagnostics)

        event_diagnostics = search_event_cutflow_masks(self.events.AnalysisEvent)
        diagnostics.update(event_diagnostics)
        event_search_kinematics = event_diagnostics["event_dijetDphi2p5"]
        for name, mask in track_diagnostics.items():
            diagnostics[f"eventKinematics_{name}"] = event_search_kinematics & mask
        self.events["SearchDiag"] = ak.zip(diagnostics)
        if self._search_diagnostics_enabled() and self._category_mode() == "fake_tracks":
            self._store_fake_track_diagnostics()

    def define_common_variables_before_presel(self, variation):
        pass
