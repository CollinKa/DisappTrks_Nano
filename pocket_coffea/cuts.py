"""PocketCoffea event cuts for initial search-selection validation."""

from __future__ import annotations

import os
from pathlib import Path

import awkward as ak
import numpy as np
from coffea.lumi_tools import LumiMask

from pocket_coffea.lib.cut_definition import Cut
from pocket_coffea.lib.cut_functions import apply_golden_json, get_JetVetoMap_Mask


EVENT_DIAGNOSTIC_FIELDS = [
    "event_metNoMu120",
    "event_leadingJet110",
    "event_leadingJetEta2p4",
    "event_leadingJetTightLepVeto",
    "event_dijetDphi2p5",
    "event_jetMetDphi0p5",
]

TRACK_DIAGNOSTIC_FIELDS = [
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
    "track_calo10",
    "track_missingOuter3",
    "track_electronVeto",
    "track_muonVeto",
    "track_tauVeto",
]

FAKE_TRACK_DIAGNOSTIC_FIELDS = [
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
    "track_dz0p5",
    "track_dRJet0p5",
    "track_calo10",
    "track_missingOuter3",
    "track_electronVeto",
    "track_muonVeto",
    "track_tauVeto",
    "track_d0Sideband",
    "track_NLayers4",
    "track_NLayers5",
    "track_NLayers6plus",
    "track_combinedBins",
]

FAKE_ZMUMU_DIAGNOSTIC_FIELDS = [
    "event_trigger",
    "muon_pt26",
    "muon_eta2p1",
    "muon_tight_id",
    "muon_selected_tag",
    "z_os_window",
    "sideband_NLayers4",
    "sideband_NLayers5",
    "sideband_NLayers6plus",
    "sideband_combinedBins",
]

FAKE_ZEE_DIAGNOSTIC_FIELDS = [
    "event_trigger",
    "electron_pt25",
    "electron_eta2p1",
    "electron_tight_id",
    "electron_dxy",
    "electron_dz",
    "electron_pt32",
    "z_os_window",
    "sideband_NLayers4",
    "sideband_NLayers5",
    "sideband_NLayers6plus",
    "sideband_combinedBins",
]

COMBINED_DIAGNOSTIC_FIELDS = [
    f"eventKinematics_{field}" for field in TRACK_DIAGNOSTIC_FIELDS
]

SEARCH_DIAGNOSTIC_FIELDS = (
    EVENT_DIAGNOSTIC_FIELDS + TRACK_DIAGNOSTIC_FIELDS + COMBINED_DIAGNOSTIC_FIELDS
)

PVETO_LAYERS = ("NLayers4", "NLayers5", "NLayers6plus")

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

MUON_TABLE16_FIELDS = [
    "event_singlemu_trigger",
    "event_met_filters",
    "event_jet_veto_map",
    "muon_pt26",
    "muon_eta2p1",
    "muon_tight_id",
    "muon_selected_tag",
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
    "pair_mass10",
    "track_electronVeto",
    "track_tauVeto",
    "track_calo10",
    "track_probe_before_layer",
    "pair_zwindow",
    "pair_os",
    "layer_combinedBins",
]

ELECTRON_PVETO_DIAGNOSTIC_FIELDS = [
    "event_singleele_trigger",
    "event_met_filters",
    "event_jet_veto_map",
    "electron_pt35",
    "electron_eta2p1",
    "electron_tight_id",
    "electron_dxy",
    "electron_dz",
    "electron_selected_tag",
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
    "pair_mass10",
    "track_muonVeto",
    "track_tauVeto",
    "track_calo10",
    "track_probe_before_layer",
    "pair_zwindow",
    "pair_os",
    "layer_combinedBins",
    "pair_pass_electron_pveto",
]

TAU_PVETO_DIAGNOSTIC_FIELDS = [
    "event_trigger",
    "event_met_filters",
    "event_jet_veto_map",
    "tag_pt",
    "tag_eta2p1",
    "tag_tight_id",
    "tag_selected",
    "tag_low_mt",
    "track_pt30",
    "track_eta2p1",
    "track_noDTWheelGap",
    "track_noECALCrack",
    "track_noCSCTransition",
    "track_noTOBCrack",
    "track_fiducialECAL",
    "track_pixelHits4",
    "track_noMissingInner",
    "track_noMissingMiddle",
    "track_chargedIso0p05",
    "track_dxy0p02",
    "track_dz0p5",
    "track_electronVeto",
    "track_muonVeto",
    "pair_masswindow",
    "pair_os",
    "layer_combinedBins",
    "pair_pass_tau_pveto",
    "pair_ss_masswindow",
    "pair_ss_pass_tau_pveto",
]


def _all_true(events):
    return np.ones(len(events), dtype=bool)


def _all_false(events):
    return np.zeros(len(events), dtype=bool)


def _hlt_or(events, names):
    if "HLT" not in events.fields:
        return _all_false(events)

    mask = None
    for name in names:
        if name in events.HLT.fields:
            mask = events.HLT[name] if mask is None else (mask | events.HLT[name])
    return mask if mask is not None else _all_false(events)


def _metadata_era(events):
    return str(getattr(events, "metadata", {}).get("era", ""))


def _pocketcoffea_run3_year_key(year, events=None, processor_params=None):
    """Map our plain Run-3 year metadata onto PocketCoffea parameter keys."""
    year = str(year)
    available = ()
    if processor_params is not None:
        for container_name in ("lumi", "event_flags", "jet_scale_factors"):
            container = getattr(processor_params, container_name, None)
            if container is None:
                continue
            if container_name == "lumi":
                maybe = getattr(container, "goldenJSON", {})
            elif container_name == "jet_scale_factors":
                maybe = getattr(container, "vetomaps", {})
            else:
                maybe = container
            try:
                available = tuple(maybe.keys())
                break
            except AttributeError:
                pass

    if year in available:
        return year

    era = _metadata_era(events)
    if year == "2022":
        return "2022_preEE" if era in ("C", "D") else "2022_postEE"
    if year == "2023":
        return "2023_preBPix" if era == "C" else "2023_postBPix"
    return year


def _container_has_key(container, key):
    try:
        return str(key) in container
    except Exception:
        try:
            container[str(key)]
        except Exception:
            return False
        return True


def _event_flags_year_key(year, events, processor_params):
    mapped_year = _pocketcoffea_run3_year_key(year, events, processor_params)
    if _container_has_key(processor_params.event_flags, mapped_year):
        return mapped_year

    # PocketCoffea may not yet define explicit event-flag lists for newly added
    # Run-3/Run-3-extension data-taking years.  Use the 2024 NanoAODv15/data-era
    # list as the closest available fallback.
    if str(year) in ("2025", "2026") and _container_has_key(
        processor_params.event_flags, "2024"
    ):
        return "2024"

    return mapped_year


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


def _jet_id_compute_year(processor_params, mapped_year):
    year = str(mapped_year)
    try:
        if year in processor_params.jet_scale_factors.jet_id:
            return year
    except Exception:
        pass

    # PocketCoffea may not yet carry explicit jet-ID correction keys for newly
    # added NanoAODv15 data-taking years.  Use the 2024 v15 jet-ID correction as
    # a compatibility fallback for the jet-ID recompute only; the jet-veto-map
    # payload itself is still selected with mapped_year.
    if year in ("2025", "2026"):
        try:
            if "2024" in processor_params.jet_scale_factors.jet_id:
                return "2024"
        except Exception:
            pass
    return year


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
    """Evaluate the Run-3 JME jet-veto map using a concrete local payload."""
    import correctionlib

    jets = events["Jet"]
    if "jetId" in jets.fields:
        jet_id = jets.jetId
    else:
        # Only recompute if the stored NanoAOD jetId branch is absent.  This
        # avoids requiring a PocketCoffea jet-ID correction entry for newly
        # added data-taking years such as 2025.
        from pocket_coffea.lib.jets import compute_jetId

        jet_id = compute_jetId(
            events,
            "Jet",
            processor_params,
            _jet_id_compute_year(processor_params, mapped_year),
        )
    jets = ak.with_field(jets, jet_id, "jetId_corrected")
    mask_for_veto_map = (
        (jets["jetId_corrected"] >= 6)
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


def _golden_json_lumi(events, params, year, processor_params, sample, isMC, **kwargs):
    if isMC:
        return _all_true(events)

    mapped_year = _pocketcoffea_run3_year_key(year, events, processor_params)
    local_json = _local_golden_json_path(mapped_year)
    if local_json is not None:
        return LumiMask(str(local_json))(events.run, events.luminosityBlock)
    payload_mask = _payload_golden_json_mask(events, mapped_year)
    if payload_mask is not None:
        return payload_mask

    try:
        return apply_golden_json(
            events,
            params=params,
            year=mapped_year,
            processor_params=processor_params,
            sample=sample,
            isMC=False,
            **kwargs,
        )
    except Exception:
        if os.environ.get("DISAPPTRKS_ALLOW_MISSING_GOLDEN_JSON", "").lower() in (
            "1",
            "true",
            "yes",
            "on",
        ):
            return _all_true(events)
        raise RuntimeError(
            "Could not apply the golden JSON lumimask. Provide local JSONs in "
            "pocket_coffea/data/golden_jsons, set DISAPPTRKS_GOLDEN_JSON_DIR, "
            "or explicitly set DISAPPTRKS_ALLOW_MISSING_GOLDEN_JSON=1 for a "
            "non-production diagnostic run."
        )


def _event_flags(events, params, year, processor_params, sample, isMC, **kwargs):
    if "Flag" in events.fields and "METFilters" in events.Flag.fields:
        return events.Flag.METFilters
    if "METFilters" in events.fields:
        return events.METFilters

    mapped_year = _event_flags_year_key(year, events, processor_params)
    flags = list(processor_params.event_flags[mapped_year])
    if not isMC and _container_has_key(processor_params.event_flags_data, mapped_year):
        flags += list(processor_params.event_flags_data[mapped_year])

    mask = _all_true(events)
    for flag in flags:
        if "Flag" in events.fields and flag in events.Flag.fields:
            mask = mask & events.Flag[flag].to_numpy()
        elif flag in events.fields:
            mask = mask & events[flag].to_numpy()
        else:
            # Some custom NanoAODs only persist the combined METFilters bit.  If
            # that was absent too, keep missing optional flags as pass-through
            # so lightweight local files remain usable.
            continue
    return mask


def _jet_veto_map(events, params, year, processor_params, sample, isMC, **kwargs):
    if "Flag" in events.fields and "jetVeto2022" in events.Flag.fields:
        return events.Flag.jetVeto2022
    if "jetVeto2022" in events.fields:
        return events.jetVeto2022

    mapped_year = _pocketcoffea_run3_year_key(year, events, processor_params)
    payload = _local_jet_veto_map_path(mapped_year)
    if payload is None:
        payload = _configured_jet_veto_map_path(processor_params, mapped_year)
    if payload is None:
        payload = _cvmfs_jet_veto_map_path(mapped_year)
    if payload is not None:
        return _evaluate_jet_veto_map(events, processor_params, mapped_year, payload)

    try:
        return get_JetVetoMap_Mask(
            events,
            params=params,
            year=mapped_year,
            processor_params=processor_params,
            sample=sample,
            isMC=isMC,
            **kwargs,
        )
    except Exception as exc:
        if os.environ.get("DISAPPTRKS_ALLOW_MISSING_JET_VETO_MAP", "").lower() in (
            "1",
            "true",
            "yes",
            "on",
        ):
            return _all_true(events)
        raise RuntimeError(
            "Could not apply the JME jet-veto map. Provide local payloads in "
            "pocket_coffea/data/jet_veto_maps, set DISAPPTRKS_JET_VETO_MAP_DIR, "
            "or explicitly set DISAPPTRKS_ALLOW_MISSING_JET_VETO_MAP=1 for a "
            "non-production diagnostic run."
        ) from exc


def _has_disappearing_track(events, params, **kwargs):
    return events.nIsoTrackSearch >= params["minimum"]


def _has_count(events, params, **kwargs):
    return events[params["field"]] >= params["minimum"]


def _search_diagnostic(events, params, **kwargs):
    return events.SearchDiag[params["field"]]


def _fake_track_diagnostic(events, params, **kwargs):
    return events.FakeTrackDiag[params["field"]]


def _fake_z_control_diagnostic(events, params, **kwargs):
    return events[params["collection"]][params["field"]]


def _muon_table16_diagnostic(events, params, **kwargs):
    return events.MuonTable16Diag[params["field"]]


def _electron_pveto_diagnostic(events, params, **kwargs):
    return events.ElectronPVetoDiag[params["field"]]


def _tau_pveto_diagnostic(events, params, **kwargs):
    return events[params["collection"]][params["field"]]


def _search_kinematics(events, params, **kwargs):
    event = events.AnalysisEvent
    return (
        (event.METNoMu_pt >= params["met_min"])
        & (event.leadingJet_pt > params["jet_pt_min"])
        & (abs(event.leadingJet_eta) < params["jet_eta_max"])
        & event.leadingJet_tightLepVeto
        & ((event.dijetMaxDeltaPhi < 0.0) | (event.dijetMaxDeltaPhi < params["dijet_dphi_max"]))
        & (event.leadingJetMETNoMuDeltaPhi >= params["jet_met_dphi_min"])
    )


def _single_muon_hlt(events, params, **kwargs):
    return _hlt_or(events, params["paths"])


def _single_electron_hlt(events, params, **kwargs):
    return _hlt_or(events, params["paths"])


def _met_hlt(events, params, **kwargs):
    return _hlt_or(events, params["paths"])


has_disappearing_track = Cut(
    name="has_disappearing_track",
    params={"minimum": 1},
    function=_has_disappearing_track,
)

golden_json_lumi = Cut(
    name="golden_json_lumi",
    params={},
    function=_golden_json_lumi,
)

event_flags = Cut(
    name="event_flags",
    params={},
    function=_event_flags,
)

jet_veto_map = Cut(
    name="jet_veto_map",
    params={},
    function=_jet_veto_map,
)

single_muon_hlt = Cut(
    name="single_muon_hlt",
    params={"paths": ("IsoMu24",)},
    function=_single_muon_hlt,
)

single_electron_hlt = Cut(
    name="single_electron_hlt",
    params={
        "paths": (
            "Ele32_WPTight_Gsf",
            "Ele32_WPTight_Gsf_L1DoubleEG",
            "Ele32_WPTight_Gsf_DoubleL1EG",
        )
    },
    function=_single_electron_hlt,
)

met_hlt = Cut(
    name="met_hlt",
    params={
        "paths": (
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
    },
    function=_met_hlt,
)

has_muon_tag = Cut(
    name="has_muon_tag",
    params={"field": "nMuonTag", "minimum": 1},
    function=_has_count,
)

has_muon_veto_probe_track = Cut(
    name="has_muon_veto_probe_track",
    params={"field": "nMuonVetoProbeTrack", "minimum": 1},
    function=_has_count,
)

has_muon_veto_tag_probe_pair = Cut(
    name="has_muon_veto_tag_probe_pair",
    params={"field": "nMuonVetoTagProbePair", "minimum": 1},
    function=_has_count,
)

has_muon_veto_os_pair = Cut(
    name="has_muon_veto_os_pair",
    params={"field": "nMuonVetoTagProbePairOS", "minimum": 1},
    function=_has_count,
)

has_muon_veto_os_mass10_pair = Cut(
    name="has_muon_veto_os_mass10_pair",
    params={"field": "nMuonVetoTagProbePairOSMass10", "minimum": 1},
    function=_has_count,
)

has_muon_veto_ss_pair = Cut(
    name="has_muon_veto_ss_pair",
    params={"field": "nMuonVetoTagProbePairSS", "minimum": 1},
    function=_has_count,
)

has_muon_veto_ss_mass10_pair = Cut(
    name="has_muon_veto_ss_mass10_pair",
    params={"field": "nMuonVetoTagProbePairSSMass10", "minimum": 1},
    function=_has_count,
)

has_muon_veto_zwindow_pair = Cut(
    name="has_muon_veto_zwindow_pair",
    params={"field": "nMuonVetoTagProbePairZWindow", "minimum": 1},
    function=_has_count,
)

has_muon_veto_zwindow_pass_pair = Cut(
    name="has_muon_veto_zwindow_pass_pair",
    params={"field": "nMuonVetoTagProbePairZWindowPass", "minimum": 1},
    function=_has_count,
)

has_muon_veto_zwindow_fail_pair = Cut(
    name="has_muon_veto_zwindow_fail_pair",
    params={"field": "nMuonVetoTagProbePairZWindowFail", "minimum": 1},
    function=_has_count,
)

has_muon_pveto_zwindow_pass_pair = Cut(
    name="has_muon_pveto_zwindow_pass_pair",
    params={"field": "nMuonPVetoTagProbePairZWindowPass", "minimum": 1},
    function=_has_count,
)

has_muon_veto_ss_zwindow_pair = Cut(
    name="has_muon_veto_ss_zwindow_pair",
    params={"field": "nMuonVetoTagProbePairSSZWindow", "minimum": 1},
    function=_has_count,
)

has_muon_veto_ss_zwindow_pass_pair = Cut(
    name="has_muon_veto_ss_zwindow_pass_pair",
    params={"field": "nMuonVetoTagProbePairSSZWindowPass", "minimum": 1},
    function=_has_count,
)

has_muon_veto_ss_zwindow_fail_pair = Cut(
    name="has_muon_veto_ss_zwindow_fail_pair",
    params={"field": "nMuonVetoTagProbePairSSZWindowFail", "minimum": 1},
    function=_has_count,
)

has_muon_pveto_ss_zwindow_pass_pair = Cut(
    name="has_muon_pveto_ss_zwindow_pass_pair",
    params={"field": "nMuonPVetoTagProbePairSSZWindowPass", "minimum": 1},
    function=_has_count,
)

muon_pveto_layer_cuts = {}
for layer in PVETO_LAYERS:
    muon_pveto_layer_cuts[f"muon_veto_zwindow_{layer}"] = Cut(
        name=f"has_muon_veto_zwindow_pair_{layer}",
        params={"field": f"nMuonVetoTagProbePairZWindow_{layer}", "minimum": 1},
        function=_has_count,
    )
    muon_pveto_layer_cuts[f"muon_pveto_zwindow_pass_{layer}"] = Cut(
        name=f"has_muon_pveto_zwindow_pass_pair_{layer}",
        params={"field": f"nMuonPVetoTagProbePairZWindowPass_{layer}", "minimum": 1},
        function=_has_count,
    )
    muon_pveto_layer_cuts[f"muon_veto_ss_zwindow_{layer}"] = Cut(
        name=f"has_muon_veto_ss_zwindow_pair_{layer}",
        params={"field": f"nMuonVetoTagProbePairSSZWindow_{layer}", "minimum": 1},
        function=_has_count,
    )
    muon_pveto_layer_cuts[f"muon_pveto_ss_zwindow_pass_{layer}"] = Cut(
        name=f"has_muon_pveto_ss_zwindow_pass_pair_{layer}",
        params={"field": f"nMuonPVetoTagProbePairSSZWindowPass_{layer}", "minimum": 1},
        function=_has_count,
    )


def _make_count_cut(name, field):
    return Cut(
        name=f"has_{name}",
        params={"field": field, "minimum": 1},
        function=_has_count,
    )


lepton_pveto_cuts = {
    "electron_veto_tag": _make_count_cut("electron_veto_tag", "nElectronTag"),
    "electron_veto_probe": _make_count_cut(
        "electron_veto_probe", "nElectronVetoProbeTrack"
    ),
    "electron_veto_pair": _make_count_cut(
        "electron_veto_pair", "nElectronTagProbePair"
    ),
    "electron_veto_zwindow": _make_count_cut(
        "electron_veto_zwindow", "nElectronTagProbePairOSMassWindow"
    ),
    "electron_pveto_zwindow_pass": _make_count_cut(
        "electron_pveto_zwindow_pass", "nElectronPVetoTagProbePairMassWindowPass"
    ),
    "electron_veto_ss_zwindow": _make_count_cut(
        "electron_veto_ss_zwindow", "nElectronTagProbePairSSMassWindow"
    ),
    "electron_pveto_ss_zwindow_pass": _make_count_cut(
        "electron_pveto_ss_zwindow_pass",
        "nElectronPVetoTagProbePairSSMassWindowPass",
    ),
    "tau_mu_veto_tag": _make_count_cut("tau_mu_veto_tag", "nMuonLowMTTag"),
    "tau_mu_veto_probe": _make_count_cut("tau_mu_veto_probe", "nTauVetoProbeTrack"),
    "tau_mu_veto_pair": _make_count_cut("tau_mu_veto_pair", "nTauMuTagProbePair"),
    "tau_mu_veto_masswindow": _make_count_cut(
        "tau_mu_veto_masswindow", "nTauMuTagProbePairOSMassWindow"
    ),
    "tau_mu_pveto_masswindow_pass": _make_count_cut(
        "tau_mu_pveto_masswindow_pass", "nTauMuPVetoTagProbePairMassWindowPass"
    ),
    "tau_mu_veto_ss_masswindow": _make_count_cut(
        "tau_mu_veto_ss_masswindow", "nTauMuTagProbePairSSMassWindow"
    ),
    "tau_mu_pveto_ss_masswindow_pass": _make_count_cut(
        "tau_mu_pveto_ss_masswindow_pass",
        "nTauMuPVetoTagProbePairSSMassWindowPass",
    ),
    "tau_ele_veto_tag": _make_count_cut("tau_ele_veto_tag", "nElectronLowMTTag"),
    "tau_ele_veto_probe": _make_count_cut("tau_ele_veto_probe", "nTauVetoProbeTrack"),
    "tau_ele_veto_pair": _make_count_cut("tau_ele_veto_pair", "nTauEleTagProbePair"),
    "tau_ele_veto_masswindow": _make_count_cut(
        "tau_ele_veto_masswindow", "nTauEleTagProbePairOSMassWindow"
    ),
    "tau_ele_pveto_masswindow_pass": _make_count_cut(
        "tau_ele_pveto_masswindow_pass", "nTauElePVetoTagProbePairMassWindowPass"
    ),
    "tau_ele_veto_ss_masswindow": _make_count_cut(
        "tau_ele_veto_ss_masswindow", "nTauEleTagProbePairSSMassWindow"
    ),
    "tau_ele_pveto_ss_masswindow_pass": _make_count_cut(
        "tau_ele_pveto_ss_masswindow_pass",
        "nTauElePVetoTagProbePairSSMassWindowPass",
    ),
}

fake_track_cuts = {
    "fake_basic3hits_d0_signal": _make_count_cut(
        "fake_basic3hits_d0_signal",
        "nFakeBasic3HitsD0Signal",
    ),
    "fake_basic3hits_d0_sideband": _make_count_cut(
        "fake_basic3hits_d0_sideband",
        "nFakeBasic3HitsD0Sideband",
    ),
    "fake_zmumu_control": _make_count_cut(
        "fake_zmumu_control",
        "nFakeZMuMuControl",
    ),
    "fake_zee_control": _make_count_cut(
        "fake_zee_control",
        "nFakeZeeControl",
    ),
}

for layer in (*PVETO_LAYERS, "combinedBins"):
    fake_track_cuts[f"fake_control_{layer}"] = _make_count_cut(
        f"fake_control_{layer}",
        f"nFakeControl_{layer}",
    )
    fake_track_cuts[f"fake_zmumu_sideband_{layer}"] = _make_count_cut(
        f"fake_zmumu_sideband_{layer}",
        f"nFakeZMuMuSideband_{layer}",
    )
    fake_track_cuts[f"fake_zee_sideband_{layer}"] = _make_count_cut(
        f"fake_zee_sideband_{layer}",
        f"nFakeZeeSideband_{layer}",
    )

for layer in PVETO_LAYERS:
    lepton_pveto_cuts[f"electron_veto_zwindow_{layer}"] = _make_count_cut(
        f"electron_veto_zwindow_{layer}",
        f"nElectronTagProbePairMassWindow_{layer}",
    )
    lepton_pveto_cuts[f"electron_pveto_zwindow_pass_{layer}"] = _make_count_cut(
        f"electron_pveto_zwindow_pass_{layer}",
        f"nElectronPVetoTagProbePairMassWindowPass_{layer}",
    )
    lepton_pveto_cuts[f"electron_veto_ss_zwindow_{layer}"] = _make_count_cut(
        f"electron_veto_ss_zwindow_{layer}",
        f"nElectronTagProbePairSSMassWindow_{layer}",
    )
    lepton_pveto_cuts[f"electron_pveto_ss_zwindow_pass_{layer}"] = _make_count_cut(
        f"electron_pveto_ss_zwindow_pass_{layer}",
        f"nElectronPVetoTagProbePairSSMassWindowPass_{layer}",
    )
    for tau_prefix, field_prefix, category_prefix in (
        ("tau_mu", "TauMu", "tau_mu"),
        ("tau_ele", "TauEle", "tau_ele"),
    ):
        lepton_pveto_cuts[f"{category_prefix}_veto_masswindow_{layer}"] = _make_count_cut(
            f"{category_prefix}_veto_masswindow_{layer}",
            f"n{field_prefix}TagProbePairMassWindow_{layer}",
        )
        lepton_pveto_cuts[f"{category_prefix}_pveto_masswindow_pass_{layer}"] = (
            _make_count_cut(
                f"{category_prefix}_pveto_masswindow_pass_{layer}",
                f"n{field_prefix}PVetoTagProbePairMassWindowPass_{layer}",
            )
        )
        lepton_pveto_cuts[f"{category_prefix}_veto_ss_masswindow_{layer}"] = (
            _make_count_cut(
                f"{category_prefix}_veto_ss_masswindow_{layer}",
                f"n{field_prefix}TagProbePairSSMassWindow_{layer}",
            )
        )
        lepton_pveto_cuts[f"{category_prefix}_pveto_ss_masswindow_pass_{layer}"] = (
            _make_count_cut(
                f"{category_prefix}_pveto_ss_masswindow_pass_{layer}",
                f"n{field_prefix}PVetoTagProbePairSSMassWindowPass_{layer}",
            )
        )

lepton_background_cuts = {}
for layer in (*PVETO_LAYERS, "combinedBins"):
    for category_prefix, field_prefix in (
        ("muon", "Muon"),
        ("electron", "Electron"),
        ("tau_mu", "TauMu"),
        ("tau_ele", "TauEle"),
    ):
        lepton_background_cuts[f"{category_prefix}_background_control_{layer}"] = (
            _make_count_cut(
                f"{category_prefix}_background_control_{layer}",
                f"n{field_prefix}BackgroundControl_{layer}",
            )
        )
        lepton_background_cuts[f"{category_prefix}_background_offline_{layer}"] = (
            _make_count_cut(
                f"{category_prefix}_background_offline_{layer}",
                f"n{field_prefix}BackgroundOffline_{layer}",
            )
        )
        lepton_background_cuts[f"{category_prefix}_background_trigger_{layer}"] = (
            _make_count_cut(
                f"{category_prefix}_background_trigger_{layer}",
                f"n{field_prefix}BackgroundTrigger_{layer}",
            )
        )

search_kinematics = Cut(
    name="search_kinematics",
    params={
        "met_min": 120.0,
        "jet_pt_min": 110.0,
        "jet_eta_max": 2.4,
        "jet_met_dphi_min": 0.5,
        "dijet_dphi_max": 2.5,
    },
    function=_search_kinematics,
)

# Defining the basic selection from the AN. The MET triggers are applied
# as preselections which is why they do not show up here.
basic_event_selection = Cut(
    name="basic_event_selection",
    params={
        "met_min": 120.0,
        "jet_pt_min": 110.0,
        "jet_eta_max": 2.4,
        "dijet_dphi_max": 2.5,
        "jet_met_dphi_min": 0.5,
    },
    function=_search_kinematics,
)

basic_selection = basic_event_selection
isolated_track_selection = Cut(
    name="isolated_track_selection",
    params={"field": "nIsoTrackIsolated", "minimum": 1},
    function=_has_count,
)
candidate_track_selection = Cut(
    name="candidate_track_selection",
    params={"field": "nIsoTrackCandidate", "minimum": 1},
    function=_has_count,
)
disappearing_track_selection = has_disappearing_track

search_diagnostic_cuts = {
    field: Cut(
        name=field,
        params={"field": field},
        function=_search_diagnostic,
    )
    for field in SEARCH_DIAGNOSTIC_FIELDS
}

fake_track_diagnostic_cuts = {
    field: Cut(
        name=f"fake_track_diag_{field}",
        params={"field": field},
        function=_fake_track_diagnostic,
    )
    for field in FAKE_TRACK_DIAGNOSTIC_FIELDS
}

fake_z_control_diagnostic_cuts = {
    f"{mode}_{field}": Cut(
        name=f"fake_{mode}_diag_{field}",
        params={"collection": collection, "field": field},
        function=_fake_z_control_diagnostic,
    )
    for mode, collection, fields in (
        ("zmumu", "FakeZMuMuDiag", FAKE_ZMUMU_DIAGNOSTIC_FIELDS),
        ("zee", "FakeZeeDiag", FAKE_ZEE_DIAGNOSTIC_FIELDS),
    )
    for field in fields
}

muon_table16_cuts = {
    field: Cut(
        name=f"muon_table16_{field}",
        params={"field": field},
        function=_muon_table16_diagnostic,
    )
    for field in MUON_TABLE16_FIELDS
}

electron_pveto_diagnostic_cuts = {
    field: Cut(
        name=f"electron_pveto_diag_{field}",
        params={"field": field},
        function=_electron_pveto_diagnostic,
    )
    for field in ELECTRON_PVETO_DIAGNOSTIC_FIELDS
}

tau_pveto_diagnostic_cuts = {
    f"{mode}_{field}": Cut(
        name=f"{mode}_pveto_diag_{field}",
        params={"collection": collection, "field": field},
        function=_tau_pveto_diagnostic,
    )
    for mode, collection in (
        ("tau_mu", "TauMuPVetoDiag"),
        ("tau_ele", "TauElePVetoDiag"),
    )
    for field in TAU_PVETO_DIAGNOSTIC_FIELDS
}
