"""PocketCoffea workflow for the custom disappearing-track NanoAOD."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import awkward as ak
import numpy as np
import uproot
from coffea.lumi_tools import LumiMask

from pocket_coffea.workflows.base import BaseProcessorABC
from pocket_coffea.utils.skim import copy_file, uproot_writeable

from disapptrks.selections import (
    add_electron_derived_fields,
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
    fake_track_base_mask,
    fake_track_layer_cut,
    fake_track_no_d0_mask,
    fiducial_map_probe_track_mask,
    generic_probe_pair_layer_mask,
    hadronic_tau_control_object_mask,
    invariant_mass,
    isolated_track_selection_mask,
    layer_mask,
    lepton_veto_probe_track_mask,
    low_mt_mask,
    mass10_muon_probe_pair_mask,
    mass_window_pair_mask,
    met_no_mu_minus_lepton,
    minimum_delta_r,
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
    generic_probe_pair_layer_mask,
    search_event_cutflow_masks,
    search_track_cutflow_masks,
    search_track_mask,
    single_electron_trigger_mask,
    trigger_matched_track_mask,
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
from disapptrks.triggers import ISO_MUON_REFERENCE_TRIGGER, tau_cross_trigger_for_year

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


# Group-shared fiducial maps, kept current for every run period. Used automatically
# when no explicit DISAPPTRKS_*_FIDUCIAL_MAP_JSON/_DIR override is set, so a job
# doesn't need era-specific paths hand-configured (and can't silently pick up a
# stale local copy). Override the base location with DISAPPTRKS_FIDUCIAL_MAP_EOS_BASE
# if the group's shared space ever moves.
FIDUCIAL_MAP_EOS_BASE_DEFAULT = (
    "root://cmseos.fnal.gov//store/group/lpcdisapptrks/fiducialmaps"
)

# self._year -> the era string used in the shared maps' filenames. Mirrors the same
# 2022/2023 run-period split _jet_veto_map_parameter_year already encodes for the jet
# veto maps; kept as a separate table here since it's a different auxiliary file with
# its own naming convention, not because the underlying physics grouping differs.
FIDUCIAL_MAP_ERAS = {
    "2022_preEE": "2022CD",
    "2022_postEE": "2022EFG",
    "2023_preBPix": "2023C",
    "2023_postBPix": "2023D",
    "2024": "2024",
    "2025": "2025",
    "2026": "2026",
}


def _fiducial_map_era(year, era) -> str:
    year = str(year)
    if year in FIDUCIAL_MAP_ERAS:
        return FIDUCIAL_MAP_ERAS[year]
    # Defensive fallback for a bare year without a pre/post suffix, disambiguated
    # by the era letter -- only expected for an older dataset JSON that predates
    # the compound year convention confirmed elsewhere in this codebase.
    if year == "2022":
        return "2022CD" if str(era) in ("C", "D") else "2022EFG"
    if year == "2023":
        return "2023C" if str(era) == "C" else "2023D"
    return year


def _eos_fiducial_map_urls(flavor: str, mapped_era: str) -> list[str]:
    base = os.environ.get(
        "DISAPPTRKS_FIDUCIAL_MAP_EOS_BASE", FIDUCIAL_MAP_EOS_BASE_DEFAULT
    )
    return [f"{base}/{flavor}_fiducial_map_{mapped_era}_v2.json"]


def _read_eos_json(url: str):
    try:
        from XRootD import client
    except ImportError as err:
        raise ImportError(
            "Install XRootD python bindings with: conda install -c conda-forge xrootd"
        ) from err

    xrd_file = client.File()
    status, _ = xrd_file.open(url)
    if not status.ok:
        xrd_file.close()
        return None
    try:
        _, data = xrd_file.read()
    finally:
        xrd_file.close()
    if not data:
        return None
    return json.loads(data)


def _eos_fiducial_hot_spots(flavor: str, year, era):
    mapped_era = _fiducial_map_era(year, era)
    for url in _eos_fiducial_map_urls(flavor, mapped_era):
        payload = _read_eos_json(url)
        if payload is None:
            continue
        hot_spots = tuple(payload.get("hot_spots", ()))
        if (
            not hot_spots
            and os.environ.get("DISAPPTRKS_REQUIRE_FIDUCIAL_MAPS", "").lower()
            in ("1", "true", "yes", "on")
        ):
            raise ValueError(f"Fiducial map {url} has no hot spots")
        print(f"Loaded {len(hot_spots)} {flavor} fiducial-map hot spot(s) from {url}")
        return hot_spots
    return None


def _load_fiducial_hot_spots(
    flavor: str, year=None, era=None
) -> tuple[dict[str, float], ...]:
    hot_spot_env_name = f"DISAPPTRKS_{flavor.upper()}_FIDUCIAL_HOT_SPOTS_JSON"
    hot_spot_env = os.environ.get(hot_spot_env_name)
    if hot_spot_env:
        hot_spots = tuple(json.loads(hot_spot_env))
        if (
            not hot_spots
            and os.environ.get("DISAPPTRKS_REQUIRE_FIDUCIAL_MAPS", "").lower()
            in ("1", "true", "yes", "on")
        ):
            raise ValueError(f"Embedded {flavor} fiducial map has no hot spots")
        print(f"Loaded {len(hot_spots)} embedded {flavor} fiducial-map hot spot(s)")
        return hot_spots

    path = _fiducial_map_path(flavor)
    if path is not None:
        print(f"Opening {flavor} fiducial map from {path} in {Path.cwd()}")
        with path.open() as handle:
            payload = json.load(handle)
        hot_spots = tuple(payload.get("hot_spots", ()))
        if (
            not hot_spots
            and os.environ.get("DISAPPTRKS_REQUIRE_FIDUCIAL_MAPS", "").lower()
            in ("1", "true", "yes", "on")
        ):
            raise ValueError(f"Fiducial map {path} has no hot spots")
        print(f"Loaded {len(hot_spots)} {flavor} fiducial-map hot spot(s) from {path}")
        return hot_spots

    # No explicit override configured -- automatically resolve the era-appropriate
    # map from the group's shared EOS space, based on the dataset's own year/era,
    # rather than requiring a hand-set local path per era.
    if year is not None:
        hot_spots = _eos_fiducial_hot_spots(flavor, year, era)
        if hot_spots is not None:
            return hot_spots

    if os.environ.get("DISAPPTRKS_REQUIRE_FIDUCIAL_MAPS", "").lower() in (
        "1",
        "true",
        "yes",
        "on",
    ):
        raise FileNotFoundError(
            f"No fiducial-map path configured or found on the shared EOS space for "
            f"{flavor} (year={year!r}, era={era!r}). Set "
            f"DISAPPTRKS_{flavor.upper()}_FIDUCIAL_MAP_JSON or "
            "DISAPPTRKS_FIDUCIAL_MAP_DIR to override."
        )
    return ()


# Group-shared canonical dataset JSONs, published with ``disapptrks make-dataset-json
# --publish`` so every checkout runs against the same input-file lists instead of each
# person's local `datasets/` accumulating its own ad hoc copies. Mirrors the
# FIDUCIAL_MAP_EOS_BASE_DEFAULT pattern above; override with DISAPPTRKS_DATASET_EOS_BASE
# if the group's shared space ever moves.
DATASET_JSON_EOS_BASE_DEFAULT = (
    "root://cmseos.fnal.gov//store/group/lpcdisapptrks/dataset_jsons"
)


def _eos_dataset_json_url(name: str) -> str:
    base = os.environ.get("DISAPPTRKS_DATASET_EOS_BASE", DATASET_JSON_EOS_BASE_DEFAULT)
    return f"{base}/{name}.json"


def resolve_dataset_json(value: str, *, localdir: str) -> str:
    """Resolve DISAPPTRKS_DATASET_JSON to a local file path.

    ``value`` is tried, in order, as: an absolute path; a path relative to the
    current directory; a path relative to ``localdir`` (the existing dev/local
    override -- unchanged from before this function existed); and finally a
    canonical dataset name (e.g. ``eos_2023C_Muon`` or ``eos_2023C_Muon_OSUv2``,
    the ``.json`` suffix optional) published to the group's shared EOS
    dataset-JSON space, which is fetched here on demand since PocketCoffea
    needs a real local file rather than a URL.
    """
    if os.path.isabs(value) or os.path.exists(value):
        return value
    local_candidate = os.path.join(localdir, value)
    if os.path.exists(local_candidate):
        return local_candidate

    name = value[:-5] if value.endswith(".json") else value
    url = _eos_dataset_json_url(name)
    payload = _read_eos_json(url)
    if payload is None:
        raise FileNotFoundError(
            f"Dataset JSON {value!r} was not found locally under {localdir} and "
            f"not found on the shared EOS space at {url}. Use an existing "
            "canonical name (published with `disapptrks make-dataset-json "
            "--publish`) or a local/absolute path for a dev override."
        )
    fd, tmp_path = tempfile.mkstemp(prefix=f"disapptrks-dataset-{name}-", suffix=".json")
    with os.fdopen(fd, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)
    print(f"Resolved dataset JSON {value!r} from {url}")
    return tmp_path


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
    return ak.ones_like(events.event, dtype=bool)


def _hlt_or_mask(events, names):
    mask = None
    for name in names:
        if "HLT" in events.fields and name in events.HLT.fields:
            mask = events.HLT[name] if mask is None else (mask | events.HLT[name])
    return mask if mask is not None else _all_true_like(events) & False


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
    return _hlt_or_mask(events, ("IsoMu24",))


def _tau_probability_single_muon_trigger_mask(events):
    return _hlt_or_mask(events, (ISO_MUON_REFERENCE_TRIGGER,))


def _muon_tau_trigger_mask(events, year):
    return _hlt_or_mask(events, (tau_cross_trigger_for_year(year),))


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
    return _hlt_or_mask(events, names)


def _leading_jet_delta_phi(events, phi):
    import awkward as ak

    good_jets = events.Jet[
        (events.Jet.pt > 30.0)
        & (abs(events.Jet.eta) < 4.5)
        & run3_tight_lepton_veto_jet_mask(events.Jet)
    ]
    leading_phi = ak.fill_none(ak.firsts(good_jets.phi), 0.0)
    return abs(delta_phi(leading_phi, phi))


def _lepton_background_track_mask(
    tracks,
    *,
    flavor: str,
    layer: str,
    matched_object_d_r=None,
    require_dedx_max_over_median: bool = True,
):
    mask = base_probe_track_mask(
        tracks,
        pt_min=55.0,
        layer=layer,
        # Legacy TauTagPt55 explicitly removes cutTrkJetDeltaPhi because the
        # tau-matched track naturally lies in the reconstructed tau jet.
        apply_jet_cut=(flavor != "tau"),
        apply_calo_cut=(flavor == "muon"),
        apply_outer_hits_cut=False,
        # The Poffline/Pmiss control track is the same kind of track the
        # signal selection targets, so it gets the same high-purity/dE/dx
        # requirement as the Pveto probe tracks (base_probe_track_mask
        # already applies high-purity by default; this adds dE/dx).
        require_dedx_max_over_median=require_dedx_max_over_median,
    )
    if flavor == "electron":
        mask = mask & ((tracks.dRMinElectron >= 0.0) & (tracks.dRMinElectron < 0.1))
    elif flavor == "muon":
        mask = mask & ((tracks.dRMinMuon >= 0.0) & (tracks.dRMinMuon < 0.1))
    elif flavor == "tau":
        if matched_object_d_r is None:
            raise ValueError("tau background track selection requires selected-tau dR")
        mask = mask & ((matched_object_d_r >= 0.0) & (matched_object_d_r < 0.1))
    else:
        raise ValueError(f"unknown lepton background flavor: {flavor}")
    return mask


def _lepton_background_tag_pt55_event_mask(analysis_event):
    """Event-side cuts in the legacy Electron/Muon/TauTagPt55 channels."""

    return (
        (analysis_event.leadingJet_pt > 110.0)
        & (abs(analysis_event.leadingJet_eta) < 2.4)
        & analysis_event.leadingJet_tightLepVeto
        & (
            (analysis_event.dijetMaxDeltaPhi < 0.0)
            | (analysis_event.dijetMaxDeltaPhi < 2.5)
        )
    )


def _z_to_mumu_control_mask(events, muons):
    selected = muons[muon_tag_mask(muons)]
    first, second = ak.unzip(ak.combinations(selected, 2, axis=1))
    pair_mass = invariant_mass(first, second, first_mass=MUON_MASS, second_mass=MUON_MASS)
    os_z = (first.charge * second.charge < 0) & (abs(pair_mass - Z_MASS) < 10.0)
    return _single_muon_trigger_mask(events) & (ak.num(selected) == 2) & ak.any(os_z, axis=1)


def _z_to_mumu_control_diagnostics(
    events, muons, *, fiducial_hot_spots=(), require_dedx_max_over_median=True
):
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
    base_mask = _fake_track_base_mask_with_fiducial(
        events.IsoTrack, d0_region="sideband", fiducial_hot_spots=fiducial_hot_spots
    )
    for layer in (*PVETO_LAYERS, "combinedBins"):
        diagnostics[f"sideband_{layer}"] = (
            _fake_track_count_for_control(
                events,
                z_os_window,
                layer=layer,
                d0_region="sideband",
                fiducial_hot_spots=fiducial_hot_spots,
                require_dedx_max_over_median=require_dedx_max_over_median,
                base_mask=base_mask,
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
    trigger_match_ok = (
        electrons.matchedSingleElectron
        if "matchedSingleElectron" in electrons.fields
        else True
    )
    return (
        (electrons.pt > pt_min)
        & (abs(electrons.eta) < eta_max)
        & (electrons.cutBased >= 4)
        & dxy_ok
        & dz_ok
        & trigger_match_ok
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


def _z_to_ee_control_diagnostics(
    events, electrons, *, fiducial_hot_spots=(), require_dedx_max_over_median=True
):
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
    base_mask = _fake_track_base_mask_with_fiducial(
        events.IsoTrack, d0_region="sideband", fiducial_hot_spots=fiducial_hot_spots
    )
    for layer in (*PVETO_LAYERS, "combinedBins"):
        diagnostics[f"sideband_{layer}"] = (
            _fake_track_count_for_control(
                events,
                z_os_window,
                layer=layer,
                d0_region="sideband",
                fiducial_hot_spots=fiducial_hot_spots,
                require_dedx_max_over_median=require_dedx_max_over_median,
                base_mask=base_mask,
            )
            >= 1
        )
    return diagnostics


def _fake_track_mask(
    tracks,
    *,
    layer,
    d0_region,
    fiducial_hot_spots=(),
    sideband_min: float = 0.05,
    sideband_max: float = 0.50,
    require_high_purity: bool = True,
    require_dedx_max_over_median: bool = True,
):
    mask = fake_track_no_d0_mask(
        tracks,
        layer=layer,
        d0_region=d0_region,
        sideband_min=sideband_min,
        sideband_max=sideband_max,
        require_high_purity=require_high_purity,
        require_dedx_max_over_median=require_dedx_max_over_median,
    )
    if fiducial_hot_spots:
        mask = mask & _outside_fiducial_hot_spots(tracks, fiducial_hot_spots)
    return mask


def _fake_track_base_mask_with_fiducial(
    tracks,
    *,
    d0_region,
    fiducial_hot_spots=(),
    sideband_min: float = 0.05,
    sideband_max: float = 0.50,
):
    """Layer-independent term of ``_fake_track_mask``, computed once.

    A caller looping over several layer bins for the same tracks/d0_region
    should compute this once and combine it with ``fake_track_layer_cut``
    per bin (see ``base_mask=`` on ``_fake_track_count_for_control`` and
    friends), instead of calling ``_fake_track_mask`` fresh for every layer
    and re-evaluating the same layer-independent terms each time.
    """

    mask = fake_track_base_mask(
        tracks,
        d0_region=d0_region,
        sideband_min=sideband_min,
        sideband_max=sideband_max,
    )
    if fiducial_hot_spots:
        mask = mask & _outside_fiducial_hot_spots(tracks, fiducial_hot_spots)
    return mask


def _high_purity_study_tracks_for_control(
    events, control_mask, *, fiducial_hot_spots=()
):
    """Sideband candidates before the analysis high-purity requirement.

    All other nominal fake-track sideband requirements are retained.  Keeping
    this collection separate prevents the study from changing the background
    estimate's nominal high-purity selection.
    """

    candidate_mask = _fake_track_mask(
        events.IsoTrack,
        layer="combinedBins",
        d0_region="sideband",
        fiducial_hot_spots=fiducial_hot_spots,
        require_high_purity=False,
    )
    source_indices = ak.local_index(events.IsoTrack, axis=1)[candidate_mask]
    tracks = events.IsoTrack[candidate_mask]
    control_track_mask, _ = ak.broadcast_arrays(control_mask, tracks.pt)
    tracks = tracks[control_track_mask]
    source_indices = source_indices[control_track_mask]
    return ak.with_field(tracks, source_indices, "sourceIsoTrackIdx")


def _dedx_hits_grouped_by_track(events, tracks):
    """Group DeDxHitInfo rows under each selected IsoTrack row."""
    if "IsoTrackDeDxHit" not in events.fields:
        raise AttributeError(
            "high-purity dE/dx hit histograms require the IsoTrackDeDxHit table"
        )
    hits = events.IsoTrackDeDxHit
    pairs = ak.cartesian(
        {
            "selectedTrackIdx": tracks.sourceIsoTrackIdx,
            "hit": hits,
        },
        axis=1,
        nested=True,
    )
    return pairs.hit[
        pairs.hit.isoTrackIdx == pairs.selectedTrackIdx
    ]


def _prepare_dedx_hits_for_histograms(selected):
    """Add detector fields and mask nonphysical hit-level sentinels."""

    selected = ak.with_field(
        selected,
        10 * selected.subdet + selected.layer,
        "detectorLayer",
    )
    # Pixel cluster sizes are encoded as -1 for strip hits.  Mask them rather
    # than letting the sentinel dominate the physical pixel distributions.
    for field in ("pixelSize", "pixelSizeX", "pixelSizeY"):
        selected = ak.with_field(
            selected,
            ak.mask(selected[field], selected.isPixel != 0),
            field,
        )
    selected = ak.with_field(
        selected,
        ak.mask(selected.passesStripShapeSelection, selected.isPixel == 0),
        "stripPassesShapeSelection",
    )
    return selected


def _dedx_hits_for_high_purity_tracks(events, tracks):
    """Select DeDxHitInfo rows associated with a jagged IsoTrack collection."""

    grouped_hits = _dedx_hits_grouped_by_track(events, tracks)
    return _prepare_dedx_hits_for_histograms(
        ak.flatten(grouped_hits, axis=2)
    )


def _dedx_track_summaries(tracks, grouped_hits):
    """Attach per-track summaries of associated retained dE/dx hits."""

    dedx = grouped_hits.dEdx
    n_hits = ak.num(dedx, axis=-1)
    layers = tracks.hp_nValidTrackerHits
    if "hp_trackerLayersWithMeasurement" in tracks.fields:
        layers = tracks.hp_trackerLayersWithMeasurement

    sorted_dedx = ak.sort(dedx, axis=-1)
    positions = ak.local_index(sorted_dedx, axis=-1)
    lower_middle = (n_hits - 1) // 2
    upper_middle = n_hits // 2
    middle_values = sorted_dedx[
        (positions == lower_middle) | (positions == upper_middle)
    ]
    median = ak.mean(middle_values, axis=-1, mask_identity=True)
    maximum = ak.max(dedx, axis=-1, mask_identity=True)
    minimum = ak.min(dedx, axis=-1, mask_identity=True)
    dedx_sum = ak.sum(dedx, axis=-1)
    truncated_denominator = ak.where(n_hits > 1, n_hits - 1, 1)
    truncated_mean = ak.where(
        n_hits > 1,
        (dedx_sum - maximum) / truncated_denominator,
        maximum,
    )
    safe_median = ak.where(median > 0, median, 1.0)
    maximum_over_median = ak.mask(
        maximum / safe_median,
        median > 0,
    )

    strip_hits = grouped_hits[grouped_hits.isPixel == 0]
    n_strip_hits = ak.num(strip_hits, axis=-1)
    n_strip_shape_failures = ak.sum(
        strip_hits.passesStripShapeSelection == 0,
        axis=-1,
    )
    strip_denominator = ak.where(n_strip_hits > 0, n_strip_hits, 1)
    strip_failure_fraction = ak.mask(
        n_strip_shape_failures / strip_denominator,
        n_strip_hits > 0,
    )

    summaries = tracks
    fields = {
        "nRetainedDeDxHits": n_hits,
        "nRetainedDeDxHitsMinusLayers": n_hits - layers,
        "dEdxMedian": median,
        "dEdxTruncatedMeanDropMaximum": truncated_mean,
        "dEdxMaximum": maximum,
        "dEdxStdDev": ak.std(dedx, axis=-1, mask_identity=True),
        "dEdxRange": maximum - minimum,
        "dEdxMaximumOverMedian": maximum_over_median,
        "nDeDxHitsAbove10": ak.sum(dedx >= 10.0, axis=-1),
        "nDeDxHitsAbove20": ak.sum(dedx >= 20.0, axis=-1),
        "nStripDeDxHits": n_strip_hits,
        "nStripShapeFailures": n_strip_shape_failures,
        "stripShapeFailureFraction": strip_failure_fraction,
    }
    for field, values in fields.items():
        summaries = ak.with_field(summaries, values, field)
    return summaries


def _fake_track_count_for_control(
    events,
    control_mask,
    *,
    layer,
    d0_region,
    fiducial_hot_spots=(),
    require_dedx_max_over_median=True,
    base_mask=None,
):
    if base_mask is not None:
        mask = base_mask & fake_track_layer_cut(
            events.IsoTrack,
            layer=layer,
            require_dedx_max_over_median=require_dedx_max_over_median,
        )
    else:
        mask = _fake_track_mask(
            events.IsoTrack,
            layer=layer,
            d0_region=d0_region,
            fiducial_hot_spots=fiducial_hot_spots,
            require_dedx_max_over_median=require_dedx_max_over_median,
        )
    count = ak.num(events.IsoTrack[mask])
    return ak.where(control_mask, count, 0)


def _fake_fit_tracks_for_control(
    events,
    control_mask,
    *,
    fiducial_hot_spots=(),
    require_dedx_max_over_median=True,
):
    tracks = events.IsoTrack[
        _fake_track_mask(
            events.IsoTrack,
            layer="NLayers4",
            d0_region="sideband",
            fiducial_hot_spots=fiducial_hot_spots,
            sideband_min=0.0,
            sideband_max=0.5,
            require_dedx_max_over_median=require_dedx_max_over_median,
        )
    ]
    tracks = ak.with_field(tracks, abs(tracks.dxy), "absDxy")
    control_track_mask, _ = ak.broadcast_arrays(control_mask, tracks.pt)
    return tracks[control_track_mask]


def _fake_sideband_tracks_for_control(
    events,
    control_mask,
    *,
    layer,
    fiducial_hot_spots=(),
    require_dedx_max_over_median=True,
    base_mask=None,
):
    """Return the candidates entering an event-level N_sideband numerator.

    The returned jagged collection retains every qualifying track in events
    passing the requested Z control region.  Its event population is exactly
    the one tested by ``_fake_track_count_for_control`` with the nominal
    0.05 <= |d0| < 0.50 cm sideband.
    """

    if base_mask is not None:
        mask = base_mask & fake_track_layer_cut(
            events.IsoTrack,
            layer=layer,
            require_dedx_max_over_median=require_dedx_max_over_median,
        )
    else:
        mask = _fake_track_mask(
            events.IsoTrack,
            layer=layer,
            d0_region="sideband",
            fiducial_hot_spots=fiducial_hot_spots,
            require_dedx_max_over_median=require_dedx_max_over_median,
        )
    tracks = events.IsoTrack[mask]
    control_track_mask, _ = ak.broadcast_arrays(control_mask, tracks.pt)
    return tracks[control_track_mask]


def _add_fake_sideband_track_diagnostics(
    events,
    control_mask,
    *,
    control_key,
    layer,
    fiducial_hot_spots=(),
    require_dedx_max_over_median=True,
    base_mask=None,
):
    """Attach selected-track diagnostics without changing the sideband selection."""

    if base_mask is not None:
        candidate_mask = base_mask & fake_track_layer_cut(
            events.IsoTrack,
            layer=layer,
            require_dedx_max_over_median=require_dedx_max_over_median,
        )
    else:
        candidate_mask = _fake_track_mask(
            events.IsoTrack,
            layer=layer,
            d0_region="sideband",
            fiducial_hot_spots=fiducial_hot_spots,
            require_dedx_max_over_median=require_dedx_max_over_median,
        )
    indices = ak.local_index(events.IsoTrack, axis=1)[candidate_mask]
    candidates = events.IsoTrack[candidate_mask]
    control_candidate_mask, _ = ak.broadcast_arrays(control_mask, candidates.pt)
    indices = indices[control_candidate_mask]
    candidates = candidates[control_candidate_mask]
    candidates = ak.with_field(candidates, indices, "isoTrackIdx")

    prefix = f"Fake{control_key}Sideband"
    events[f"{prefix}Track_{layer}"] = candidates
    events[f"n{prefix}Candidates_{layer}"] = ak.num(candidates)
    events[f"n{prefix}HighPurityCandidates_{layer}"] = ak.sum(
        candidates.isHighPurityTrack, axis=1
    )


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
    try:
        cwd = Path.cwd()
    except OSError:
        cwd = None
    if cwd is not None:
        search_dirs.append(cwd / "data" / "golden_jsons")
        search_dirs.append(cwd / "golden_jsons")

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
    mapped_year = str(mapped_year)
    payload = GOLDEN_JSON_PAYLOADS.get(mapped_year)
    if payload is None:
        payload = GOLDEN_JSON_PAYLOADS.get(mapped_year.split("_", 1)[0])
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
    def process(self, events):
        """Run a chunk from a stable working directory.

        Condor may remove the sandbox directory while a long-lived Dask worker
        is still alive.  Python path handling and Numba compilation both fail
        when the process cwd no longer exists, even when all required payloads
        are otherwise available.
        """

        try:
            os.getcwd()
        except OSError:
            os.chdir(tempfile.gettempdir())
        return super().process(events)

    def export_skimmed_chunk(self):
        """Export a skim through stable worker-local scratch.

        Dask workers can outlive the Condor sandbox directory that was their
        original current working directory.  Upstream PocketCoffea writes a
        bare relative filename there, which then fails with ENOENT.  Use an
        explicit temporary directory and retain the upstream output metadata.
        """

        if self._category_mode() != "z_sideband_skim":
            return super().export_skimmed_chunk()

        if self._isMC:
            skimmed_sumw = ak.sum(self.events.genWeight)
            if skimmed_sumw == 0:
                self.events["skimRescaleGenWeight"] = np.zeros(
                    self.nEvents_after_skim
                )
            else:
                self.events["skimRescaleGenWeight"] = (
                    np.ones(self.nEvents_after_skim)
                    * self.output["sum_genweights"][self._dataset]
                    / skimmed_sumw
                )
            self.output["sum_genweights_skimmed"] = {
                self._dataset: skimmed_sumw
            }

        filename = "__".join(
            [
                self._dataset,
                str(self.events.metadata["fileuuid"]),
                str(self.events.metadata["entrystart"]),
                str(self.events.metadata["entrystop"]),
            ]
        ) + ".root"
        with tempfile.TemporaryDirectory(prefix="disapptrks-skim-") as scratch:
            local_path = os.path.join(scratch, filename)
            with uproot.recreate(local_path, compression=uproot.ZSTD(5)) as fout:
                fout["Events"] = uproot_writeable(self.events)
            copy_file(
                filename,
                scratch,
                self.cfg.save_skimmed_files_folder,
                subdirs=[self._dataset],
            )

        self.output["skimmed_files"] = {
            self._dataset: [
                os.path.join(
                    self.cfg.save_skimmed_files_folder,
                    self._dataset,
                    filename,
                )
            ]
        }
        self.output["nskimmed_events"] = {
            self._dataset: [self.nEvents_after_skim]
        }

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

    def _fake_sideband_histograms_enabled(self):
        try:
            return bool(self.params.disapptrks.fake_sideband_histograms)
        except Exception:
            return os.environ.get(
                "DISAPPTRKS_ENABLE_FAKE_SIDEBAND_HISTOGRAMS", "1"
            ).lower() in ("1", "true", "yes", "on")

    def _fake_track_dedx_cut_enabled(self):
        try:
            return bool(self.params.disapptrks.fake_track_require_dedx_cut)
        except Exception:
            return os.environ.get(
                "DISAPPTRKS_FAKE_TRACK_REQUIRE_DEDX_CUT", "1"
            ).lower() in ("1", "true", "yes", "on")

    def _lepton_background_dedx_cut_enabled(self):
        try:
            return bool(self.params.disapptrks.lepton_background_require_dedx_cut)
        except Exception:
            return os.environ.get(
                "DISAPPTRKS_LEPTON_BACKGROUND_REQUIRE_DEDX_CUT", "1"
            ).lower() in ("1", "true", "yes", "on")

    def _high_purity_dedx_histograms_enabled(self):
        try:
            return bool(self.params.disapptrks.high_purity_dedx_histograms)
        except Exception:
            return os.environ.get(
                "DISAPPTRKS_ENABLE_HIGH_PURITY_DEDX_HISTOGRAMS", "0"
            ).lower() in ("1", "true", "yes", "on")

    def _signal_dedx_histograms_enabled(self):
        try:
            return bool(self.params.disapptrks.signal_dedx_histograms)
        except Exception:
            return os.environ.get(
                "DISAPPTRKS_ENABLE_SIGNAL_DEDX_HISTOGRAMS", "0"
            ).lower() in ("1", "true", "yes", "on")

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
                flavor, year=self._year, era=self._era
            )
        return self._disapptrks_fiducial_hot_spots[flavor]

    def _lepton_fiducial_hot_spots(self, *flavors):
        if not flavors:
            flavors = ("electron", "muon")
        hot_spots = ()
        for flavor in flavors:
            hot_spots += self._fiducial_hot_spots(flavor)
        return hot_spots

    def _event_int_like(self, value):
        return ak.ones_like(self.events.event, dtype=np.int64) * int(value)

    def _store_trigger_efficiency_counts(
        self,
        *,
        prefix,
        pairs,
        mass_mask_function,
    ):
        pair_mask = mass_mask_function(pairs) & (pairs.probe_pt > 55.0)
        os_pairs = pairs[pair_mask & pairs.os]
        ss_pairs = pairs[pair_mask & pairs.ss]
        os_trigger_pairs = pairs[pair_mask & pairs.os & pairs.probe_firesTrigger]
        ss_trigger_pairs = pairs[pair_mask & pairs.ss & pairs.probe_firesTrigger]
        self.events[f"n{prefix}TriggerEffProbesPT55"] = ak.values_astype(
            ak.num(os_pairs),
            np.int64,
        )
        self.events[f"n{prefix}TriggerEffProbesSSPT55"] = ak.values_astype(
            ak.num(ss_pairs),
            np.int64,
        )
        self.events[f"n{prefix}TriggerEffProbesFiringTrigger"] = ak.values_astype(
            ak.num(os_trigger_pairs),
            np.int64,
        )
        self.events[f"n{prefix}TriggerEffSSProbesFiringTrigger"] = ak.values_astype(
            ak.num(ss_trigger_pairs),
            np.int64,
        )
        for layer in PVETO_LAYERS:
            layer_pair_mask = pair_mask & generic_probe_pair_layer_mask(pairs, layer)
            layer_os_pairs = pairs[layer_pair_mask & pairs.os]
            layer_ss_pairs = pairs[layer_pair_mask & pairs.ss]
            layer_os_trigger_pairs = pairs[
                layer_pair_mask & pairs.os & pairs.probe_firesTrigger
            ]
            layer_ss_trigger_pairs = pairs[
                layer_pair_mask & pairs.ss & pairs.probe_firesTrigger
            ]
            self.events[f"n{prefix}TriggerEffProbesPT55_{layer}"] = ak.values_astype(
                ak.num(layer_os_pairs),
                np.int64,
            )
            self.events[f"n{prefix}TriggerEffProbesSSPT55_{layer}"] = ak.values_astype(
                ak.num(layer_ss_pairs),
                np.int64,
            )
            self.events[
                f"n{prefix}TriggerEffProbesFiringTrigger_{layer}"
            ] = ak.values_astype(
                ak.num(layer_os_trigger_pairs),
                np.int64,
            )
            self.events[
                f"n{prefix}TriggerEffSSProbesFiringTrigger_{layer}"
            ] = ak.values_astype(
                ak.num(layer_ss_trigger_pairs),
                np.int64,
            )

    def _store_fiducial_hot_spot_counts(self, flavors=("electron", "muon")):
        self.events["nElectronFiducialHotSpotsLoaded"] = self._event_int_like(
            len(self._fiducial_hot_spots("electron")) if "electron" in flavors else 0
        )
        self.events["nMuonFiducialHotSpotsLoaded"] = self._event_int_like(
            len(self._fiducial_hot_spots("muon")) if "muon" in flavors else 0
        )

    def _apply_lepton_fiducial_maps_to_track_cutflow(self, masks, *flavors):
        fiducial_hot_spots = self._lepton_fiducial_hot_spots(*flavors)
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

    def _lepton_background_categories_enabled(self):
        try:
            return bool(self.params.disapptrks.lepton_background_categories)
        except Exception:
            pass
        return os.environ.get(
            "DISAPPTRKS_ENABLE_LEPTON_BACKGROUND_CATEGORIES",
            "",
        ).lower() in ("1", "true", "yes", "on")

    def _store_lepton_pveto_pairs(
        self,
        *,
        prefix,
        flavor,
        tags,
        probes,
        tag_mass,
        probe_mass,
        window_low,
        window_high,
        pass_mask_function,
        fiducial_hot_spots=(),
    ):
        probes = ak.with_field(
            probes,
            trigger_matched_track_mask(self.events, probes, flavor=flavor),
            "firesTrigger",
        )
        pairs = build_lepton_veto_tag_probe_pairs(
            tags,
            probes,
            tag_mass=tag_mass,
            probe_mass=probe_mass,
        )
        pveto_pass_mask = pass_mask_function(pairs) & _outside_fiducial_hot_spots(
            pairs, fiducial_hot_spots
        )
        # Hoisted out of the per-field/per-layer fills below: both are
        # layer-independent and were previously recomputed on every use
        # (5x each across this function).
        os_mass_window_mask = os_mass_window_pair_mask(pairs, window_low, window_high)
        ss_mass_window_mask = ss_mass_window_pair_mask(pairs, window_low, window_high)
        self.events[f"{prefix}TagProbePair"] = pairs
        self.events[f"{prefix}TagProbePairMassWindow"] = pairs[
            mass_window_pair_mask(pairs, window_low, window_high)
        ]
        self.events[f"{prefix}TagProbePairOSMassWindow"] = pairs[os_mass_window_mask]
        self.events[f"{prefix}TagProbePairSSMassWindow"] = pairs[ss_mass_window_mask]
        self._store_trigger_efficiency_counts(
            prefix=prefix,
            pairs=pairs,
            mass_mask_function=lambda pair: mass_window_pair_mask(
                pair,
                window_low,
                window_high,
            ),
        )
        self.events[f"{prefix}PVetoTagProbePairMassWindowPass"] = pairs[
            os_mass_window_mask & pveto_pass_mask
        ]
        self.events[f"{prefix}PVetoTagProbePairSSMassWindowPass"] = pairs[
            ss_mass_window_mask & pveto_pass_mask
        ]
        for layer in PVETO_LAYERS:
            pair_layer_mask = generic_probe_pair_layer_mask(pairs, layer)
            self.events[f"{prefix}TagProbePairMassWindow_{layer}"] = pairs[
                os_mass_window_mask & pair_layer_mask
            ]
            self.events[f"{prefix}PVetoTagProbePairMassWindowPass_{layer}"] = pairs[
                os_mass_window_mask & pair_layer_mask & pveto_pass_mask
            ]
            self.events[f"{prefix}TagProbePairSSMassWindow_{layer}"] = pairs[
                ss_mass_window_mask & pair_layer_mask
            ]
            self.events[f"{prefix}PVetoTagProbePairSSMassWindowPass_{layer}"] = pairs[
                ss_mass_window_mask & pair_layer_mask & pveto_pass_mask
            ]

    def _store_lepton_background_controls(
        self,
        *,
        prefix,
        flavor,
        tags,
        event_quality,
        required_event_mask=None,
        diagnostic_cross_trigger=None,
        diagnostic_reference_trigger=None,
        fiducial_hot_spots=(),
        met_cut=120.0,
        phi_cut=0.5,
    ):
        met_pt, met_phi = met_no_mu_minus_lepton(
            self.events,
            tags,
            flavor=flavor,
        )
        if required_event_mask is None:
            required_event_mask = ak.ones_like(event_quality, dtype=bool)
        tag_event = (
            event_quality
            & required_event_mask
            & (ak.num(tags) >= 1)
            & _lepton_background_tag_pt55_event_mask(self.events.AnalysisEvent)
        )
        offline_event = (
            tag_event
            & (met_pt >= met_cut)
            & (_leading_jet_delta_phi(self.events, met_phi) >= phi_cut)
        )
        trigger_event = offline_event & _met_trigger_mask(self.events)
        met_trigger_event = tag_event & _met_trigger_mask(self.events)
        dphi = _leading_jet_delta_phi(self.events, met_phi)
        matched_object_d_r = (
            minimum_delta_r(self.events.IsoTrack, tags)
            if flavor == "tau"
            else None
        )
        for layer in (*PVETO_LAYERS, "combinedBins"):
            track_mask = _lepton_background_track_mask(
                self.events.IsoTrack,
                flavor=flavor,
                layer=layer,
                matched_object_d_r=matched_object_d_r,
                require_dedx_max_over_median=self._lepton_background_dedx_cut_enabled(),
            )
            track_mask = track_mask & _outside_fiducial_hot_spots(
                self.events.IsoTrack,
                fiducial_hot_spots,
            )
            has_track = ak.num(self.events.IsoTrack[track_mask]) >= 1
            control_event = tag_event & has_track
            self.events[f"n{prefix}BackgroundControl_{layer}"] = ak.values_astype(
                control_event,
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
            self.events[f"n{prefix}BackgroundMetMinusOnePt_{layer}"] = ak.where(
                control_event,
                met_pt,
                -1.0,
            )
            self.events[f"n{prefix}BackgroundMetMinusOnePtTrig_{layer}"] = ak.where(
                met_trigger_event & has_track,
                met_pt,
                -1.0,
            )
            self.events[f"n{prefix}BackgroundMetNoMuPt_{layer}"] = ak.where(
                control_event,
                self.events.MetNoMu.pt,
                -1.0,
            )
            self.events[f"n{prefix}BackgroundMetNoMuPtTrig_{layer}"] = ak.where(
                met_trigger_event & has_track,
                self.events.MetNoMu.pt,
                -1.0,
            )
            self.events[
                f"n{prefix}BackgroundDeltaPhiMetJetLeadingVsMetMinusOnePt_{layer}"
            ] = ak.where(
                control_event,
                dphi,
                -1.0,
            )

        if flavor == "tau":
            self._store_tau_background_diagnostics(
                tags=tags,
                event_quality=event_quality,
                required_lepton=required_event_mask,
                cross_trigger=diagnostic_cross_trigger,
                reference_trigger=diagnostic_reference_trigger,
                met_pt=met_pt,
                met_phi=met_phi,
                matched_object_d_r=matched_object_d_r,
            )

    def _store_tau_background_diagnostics(
        self,
        *,
        tags,
        event_quality,
        required_lepton,
        cross_trigger,
        reference_trigger,
        met_pt,
        met_phi,
        matched_object_d_r,
    ):
        """Store cumulative native-PocketCoffea stages for the tau control."""

        taus = self.events.Tau
        tau_pt = taus.pt > 50.0
        tau_eta = tau_pt & (abs(taus.eta) < 2.1)
        tau_dm = tau_eta & taus.idDecayModeNewDMs
        if "idDeepTau2018v2p5VSjet" in taus.fields:
            tau_lepton_id = (
                tau_dm
                & (taus.idDeepTau2018v2p5VSe >= 1)
                & (taus.idDeepTau2018v2p5VSmu >= 1)
            )
            tau_tight = tau_lepton_id & (taus.idDeepTau2018v2p5VSjet >= 6)
        else:
            tau_lepton_id = (
                tau_dm
                & (taus.rawDeepTau2018v2p5VSe > 0.099)
                & (taus.rawDeepTau2018v2p5VSmu > 0.2949)
            )
            tau_tight = tau_lepton_id & (taus.rawDeepTau2018v2p5VSjet > 0.8841)

        cumulative = event_quality
        diagnostics = {"event_quality": cumulative}
        if cross_trigger is None:
            cross_trigger = required_lepton
        cumulative = cumulative & cross_trigger
        diagnostics["event_cross_trigger"] = cumulative
        if reference_trigger is None:
            reference_trigger = required_lepton
        cumulative = cumulative & reference_trigger
        diagnostics["event_reference_muon_trigger"] = cumulative
        for name, tau_mask in (
            ("tau_pt50", tau_pt),
            ("tau_eta2p1", tau_eta),
            ("tau_decay_mode", tau_dm),
            ("tau_lepton_rejection", tau_lepton_id),
            ("tau_tight_vsjet", tau_tight),
        ):
            diagnostics[name] = cumulative & (ak.num(taus[tau_mask]) >= 1)

        cumulative = diagnostics["tau_tight_vsjet"] & (
            ak.num(tags) >= 1
        ) & _lepton_background_tag_pt55_event_mask(self.events.AnalysisEvent)
        diagnostics["event_jet_selection"] = cumulative

        track_masks = search_track_cutflow_masks(self.events.IsoTrack)
        track_stage_names = (
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
        )
        for name in track_stage_names:
            diagnostics[name] = cumulative & (
                ak.num(self.events.IsoTrack[track_masks[name]]) >= 1
            )

        # TauTagPt55 stops before the generic track--jet separation requirement.
        tau_track_base = track_masks["track_dz0p5"]
        tau_match = (
            (matched_object_d_r >= 0.0)
            & (matched_object_d_r < 0.1)
        )
        matched_base = tau_track_base & tau_match
        diagnostics["track_tauMatch0p1"] = cumulative & (
            ak.num(self.events.IsoTrack[matched_base]) >= 1
        )
        offline_event = (
            cumulative
            & (met_pt >= 120.0)
            & (_leading_jet_delta_phi(self.events, met_phi) >= 0.5)
        )
        trigger_event = offline_event & _met_trigger_mask(self.events)
        for layer in (*PVETO_LAYERS, "combinedBins"):
            layer_tracks = matched_base & layer_mask(self.events.IsoTrack, layer)
            has_track = ak.num(self.events.IsoTrack[layer_tracks]) >= 1
            diagnostics[f"track_{layer}"] = cumulative & has_track
            diagnostics[f"offline_{layer}"] = offline_event & has_track
            diagnostics[f"trigger_{layer}"] = trigger_event & has_track

        self.events["TauBackgroundDiag"] = ak.zip(diagnostics)

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
            self._category_mode() == "tau_trigger_probability"
            and not self._full_workflow_enabled()
        ):
            return
        if self._category_mode() == "high_purity_study":
            control = self._fake_track_controls()[0]
            if control == "zmumu":
                self.events["Muon"] = add_muon_derived_fields(self.events)
            elif control == "zee":
                self.events["Electron"] = add_electron_derived_fields(self.events)
            else:
                raise ValueError("high_purity_study supports only zmumu or zee")
            self.events["IsoTrack"] = add_isotrack_derived_fields(self.events)
            # Permit schema-transition validation files to run. Histograms for
            # newly added fields remain empty (the sentinel is below range)
            # instead of aborting the whole job.
            for field in (
                "innerPx", "innerPy", "innerPz", "innerPt",
                "outerPx", "outerPy", "outerPz", "outerPt",
            ):
                if field not in self.events.IsoTrack.fields:
                    self.events["IsoTrack"] = ak.with_field(
                        self.events.IsoTrack,
                        ak.ones_like(self.events.IsoTrack.pt) * -999.0,
                        field,
                    )
            # The older schema has the same reco::Track covariance errors
            # under generic names.  Reference-point translation supplies no
            # separate covariance estimate, so these are valid aliases.
            for field, source in (
                ("dxyBSErr", "dxyErr"), ("dzBSErr", "dzErr"),
                ("dxyClosestPVErr", "dxyErr"), ("dzClosestPVErr", "dzErr"),
            ):
                if field not in self.events.IsoTrack.fields:
                    self.events["IsoTrack"] = ak.with_field(
                        self.events.IsoTrack, self.events.IsoTrack[source], field
                    )
            control_mask = (
                _z_to_mumu_control_mask(self.events, self.events.Muon)
                if control == "zmumu"
                else _z_to_ee_control_mask(self.events, self.events.Electron)
            )
            hot_spots = self._lepton_fiducial_hot_spots("electron", "muon")
            self.events["HighPurityStudyTrack"] = (
                _high_purity_study_tracks_for_control(
                    self.events,
                    control_mask,
                    fiducial_hot_spots=hot_spots,
                )
            )
            if self._high_purity_dedx_histograms_enabled():
                if "IsoTrackDeDxHit" not in self.events.fields:
                    raise AttributeError(
                        "high-purity dE/dx hit histograms require the "
                        "IsoTrackDeDxHit table"
                    )
                self.events["nIsoTrackDeDxHit"] = ak.mask(
                    ak.num(self.events.IsoTrackDeDxHit),
                    ak.num(self.events.HighPurityStudyTrack) > 0,
                )
                for layer in self.params.disapptrks.high_purity_study_layers:
                    layer_tracks = self.events.HighPurityStudyTrack[
                        layer_mask(self.events.HighPurityStudyTrack, layer)
                    ]
                    selected_tracks = layer_tracks[
                        layer_tracks.isHighPurityTrack
                    ]
                    collection = f"HighPurityStudyDeDxHit_pass_{layer}"
                    grouped_hits = _dedx_hits_grouped_by_track(
                        self.events,
                        selected_tracks,
                    )
                    hits = _prepare_dedx_hits_for_histograms(
                        ak.flatten(grouped_hits, axis=2)
                    )
                    self.events[collection] = hits
                    self.events[f"n{collection}"] = ak.mask(
                        ak.num(hits), ak.num(selected_tracks) > 0
                    )
                    self.events[
                        f"HighPurityStudyDeDxTrack_pass_{layer}"
                    ] = _dedx_track_summaries(selected_tracks, grouped_hits)
            return
        if (
            "Electron" in self.events.fields
            and
            self._mode_enabled(
                "electron_pveto",
                "tau_ele_pveto",
                "electron_pmiss_poffline",
                "tau_ele_pmiss_poffline",
                "fake_zee",
                "fiducial_maps",
            )
        ):
            self.events["Electron"] = add_electron_derived_fields(self.events)
        if (
            self._mode_enabled(
                "muon_pveto",
                "tau_mu_pveto",
                "muon_pmiss_poffline",
                "tau_mu_pmiss_poffline",
                "fake_zmumu",
                "fiducial_maps",
            )
            or self._category_mode() == "fake_tracks"
        ):
            self.events["Muon"] = add_muon_derived_fields(self.events)
        self.events["IsoTrack"] = add_isotrack_derived_fields(self.events)
        if self._lepton_background_dedx_cut_enabled() and self._mode_enabled(
            "muon_pveto",
            "electron_pveto",
            "tau_mu_pveto",
            "tau_ele_pveto",
            "muon_pmiss_poffline",
            "electron_pmiss_poffline",
            "tau_mu_pmiss_poffline",
            "tau_ele_pmiss_poffline",
            "tau_pmiss_poffline",
        ):
            # Attach dE/dx summaries before any Pveto probe-track mask or
            # Poffline/Pmiss control-track mask is built below, so
            # `muon_veto_probe_track_mask`, `lepton_veto_probe_track_mask`,
            # `tau_veto_probe_track_mask`, and `_lepton_background_track_mask`
            # can all apply the dE/dx cut (see selections.py) -- the
            # Poffline/Pmiss control track selects the same kind of track the
            # signal selection targets, so it needs the same requirement as
            # the Pveto probe tracks. Scoped to the modes that actually build
            # one of those masks -- other category modes may not carry the
            # `IsoTrackDeDxHit` branch.
            self._attach_lepton_background_dedx_fields()
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

        fiducial_count_flavors = []
        if self._mode_enabled("electron_pveto"):
            fiducial_count_flavors.append("electron")
        if self._mode_enabled("muon_pveto"):
            fiducial_count_flavors.append("muon")
        if fiducial_count_flavors:
            self._store_fiducial_hot_spot_counts(tuple(fiducial_count_flavors))

        tag_met = None
        if self._mode_enabled("muon_pveto", "muon_pmiss_poffline"):
            self.events["MuonTag"] = self.events.Muon[muon_tag_mask(self.events.Muon)]

        if self._mode_enabled("muon_pveto"):
            self.events["MuonVetoProbeTrack"] = self.events.IsoTrack[
                muon_veto_probe_track_mask(
                    self.events.IsoTrack,
                    require_dedx_max_over_median=self._lepton_background_dedx_cut_enabled(),
                )
            ]
            self.events["MuonVetoProbeTrack"] = ak.with_field(
                self.events.MuonVetoProbeTrack,
                trigger_matched_track_mask(
                    self.events,
                    self.events.MuonVetoProbeTrack,
                    flavor="muon",
                ),
                "firesTrigger",
            )
            muon_veto_pairs = build_muon_veto_tag_probe_pairs(
                self.events.MuonTag, self.events.MuonVetoProbeTrack
            )
            lepton_fiducial_hot_spots = self._fiducial_hot_spots("muon")
            muon_pveto_pass_no_fiducial_mask = muon_pveto_pair_pass_mask(
                muon_veto_pairs
            )
            outside_fiducial_map_mask = _outside_fiducial_hot_spots(
                muon_veto_pairs,
                lepton_fiducial_hot_spots,
            )
            muon_pveto_pass_mask = (
                muon_pveto_pass_no_fiducial_mask & outside_fiducial_map_mask
            )
            # Hoisted out of the per-field/per-layer fills below: each of these
            # is layer-independent and was previously recomputed on every
            # field access (8x for the Z-window OS/SS masks alone).
            os_z_window_mask = os_z_window_muon_probe_pair_mask(muon_veto_pairs)
            ss_z_window_mask = ss_z_window_muon_probe_pair_mask(muon_veto_pairs)
            muon_veto_pass_mask = muon_veto_pair_pass_mask(muon_veto_pairs)
            muon_veto_fail_mask = muon_veto_pair_fail_mask(muon_veto_pairs)
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
                os_z_window_mask
            ]
            self._store_trigger_efficiency_counts(
                prefix="Muon",
                pairs=muon_veto_pairs,
                mass_mask_function=z_window_muon_probe_pair_mask,
            )
            self.events["MuonVetoTagProbePairZWindowPass"] = muon_veto_pairs[
                os_z_window_mask & muon_veto_pass_mask
            ]
            self.events["MuonVetoTagProbePairZWindowFail"] = muon_veto_pairs[
                os_z_window_mask & muon_veto_fail_mask
            ]
            self.events["MuonPVetoTagProbePairZWindowPass"] = muon_veto_pairs[
                os_z_window_mask & muon_pveto_pass_mask
            ]
            self.events["MuonPVetoTagProbePairZWindowPassNoFiducial"] = (
                muon_veto_pairs[
                    os_z_window_mask & muon_pveto_pass_no_fiducial_mask
                ]
            )
            self.events["MuonPVetoTagProbePairZWindowFiducialRejected"] = (
                muon_veto_pairs[
                    os_z_window_mask
                    & muon_pveto_pass_no_fiducial_mask
                    & ~outside_fiducial_map_mask
                ]
            )
            self.events["MuonVetoTagProbePairSSZWindow"] = muon_veto_pairs[
                ss_z_window_mask
            ]
            self.events["MuonVetoTagProbePairSSZWindowPass"] = muon_veto_pairs[
                ss_z_window_mask & muon_veto_pass_mask
            ]
            self.events["MuonVetoTagProbePairSSZWindowFail"] = muon_veto_pairs[
                ss_z_window_mask & muon_veto_fail_mask
            ]
            self.events["MuonPVetoTagProbePairSSZWindowPass"] = muon_veto_pairs[
                ss_z_window_mask & muon_pveto_pass_mask
            ]
            self.events["MuonPVetoTagProbePairSSZWindowPassNoFiducial"] = (
                muon_veto_pairs[
                    ss_z_window_mask & muon_pveto_pass_no_fiducial_mask
                ]
            )
            self.events["MuonPVetoTagProbePairSSZWindowFiducialRejected"] = (
                muon_veto_pairs[
                    ss_z_window_mask
                    & muon_pveto_pass_no_fiducial_mask
                    & ~outside_fiducial_map_mask
                ]
            )
            for layer in PVETO_LAYERS:
                pair_layer_mask = muon_probe_pair_layer_mask(muon_veto_pairs, layer)
                self.events[f"MuonVetoTagProbePairZWindow_{layer}"] = muon_veto_pairs[
                    os_z_window_mask & pair_layer_mask
                ]
                self.events[f"MuonPVetoTagProbePairZWindowPass_{layer}"] = muon_veto_pairs[
                    os_z_window_mask & pair_layer_mask & muon_pveto_pass_mask
                ]
                self.events[f"MuonVetoTagProbePairSSZWindow_{layer}"] = muon_veto_pairs[
                    ss_z_window_mask & pair_layer_mask
                ]
                self.events[f"MuonPVetoTagProbePairSSZWindowPass_{layer}"] = muon_veto_pairs[
                    ss_z_window_mask & pair_layer_mask & muon_pveto_pass_mask
                ]

        if self._mode_enabled("electron_pveto", "electron_pmiss_poffline"):
            self.events["ElectronTag"] = self.events.Electron[
                electron_tag_mask(self.events.Electron, self.events)
            ]

        if self._mode_enabled("electron_pveto"):
            self.events["ElectronVetoProbeTrack"] = self.events.IsoTrack[
                lepton_veto_probe_track_mask(
                    self.events.IsoTrack,
                    measured_veto="electron",
                    require_dedx_max_over_median=self._lepton_background_dedx_cut_enabled(),
                )
            ]
            self._store_lepton_pveto_pairs(
                prefix="Electron",
                flavor="electron",
                tags=self.events.ElectronTag,
                probes=self.events.ElectronVetoProbeTrack,
                tag_mass=ELECTRON_MASS,
                probe_mass=ELECTRON_MASS,
                window_low=91.1876 - 10.0,
                window_high=91.1876 + 10.0,
                pass_mask_function=electron_pveto_pair_pass_mask,
                fiducial_hot_spots=self._fiducial_hot_spots("electron"),
            )

        if self._mode_enabled(
            "tau_mu_pveto",
            "tau_ele_pveto",
        ):
            tag_met = _met_for_transverse_mass(self.events)

        if self._mode_enabled("tau_mu_pveto", "tau_ele_pveto"):
            self.events["TauVetoProbeTrack"] = self.events.IsoTrack[
                tau_veto_probe_track_mask(
                    self.events.IsoTrack,
                    require_dedx_max_over_median=self._lepton_background_dedx_cut_enabled(),
                )
            ]

        if self._mode_enabled("tau_mu_pveto"):
            tau_mu_tag_mask = muon_tag_progression_masks(self.events.Muon)[
                "muon_selected_tag"
            ]
            self.events["MuonLowMTTag"] = self.events.Muon[tau_mu_tag_mask][
                low_mt_mask(self.events.Muon[tau_mu_tag_mask], tag_met)
            ]

        if self._mode_enabled("tau_mu_pveto"):
            self._store_lepton_pveto_pairs(
                prefix="TauMu",
                flavor="tau_mu",
                tags=self.events.MuonLowMTTag,
                probes=self.events.TauVetoProbeTrack,
                tag_mass=MUON_MASS,
                probe_mass=MUON_MASS,
                window_low=91.1876 - 50.0,
                window_high=91.1876 - 15.0,
                pass_mask_function=tau_pveto_pair_pass_mask,
                fiducial_hot_spots=self._lepton_fiducial_hot_spots("muon"),
            )

        if self._mode_enabled("tau_ele_pveto"):
            tau_ele_tag_mask = _z_electron_tag_mask(self.events.Electron, pt_min=32.0)
            self.events["ElectronLowMTTag"] = self.events.Electron[tau_ele_tag_mask][
                low_mt_mask(self.events.Electron[tau_ele_tag_mask], tag_met)
            ]

        if self._mode_enabled(
            "tau_mu_pveto",
            "tau_ele_pveto",
            "tau_mu_pmiss_poffline",
            "tau_ele_pmiss_poffline",
            "tau_pmiss_poffline",
        ):
            self.events["TauControlTag"] = self._select_one_tau_control_tag()

        if self._mode_enabled("tau_ele_pveto"):
            self._store_lepton_pveto_pairs(
                prefix="TauEle",
                flavor="tau_ele",
                tags=self.events.ElectronLowMTTag,
                probes=self.events.TauVetoProbeTrack,
                tag_mass=ELECTRON_MASS,
                probe_mass=ELECTRON_MASS,
                window_low=91.1876 - 50.0,
                window_high=91.1876 - 15.0,
                pass_mask_function=tau_pveto_pair_pass_mask,
                fiducial_hot_spots=self._lepton_fiducial_hot_spots("electron"),
            )

        store_background_controls_in_pveto = self._lepton_background_categories_enabled()
        if (
            store_background_controls_in_pveto
            and self._mode_enabled(
                "muon_pveto",
                "electron_pveto",
                "tau_mu_pveto",
                "tau_ele_pveto",
            )
        ) or self._mode_enabled(
            "muon_pmiss_poffline",
            "electron_pmiss_poffline",
            "tau_mu_pmiss_poffline",
            "tau_ele_pmiss_poffline",
            "tau_pmiss_poffline",
        ):
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
            if (
                self._mode_enabled("tau_pmiss_poffline")
                and "TauControlTag" in self.events.fields
            ):
                cross_trigger = _muon_tau_trigger_mask(self.events, self._year)
                reference_trigger = _tau_probability_single_muon_trigger_mask(
                    self.events
                )
                self._store_lepton_background_controls(
                    prefix="Tau",
                    flavor="tau",
                    tags=self.events.TauControlTag,
                    event_quality=event_quality,
                    # N_ctrl/Poffline/Pmiss are measured on the single-muon
                    # reference-triggered baseline only -- matching legacy
                    # TauTagPt55 (BackgroundEstimation/python/TauTagProbeSelections.py,
                    # TauTagSkim.triggers = triggersSingleMu), the channel
                    # actually used for the dissertation's tau background
                    # estimate. The cross-trigger dependence enters separately
                    # via tau_trigger_probability's P(tau) = P(cross)/P(single)
                    # factor -- requiring it here too would pre-condition this
                    # control sample on the cross-trigger, which the legacy
                    # code never did. The framework skim also now only
                    # requires the reference trigger, so this is redundant
                    # with skim; keep it explicit anyway so no-skim
                    # diagnostics stay faithful (diagnostic_cross_trigger
                    # below still records the cross-trigger's own diagnostic
                    # categories, just not as a selection requirement).
                    required_event_mask=reference_trigger,
                    diagnostic_cross_trigger=cross_trigger,
                    diagnostic_reference_trigger=reference_trigger,
                    fiducial_hot_spots=(),
                )
            if (
                (
                    self._mode_enabled("muon_pmiss_poffline")
                    or (
                        store_background_controls_in_pveto
                        and self._mode_enabled("muon_pveto")
                    )
                )
                and "MuonTag" in self.events.fields
            ):
                self._store_lepton_background_controls(
                    prefix="Muon",
                    flavor="muon",
                    tags=self.events.MuonTag,
                    event_quality=event_quality,
                    fiducial_hot_spots=self._fiducial_hot_spots("muon"),
                )
            if (
                (
                    self._mode_enabled("electron_pmiss_poffline")
                    or (
                        store_background_controls_in_pveto
                        and self._mode_enabled("electron_pveto")
                    )
                )
                and "ElectronTag" in self.events.fields
            ):
                self._store_lepton_background_controls(
                    prefix="Electron",
                    flavor="electron",
                    tags=self.events.ElectronTag,
                    event_quality=event_quality,
                    fiducial_hot_spots=self._fiducial_hot_spots("electron"),
                )
            if (
                (
                    self._mode_enabled("tau_mu_pmiss_poffline")
                    or (
                        store_background_controls_in_pveto
                        and self._mode_enabled("tau_mu_pveto")
                    )
                )
                and "TauControlTag" in self.events.fields
            ):
                self._store_lepton_background_controls(
                    prefix="TauMu",
                    flavor="tau",
                    tags=self.events.TauControlTag,
                    event_quality=event_quality,
                    required_event_mask=(
                        ak.num(self.events.MuonLowMTTag) >= 1
                        if self._mode_enabled("tau_mu_pveto")
                        else None
                    ),
                    fiducial_hot_spots=(),
                )
            if (
                (
                    self._mode_enabled("tau_ele_pmiss_poffline")
                    or (
                        store_background_controls_in_pveto
                        and self._mode_enabled("tau_ele_pveto")
                    )
                )
                and "TauControlTag" in self.events.fields
            ):
                self._store_lepton_background_controls(
                    prefix="TauEle",
                    flavor="tau",
                    tags=self.events.TauControlTag,
                    event_quality=event_quality,
                    required_event_mask=(
                        ak.num(self.events.ElectronLowMTTag) >= 1
                        if self._mode_enabled("tau_ele_pveto")
                        else None
                    ),
                    fiducial_hot_spots=(),
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
        search_mask = search_track_mask(self.events.IsoTrack)
        search_no_high_purity_mask = search_track_mask(
            self.events.IsoTrack,
            require_high_purity=False,
        )
        if self._category_mode() == "signal_acceptance":
            fiducial_hot_spots = self._lepton_fiducial_hot_spots(
                "electron", "muon"
            )
            if fiducial_hot_spots:
                fiducial_mask = _outside_fiducial_hot_spots(
                    self.events.IsoTrack,
                    fiducial_hot_spots,
                )
                search_mask = search_mask & fiducial_mask
                search_no_high_purity_mask = (
                    search_no_high_purity_mask & fiducial_mask
                )
        self.events["IsoTrackSearch"] = self.events.IsoTrack[search_mask]
        self.events["IsoTrackSearchNoHighPurity"] = self.events.IsoTrack[
            search_no_high_purity_mask
        ]
        self.events["IsoTrackSearchHighPurity"] = self.events.IsoTrackSearch[
            self.events.IsoTrackSearch.isHighPurityTrack
        ]
        if (
            self._category_mode() == "signal_acceptance"
            and self._signal_dedx_histograms_enabled()
        ):
            if "IsoTrackDeDxHit" not in self.events.fields:
                raise AttributeError(
                    "signal dE/dx histograms require the IsoTrackDeDxHit table"
                )

            source_indices = ak.local_index(self.events.IsoTrack, axis=1)[
                search_mask
            ]
            selected_tracks = ak.with_field(
                self.events.IsoTrack[search_mask],
                source_indices,
                "sourceIsoTrackIdx",
            )
            full_event_mask = search_event_cutflow_masks(
                self.events.AnalysisEvent
            )["event_jetMetDphi0p5"]
            selected_event_mask, _ = ak.broadcast_arrays(
                full_event_mask,
                selected_tracks.pt,
            )
            selected_tracks = selected_tracks[selected_event_mask]

            for layer in self.params.disapptrks.signal_dedx_layers:
                layer_tracks = selected_tracks[
                    layer_mask(selected_tracks, layer)
                ]
                grouped_hits = _dedx_hits_grouped_by_track(
                    self.events,
                    layer_tracks,
                )
                self.events[f"SignalDeDxTrack_{layer}"] = (
                    _dedx_track_summaries(layer_tracks, grouped_hits)
                )

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
        self.events["nIsoTrackSearchNoHighPurity"] = ak.num(
            self.events.IsoTrackSearchNoHighPurity
        )
        self.events["nIsoTrackSearchHighPurity"] = ak.num(
            self.events.IsoTrackSearchHighPurity
        )
        for layer in ("NLayers4", "NLayers5", "NLayers6plus"):
            self.events[f"nIsoTrackSearchNoHighPurity_{layer}"] = ak.num(
                self.events.IsoTrackSearchNoHighPurity[
                    layer_mask(self.events.IsoTrackSearchNoHighPurity, layer)
                ]
            )
            self.events[f"nIsoTrackSearch_{layer}"] = ak.num(
                self.events.IsoTrackSearch[layer_mask(self.events.IsoTrackSearch, layer)]
            )
            self.events[f"nIsoTrackSearchHighPurity_{layer}"] = ak.num(
                self.events.IsoTrackSearchHighPurity[
                    layer_mask(self.events.IsoTrackSearchHighPurity, layer)
                ]
            )
        self.events["nIsoTrackSearchNoHighPurity_combinedBins"] = (
            self.events.nIsoTrackSearchNoHighPurity
        )
        self.events["nIsoTrackSearch_combinedBins"] = self.events.nIsoTrackSearch

    def _has_eta_leg(self, collection_name, eta_max=2.1):
        if collection_name not in self.events.fields:
            return _all_true_like(self.events) & False
        collection = self.events[collection_name]
        if "eta" not in collection.fields:
            return _all_true_like(self.events) & False
        return ak.num(collection[abs(collection.eta) < eta_max]) >= 1

    def _store_tau_trigger_probability_counts(self):
        muon_eta_leg = self._has_eta_leg("Muon", eta_max=2.1)
        tau_eta_leg = self._has_eta_leg("Tau", eta_max=2.1)
        eta_legs = muon_eta_leg & tau_eta_leg
        cross_trigger = _muon_tau_trigger_mask(self.events, self._year)
        single_muon_trigger = _tau_probability_single_muon_trigger_mask(self.events)
        # Legacy/Dissertation Eq. 7.7-7.8: P(tau) = P(muon+tau) / P(muon),
        # each measured independently over the same eta-accepted baseline
        # sample (denominator is not required to be a subset of numerator).
        numerator = eta_legs & cross_trigger
        denominator = eta_legs & single_muon_trigger

        self.events["nTauTriggerProbabilityMuonEtaLeg"] = ak.values_astype(
            muon_eta_leg, np.int64
        )
        self.events["nTauTriggerProbabilityTauEtaLeg"] = ak.values_astype(
            tau_eta_leg, np.int64
        )
        self.events["nTauTriggerProbabilityDenominator"] = ak.values_astype(
            denominator, np.int64
        )
        self.events["nTauTriggerProbabilityNumerator"] = ak.values_astype(
            numerator, np.int64
        )

    def _select_one_tau_control_tag(self):
        """Choose one Table-27 tau reproducibly when an event has several."""

        selected = self.events.Tau[
            hadronic_tau_control_object_mask(self.events.Tau)
        ]
        counts = ak.num(selected)
        safe_counts = ak.where(counts > 0, counts, 1)
        # The AN chooses a passing tau randomly.  Event-number modulo gives a
        # deterministic, reproducible pseudo-random choice with no pT ordering
        # preference.
        chosen_index = self.events.event % safe_counts
        local_index = ak.local_index(selected, axis=1)
        return selected[local_index == chosen_index]

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
        self.events["nMuonPVetoTagProbePairZWindowPassNoFiducial"] = ak.num(
            self.events.MuonPVetoTagProbePairZWindowPassNoFiducial
        )
        self.events["nMuonPVetoTagProbePairZWindowFiducialRejected"] = ak.num(
            self.events.MuonPVetoTagProbePairZWindowFiducialRejected
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
        self.events["nMuonPVetoTagProbePairSSZWindowPassNoFiducial"] = ak.num(
            self.events.MuonPVetoTagProbePairSSZWindowPassNoFiducial
        )
        self.events["nMuonPVetoTagProbePairSSZWindowFiducialRejected"] = ak.num(
            self.events.MuonPVetoTagProbePairSSZWindowFiducialRejected
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

    def _attach_dedx_track_summary_fields(self):
        """Attach per-track dE/dx summaries onto ``events.IsoTrack``.

        Both the fake-track and lepton-background probe-track selections
        read ``dEdxMaximumOverMedian`` off the tracks they are given
        (mirroring how they read ``isHighPurityTrack``), so this collection
        must exist on the full track collection before any of those category
        masks are built from it.  A no-op if already attached earlier in the
        same event pass (e.g. by the fake-track and lepton-background paths
        both running under a combined ``muon_backgrounds``/``egamma_backgrounds``/
        ``all`` category mode).
        """
        if "dEdxMaximumOverMedian" in self.events.IsoTrack.fields:
            return
        tracks = ak.with_field(
            self.events.IsoTrack,
            ak.local_index(self.events.IsoTrack, axis=1),
            "sourceIsoTrackIdx",
        )
        grouped_hits = _dedx_hits_grouped_by_track(self.events, tracks)
        self.events["IsoTrack"] = _dedx_track_summaries(tracks, grouped_hits)

    def _attach_fake_track_dedx_fields(self):
        self._attach_dedx_track_summary_fields()

    def _attach_lepton_background_dedx_fields(self):
        self._attach_dedx_track_summary_fields()

    def _count_fake_track_fields(self, *, controls=("basic", "zmumu", "zee")):
        require_dedx_max_over_median = self._fake_track_dedx_cut_enabled()
        if require_dedx_max_over_median:
            self._attach_fake_track_dedx_fields()
        fake_fiducial_hot_spots = self._lepton_fiducial_hot_spots("electron", "muon")
        fake_basic3hits_d0_signal = self.events.IsoTrack[
            _fake_track_mask(
                self.events.IsoTrack,
                layer="NLayers4",
                d0_region="signal",
                fiducial_hot_spots=fake_fiducial_hot_spots,
                require_dedx_max_over_median=require_dedx_max_over_median,
            )
        ]
        fake_basic3hits_d0_sideband = self.events.IsoTrack[
            _fake_track_mask(
                self.events.IsoTrack,
                layer="NLayers4",
                d0_region="sideband",
                fiducial_hot_spots=fake_fiducial_hot_spots,
                require_dedx_max_over_median=require_dedx_max_over_median,
            )
        ]

        fake_sideband_base_mask = _fake_track_base_mask_with_fiducial(
            self.events.IsoTrack,
            d0_region="sideband",
            fiducial_hot_spots=fake_fiducial_hot_spots,
        )

        if "basic" in controls:
            self.events["nFakeBasic3HitsD0Signal"] = ak.num(fake_basic3hits_d0_signal)
            self.events["nFakeBasic3HitsD0Sideband"] = ak.num(fake_basic3hits_d0_sideband)
            for layer in (*PVETO_LAYERS, "combinedBins"):
                self.events[f"nFakeControl_{layer}"] = ak.num(
                    self.events.IsoTrack[
                        fake_sideband_base_mask
                        & fake_track_layer_cut(
                            self.events.IsoTrack,
                            layer=layer,
                            require_dedx_max_over_median=require_dedx_max_over_median,
                        )
                    ]
                )

        if "zmumu" in controls:
            fake_zmumu_control = _z_to_mumu_control_mask(self.events, self.events.Muon)
            self.events["nFakeZMuMuControl"] = ak.values_astype(
                fake_zmumu_control, np.int64
            )
            self.events["FakeZMuMuDiag"] = ak.zip(
                _z_to_mumu_control_diagnostics(
                    self.events,
                    self.events.Muon,
                    fiducial_hot_spots=fake_fiducial_hot_spots,
                    require_dedx_max_over_median=require_dedx_max_over_median,
                )
            )
            self.events["FakeZMuMuFitTrack"] = _fake_fit_tracks_for_control(
                self.events,
                fake_zmumu_control,
                fiducial_hot_spots=fake_fiducial_hot_spots,
                require_dedx_max_over_median=require_dedx_max_over_median,
            )
            for layer in (*PVETO_LAYERS, "combinedBins"):
                self.events[f"FakeZMuMuSidebandTrack_{layer}"] = (
                    _fake_sideband_tracks_for_control(
                        self.events,
                        fake_zmumu_control,
                        layer=layer,
                        fiducial_hot_spots=fake_fiducial_hot_spots,
                        require_dedx_max_over_median=require_dedx_max_over_median,
                        base_mask=fake_sideband_base_mask,
                    )
                )
                self.events[f"nFakeZMuMuSideband_{layer}"] = _fake_track_count_for_control(
                    self.events,
                    fake_zmumu_control,
                    layer=layer,
                    d0_region="sideband",
                    fiducial_hot_spots=fake_fiducial_hot_spots,
                    require_dedx_max_over_median=require_dedx_max_over_median,
                    base_mask=fake_sideband_base_mask,
                )
                if self._fake_sideband_histograms_enabled():
                    _add_fake_sideband_track_diagnostics(
                        self.events,
                        fake_zmumu_control,
                        control_key="ZMuMu",
                        layer=layer,
                        fiducial_hot_spots=fake_fiducial_hot_spots,
                        require_dedx_max_over_median=require_dedx_max_over_median,
                        base_mask=fake_sideband_base_mask,
                    )

        if "zee" in controls:
            fake_zee_control = _z_to_ee_control_mask(self.events, self.events.Electron)
            self.events["nFakeZeeControl"] = ak.values_astype(fake_zee_control, np.int64)
            self.events["FakeZeeDiag"] = ak.zip(
                _z_to_ee_control_diagnostics(
                    self.events,
                    self.events.Electron,
                    fiducial_hot_spots=fake_fiducial_hot_spots,
                    require_dedx_max_over_median=require_dedx_max_over_median,
                )
            )
            self.events["FakeZeeFitTrack"] = _fake_fit_tracks_for_control(
                self.events,
                fake_zee_control,
                fiducial_hot_spots=fake_fiducial_hot_spots,
                require_dedx_max_over_median=require_dedx_max_over_median,
            )
            for layer in (*PVETO_LAYERS, "combinedBins"):
                self.events[f"FakeZeeSidebandTrack_{layer}"] = (
                    _fake_sideband_tracks_for_control(
                        self.events,
                        fake_zee_control,
                        layer=layer,
                        fiducial_hot_spots=fake_fiducial_hot_spots,
                        require_dedx_max_over_median=require_dedx_max_over_median,
                        base_mask=fake_sideband_base_mask,
                    )
                )
                self.events[f"nFakeZeeSideband_{layer}"] = _fake_track_count_for_control(
                    self.events,
                    fake_zee_control,
                    layer=layer,
                    d0_region="sideband",
                    fiducial_hot_spots=fake_fiducial_hot_spots,
                    require_dedx_max_over_median=require_dedx_max_over_median,
                    base_mask=fake_sideband_base_mask,
                )
                if self._fake_sideband_histograms_enabled():
                    _add_fake_sideband_track_diagnostics(
                        self.events,
                        fake_zee_control,
                        control_key="Zee",
                        layer=layer,
                        fiducial_hot_spots=fake_fiducial_hot_spots,
                        require_dedx_max_over_median=require_dedx_max_over_median,
                        base_mask=fake_sideband_base_mask,
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
            muon_veto_probe_track_cutflow_masks(
                self.events.IsoTrack,
                require_dedx_max_over_median=self._lepton_background_dedx_cut_enabled(),
            ),
            "muon",
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
            muon_veto_probe_track_cutflow_masks(
                self.events.IsoTrack,
                require_dedx_max_over_median=self._lepton_background_dedx_cut_enabled(),
            ),
            "electron",
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
        tau_track_masks = tau_veto_probe_track_cutflow_masks(
            self.events.IsoTrack,
            require_dedx_max_over_median=self._lepton_background_dedx_cut_enabled(),
        )

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
            combined_layer_mask = generic_probe_pair_layer_mask(
                pairs, "combinedBins"
            )
            diagnostics.update(
                {
                    "pair_masswindow": ak.num(pairs[mass_window]) >= 1,
                    "pair_os": ak.num(pairs[os_mass_window]) >= 1,
                    "layer_combinedBins": ak.num(
                        pairs[os_mass_window & combined_layer_mask]
                    ) >= 1,
                    "pair_pass_tau_pveto": ak.num(
                        pairs[
                            os_mass_window
                            & combined_layer_mask
                            & tau_pveto_pair_pass_mask(pairs)
                        ]
                    )
                    >= 1,
                    "pair_ss_masswindow": ak.num(pairs[ss_mass_window]) >= 1,
                    "pair_ss_pass_tau_pveto": ak.num(
                        pairs[
                            ss_mass_window
                            & combined_layer_mask
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

    def _store_signal_acceptance_cutflows(self):
        """Store paired cumulative signal cutflows before/after high purity."""

        event_masks = search_event_cutflow_masks(self.events.AnalysisEvent)
        basic_event_mask = event_masks["event_jetMetDphi0p5"]
        fiducial_hot_spots = self._lepton_fiducial_hot_spots(
            "electron", "muon"
        )
        fiducial_mask = (
            _outside_fiducial_hot_spots(
                self.events.IsoTrack,
                fiducial_hot_spots,
            )
            if fiducial_hot_spots
            else None
        )

        def apply_fiducial_mask(track_masks):
            updated = {}
            after_fiducial_selection = False
            for name, track_mask in track_masks.items():
                if name == "track_fiducialECAL":
                    after_fiducial_selection = True
                if after_fiducial_selection and fiducial_mask is not None:
                    track_mask = track_mask & fiducial_mask
                updated[name] = track_mask
            return updated

        # CartesianSelection combines these per-track cumulative stages with
        # independent layer and high-purity axes.  Keeping them jagged ensures
        # all requirements are applied to the same candidate track.
        generic_track_masks = apply_fiducial_mask(
            search_track_cutflow_masks(
                self.events.IsoTrack,
                layer="combinedBins",
                require_high_purity=False,
            )
        )
        self.events["SignalAcceptanceBasicEvent"] = basic_event_mask
        self.events["SignalAcceptanceTrackStages"] = ak.zip(generic_track_masks)

        common_diagnostics = dict(event_masks)
        for name, track_mask in generic_track_masks.items():
            if name == "track_layers4plus":
                break
            common_diagnostics[name] = basic_event_mask & (
                ak.num(self.events.IsoTrack[track_mask]) >= 1
            )
        self.events["SignalAcceptanceCommon"] = ak.zip(common_diagnostics)

        layer_entry_diagnostics = {}
        layer_entry_mask = generic_track_masks["track_layers4plus"]
        for layer in (*PVETO_LAYERS, "combinedBins"):
            selected_tracks = layer_entry_mask & layer_mask(self.events.IsoTrack, layer)
            layer_entry_diagnostics[layer] = basic_event_mask & (
                ak.num(self.events.IsoTrack[selected_tracks]) >= 1
            )
        self.events["SignalAcceptanceLayerEntry"] = ak.zip(
            layer_entry_diagnostics
        )

    def _store_fake_track_diagnostics(self):
        event_search_kinematics = search_event_cutflow_masks(
            self.events.AnalysisEvent
        )["event_dijetDphi2p5"]
        diagnostics = {}
        masks = self._apply_lepton_fiducial_maps_to_track_cutflow(
            fake_track_sideband_cutflow_masks(self.events.IsoTrack),
            "electron",
            "muon",
        )
        for name, mask in masks.items():
            n_name = f"nFakeDiag{name[0].upper()}{name[1:]}"
            self.events[n_name] = ak.num(self.events.IsoTrack[mask])
            diagnostics[name] = event_search_kinematics & (self.events[n_name] >= 1)
        self.events["FakeTrackDiag"] = ak.zip(diagnostics)

    def _count_objects_mode_aware(self, variation):
        mode = self._category_mode()
        if mode == "high_purity_study":
            return
        if mode != "tau_trigger_probability":
            self._count_common_search_fields()

        if mode == "signal_acceptance":
            self._store_signal_acceptance_cutflows()
        elif mode == "muon_pveto":
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
        elif mode == "muon_pmiss_poffline":
            self.events["nMuonTag"] = ak.num(self.events.MuonTag)
        elif mode == "electron_pmiss_poffline":
            self.events["nElectronTag"] = ak.num(self.events.ElectronTag)
        elif mode == "tau_mu_pmiss_poffline":
            pass
        elif mode == "tau_ele_pmiss_poffline":
            pass
        elif mode == "tau_pmiss_poffline":
            pass
        elif mode == "tau_trigger_probability":
            self._store_tau_trigger_probability_counts()
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
        self.events["nIsoTrackSearchNoHighPurity"] = ak.num(
            self.events.IsoTrackSearchNoHighPurity
        )
        for layer in PVETO_LAYERS:
            self.events[f"nIsoTrackSearchNoHighPurity_{layer}"] = ak.num(
                self.events.IsoTrackSearchNoHighPurity[
                    layer_mask(self.events.IsoTrackSearchNoHighPurity, layer)
                ]
            )
            self.events[f"nIsoTrackSearch_{layer}"] = ak.num(
                self.events.IsoTrackSearch[
                    layer_mask(self.events.IsoTrackSearch, layer)
                ]
            )
        self.events["nIsoTrackSearchNoHighPurity_combinedBins"] = (
            self.events.nIsoTrackSearchNoHighPurity
        )
        self.events["nIsoTrackSearch_combinedBins"] = self.events.nIsoTrackSearch

        fake_fiducial_hot_spots = self._lepton_fiducial_hot_spots("electron", "muon")
        require_dedx_max_over_median = self._fake_track_dedx_cut_enabled()
        fake_basic3hits_d0_signal = self.events.IsoTrack[
            _fake_track_mask(
                self.events.IsoTrack,
                layer="NLayers4",
                d0_region="signal",
                fiducial_hot_spots=fake_fiducial_hot_spots,
                require_dedx_max_over_median=require_dedx_max_over_median,
            )
        ]
        fake_basic3hits_d0_sideband = self.events.IsoTrack[
            _fake_track_mask(
                self.events.IsoTrack,
                layer="NLayers4",
                d0_region="sideband",
                fiducial_hot_spots=fake_fiducial_hot_spots,
                require_dedx_max_over_median=require_dedx_max_over_median,
            )
        ]
        self.events["nFakeBasic3HitsD0Signal"] = ak.num(fake_basic3hits_d0_signal)
        self.events["nFakeBasic3HitsD0Sideband"] = ak.num(fake_basic3hits_d0_sideband)
        fake_sideband_base_mask = _fake_track_base_mask_with_fiducial(
            self.events.IsoTrack,
            d0_region="sideband",
            fiducial_hot_spots=fake_fiducial_hot_spots,
        )
        for layer in (*PVETO_LAYERS, "combinedBins"):
            self.events[f"nFakeControl_{layer}"] = ak.num(
                self.events.IsoTrack[
                    fake_sideband_base_mask
                    & fake_track_layer_cut(
                        self.events.IsoTrack,
                        layer=layer,
                        require_dedx_max_over_median=require_dedx_max_over_median,
                    )
                ]
            )

        fake_zmumu_control = _z_to_mumu_control_mask(self.events, self.events.Muon)
        fake_zee_control = _z_to_ee_control_mask(self.events, self.events.Electron)
        self.events["nFakeZMuMuControl"] = ak.values_astype(fake_zmumu_control, np.int64)
        self.events["nFakeZeeControl"] = ak.values_astype(fake_zee_control, np.int64)
        self.events["FakeZMuMuDiag"] = ak.zip(
            _z_to_mumu_control_diagnostics(
                self.events,
                self.events.Muon,
                fiducial_hot_spots=fake_fiducial_hot_spots,
                require_dedx_max_over_median=require_dedx_max_over_median,
            )
        )
        self.events["FakeZeeDiag"] = ak.zip(
            _z_to_ee_control_diagnostics(
                self.events,
                self.events.Electron,
                fiducial_hot_spots=fake_fiducial_hot_spots,
                require_dedx_max_over_median=require_dedx_max_over_median,
            )
        )
        self.events["FakeZMuMuFitTrack"] = _fake_fit_tracks_for_control(
            self.events,
            fake_zmumu_control,
            fiducial_hot_spots=fake_fiducial_hot_spots,
            require_dedx_max_over_median=require_dedx_max_over_median,
        )
        self.events["FakeZeeFitTrack"] = _fake_fit_tracks_for_control(
            self.events,
            fake_zee_control,
            fiducial_hot_spots=fake_fiducial_hot_spots,
            require_dedx_max_over_median=require_dedx_max_over_median,
        )
        for layer in (*PVETO_LAYERS, "combinedBins"):
            self.events[f"FakeZMuMuSidebandTrack_{layer}"] = (
                _fake_sideband_tracks_for_control(
                    self.events,
                    fake_zmumu_control,
                    layer=layer,
                    fiducial_hot_spots=fake_fiducial_hot_spots,
                    require_dedx_max_over_median=require_dedx_max_over_median,
                    base_mask=fake_sideband_base_mask,
                )
            )
            self.events[f"FakeZeeSidebandTrack_{layer}"] = (
                _fake_sideband_tracks_for_control(
                    self.events,
                    fake_zee_control,
                    layer=layer,
                    fiducial_hot_spots=fake_fiducial_hot_spots,
                    require_dedx_max_over_median=require_dedx_max_over_median,
                    base_mask=fake_sideband_base_mask,
                )
            )
            self.events[f"nFakeZMuMuSideband_{layer}"] = _fake_track_count_for_control(
                self.events,
                fake_zmumu_control,
                layer=layer,
                d0_region="sideband",
                fiducial_hot_spots=fake_fiducial_hot_spots,
                require_dedx_max_over_median=require_dedx_max_over_median,
                base_mask=fake_sideband_base_mask,
            )
            self.events[f"nFakeZeeSideband_{layer}"] = _fake_track_count_for_control(
                self.events,
                fake_zee_control,
                layer=layer,
                d0_region="sideband",
                fiducial_hot_spots=fake_fiducial_hot_spots,
                require_dedx_max_over_median=require_dedx_max_over_median,
                base_mask=fake_sideband_base_mask,
            )
            if self._fake_sideband_histograms_enabled():
                _add_fake_sideband_track_diagnostics(
                    self.events,
                    fake_zmumu_control,
                    control_key="ZMuMu",
                    layer=layer,
                    fiducial_hot_spots=fake_fiducial_hot_spots,
                    require_dedx_max_over_median=require_dedx_max_over_median,
                    base_mask=fake_sideband_base_mask,
                )
                _add_fake_sideband_track_diagnostics(
                    self.events,
                    fake_zee_control,
                    control_key="Zee",
                    layer=layer,
                    fiducial_hot_spots=fake_fiducial_hot_spots,
                    require_dedx_max_over_median=require_dedx_max_over_median,
                    base_mask=fake_sideband_base_mask,
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
            muon_veto_probe_track_cutflow_masks(
                self.events.IsoTrack,
                require_dedx_max_over_median=self._lepton_background_dedx_cut_enabled(),
            ),
            "muon",
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
            muon_veto_probe_track_cutflow_masks(
                self.events.IsoTrack,
                require_dedx_max_over_median=self._lepton_background_dedx_cut_enabled(),
            ),
            "electron",
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

        tau_track_masks = tau_veto_probe_track_cutflow_masks(
            self.events.IsoTrack,
            require_dedx_max_over_median=self._lepton_background_dedx_cut_enabled(),
        )

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
            combined_layer_mask = generic_probe_pair_layer_mask(
                pairs, "combinedBins"
            )
            diagnostics.update(
                {
                    "pair_masswindow": ak.num(pairs[mass_window]) >= 1,
                    "pair_os": ak.num(pairs[os_mass_window]) >= 1,
                    "layer_combinedBins": ak.num(
                        pairs[os_mass_window & combined_layer_mask]
                    ) >= 1,
                    "pair_pass_tau_pveto": ak.num(
                        pairs[
                            os_mass_window
                            & combined_layer_mask
                            & tau_pveto_pair_pass_mask(pairs)
                        ]
                    )
                    >= 1,
                    "pair_ss_masswindow": ak.num(pairs[ss_mass_window]) >= 1,
                    "pair_ss_pass_tau_pveto": ak.num(
                        pairs[
                            ss_mass_window
                            & combined_layer_mask
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
