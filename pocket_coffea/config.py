"""Minimal PocketCoffea configuration for local custom-NanoAOD validation.

Run from this directory after installing this package and the sibling
PocketCoffea checkout.
"""

from __future__ import annotations

import os
import json
from pathlib import Path

import cloudpickle
from omegaconf import OmegaConf

import disapptrks
import disapptrks.selections as disapptrks_selections

from pocket_coffea.parameters import defaults
from pocket_coffea.parameters.cuts import passthrough
from pocket_coffea.lib.categorization import StandardSelection
from pocket_coffea.lib.hist_manager import Axis, HistConf
from pocket_coffea.utils.configurator import Configurator

import cuts
import workflow
from cuts import (
    event_flags,
    basic_event_selection,
    candidate_track_selection,
    disappearing_track_selection,
    isolated_track_selection,
    electron_pveto_diagnostic_cuts,
    fake_track_cuts,
    golden_json_lumi,
    has_disappearing_track,
    jet_veto_map,
    lepton_background_cuts,
    lepton_pveto_cuts,
    met_hlt,
    muon_table16_cuts,
    muon_pveto_layer_cuts,
    EVENT_DIAGNOSTIC_FIELDS,
    fake_track_diagnostic_cuts,
    fake_z_control_diagnostic_cuts,
    search_diagnostic_cuts,
    search_kinematics,
    single_electron_hlt,
    single_muon_hlt,
    tau_pveto_diagnostic_cuts,
)
from cuts import (
    has_muon_veto_os_mass10_pair,
    has_muon_veto_os_pair,
    has_muon_pveto_ss_zwindow_pass_pair,
    has_muon_pveto_zwindow_pass_pair,
    has_muon_veto_ss_mass10_pair,
    has_muon_veto_ss_pair,
    has_muon_tag,
    has_muon_veto_probe_track,
    has_muon_veto_tag_probe_pair,
    has_muon_veto_ss_zwindow_fail_pair,
    has_muon_veto_ss_zwindow_pair,
    has_muon_veto_ss_zwindow_pass_pair,
    has_muon_veto_zwindow_fail_pair,
    has_muon_veto_zwindow_pair,
    has_muon_veto_zwindow_pass_pair,
)
from workflow import DisappTrksProcessor

cloudpickle.register_pickle_by_value(cuts)
cloudpickle.register_pickle_by_value(workflow)
# Dask workers can run in a slightly different import context inside the LPC
# container. Ship local analysis modules by value instead of requiring workers
# to import them from the same filesystem path.
cloudpickle.register_pickle_by_value(disapptrks)
cloudpickle.register_pickle_by_value(disapptrks_selections)

localdir = os.path.dirname(os.path.abspath(__file__))


def _local_golden_json_path_for_config(year: str) -> Path | None:
    filenames = [
        f"Cert_Collisions{year}_Golden.json",
    ]
    if year == "2026":
        filenames.extend(
            [
                "Collisions26_MLEnhancedGolden_Latest.json",
                "Cert_Collisions2026_Golden.json",
            ]
        )

    search_dirs = []
    env_dir = os.environ.get("DISAPPTRKS_GOLDEN_JSON_DIR")
    if env_dir:
        search_dirs.append(Path(env_dir))
    search_dirs.append(Path(localdir) / "data" / "golden_jsons")
    search_dirs.append(Path.cwd() / "data" / "golden_jsons")
    search_dirs.append(Path.cwd() / "golden_jsons")

    for directory in search_dirs:
        candidates = [directory / filename for filename in filenames]
        if year == "2026":
            candidates.extend(sorted(directory.glob("Collisions26*Golden*.json")))
        candidates.extend(sorted(directory.glob(f"Cert_Collisions{year}*_Golden.json")))
        for candidate in candidates:
            if candidate.exists():
                return candidate.resolve()
    return None


def _install_embedded_golden_json(year: str, params) -> None:
    path = _local_golden_json_path_for_config(year)
    if path is None:
        return

    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    cuts.GOLDEN_JSON_PAYLOADS[year] = payload
    workflow.GOLDEN_JSON_PAYLOADS[year] = payload
    try:
        params.lumi.goldenJSON[year] = str(path)
    except Exception:
        try:
            params["lumi"]["goldenJSON"][year] = str(path)
        except Exception:
            pass


def _install_cvmfs_resolver_fallback():
    """Let PocketCoffea load defaults on workers without cms-griddata CVMFS.

    PocketCoffea's default resolver inspects
    ``/cvmfs/cms-griddata.cern.ch/cat/metadata`` while building the analysis
    parameters.  Some OSU Condor workers do not mount that path.  For this
    analysis we only need the Run-3 JME jet-veto-map payloads, which are
    transferred with the job into ``data/jet_veto_maps``.  This fallback keeps
    parameter resolution alive in that environment while preserving the normal
    PocketCoffea behavior anywhere CVMFS is available.
    """

    original_setup = defaults.setup_cvmfs_resolver
    metadata_root = Path("/cvmfs/cms-griddata.cern.ch/cat/metadata")

    def setup_cvmfs_resolver(group_tags=None):
        if metadata_root.exists():
            return original_setup(group_tags)

        def cvmfs_path_resolver(period: str, group: str, file: str, tag=None) -> str:
            if group == "JME" and file == "jetvetomaps.json.gz":
                search_dirs = []
                env_dir = os.environ.get("DISAPPTRKS_JET_VETO_MAP_DIR")
                if env_dir:
                    search_dirs.append(Path(env_dir))
                search_dirs.append(Path(localdir) / "data" / "jet_veto_maps")
                search_dirs.append(Path.cwd() / "data" / "jet_veto_maps")
                search_dirs.append(Path.cwd() / "jet_veto_maps")

                for directory in search_dirs:
                    candidates = [
                        directory / f"{period}_jetvetomaps.json.gz",
                        directory / period / "jetvetomaps.json.gz",
                    ]
                    for candidate in candidates:
                        if candidate.exists():
                            return str(candidate)

            resolved_tag = tag
            if resolved_tag is None and group_tags and group in group_tags:
                resolved_tag = group_tags[group].get(period)
            if resolved_tag is None:
                resolved_tag = "latest"
            return str(metadata_root / group / period / resolved_tag / file)

        OmegaConf.register_new_resolver("cvmfs", cvmfs_path_resolver, replace=True)

    defaults.setup_cvmfs_resolver = setup_cvmfs_resolver


_install_cvmfs_resolver_fallback()
parameters = defaults.get_default_parameters()
_install_embedded_golden_json("2026", parameters)
cloudpickle.register_pickle_by_value(cuts)
cloudpickle.register_pickle_by_value(workflow)
dataset_json = os.environ.get(
    "DISAPPTRKS_DATASET_JSON",
    f"{localdir}/datasets/local_2024F_muon.json",
)
if not os.path.isabs(dataset_json) and not os.path.exists(dataset_json):
    dataset_json = os.path.join(localdir, dataset_json)


def _unique_dataset_metadata_values(json_path, key):
    try:
        with open(json_path, encoding="utf-8") as handle:
            datasets = json.load(handle)
    except OSError:
        return []

    values = {
        dataset.get("metadata", {}).get(key)
        for dataset in datasets.values()
        if dataset.get("metadata", {}).get(key) is not None
    }
    return sorted(values)


def _single_or_none(values):
    return values[0] if len(values) == 1 else None


dataset_sample = os.environ.get("DISAPPTRKS_DATASET_SAMPLE")
if dataset_sample is None:
    dataset_sample = _single_or_none(_unique_dataset_metadata_values(dataset_json, "sample"))

dataset_year = os.environ.get("DISAPPTRKS_DATASET_YEAR")
if dataset_year is None:
    dataset_year = _single_or_none(_unique_dataset_metadata_values(dataset_json, "year"))

dataset_filter = {}
if dataset_sample:
    dataset_filter["samples"] = [dataset_sample]
if dataset_year:
    dataset_filter["year"] = [dataset_year]
enable_search_diagnostics = os.environ.get(
    "DISAPPTRKS_ENABLE_SEARCH_DIAGNOSTICS", ""
).lower() in ("1", "true", "yes", "on")
enable_lepton_background_categories = os.environ.get(
    "DISAPPTRKS_ENABLE_LEPTON_BACKGROUND_CATEGORIES", ""
).lower() in ("1", "true", "yes", "on")
category_mode = os.environ.get("DISAPPTRKS_CATEGORY_MODE", "muon_pveto")
fake_track_control_mode = os.environ.get("DISAPPTRKS_FAKE_TRACK_CONTROL", "basic").lower()
if fake_track_control_mode in ("jetmet", "basic_selection"):
    fake_track_control_mode = "basic"
if fake_track_control_mode not in ("basic", "zmumu", "zee"):
    raise ValueError(
        "Unknown DISAPPTRKS_FAKE_TRACK_CONTROL="
        f"{fake_track_control_mode!r}. Expected one of basic, zmumu, zee."
    )
parameters["disapptrks"] = {
    "category_mode": category_mode,
    "fake_track_control": fake_track_control_mode,
    "full_workflow": os.environ.get("DISAPPTRKS_FULL_WORKFLOW", "").lower()
    in ("1", "true", "yes", "on"),
    "full_variables": os.environ.get("DISAPPTRKS_FULL_VARIABLES", "").lower()
    in ("1", "true", "yes", "on"),
    "search_diagnostics": enable_search_diagnostics,
}
data_quality_cuts = [golden_json_lumi, event_flags, jet_veto_map]


def _skim_cuts_for_mode(mode, sample):
    sample = sample or ""
    if mode in ("muon_pveto", "tau_mu_pveto", "muon_backgrounds"):
        return [single_muon_hlt]
    if mode in ("electron_pveto", "tau_ele_pveto", "egamma_backgrounds"):
        return [single_electron_hlt]
    if mode == "fiducial_maps":
        if sample == "DATA_Muon":
            return [single_muon_hlt]
        if sample == "DATA_EGamma":
            return [single_electron_hlt]
    if mode == "fake_tracks":
        if sample == "DATA_Muon":
            return [single_muon_hlt]
        if sample == "DATA_EGamma":
            return [single_electron_hlt]
        if sample in ("DATA_JetMET", "DATA_MET"):
            return [met_hlt]
    return []


skim_cuts = _skim_cuts_for_mode(category_mode, dataset_sample)
if os.environ.get("DISAPPTRKS_DISABLE_HLT_SKIM", "").lower() in ("1", "true", "yes", "on"):
    skim_cuts = []
enable_generic_diagnostics = enable_search_diagnostics and (
    category_mode == "all"
    or (category_mode == "fake_tracks" and fake_track_control_mode == "basic")
)
if enable_generic_diagnostics:
    diagnostic_fields = (
        EVENT_DIAGNOSTIC_FIELDS
        if category_mode == "fake_tracks"
        else tuple(search_diagnostic_cuts)
    )
    diagnostic_categories = {
        f"diag_{name}": [search_diagnostic_cuts[name]]
        for name in diagnostic_fields
    }
else:
    diagnostic_categories = {}
fake_track_diagnostic_categories = (
    {
        f"fake_track_diag_{name}": [cut]
        for name, cut in fake_track_diagnostic_cuts.items()
    }
    if (
        enable_search_diagnostics
        and category_mode == "fake_tracks"
        and fake_track_control_mode == "basic"
    )
    else {}
)
fake_z_control_diagnostic_categories = {
    cut.name: [cut] for cut in fake_z_control_diagnostic_cuts.values()
}
muon_pveto_layer_categories = {
    name: [cut] for name, cut in muon_pveto_layer_cuts.items()
}
lepton_pveto_categories = {name: [cut] for name, cut in lepton_pveto_cuts.items()}
lepton_background_categories = {
    name: [cut] for name, cut in lepton_background_cuts.items()
}
fake_track_categories = {name: [cut] for name, cut in fake_track_cuts.items()}
muon_table16_categories = {
    f"muon_table16_{name}": [cut] for name, cut in muon_table16_cuts.items()
}
electron_pveto_diagnostic_categories = {
    f"electron_pveto_diag_{name}": [cut]
    for name, cut in electron_pveto_diagnostic_cuts.items()
}
tau_pveto_diagnostic_categories = {
    f"tau_pveto_diag_{name}": [cut]
    for name, cut in tau_pveto_diagnostic_cuts.items()
}

common_categories = {
    "inclusive": [passthrough],
    "basic_selection": [basic_event_selection],
    "isolated_track_selection": [basic_event_selection, isolated_track_selection],
    "candidate_track_selection": [
        basic_event_selection,
        isolated_track_selection,
        candidate_track_selection,
    ],
    "disappearing_track_selection": [
        basic_event_selection,
        isolated_track_selection,
        candidate_track_selection,
        disappearing_track_selection,
    ],
    "search": [
        basic_event_selection,
        isolated_track_selection,
        candidate_track_selection,
        disappearing_track_selection,
    ],
}
muon_pveto_categories = {
    "muon_veto_tag": [has_muon_tag],
    "muon_veto_probe": [has_muon_tag, has_muon_veto_probe_track],
    "muon_veto_pair": [has_muon_veto_tag_probe_pair],
    "muon_veto_pair_os": [has_muon_veto_os_pair],
    "muon_veto_pair_os_mass10": [has_muon_veto_os_mass10_pair],
    "muon_veto_pair_ss": [has_muon_veto_ss_pair],
    "muon_veto_pair_ss_mass10": [has_muon_veto_ss_mass10_pair],
    "muon_veto_zwindow": [has_muon_veto_zwindow_pair],
    "muon_veto_zwindow_pass": [has_muon_veto_zwindow_pass_pair],
    "muon_veto_zwindow_fail": [has_muon_veto_zwindow_fail_pair],
    "muon_pveto_zwindow_pass": [has_muon_pveto_zwindow_pass_pair],
    "muon_veto_ss_zwindow": [has_muon_veto_ss_zwindow_pair],
    "muon_veto_ss_zwindow_pass": [has_muon_veto_ss_zwindow_pass_pair],
    "muon_veto_ss_zwindow_fail": [has_muon_veto_ss_zwindow_fail_pair],
    "muon_pveto_ss_zwindow_pass": [has_muon_pveto_ss_zwindow_pass_pair],
}


def _categories_with_prefix(categories, *prefixes):
    return {
        name: cut
        for name, cut in categories.items()
        if any(name.startswith(prefix) for prefix in prefixes)
    }


def _categories_with_exact_or_prefix(categories, exact=(), prefixes=()):
    return {
        name: cut
        for name, cut in categories.items()
        if name in exact or any(name.startswith(prefix) for prefix in prefixes)
    }


fake_track_basic_categories = _categories_with_exact_or_prefix(
    fake_track_categories,
    exact=("fake_basic3hits_d0_signal", "fake_basic3hits_d0_sideband"),
    prefixes=("fake_control_",),
)
fake_track_zmumu_categories = _categories_with_exact_or_prefix(
    fake_track_categories,
    exact=("fake_zmumu_control",),
    prefixes=("fake_zmumu_sideband_",),
)
fake_track_zee_categories = _categories_with_exact_or_prefix(
    fake_track_categories,
    exact=("fake_zee_control",),
    prefixes=("fake_zee_sideband_",),
)
pveto_lepton_background_categories = (
    lepton_background_categories if enable_lepton_background_categories else {}
)


if category_mode == "muon_pveto":
    selected_categories = {
        **common_categories,
        **muon_pveto_categories,
        **muon_table16_categories,
        **muon_pveto_layer_categories,
        **_categories_with_prefix(pveto_lepton_background_categories, "muon_"),
    }
elif category_mode == "electron_pveto":
    selected_categories = {
        **common_categories,
        **_categories_with_prefix(lepton_pveto_categories, "electron_"),
        **_categories_with_prefix(pveto_lepton_background_categories, "electron_"),
        **electron_pveto_diagnostic_categories,
    }
elif category_mode == "tau_mu_pveto":
    selected_categories = {
        **common_categories,
        **_categories_with_prefix(lepton_pveto_categories, "tau_mu_"),
        **_categories_with_prefix(pveto_lepton_background_categories, "tau_mu_"),
        **_categories_with_prefix(tau_pveto_diagnostic_categories, "tau_pveto_diag_tau_mu_"),
    }
elif category_mode == "tau_ele_pveto":
    selected_categories = {
        **common_categories,
        **_categories_with_prefix(lepton_pveto_categories, "tau_ele_"),
        **_categories_with_prefix(pveto_lepton_background_categories, "tau_ele_"),
        **_categories_with_prefix(tau_pveto_diagnostic_categories, "tau_pveto_diag_tau_ele_"),
    }
elif category_mode == "fake_tracks":
    if fake_track_control_mode == "zmumu":
        selected_categories = {
            "inclusive": common_categories["inclusive"],
            **fake_track_zmumu_categories,
            **_categories_with_prefix(fake_z_control_diagnostic_categories, "fake_zmumu_"),
        }
    elif fake_track_control_mode == "zee":
        selected_categories = {
            "inclusive": common_categories["inclusive"],
            **fake_track_zee_categories,
            **_categories_with_prefix(fake_z_control_diagnostic_categories, "fake_zee_"),
        }
    else:
        selected_categories = {
            **common_categories,
            **fake_track_basic_categories,
        }
elif category_mode == "muon_backgrounds":
    selected_categories = {
        **common_categories,
        **muon_pveto_categories,
        **muon_pveto_layer_categories,
        **_categories_with_prefix(lepton_pveto_categories, "tau_mu_"),
        **_categories_with_prefix(lepton_background_categories, "muon_", "tau_mu_"),
        **fake_track_zmumu_categories,
    }
elif category_mode == "egamma_backgrounds":
    selected_categories = {
        **common_categories,
        **_categories_with_prefix(lepton_pveto_categories, "electron_", "tau_ele_"),
        **_categories_with_prefix(lepton_background_categories, "electron_", "tau_ele_"),
        **fake_track_zee_categories,
    }
elif category_mode == "fiducial_maps":
    selected_categories = {
        "inclusive": common_categories["inclusive"],
    }
elif category_mode == "all":
    selected_categories = {
        **common_categories,
        **muon_pveto_categories,
        **lepton_pveto_categories,
        **lepton_background_categories,
        **fake_track_categories,
        **muon_table16_categories,
        **electron_pveto_diagnostic_categories,
        **tau_pveto_diagnostic_categories,
        **muon_pveto_layer_categories,
        **fake_z_control_diagnostic_categories,
    }
else:
    raise ValueError(
        "Unknown DISAPPTRKS_CATEGORY_MODE="
        f"{category_mode!r}. Expected one of muon_pveto, electron_pveto, "
        "tau_mu_pveto, tau_ele_pveto, fake_tracks, muon_backgrounds, "
        "egamma_backgrounds, fiducial_maps, all."
    )

selected_categories = {
    **selected_categories,
    **diagnostic_categories,
    **fake_track_diagnostic_categories,
}

pveto_layers = ("NLayers4", "NLayers5", "NLayers6plus")
outer_hit_variants = {
    "missingOuterHits": "tracker layers without measurement",
    "hp_nLostHitsOuter": "lost hits",
    "hp_nLostTrackerHitsOuter": "lost tracker hits",
    "hp_nLostPixelHitsOuter": "lost pixel hits",
    "hp_nLostStripHitsOuter": "lost strip hits",
    "hp_pixelLayersWithoutMeasurementOuter": "pixel layers without measurement",
    "hp_stripLayersWithoutMeasurementOuter": "strip layers without measurement",
}


def _event_count_hist(field, label, bins=50):
    return HistConf(
        [
            Axis(
                coll="events",
                field=field,
                bins=bins,
                start=0,
                stop=bins,
                label=label,
            )
        ]
    )


def _env_flag(name, default=False):
    value = os.environ.get(name)
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "on")


def _variables_for_mode(mode, variables):
    """Return the histogram variables needed for the active production mode.

    Full diagnostic histograms dominate runtime for large production jobs.  The
    default is therefore a minimal variable set that preserves the counts and
    fake-track fit histograms needed by the table-making commands.  Set
    ``DISAPPTRKS_FULL_VARIABLES=1`` for exploratory/debugging jobs that need the
    full histogram suite.
    """

    if mode == "all" or _env_flag("DISAPPTRKS_FULL_VARIABLES"):
        return variables
    if _env_flag("DISAPPTRKS_DISABLE_MINIMAL_VARIABLES"):
        return variables

    fake_track_prefixes = {
        "basic": (),
        "zmumu": ("fakeZMuMuFitTrack_",),
        "zee": ("fakeZeeFitTrack_",),
    }[fake_track_control_mode]
    prefixes_by_mode = {
        "muon_pveto": ("nMuon",),
        "electron_pveto": ("nElectron",),
        "tau_mu_pveto": ("nTauMu",),
        "tau_ele_pveto": ("nTauEle",),
        "fake_tracks": fake_track_prefixes,
        "muon_backgrounds": ("nMuon", "nTauMu", "fakeZMuMuFitTrack_"),
        "egamma_backgrounds": ("nElectron", "nTauEle", "fakeZeeFitTrack_"),
        "fiducial_maps": ("electronFiducial", "muonFiducial"),
    }
    prefixes = prefixes_by_mode.get(mode)
    if prefixes is None:
        return variables
    return {
        name: variable
        for name, variable in variables.items()
        if any(name.startswith(prefix) for prefix in prefixes)
    }


lepton_pair_count_variables = {}
for prefix, label in (
    ("Electron", "electron"),
    ("TauMu", r"tau-muon"),
    ("TauEle", r"tau-electron"),
):
    lepton_pair_count_variables.update(
        {
            f"n{prefix}TagProbePair": _event_count_hist(
                f"n{prefix}TagProbePair",
                f"N({label} tag-probe pairs)",
            ),
            f"n{prefix}TagProbePairMassWindow": _event_count_hist(
                f"n{prefix}TagProbePairMassWindow",
                f"N({label} tag-probe pairs in mass window)",
            ),
            f"n{prefix}TagProbePairOSMassWindow": _event_count_hist(
                f"n{prefix}TagProbePairOSMassWindow",
                f"N(OS {label} tag-probe pairs in mass window)",
            ),
            f"n{prefix}TagProbePairSSMassWindow": _event_count_hist(
                f"n{prefix}TagProbePairSSMassWindow",
                f"N(SS {label} tag-probe pairs in mass window)",
            ),
            f"n{prefix}PVetoTagProbePairMassWindowPass": _event_count_hist(
                f"n{prefix}PVetoTagProbePairMassWindowPass",
                f"N(OS {label} mass-window pairs passing Pveto numerator)",
            ),
            f"n{prefix}PVetoTagProbePairSSMassWindowPass": _event_count_hist(
                f"n{prefix}PVetoTagProbePairSSMassWindowPass",
                f"N(SS {label} mass-window pairs passing Pveto numerator)",
            ),
        }
    )
    for layer in pveto_layers:
        lepton_pair_count_variables.update(
            {
                f"n{prefix}TagProbePairMassWindow_{layer}": _event_count_hist(
                    f"n{prefix}TagProbePairMassWindow_{layer}",
                    f"N(OS {label} mass-window pairs, {layer})",
                ),
                f"n{prefix}PVetoTagProbePairMassWindowPass_{layer}": _event_count_hist(
                    f"n{prefix}PVetoTagProbePairMassWindowPass_{layer}",
                    f"N(OS {label} mass-window pairs passing Pveto numerator, {layer})",
                ),
                f"n{prefix}TagProbePairSSMassWindow_{layer}": _event_count_hist(
                    f"n{prefix}TagProbePairSSMassWindow_{layer}",
                    f"N(SS {label} mass-window pairs, {layer})",
                ),
                f"n{prefix}PVetoTagProbePairSSMassWindowPass_{layer}": _event_count_hist(
                    f"n{prefix}PVetoTagProbePairSSMassWindowPass_{layer}",
                    f"N(SS {label} mass-window pairs passing Pveto numerator, {layer})",
                ),
            }
        )

lepton_background_count_variables = {}
for prefix, label in (
    ("Muon", "muon"),
    ("Electron", "electron"),
    ("TauMu", r"tau-muon"),
    ("TauEle", r"tau-electron"),
):
    for layer in (*pveto_layers, "combinedBins"):
        lepton_background_count_variables.update(
            {
                f"n{prefix}BackgroundControl_{layer}": _event_count_hist(
                    f"n{prefix}BackgroundControl_{layer}",
                    f"N({label} lepton-background control events, {layer})",
                    bins=2,
                ),
                f"n{prefix}BackgroundOffline_{layer}": _event_count_hist(
                    f"n{prefix}BackgroundOffline_{layer}",
                    f"N({label} control events passing offline MET, {layer})",
                    bins=2,
                ),
                f"n{prefix}BackgroundTrigger_{layer}": _event_count_hist(
                    f"n{prefix}BackgroundTrigger_{layer}",
                    f"N({label} control events passing offline MET and MET trigger, {layer})",
                    bins=2,
                ),
            }
        )

fiducial_map_variables = {}
for prefix, label in (
    ("Electron", "electron"),
    ("Muon", "muon"),
):
    for stage, stage_label in (("Before", "before veto"), ("After", "after veto")):
        fiducial_map_variables[f"{label}Fiducial{stage}_eta_phi"] = HistConf(
            [
                Axis(
                    coll=f"{prefix}Fiducial{stage}",
                    field="probe_eta",
                    bins=60,
                    start=-3.0,
                    stop=3.0,
                    label=f"{label} fiducial-map probe eta ({stage_label})",
                ),
                Axis(
                    coll=f"{prefix}Fiducial{stage}",
                    field="probe_phi",
                    bins=64,
                    start=-3.2,
                    stop=3.2,
                    label=f"{label} fiducial-map probe phi ({stage_label})",
                ),
            ],
            only_categories=["inclusive"],
        )

cfg = Configurator(
    parameters=parameters,
    datasets={
        "jsons": [dataset_json],
        "filter": dataset_filter,
    },
    workflow=DisappTrksProcessor,
    calibrators=[],
    skim=skim_cuts,
    preselections=data_quality_cuts,
    categories=StandardSelection(selected_categories),
    weights={"common": {"inclusive": []}, "bysample": {}},
    weights_classes=[],
    variations={"weights": {"common": {"inclusive": []}}},
    variables=_variables_for_mode(category_mode, {
        "nIsoTrack": HistConf(
            [
                Axis(
                    coll="events",
                    field="nIsoTrack",
                    bins=20,
                    start=0,
                    stop=20,
                    label="N(isolated tracks)",
                )
            ]
        ),
        "nIsoTrackProbe": HistConf(
            [
                Axis(
                    coll="events",
                    field="nIsoTrackProbe",
                    bins=10,
                    start=0,
                    stop=10,
                    label="N(probe tracks)",
                )
            ]
        ),
        "nMuonTag": HistConf(
            [
                Axis(
                    coll="events",
                    field="nMuonTag",
                    bins=10,
                    start=0,
                    stop=10,
                    label="N(tight isolated muon tags)",
                )
            ]
        ),
        "nMuonVetoProbeTrack": HistConf(
            [
                Axis(
                    coll="events",
                    field="nMuonVetoProbeTrack",
                    bins=10,
                    start=0,
                    stop=10,
                    label="N(muon-veto probe tracks)",
                )
            ]
        ),
        "nMuonVetoTagProbePair": HistConf(
            [
                Axis(
                    coll="events",
                    field="nMuonVetoTagProbePair",
                    bins=20,
                    start=0,
                    stop=20,
                    label="N(muon tag-probe pairs)",
                )
            ]
        ),
        "nMuonVetoTagProbePairOS": HistConf(
            [
                Axis(
                    coll="events",
                    field="nMuonVetoTagProbePairOS",
                    bins=20,
                    start=0,
                    stop=20,
                    label="N(OS muon tag-probe pairs)",
                )
            ]
        ),
        "nMuonVetoTagProbePairOSMass10": HistConf(
            [
                Axis(
                    coll="events",
                    field="nMuonVetoTagProbePairOSMass10",
                    bins=20,
                    start=0,
                    stop=20,
                    label="N(OS muon tag-probe pairs with mass > 10 GeV)",
                )
            ]
        ),
        "nMuonVetoTagProbePairSS": HistConf(
            [
                Axis(
                    coll="events",
                    field="nMuonVetoTagProbePairSS",
                    bins=20,
                    start=0,
                    stop=20,
                    label="N(SS muon tag-probe pairs)",
                )
            ]
        ),
        "nMuonVetoTagProbePairSSMass10": HistConf(
            [
                Axis(
                    coll="events",
                    field="nMuonVetoTagProbePairSSMass10",
                    bins=20,
                    start=0,
                    stop=20,
                    label="N(SS muon tag-probe pairs with mass > 10 GeV)",
                )
            ]
        ),
        "nMuonVetoTagProbePairZWindow": HistConf(
            [
                Axis(
                    coll="events",
                    field="nMuonVetoTagProbePairZWindow",
                    bins=10,
                    start=0,
                    stop=10,
                    label="N(OS muon tag-probe pairs in Z window)",
                )
            ]
        ),
        "nMuonVetoTagProbePairZWindowPass": HistConf(
            [
                Axis(
                    coll="events",
                    field="nMuonVetoTagProbePairZWindowPass",
                    bins=10,
                    start=0,
                    stop=10,
                    label="N(OS Z-window pairs passing muon veto)",
                )
            ]
        ),
        "nMuonVetoTagProbePairZWindowFail": HistConf(
            [
                Axis(
                    coll="events",
                    field="nMuonVetoTagProbePairZWindowFail",
                    bins=10,
                    start=0,
                    stop=10,
                    label="N(OS Z-window pairs failing muon veto)",
                )
            ]
        ),
        "nMuonPVetoTagProbePairZWindowPass": HistConf(
            [
                Axis(
                    coll="events",
                    field="nMuonPVetoTagProbePairZWindowPass",
                    bins=10,
                    start=0,
                    stop=10,
                    label="N(OS Z-window pairs passing muon Pveto numerator)",
                )
            ]
        ),
        "nMuonVetoTagProbePairSSZWindow": HistConf(
            [
                Axis(
                    coll="events",
                    field="nMuonVetoTagProbePairSSZWindow",
                    bins=10,
                    start=0,
                    stop=10,
                    label="N(SS Z-window muon tag-probe pairs)",
                )
            ]
        ),
        "nMuonVetoTagProbePairSSZWindowPass": HistConf(
            [
                Axis(
                    coll="events",
                    field="nMuonVetoTagProbePairSSZWindowPass",
                    bins=10,
                    start=0,
                    stop=10,
                    label="N(SS Z-window pairs passing muon veto)",
                )
            ]
        ),
        "nMuonVetoTagProbePairSSZWindowFail": HistConf(
            [
                Axis(
                    coll="events",
                    field="nMuonVetoTagProbePairSSZWindowFail",
                    bins=10,
                    start=0,
                    stop=10,
                    label="N(SS Z-window pairs failing muon veto)",
                )
            ]
        ),
        "nMuonPVetoTagProbePairSSZWindowPass": HistConf(
            [
                Axis(
                    coll="events",
                    field="nMuonPVetoTagProbePairSSZWindowPass",
                    bins=10,
                    start=0,
                    stop=10,
                    label="N(SS Z-window pairs passing muon Pveto numerator)",
                )
            ]
        ),
        **{
            f"nMuonVetoTagProbePairZWindow_{layer}": HistConf(
                [
                    Axis(
                        coll="events",
                        field=f"nMuonVetoTagProbePairZWindow_{layer}",
                        bins=10,
                        start=0,
                        stop=10,
                        label=f"N(OS Z-window pairs, {layer})",
                    )
                ]
            )
            for layer in pveto_layers
        },
        **{
            f"nMuonPVetoTagProbePairZWindowPass_{layer}": HistConf(
                [
                    Axis(
                        coll="events",
                        field=f"nMuonPVetoTagProbePairZWindowPass_{layer}",
                        bins=10,
                        start=0,
                        stop=10,
                        label=f"N(OS Z-window pairs passing muon Pveto numerator, {layer})",
                    )
                ]
            )
            for layer in pveto_layers
        },
        **{
            f"nMuonVetoTagProbePairSSZWindow_{layer}": HistConf(
                [
                    Axis(
                        coll="events",
                        field=f"nMuonVetoTagProbePairSSZWindow_{layer}",
                        bins=10,
                        start=0,
                        stop=10,
                        label=f"N(SS Z-window pairs, {layer})",
                    )
                ]
            )
            for layer in pveto_layers
        },
        **{
            f"nMuonPVetoTagProbePairSSZWindowPass_{layer}": HistConf(
                [
                    Axis(
                        coll="events",
                        field=f"nMuonPVetoTagProbePairSSZWindowPass_{layer}",
                        bins=10,
                        start=0,
                        stop=10,
                        label=f"N(SS Z-window pairs passing muon Pveto numerator, {layer})",
                    )
                ]
            )
            for layer in pveto_layers
        },
        **lepton_pair_count_variables,
        **lepton_background_count_variables,
        **fiducial_map_variables,
        "nIsoTrackIsolated": HistConf(
            [
                Axis(
                    coll="events",
                    field="nIsoTrackIsolated",
                    bins=10,
                    start=0,
                    stop=10,
                    label="N(isolated tracks)",
                )
            ]
        ),
        "nIsoTrackCandidate": HistConf(
            [
                Axis(
                    coll="events",
                    field="nIsoTrackCandidate",
                    bins=10,
                    start=0,
                    stop=10,
                    label="N(candidate tracks)",
                )
            ]
        ),
        "nIsoTrackSearch": HistConf(
            [
                Axis(
                    coll="events",
                    field="nIsoTrackSearch",
                    bins=10,
                    start=0,
                    stop=10,
                    label="N(search tracks)",
                )
            ]
        ),
        "nIsoTrackSearchPreMissingOuter": HistConf(
            [
                Axis(
                    coll="events",
                    field="nIsoTrackSearchPreMissingOuter",
                    bins=10,
                    start=0,
                    stop=10,
                    label="N(search tracks before missing outer hit cut)",
                )
            ]
        ),
        "nIsoTrackSearchPreLeptonVeto": HistConf(
            [
                Axis(
                    coll="events",
                    field="nIsoTrackSearchPreLeptonVeto",
                    bins=10,
                    start=0,
                    stop=10,
                    label="N(search tracks before lepton vetoes)",
                )
            ]
        ),
        "metNoMu_pt": HistConf(
            [
                Axis(
                    coll="AnalysisEvent",
                    field="METNoMu_pt",
                    bins=60,
                    start=0,
                    stop=600,
                    label=r"$p_T^{miss,no\ \mu}$ [GeV]",
                )
            ]
        ),
        "leadingJet_pt": HistConf(
            [
                Axis(
                    coll="AnalysisEvent",
                    field="leadingJet_pt",
                    bins=60,
                    start=0,
                    stop=600,
                    label="Leading selected jet $p_T$ [GeV]",
                )
            ]
        ),
        "leadingJet_metNoMu_deltaPhi": HistConf(
            [
                Axis(
                    coll="AnalysisEvent",
                    field="leadingJetMETNoMuDeltaPhi",
                    bins=32,
                    start=0,
                    stop=3.2,
                    label=r"$\Delta\phi$(leading jet, no-$\mu$ MET)",
                )
            ]
        ),
        "dijet_max_deltaPhi": HistConf(
            [
                Axis(
                    coll="AnalysisEvent",
                    field="dijetMaxDeltaPhi",
                    bins=34,
                    start=-0.2,
                    stop=3.2,
                    label=r"max $\Delta\phi$(selected dijets)",
                )
            ]
        ),
        "probeTrack_pt": HistConf(
            [
                Axis(
                    coll="IsoTrackProbe",
                    field="pt",
                    bins=60,
                    start=0,
                    stop=300,
                    label="Probe track $p_T$ [GeV]",
                )
            ]
        ),
        "probeTrack_eta": HistConf(
            [
                Axis(
                    coll="IsoTrackProbe",
                    field="eta",
                    bins=50,
                    start=-2.5,
                    stop=2.5,
                    label="Probe track eta",
                )
            ]
        ),
        "probeTrack_phi": HistConf(
            [
                Axis(
                    coll="IsoTrackProbe",
                    field="phi",
                    bins=64,
                    start=-3.2,
                    stop=3.2,
                    label="Probe track phi",
                )
            ]
        ),
        "probeTrack_caloEnergy": HistConf(
            [
                Axis(
                    coll="IsoTrackProbe",
                    field="caloEnergy",
                    bins=60,
                    start=0,
                    stop=60,
                    label="Probe track calorimeter energy [GeV]",
                )
            ]
        ),
        "probeTrack_dRMinJet": HistConf(
            [
                Axis(
                    coll="IsoTrackProbe",
                    field="dRMinJet",
                    bins=60,
                    start=-1,
                    stop=5,
                    label=r"Probe track min $\Delta R$(jet)",
                )
            ]
        ),
        "muonTag_pt": HistConf(
            [
                Axis(
                    coll="MuonTag",
                    field="pt",
                    bins=60,
                    start=0,
                    stop=300,
                    label="Muon tag $p_T$ [GeV]",
                )
            ]
        ),
        "muonTag_eta": HistConf(
            [
                Axis(
                    coll="MuonTag",
                    field="eta",
                    bins=50,
                    start=-2.5,
                    stop=2.5,
                    label="Muon tag eta",
                )
            ]
        ),
        "muonVetoProbeTrack_pt": HistConf(
            [
                Axis(
                    coll="MuonVetoProbeTrack",
                    field="pt",
                    bins=60,
                    start=0,
                    stop=300,
                    label="Muon-veto probe track $p_T$ [GeV]",
                )
            ]
        ),
        "muonVetoProbeTrack_eta": HistConf(
            [
                Axis(
                    coll="MuonVetoProbeTrack",
                    field="eta",
                    bins=50,
                    start=-2.5,
                    stop=2.5,
                    label="Muon-veto probe track eta",
                )
            ]
        ),
        "muonVetoProbeTrack_dRMinMuon": HistConf(
            [
                Axis(
                    coll="MuonVetoProbeTrack",
                    field="dRMinMuon",
                    bins=60,
                    start=-1,
                    stop=5,
                    label=r"Probe track min $\Delta R$(muon)",
                )
            ]
        ),
        "muonVetoTagProbePair_mass": HistConf(
            [
                Axis(
                    coll="MuonVetoTagProbePair",
                    field="mass",
                    bins=80,
                    start=0,
                    stop=200,
                    label="Muon tag-probe mass [GeV]",
                )
            ]
        ),
        "muonVetoTagProbePairOS_mass": HistConf(
            [
                Axis(
                    coll="MuonVetoTagProbePairOS",
                    field="mass",
                    bins=80,
                    start=0,
                    stop=200,
                    label="OS muon tag-probe mass [GeV]",
                )
            ]
        ),
        "muonVetoTagProbePairOSMass10_mass": HistConf(
            [
                Axis(
                    coll="MuonVetoTagProbePairOSMass10",
                    field="mass",
                    bins=80,
                    start=0,
                    stop=200,
                    label="OS muon tag-probe mass, mass > 10 GeV [GeV]",
                )
            ]
        ),
        "muonVetoTagProbePairZWindow_mass": HistConf(
            [
                Axis(
                    coll="MuonVetoTagProbePairZWindow",
                    field="mass",
                    bins=40,
                    start=70,
                    stop=110,
                    label="OS Z-window muon tag-probe mass [GeV]",
                )
            ]
        ),
        "muonVetoTagProbePairZWindow_probe_dRMinMuon": HistConf(
            [
                Axis(
                    coll="MuonVetoTagProbePairZWindow",
                    field="probe_dRMinMuon",
                    bins=60,
                    start=-1,
                    stop=5,
                    label=r"OS Z-window probe min $\Delta R$(muon)",
                )
            ]
        ),
        "muonVetoTagProbePairZWindowPass_probe_dRMinMuon": HistConf(
            [
                Axis(
                    coll="MuonVetoTagProbePairZWindowPass",
                    field="probe_dRMinMuon",
                    bins=60,
                    start=-1,
                    stop=5,
                    label=r"OS Z-window passing-probe min $\Delta R$(muon)",
                )
            ]
        ),
        "muonVetoTagProbePairZWindowFail_probe_dRMinMuon": HistConf(
            [
                Axis(
                    coll="MuonVetoTagProbePairZWindowFail",
                    field="probe_dRMinMuon",
                    bins=60,
                    start=-1,
                    stop=5,
                    label=r"OS Z-window failing-probe min $\Delta R$(muon)",
                )
            ]
        ),
        "fakeZMuMuFitTrack_absDxy": HistConf(
            [
                Axis(
                    coll="FakeZMuMuFitTrack",
                    field="absDxy",
                    bins=50,
                    start=0.0,
                    stop=0.5,
                    label=r"Z$\to\mu\mu$ fake-track $|d_{0}|$ [cm]",
                )
            ],
            only_categories=["inclusive"],
        ),
        "fakeZMuMuFitTrack_dxy": HistConf(
            [
                Axis(
                    coll="FakeZMuMuFitTrack",
                    field="dxy",
                    bins=25,
                    start=-0.5,
                    stop=0.5,
                    label=r"Z$\to\mu\mu$ fake-track $d_{0}$ [cm]",
                )
            ],
            only_categories=["inclusive"],
        ),
        "fakeZeeFitTrack_absDxy": HistConf(
            [
                Axis(
                    coll="FakeZeeFitTrack",
                    field="absDxy",
                    bins=50,
                    start=0.0,
                    stop=0.5,
                    label=r"Z$\to ee$ fake-track $|d_{0}|$ [cm]",
                )
            ],
            only_categories=["inclusive"],
        ),
        "fakeZeeFitTrack_dxy": HistConf(
            [
                Axis(
                    coll="FakeZeeFitTrack",
                    field="dxy",
                    bins=25,
                    start=-0.5,
                    stop=0.5,
                    label=r"Z$\to ee$ fake-track $d_{0}$ [cm]",
                )
            ],
            only_categories=["inclusive"],
        ),
        **{
            f"allTrack_{field}": HistConf(
                [
                    Axis(
                        coll="IsoTrack",
                        field=field,
                        bins=21,
                        start=-0.5,
                        stop=20.5,
                        label=f"All isolated-track outer {label}",
                    )
                ]
            )
            for field, label in outer_hit_variants.items()
        },
        **{
            f"probeTrack_{field}": HistConf(
                [
                    Axis(
                        coll="IsoTrackProbe",
                        field=field,
                        bins=21,
                        start=-0.5,
                        stop=20.5,
                        label=f"Probe-track outer {label}",
                    )
                ]
            )
            for field, label in outer_hit_variants.items()
        },
        **{
            f"preMissingOuterTrack_{field}": HistConf(
                [
                    Axis(
                        coll="IsoTrackSearchPreMissingOuter",
                        field=field,
                        bins=21,
                        start=-0.5,
                        stop=20.5,
                        label=f"Search preselection track outer {label}",
                    )
                ]
            )
            for field, label in outer_hit_variants.items()
        },
        **{
            f"preLeptonVetoTrack_{field}": HistConf(
                [
                    Axis(
                        coll="IsoTrackSearchPreLeptonVeto",
                        field=field,
                        bins=21,
                        start=-0.5,
                        stop=20.5,
                        label=f"Search pre-lepton-veto track outer {label}",
                    )
                ]
            )
            for field, label in outer_hit_variants.items()
        },
        "searchTrack_pt": HistConf(
            [
                Axis(
                    coll="IsoTrackSearch",
                    field="pt",
                    bins=60,
                    start=0,
                    stop=300,
                    label="Search track $p_T$ [GeV]",
                )
            ]
        ),
        "searchTrack_eta_phi": HistConf(
            [
                Axis(
                    coll="IsoTrackSearch",
                    field="eta",
                    bins=50,
                    start=-2.5,
                    stop=2.5,
                    label="Search track eta",
                ),
                Axis(
                    coll="IsoTrackSearch",
                    field="phi",
                    bins=64,
                    start=-3.2,
                    stop=3.2,
                    label="Search track phi",
                ),
            ]
        ),
    }),
    columns={},
)
