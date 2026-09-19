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
from pocket_coffea.lib.categorization import (
    CartesianSelection,
    MultiCut,
    StandardSelection,
)
from pocket_coffea.lib.columns_manager import ColOut
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
    has_high_purity_disappearing_track,
    high_purity_study_layer_axis_cuts,
    high_purity_study_selection_axis_cuts,
    jet_veto_map,
    lepton_background_cuts,
    lepton_pveto_cuts,
    met_hlt,
    muon_tau_hlt,
    muon_table16_cuts,
    muon_pveto_layer_cuts,
    EVENT_DIAGNOSTIC_FIELDS,
    fake_track_diagnostic_cuts,
    fake_z_control_diagnostic_cuts,
    search_diagnostic_cuts,
    tau_background_diagnostic_cuts,
    search_kinematics,
    SIGNAL_ACCEPTANCE_CARTESIAN_FIELDS,
    signal_acceptance_common_cutflow_cuts,
    signal_acceptance_layer_axis_cuts,
    signal_acceptance_layer_entry_cuts,
    signal_acceptance_layer_cuts,
    signal_acceptance_stage_cuts,
    signal_acceptance_variant_axis_cuts,
    single_electron_hlt,
    single_muon_hlt,
    tau_trigger_probability_hlt,
    tau_pveto_diagnostic_cuts,
    z_sideband_skim_cuts,
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
    try:
        cwd = Path.cwd()
    except OSError:
        cwd = None
    if cwd is not None:
        search_dirs.append(cwd / "data" / "golden_jsons")
        search_dirs.append(cwd / "golden_jsons")

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
for _golden_json_year in ("2022", "2023", "2024", "2025", "2026"):
    _install_embedded_golden_json(_golden_json_year, parameters)
cloudpickle.register_pickle_by_value(cuts)
cloudpickle.register_pickle_by_value(workflow)
dataset_json = os.environ.get(
    "DISAPPTRKS_DATASET_JSON",
    f"{localdir}/datasets/local_2024F_muon.json",
)
dataset_json = workflow.resolve_dataset_json(dataset_json, localdir=localdir)


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
enable_pveto_diagnostics = os.environ.get(
    "DISAPPTRKS_ENABLE_PVETO_DIAGNOSTICS", ""
).lower() in ("1", "true", "yes", "on")
category_mode = os.environ.get("DISAPPTRKS_CATEGORY_MODE", "muon_pveto")
fake_track_control_mode = os.environ.get("DISAPPTRKS_FAKE_TRACK_CONTROL", "basic").lower()
high_purity_study_layers = tuple(
    item.strip()
    for item in os.environ.get("DISAPPTRKS_HIGH_PURITY_STUDY_LAYERS", "NLayers4").split(",")
    if item.strip()
)
if any(layer not in ("NLayers4", "NLayers5", "NLayers6plus", "combinedBins") for layer in high_purity_study_layers):
    raise ValueError("DISAPPTRKS_HIGH_PURITY_STUDY_LAYERS contains an unknown layer bin")
enable_fake_sideband_histograms = os.environ.get(
    "DISAPPTRKS_ENABLE_FAKE_SIDEBAND_HISTOGRAMS", "1"
).lower() in ("1", "true", "yes", "on")
fake_track_require_dedx_cut = os.environ.get(
    "DISAPPTRKS_FAKE_TRACK_REQUIRE_DEDX_CUT", "1"
).lower() in ("1", "true", "yes", "on")
lepton_background_require_dedx_cut = os.environ.get(
    "DISAPPTRKS_LEPTON_BACKGROUND_REQUIRE_DEDX_CUT", "1"
).lower() in ("1", "true", "yes", "on")
enable_high_purity_dedx_histograms = os.environ.get(
    "DISAPPTRKS_ENABLE_HIGH_PURITY_DEDX_HISTOGRAMS", "0"
).lower() in ("1", "true", "yes", "on")
enable_signal_dedx_histograms = os.environ.get(
    "DISAPPTRKS_ENABLE_SIGNAL_DEDX_HISTOGRAMS", "0"
).lower() in ("1", "true", "yes", "on")
signal_dedx_layers = tuple(
    item.strip()
    for item in os.environ.get(
        "DISAPPTRKS_SIGNAL_DEDX_LAYERS",
        "NLayers4,NLayers5,NLayers6plus",
    ).split(",")
    if item.strip()
)
if any(
    layer not in ("NLayers4", "NLayers5", "NLayers6plus", "combinedBins")
    for layer in signal_dedx_layers
):
    raise ValueError("DISAPPTRKS_SIGNAL_DEDX_LAYERS contains an unknown layer bin")
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
    "fake_sideband_histograms": enable_fake_sideband_histograms,
    "fake_track_require_dedx_cut": fake_track_require_dedx_cut,
    "lepton_background_require_dedx_cut": lepton_background_require_dedx_cut,
    "high_purity_study_layers": high_purity_study_layers,
    "high_purity_dedx_histograms": enable_high_purity_dedx_histograms,
    "signal_dedx_histograms": enable_signal_dedx_histograms,
    "signal_dedx_layers": signal_dedx_layers,
    "full_workflow": os.environ.get("DISAPPTRKS_FULL_WORKFLOW", "").lower()
    in ("1", "true", "yes", "on"),
    "full_variables": os.environ.get("DISAPPTRKS_FULL_VARIABLES", "").lower()
    in ("1", "true", "yes", "on"),
    "search_diagnostics": enable_search_diagnostics,
    "pveto_diagnostics": enable_pveto_diagnostics,
    "lepton_background_categories": enable_lepton_background_categories,
}
data_quality_cuts = [golden_json_lumi, event_flags, jet_veto_map]


def _skim_cuts_for_mode(mode, sample):
    sample = sample or ""
    if mode in (
        "muon_pveto",
        "tau_mu_pveto",
        "muon_pmiss_poffline",
        "tau_mu_pmiss_poffline",
        "muon_backgrounds",
    ):
        return [single_muon_hlt]
    if mode == "tau_trigger_probability":
        return [tau_trigger_probability_hlt]
    if mode == "tau_pmiss_poffline":
        # N_ctrl/Poffline/Pmiss are measured on the single-muon-triggered
        # baseline, matching the legacy DisappTrks BackgroundEstimation
        # channel actually used for the dissertation's tau estimate
        # (TauTagPt55 -> TauTagSkim.triggers = triggersSingleMu in
        # BackgroundEstimation/python/TauTagProbeSelections.py). The
        # cross-trigger dependence enters separately via tau_trigger_probability's
        # P(tau) = P(cross)/P(single) factor, not by pre-conditioning this
        # control sample on the cross-trigger having fired.
        return [single_muon_hlt]
    if mode in (
        "electron_pveto",
        "tau_ele_pveto",
        "electron_pmiss_poffline",
        "tau_ele_pmiss_poffline",
        "egamma_backgrounds",
    ):
        return [single_electron_hlt]
    if mode == "fiducial_maps":
        if sample == "DATA_Muon":
            return [single_muon_hlt]
        if sample == "DATA_EGamma":
            return [single_electron_hlt]
    if mode in ("fake_tracks", "high_purity_study"):
        if sample == "DATA_Muon":
            return [single_muon_hlt]
        if sample == "DATA_EGamma":
            return [single_electron_hlt]
        if sample in ("DATA_JetMET", "DATA_MET"):
            return [met_hlt]
    if mode == "signal_acceptance":
        return [met_hlt]
    return []


skim_cuts = _skim_cuts_for_mode(category_mode, dataset_sample)
skim_output = None
if category_mode == "high_purity_study":
    if fake_track_control_mode not in ("zmumu", "zee"):
        raise ValueError(
            "high_purity_study requires DISAPPTRKS_FAKE_TRACK_CONTROL=zmumu or zee"
        )
    # This raw-Nano preselection is deliberately inclusive with respect to the
    # exact downstream Z control and fake-sideband selection.  Applying it at
    # the framework skim stage avoids constructing expensive derived track
    # quantities for the overwhelming majority of irrelevant data events.
    skim_cuts = [z_sideband_skim_cuts[fake_track_control_mode]]
if category_mode == "z_sideband_skim":
    if fake_track_control_mode not in ("zmumu", "zee"):
        raise ValueError(
            "z_sideband_skim requires DISAPPTRKS_FAKE_TRACK_CONTROL=zmumu or zee"
        )
    skim_output = os.environ.get("DISAPPTRKS_SKIM_OUTPUT")
    if not skim_output:
        raise ValueError(
            "z_sideband_skim requires DISAPPTRKS_SKIM_OUTPUT to name the ROOT output directory"
        )
    skim_cuts = [z_sideband_skim_cuts[fake_track_control_mode]]
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
tau_background_diagnostic_categories = {
    f"tau_background_diag_{name}": [cut]
    for name, cut in tau_background_diagnostic_cuts.items()
}
fake_track_categories = {name: [cut] for name, cut in fake_track_cuts.items()}
include_pveto_diagnostics = enable_pveto_diagnostics or category_mode == "all"
muon_table16_categories = (
    {f"muon_table16_{name}": [cut] for name, cut in muon_table16_cuts.items()}
    if include_pveto_diagnostics
    else {}
)
electron_pveto_diagnostic_categories = (
    {
        f"electron_pveto_diag_{name}": [cut]
        for name, cut in electron_pveto_diagnostic_cuts.items()
    }
    if include_pveto_diagnostics
    else {}
)
tau_pveto_diagnostic_categories = (
    {
        f"tau_pveto_diag_{name}": [cut]
        for name, cut in tau_pveto_diagnostic_cuts.items()
    }
    if include_pveto_diagnostics
    else {}
)

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
    "disappearing_track_selection_high_purity": [
        basic_event_selection,
        isolated_track_selection,
        candidate_track_selection,
        has_high_purity_disappearing_track,
    ],
}
for _name, _cut in signal_acceptance_layer_cuts.items():
    common_categories[_name] = [
        basic_event_selection,
        _cut,
    ]
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


# A category_mode with exactly one active cumulative-cutflow-diagnostic family
# (Table-16-style: one Cut per stage, each just checking one precomputed boolean
# field) sets this instead of merging that family flat into `selected_categories`.
# Per pocketcoffea-conventions, a cumulative-stage axis like this is a MultiCut
# axis, not N one-off StandardSelection categories -- see the `category_selection`
# assembly below, which wraps it in a CartesianSelection with its own PackedSelection
# budget instead of counting it against the 64-mask cap shared by everything else in
# the mode. `cuts_names` is chosen to exactly reproduce today's flat category names,
# so no downstream consumer (cli.py/tables.py) needs to change.
active_diagnostic_categories = {}

if category_mode == "muon_pveto":
    active_diagnostic_categories = muon_table16_categories
    selected_categories = {
        **common_categories,
        **muon_pveto_categories,
        **muon_pveto_layer_categories,
        **_categories_with_prefix(pveto_lepton_background_categories, "muon_"),
    }
elif category_mode == "electron_pveto":
    active_diagnostic_categories = electron_pveto_diagnostic_categories
    selected_categories = {
        **common_categories,
        **_categories_with_prefix(lepton_pveto_categories, "electron_"),
        **_categories_with_prefix(pveto_lepton_background_categories, "electron_"),
    }
elif category_mode == "tau_mu_pveto":
    active_diagnostic_categories = _categories_with_prefix(
        tau_pveto_diagnostic_categories, "tau_pveto_diag_tau_mu_"
    )
    selected_categories = {
        **common_categories,
        **_categories_with_prefix(lepton_pveto_categories, "tau_mu_"),
        **_categories_with_prefix(pveto_lepton_background_categories, "tau_mu_"),
    }
elif category_mode == "tau_ele_pveto":
    active_diagnostic_categories = _categories_with_prefix(
        tau_pveto_diagnostic_categories, "tau_pveto_diag_tau_ele_"
    )
    selected_categories = {
        **common_categories,
        **_categories_with_prefix(lepton_pveto_categories, "tau_ele_"),
        **_categories_with_prefix(pveto_lepton_background_categories, "tau_ele_"),
    }
elif category_mode == "muon_pmiss_poffline":
    selected_categories = {
        "inclusive": common_categories["inclusive"],
        **_categories_with_prefix(lepton_background_categories, "muon_"),
    }
elif category_mode == "electron_pmiss_poffline":
    selected_categories = {
        "inclusive": common_categories["inclusive"],
        **_categories_with_prefix(lepton_background_categories, "electron_"),
    }
elif category_mode == "tau_mu_pmiss_poffline":
    active_diagnostic_categories = tau_background_diagnostic_categories
    selected_categories = {
        "inclusive": common_categories["inclusive"],
        **_categories_with_prefix(lepton_background_categories, "tau_mu_"),
    }
elif category_mode == "tau_ele_pmiss_poffline":
    active_diagnostic_categories = tau_background_diagnostic_categories
    selected_categories = {
        "inclusive": common_categories["inclusive"],
        **_categories_with_prefix(lepton_background_categories, "tau_ele_"),
    }
elif category_mode == "tau_trigger_probability":
    selected_categories = {
        "inclusive": common_categories["inclusive"],
    }
elif category_mode == "tau_pmiss_poffline":
    active_diagnostic_categories = tau_background_diagnostic_categories
    selected_categories = {
        "inclusive": common_categories["inclusive"],
        **_categories_with_prefix(lepton_background_categories, "tau_control_"),
    }
elif category_mode == "fake_tracks":
    if fake_track_control_mode == "zmumu":
        active_diagnostic_categories = _categories_with_prefix(
            fake_z_control_diagnostic_categories, "fake_zmumu_"
        )
        selected_categories = {
            "inclusive": common_categories["inclusive"],
            **fake_track_zmumu_categories,
        }
    elif fake_track_control_mode == "zee":
        active_diagnostic_categories = _categories_with_prefix(
            fake_z_control_diagnostic_categories, "fake_zee_"
        )
        selected_categories = {
            "inclusive": common_categories["inclusive"],
            **fake_track_zee_categories,
        }
    else:
        selected_categories = {
            **common_categories,
            **fake_track_basic_categories,
        }
elif category_mode == "high_purity_study":
    if fake_track_control_mode not in ("zmumu", "zee"):
        raise ValueError(
            "high_purity_study requires DISAPPTRKS_FAKE_TRACK_CONTROL=zmumu or zee"
        )
    selected_categories = {"inclusive": common_categories["inclusive"]}
elif category_mode == "z_sideband_skim":
    selected_categories = {"inclusive": common_categories["inclusive"]}
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
elif category_mode == "signal_acceptance":
    selected_categories = {
        "inclusive": common_categories["inclusive"],
        **{
            name: category_cut_list
            for name, category_cut_list in common_categories.items()
            if name.startswith("signal_selection_")
        },
        **{
            name: [cut]
            for name, cut in signal_acceptance_common_cutflow_cuts.items()
        },
        **{
            name: [cut]
            for name, cut in signal_acceptance_layer_entry_cuts.items()
        },
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
        "tau_mu_pveto, tau_ele_pveto, muon_pmiss_poffline, "
        "electron_pmiss_poffline, tau_mu_pmiss_poffline, "
        "tau_ele_pmiss_poffline, tau_pmiss_poffline, tau_trigger_probability, "
        "fake_tracks, high_purity_study, z_sideband_skim, muon_backgrounds, "
        "egamma_backgrounds, fiducial_maps, signal_acceptance, all."
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
        "zmumu": (
            "fakeZMuMuFitTrack_",
            "fakeZMuMuSideband_",
            "nFakeZMuMuSideband",
        ),
        "zee": ("fakeZeeFitTrack_", "fakeZeeSideband_", "nFakeZeeSideband"),
    }[fake_track_control_mode]
    prefixes_by_mode = {
        "muon_pveto": ("nMuon",),
        "electron_pveto": ("nElectron",),
        "tau_mu_pveto": ("nTauMu",),
        "tau_ele_pveto": ("nTauEle",),
        "muon_pmiss_poffline": ("nMuonBackground",),
        "electron_pmiss_poffline": ("nElectronBackground",),
        "tau_mu_pmiss_poffline": ("nTauMuBackground",),
        "tau_ele_pmiss_poffline": ("nTauEleBackground",),
        "tau_pmiss_poffline": ("nTauBackground",),
        "tau_trigger_probability": ("nTauTriggerProbability",),
        "fake_tracks": fake_track_prefixes,
        "high_purity_study": ("highPurityStudy",),
        "z_sideband_skim": (),
        "muon_backgrounds": (
            "nMuon",
            "nTauMu",
            "fakeZMuMuFitTrack_",
            "fakeZMuMuSideband_",
            "nFakeZMuMuSideband",
        ),
        "egamma_backgrounds": (
            "nElectron",
            "nTauEle",
            "fakeZeeFitTrack_",
            "fakeZeeSideband_",
            "nFakeZeeSideband",
        ),
        "fiducial_maps": ("electronFiducial", "muonFiducial"),
        # The acceptance comparison itself is read directly from category
        # cutflows. Optional signal dE/dx summaries use a deliberately small
        # dedicated histogram family.
        "signal_acceptance": (
            ("signalDeDxTrack",) if enable_signal_dedx_histograms else ()
        ),
    }
    prefixes = prefixes_by_mode.get(mode)
    if prefixes is None:
        return variables
    selected = {
        name: variable
        for name, variable in variables.items()
        if any(name.startswith(prefix) for prefix in prefixes)
    }
    pveto_background_prefixes = {
        "muon_pveto": ("nMuonBackground",),
        "electron_pveto": ("nElectronBackground",),
        "tau_mu_pveto": ("nTauMuBackground",),
        "tau_ele_pveto": ("nTauEleBackground",),
    }.get(mode, ())
    if pveto_background_prefixes and not enable_lepton_background_categories:
        selected = {
            name: variable
            for name, variable in selected.items()
            if not any(name.startswith(prefix) for prefix in pveto_background_prefixes)
        }
    return selected


lepton_pair_count_variables = {}
for prefix, label in (
    ("Electron", "electron"),
    ("TauMu", r"tau-muon"),
    ("TauEle", r"tau-electron"),
):
    lepton_pair_count_variables.update(
        {
            f"n{prefix}TriggerEffProbesPT55": _event_count_hist(
                f"n{prefix}TriggerEffProbesPT55",
                f"N(OS {label} trigger-efficiency probes with pt > 55 GeV)",
            ),
            f"n{prefix}TriggerEffProbesSSPT55": _event_count_hist(
                f"n{prefix}TriggerEffProbesSSPT55",
                f"N(SS {label} trigger-efficiency probes with pt > 55 GeV)",
            ),
            f"n{prefix}TriggerEffProbesFiringTrigger": _event_count_hist(
                f"n{prefix}TriggerEffProbesFiringTrigger",
                f"N(OS {label} trigger-efficiency probes firing trigger)",
            ),
            f"n{prefix}TriggerEffSSProbesFiringTrigger": _event_count_hist(
                f"n{prefix}TriggerEffSSProbesFiringTrigger",
                f"N(SS {label} trigger-efficiency probes firing trigger)",
            ),
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
                f"n{prefix}TriggerEffProbesPT55_{layer}": _event_count_hist(
                    f"n{prefix}TriggerEffProbesPT55_{layer}",
                    f"N(OS {label} trigger-efficiency probes with pt > 55 GeV, {layer})",
                ),
                f"n{prefix}TriggerEffProbesSSPT55_{layer}": _event_count_hist(
                    f"n{prefix}TriggerEffProbesSSPT55_{layer}",
                    f"N(SS {label} trigger-efficiency probes with pt > 55 GeV, {layer})",
                ),
                f"n{prefix}TriggerEffProbesFiringTrigger_{layer}": _event_count_hist(
                    f"n{prefix}TriggerEffProbesFiringTrigger_{layer}",
                    f"N(OS {label} trigger-efficiency probes firing trigger, {layer})",
                ),
                f"n{prefix}TriggerEffSSProbesFiringTrigger_{layer}": _event_count_hist(
                    f"n{prefix}TriggerEffSSProbesFiringTrigger_{layer}",
                    f"N(SS {label} trigger-efficiency probes firing trigger, {layer})",
                ),
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

lepton_pair_count_variables.update(
    {
        "nMuonTriggerEffProbesPT55": _event_count_hist(
            "nMuonTriggerEffProbesPT55",
            "N(OS muon trigger-efficiency probes with pt > 55 GeV)",
        ),
        "nMuonTriggerEffProbesSSPT55": _event_count_hist(
            "nMuonTriggerEffProbesSSPT55",
            "N(SS muon trigger-efficiency probes with pt > 55 GeV)",
        ),
        "nMuonTriggerEffProbesFiringTrigger": _event_count_hist(
            "nMuonTriggerEffProbesFiringTrigger",
            "N(OS muon trigger-efficiency probes firing trigger)",
        ),
        "nMuonTriggerEffSSProbesFiringTrigger": _event_count_hist(
            "nMuonTriggerEffSSProbesFiringTrigger",
            "N(SS muon trigger-efficiency probes firing trigger)",
        ),
    }
)
for layer in pveto_layers:
    lepton_pair_count_variables.update(
        {
            f"nMuonTriggerEffProbesPT55_{layer}": _event_count_hist(
                f"nMuonTriggerEffProbesPT55_{layer}",
                f"N(OS muon trigger-efficiency probes with pt > 55 GeV, {layer})",
            ),
            f"nMuonTriggerEffProbesSSPT55_{layer}": _event_count_hist(
                f"nMuonTriggerEffProbesSSPT55_{layer}",
                f"N(SS muon trigger-efficiency probes with pt > 55 GeV, {layer})",
            ),
            f"nMuonTriggerEffProbesFiringTrigger_{layer}": _event_count_hist(
                f"nMuonTriggerEffProbesFiringTrigger_{layer}",
                f"N(OS muon trigger-efficiency probes firing trigger, {layer})",
            ),
            f"nMuonTriggerEffSSProbesFiringTrigger_{layer}": _event_count_hist(
                f"nMuonTriggerEffSSProbesFiringTrigger_{layer}",
                f"N(SS muon trigger-efficiency probes firing trigger, {layer})",
            ),
        }
    )

lepton_background_count_variables = {}
for prefix, label in (
    ("Muon", "muon"),
    ("Electron", "electron"),
    ("Tau", "tau"),
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
                f"n{prefix}BackgroundMetMinusOnePt_{layer}": HistConf(
                    [
                        Axis(
                            coll="events",
                            field=f"n{prefix}BackgroundMetMinusOnePt_{layer}",
                            bins=100,
                            start=0,
                            stop=1000,
                            label=f"{label} lepton-removed MET ({layer}) [GeV]",
                        )
                    ]
                ),
                f"n{prefix}BackgroundMetMinusOnePtTrig_{layer}": HistConf(
                    [
                        Axis(
                            coll="events",
                            field=f"n{prefix}BackgroundMetMinusOnePtTrig_{layer}",
                            bins=100,
                            start=0,
                            stop=1000,
                            label=(
                                f"{label} lepton-removed MET with MET trigger "
                                f"({layer}) [GeV]"
                            ),
                        )
                    ]
                ),
                f"n{prefix}BackgroundMetNoMuPt_{layer}": HistConf(
                    [
                        Axis(
                            coll="events",
                            field=f"n{prefix}BackgroundMetNoMuPt_{layer}",
                            bins=100,
                            start=0,
                            stop=1000,
                            label=f"{label} MET no mu ({layer}) [GeV]",
                        )
                    ]
                ),
                f"n{prefix}BackgroundMetNoMuPtTrig_{layer}": HistConf(
                    [
                        Axis(
                            coll="events",
                            field=f"n{prefix}BackgroundMetNoMuPtTrig_{layer}",
                            bins=100,
                            start=0,
                            stop=1000,
                            label=f"{label} MET no mu with MET trigger ({layer}) [GeV]",
                        )
                    ]
                ),
                f"n{prefix}BackgroundDeltaPhiMetJetLeadingVsMetMinusOnePt_{layer}": HistConf(
                    [
                        Axis(
                            coll="events",
                            field=f"n{prefix}BackgroundMetMinusOnePt_{layer}",
                            bins=100,
                            start=0,
                            stop=1000,
                            label=f"{label} lepton-removed MET ({layer}) [GeV]",
                        ),
                        Axis(
                            coll="events",
                            field=f"n{prefix}BackgroundDeltaPhiMetJetLeadingVsMetMinusOnePt_{layer}",
                            bins=32,
                            start=0,
                            stop=3.2,
                            label=(
                                f"{label} delta phi(leading jet, lepton-removed MET) "
                                f"({layer})"
                            ),
                        ),
                    ]
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

tau_trigger_probability_variables = {
    "nTauTriggerProbabilityDenominator": _event_count_hist(
        "nTauTriggerProbabilityDenominator",
        "N(events passing tau-trigger eta legs and the single-muon trigger)",
        bins=2,
    ),
    "nTauTriggerProbabilityNumerator": _event_count_hist(
        "nTauTriggerProbabilityNumerator",
        "N(events passing tau-trigger eta legs and the muon+tau cross-trigger)",
        bins=2,
    ),
    "nTauTriggerProbabilityMuonEtaLeg": _event_count_hist(
        "nTauTriggerProbabilityMuonEtaLeg",
        "N(events with at least one muon |eta| < 2.1)",
        bins=2,
    ),
    "nTauTriggerProbabilityTauEtaLeg": _event_count_hist(
        "nTauTriggerProbabilityTauEtaLeg",
        "N(events with at least one tau |eta| < 2.1)",
        bins=2,
    ),
}


# Candidate-level diagnostics for the tracks in events counted by the fake
# background N_sideband numerator.  The six hit-pattern components identify
# which pixel/strip subdetectors supplied the measured layers; dE/dx is shown
# separately in each exclusive signal-region layer bin.
fake_sideband_track_variables = {}
_fake_hit_pattern_fields = {
    "hp_pixelBarrelLayersWithMeasurement": "pixel barrel layers with hits",
    "hp_pixelEndcapLayersWithMeasurement": "pixel endcap layers with hits",
    "hp_stripTIBLayersWithMeasurement": "TIB layers with hits",
    "hp_stripTIDLayersWithMeasurement": "TID layers with hits",
    "hp_stripTOBLayersWithMeasurement": "TOB layers with hits",
    "hp_stripTECLayersWithMeasurement": "TEC layers with hits",
}
for control_key, control_label in (("ZMuMu", r"Z$\to\mu\mu$"), ("Zee", r"Z$\to ee$")):
    for layer in (*pveto_layers, "combinedBins"):
        collection = f"Fake{control_key}SidebandTrack_{layer}"
        for suffix, description in (
            ("Candidates", "all sideband candidates"),
            ("HighPurityCandidates", "high-purity sideband candidates"),
        ):
            fake_sideband_track_variables[
                f"nFake{control_key}Sideband{suffix}_{layer}"
            ] = _event_count_hist(
                f"nFake{control_key}Sideband{suffix}_{layer}",
                f"N({control_label} {description}, {layer})",
                bins=20,
            )
        for field, field_label in _fake_hit_pattern_fields.items():
            fake_sideband_track_variables[
                f"fake{control_key}Sideband_{layer}_{field}"
            ] = HistConf(
                [
                    Axis(
                        coll=collection,
                        field=field,
                        bins=11,
                        start=-0.5,
                        stop=10.5,
                        label=f"{control_label} sideband {field_label} ({layer})",
                    )
                ],
                only_categories=["inclusive"],
            )
        for field, detector in (("dEdxPixel", "pixel"), ("dEdxStrip", "strip")):
            fake_sideband_track_variables[
                f"fake{control_key}Sideband_{layer}_{field}"
            ] = HistConf(
                [
                    Axis(
                        coll=collection,
                        field=field,
                        bins=100,
                        start=0.0,
                        stop=20.0,
                        label=(
                            f"{control_label} sideband {detector} dE/dx "
                            f"({layer}) [MeV/mm]"
                        ),
                    )
                ],
                only_categories=["inclusive"],
            )
        if layer != "combinedBins":
            for hit_field, detector, dedx_field in (
                ("hp_pixelBarrelLayersWithMeasurement", "pixel barrel", "dEdxPixel"),
                ("hp_pixelEndcapLayersWithMeasurement", "pixel endcap", "dEdxPixel"),
                ("hp_stripTIBLayersWithMeasurement", "TIB", "dEdxStrip"),
                ("hp_stripTIDLayersWithMeasurement", "TID", "dEdxStrip"),
                ("hp_stripTOBLayersWithMeasurement", "TOB", "dEdxStrip"),
                ("hp_stripTECLayersWithMeasurement", "TEC", "dEdxStrip"),
            ):
                fake_sideband_track_variables[
                    f"fake{control_key}Sideband_{layer}_{dedx_field}_vs_{hit_field}"
                ] = HistConf(
                    [
                        Axis(
                            coll=collection,
                            field=hit_field,
                            bins=11,
                            start=-0.5,
                            stop=10.5,
                            label=f"{detector} layers with measurement",
                        ),
                        Axis(
                            coll=collection,
                            field=dedx_field,
                            bins=78,
                            start=0.5,
                            stop=20.0,
                            label=f"{detector} dE/dx",
                        ),
                    ],
                    only_categories=["inclusive"],
                )


# Dedicated, sparse sideband study of the variables entering the CMS track
# high-purity decision. The workflow builds one pre-highPurity track collection;
# native CartesianSelection categories apply the before/pass and layer masks.
high_purity_study_variables = {}

# Fake-track candidates have long momentum and fit-quality tails.  Variable
# binning preserves useful resolution in the core without silently collapsing
# most candidates into overflow.  Underflow and overflow are still retained
# and reported by the plotting command.
_high_purity_pt_edges = (
    [float(value) for value in range(50, 601, 5)]
    + [float(value) for value in range(620, 1201, 20)]
    + [float(value) for value in range(1300, 3001, 100)]
    + [float(value) for value in range(3500, 10001, 500)]
)
_high_purity_pt_err_edges = (
    [float(value) for value in range(0, 101)]
    + [float(value) for value in range(105, 501, 5)]
    + [float(value) for value in range(520, 2001, 20)]
    + [float(value) for value in range(2100, 10001, 100)]
)
_high_purity_features = {
    "pt": (_high_purity_pt_edges, None, None, r"track $p_T$ [GeV]"),
    "eta": (84, -2.1, 2.1, r"track $\eta$"),
    "phi": (64, -3.2, 3.2, r"track $\phi$"),
    "trackPtErr": (_high_purity_pt_err_edges, None, None, r"$\delta p_T$ [GeV]"),
    "trackEtaErr": (100, 0.0, 0.05, r"$\delta\eta$"),
    "trackPhiErr": (100, 0.0, 0.05, r"$\delta\phi$ [rad]"),
    "innerPx": (120, -600.0, 600.0, r"inner-state $p_x$ [GeV]"),
    "innerPy": (120, -600.0, 600.0, r"inner-state $p_y$ [GeV]"),
    "innerPz": (160, -1600.0, 1600.0, r"inner-state $p_z$ [GeV]"),
    "innerPt": (100, 50.0, 550.0, r"inner-state $p_T$ [GeV]"),
    "outerPx": (120, -600.0, 600.0, r"outer-state $p_x$ [GeV]"),
    "outerPy": (120, -600.0, 600.0, r"outer-state $p_y$ [GeV]"),
    "outerPz": (160, -1600.0, 1600.0, r"outer-state $p_z$ [GeV]"),
    "outerPt": (100, 0.0, 550.0, r"outer-state $p_T$ [GeV]"),
    "dxyBS": (200, -1.0, 1.0, r"$d_0$ (beamspot) [cm]"),
    "dzBS": (240, -30.0, 30.0, r"$d_z$ (beamspot) [cm]"),
    "dxyClosestPV": (100, -0.5, 0.5, r"$d_0$ (closest PV) [cm]"),
    "dzClosestPV": (100, -0.5, 0.5, r"$d_z$ (closest PV) [cm]"),
    "dxyBSErr": (100, 0.0, 0.1, r"$\delta d_0$ (beamspot) [cm]"),
    "dzBSErr": (100, 0.0, 0.1, r"$\delta d_z$ (beamspot) [cm]"),
    "dxyClosestPVErr": (100, 0.0, 0.1, r"$\delta d_0$ (closest PV) [cm]"),
    "dzClosestPVErr": (100, 0.0, 0.1, r"$\delta d_z$ (closest PV) [cm]"),
    "trackChi2": (500, 0.0, 500.0, r"track $\chi^2$"),
    "trackNdof": (81, -0.5, 80.5, r"track ndof"),
    "trackNormalizedChi2": (400, 0.0, 100.0, r"track $\chi^2$/ndof"),
    "hp_nValidPixelHits": (16, -0.5, 15.5, "valid pixel hits"),
    "hp_nValidStripHits": (31, -0.5, 30.5, "valid strip hits"),
    "hp_nLostHitsInner": (11, -0.5, 10.5, "missing hits before innermost hit"),
    "hp_nLostHitsOuter": (16, -0.5, 15.5, "missing hits after outermost hit"),
    "hp_trackerLayersTotallyOffOrBadInner": (11, -0.5, 10.5, "inactive layers before innermost hit"),
    "hp_trackerLayersTotallyOffOrBadOuter": (16, -0.5, 15.5, "inactive layers after outermost hit"),
    "missingMiddleHits": (11, -0.5, 10.5, "layers without hits on track body"),
    "trackAlgo": (31, -1.5, 29.5, "track algorithm / iteration flag"),
    "trackOriginalAlgo": (31, -1.5, 29.5, "original track algorithm / iteration flag"),
}
if category_mode == "high_purity_study":
    _study_control_key = "ZMuMu" if fake_track_control_mode == "zmumu" else "Zee"
    _study_categories = [
        f"high_purity_{selection}_{layer}"
        for selection in ("before", "pass")
        for layer in high_purity_study_layers
    ]
    for _field, (_bins, _start, _stop, _label) in _high_purity_features.items():
        high_purity_study_variables[
            f"highPurityStudy{_study_control_key}_{_field}"
        ] = HistConf(
            [
                Axis(
                    coll="HighPurityStudyTrack",
                    field=_field,
                    bins=_bins,
                    start=_start,
                    stop=_stop,
                    label=_label,
                )
            ],
            only_categories=_study_categories,
        )


# Optional hit-level companion to the track-level high-purity study.  Each
# collection contains only DeDxHitInfo rows associated with the requested
# sideband-track population, so the ordinary inclusive event category can fill
# it without mixing track-shaped and hit-shaped Cartesian masks.
high_purity_dedx_hit_variables = {}
_high_purity_dedx_hit_features = {
    "isoTrackIdx": (51, -0.5, 50.5, "source IsoTrack row index"),
    "hitIdx": (51, -0.5, 50.5, "index in DeDxHitInfo payload"),
    "detId": (500, 2.5e8, 5.0e8, "raw tracker detector ID"),
    "subdet": (6, 0.5, 6.5, "tracker subdetector code"),
    "layer": (10, 0.5, 10.5, "barrel layer or endcap disk/wheel"),
    "side": (3, -0.5, 2.5, "tracker side code"),
    "isPixel": (2, -0.5, 1.5, "is pixel hit"),
    "type": (11, -0.5, 10.5, "DeDxHitInfo hit type"),
    "passesStripShapeSelection": (2, -0.5, 1.5, "passes strip-shape selection"),
    "charge": (400, 0.0, 2.0e5, "cluster charge"),
    "pathLength": (300, 0.0, 0.30, "path length through active material"),
    "dEdx": (250, 0.0, 50.0, r"per-hit dE/dx [MeV/mm]"),
    "localX": (300, -15.0, 15.0, "hit local x"),
    "localY": (300, -15.0, 15.0, "hit local y"),
    "pixelSize": (31, -0.5, 30.5, "pixel cluster size"),
    "pixelSizeX": (21, -0.5, 20.5, "pixel cluster size in local x"),
    "pixelSizeY": (31, -0.5, 30.5, "pixel cluster size in local y"),
}
_high_purity_dedx_hit_2d_features = {
    "type": _high_purity_dedx_hit_features["type"],
    "stripPassesShapeSelection": (
        2, -0.5, 1.5, "strip hit passes shape selection"
    ),
    "charge": _high_purity_dedx_hit_features["charge"],
    "pathLength": _high_purity_dedx_hit_features["pathLength"],
    "dEdx": _high_purity_dedx_hit_features["dEdx"],
    "localX": _high_purity_dedx_hit_features["localX"],
    "localY": _high_purity_dedx_hit_features["localY"],
    "pixelSize": _high_purity_dedx_hit_features["pixelSize"],
    "pixelSizeX": _high_purity_dedx_hit_features["pixelSizeX"],
    "pixelSizeY": _high_purity_dedx_hit_features["pixelSizeY"],
}
_high_purity_dedx_track_features = {
    "nRetainedDeDxHits": (
        31, -0.5, 30.5, "retained dE/dx hits on track"
    ),
    "nRetainedDeDxHitsMinusLayers": (
        31, -10.5, 20.5, "retained dE/dx hits minus measured layers"
    ),
    "dEdxMedian": (100, 0.0, 50.0, r"median per-hit dE/dx [MeV/mm]"),
    "dEdxTruncatedMeanDropMaximum": (
        100, 0.0, 50.0,
        r"mean per-hit dE/dx after dropping maximum [MeV/mm]",
    ),
    "dEdxMaximum": (100, 0.0, 50.0, r"maximum per-hit dE/dx [MeV/mm]"),
    "dEdxStdDev": (100, 0.0, 25.0, r"per-track dE/dx standard deviation [MeV/mm]"),
    "dEdxRange": (100, 0.0, 50.0, r"per-track dE/dx range [MeV/mm]"),
    "dEdxMaximumOverMedian": (
        100, 0.0, 20.0, "maximum / median per-hit dE/dx"
    ),
    "nDeDxHitsAbove10": (16, -0.5, 15.5, r"dE/dx hits $\geq10$ MeV/mm"),
    "nDeDxHitsAbove20": (16, -0.5, 15.5, r"dE/dx hits $\geq20$ MeV/mm"),
    "nStripDeDxHits": (16, -0.5, 15.5, "retained strip dE/dx hits"),
    "nStripShapeFailures": (
        16, -0.5, 15.5, "strip hits failing shape selection"
    ),
    "stripShapeFailureFraction": (
        21, -0.025, 1.025, "fraction of strip hits failing shape selection"
    ),
}
signal_dedx_track_variables = {}
if category_mode == "signal_acceptance" and enable_signal_dedx_histograms:
    for _layer in signal_dedx_layers:
        _collection = f"SignalDeDxTrack_{_layer}"
        _prefix = f"signalDeDxTrack_{_layer}"
        for _field, (_bins, _start, _stop, _label) in (
            _high_purity_dedx_track_features.items()
        ):
            signal_dedx_track_variables[f"{_prefix}_{_field}"] = HistConf(
                [
                    Axis(
                        coll=_collection,
                        field=_field,
                        bins=_bins,
                        start=_start,
                        stop=_stop,
                        label=_label,
                    )
                ],
                only_categories=["inclusive"],
            )
if category_mode == "high_purity_study" and enable_high_purity_dedx_histograms:
    _study_control_key = "ZMuMu" if fake_track_control_mode == "zmumu" else "Zee"
    high_purity_dedx_hit_variables[
        f"highPurityStudy{_study_control_key}DeDxHit_nIsoTrackDeDxHit"
    ] = HistConf(
        [
            Axis(
                coll="events",
                field="nIsoTrackDeDxHit",
                bins=101,
                start=-0.5,
                stop=100.5,
                label="N(IsoTrackDeDxHit rows in sideband event)",
            )
        ],
        only_categories=["inclusive"],
    )
    for _layer in high_purity_study_layers:
        for _selection in ("pass",):
            _collection = f"HighPurityStudyDeDxHit_{_selection}_{_layer}"
            _track_collection = (
                f"HighPurityStudyDeDxTrack_{_selection}_{_layer}"
            )
            _prefix = (
                f"highPurityStudy{_study_control_key}DeDxHit_"
                f"{_selection}_{_layer}"
            )
            _track_prefix = (
                f"highPurityStudy{_study_control_key}DeDxTrack_"
                f"{_selection}_{_layer}"
            )
            for _field, (_bins, _start, _stop, _label) in (
                _high_purity_dedx_track_features.items()
            ):
                high_purity_dedx_hit_variables[
                    f"{_track_prefix}_{_field}"
                ] = HistConf(
                    [
                        Axis(
                            coll=_track_collection,
                            field=_field,
                            bins=_bins,
                            start=_start,
                            stop=_stop,
                            label=_label,
                        )
                    ],
                    only_categories=["inclusive"],
                )
            high_purity_dedx_hit_variables[f"{_prefix}_nHits"] = HistConf(
                [
                    Axis(
                        coll="events",
                        field=f"n{_collection}",
                        bins=101,
                        start=-0.5,
                        stop=100.5,
                        label="N(associated retained dE/dx hits in event)",
                    )
                ],
                only_categories=["inclusive"],
            )
            for _field, (_bins, _start, _stop, _label) in (
                _high_purity_dedx_hit_features.items()
            ):
                high_purity_dedx_hit_variables[f"{_prefix}_{_field}"] = HistConf(
                    [
                        Axis(
                            coll=_collection,
                            field=_field,
                            bins=_bins,
                            start=_start,
                            stop=_stop,
                            label=_label,
                        )
                    ],
                    only_categories=["inclusive"],
                )
            high_purity_dedx_hit_variables[
                f"{_prefix}_subdet_vs_layer"
            ] = HistConf(
                [
                    Axis(
                        coll=_collection,
                        field="subdet",
                        bins=6,
                        start=0.5,
                        stop=6.5,
                        label="tracker subdetector code",
                    ),
                    Axis(
                        coll=_collection,
                        field="layer",
                        bins=10,
                        start=0.5,
                        stop=10.5,
                        label="layer/disk/wheel number",
                    ),
                ],
                only_categories=["inclusive"],
            )
            for _field, (_bins, _start, _stop, _label) in (
                _high_purity_dedx_hit_2d_features.items()
            ):
                high_purity_dedx_hit_variables[
                    f"{_prefix}_{_field}_vs_detectorLayer"
                ] = HistConf(
                    [
                        Axis(
                            coll=_collection,
                            field="detectorLayer",
                            bins=60,
                            start=9.5,
                            stop=69.5,
                            label="encoded detector layer (10*subdet + layer)",
                        ),
                        Axis(
                            coll=_collection,
                            field=_field,
                            bins=_bins,
                            start=_start,
                            stop=_stop,
                            label=_label,
                        ),
                    ],
                    only_categories=["inclusive"],
                )


sideband_event_columns = {
    "common": {"inclusive": [], "bycategory": {}},
    "bysample": {},
}
_sideband_manifest_track_fields = [
    "isoTrackIdx",
    "pt",
    "eta",
    "phi",
    "charge",
    "dxy",
    "dz",
    "isHighPurityTrack",
    "hp_nValidHits",
    "hp_nValidPixelHits",
    "hp_trackerLayersWithMeasurement",
    "missingInnerHits",
    "missingMiddleHits",
    "missingOuterHits",
    "pfRelIso03_chg",
    "caloEnergy",
    "dEdxPixel",
    "dEdxStrip",
]
if (
    enable_fake_sideband_histograms
    and category_mode == "fake_tracks"
    and fake_track_control_mode in ("zmumu", "zee")
):
    _manifest_control_key = "ZMuMu" if fake_track_control_mode == "zmumu" else "Zee"
    for _manifest_layer in pveto_layers:
        _manifest_category = (
            f"fake_{fake_track_control_mode}_sideband_{_manifest_layer}"
        )
        sideband_event_columns["common"]["bycategory"][_manifest_category] = [
            ColOut(
                "events",
                ["run", "luminosityBlock", "event"],
                store_size=False,
            ),
            ColOut(
                f"Fake{_manifest_control_key}SidebandTrack_{_manifest_layer}",
                _sideband_manifest_track_fields,
                flatten=True,
                store_size=True,
            ),
        ]

if category_mode == "signal_acceptance":
    category_selection = CartesianSelection(
        multicuts=[
            MultiCut(
                name="high_purity_variant",
                cuts=signal_acceptance_variant_axis_cuts,
                cuts_names=[
                    "signal_cutflow_without_high_purity",
                    "signal_cutflow_with_high_purity",
                ],
            ),
            MultiCut(
                name="layer_bin",
                cuts=signal_acceptance_layer_axis_cuts,
                cuts_names=[
                    "NLayers4",
                    "NLayers5",
                    "NLayers6plus",
                    "combinedBins",
                ],
            ),
            MultiCut(
                name="cutflow_stage",
                cuts=signal_acceptance_stage_cuts,
                cuts_names=list(SIGNAL_ACCEPTANCE_CARTESIAN_FIELDS),
            ),
        ],
        common_cats=StandardSelection(selected_categories),
    )
elif category_mode == "high_purity_study":
    category_selection = CartesianSelection(
        multicuts=[
            MultiCut(
                name="high_purity_selection",
                cuts=high_purity_study_selection_axis_cuts,
                cuts_names=["high_purity_before", "high_purity_pass"],
            ),
            MultiCut(
                name="layer_bin",
                cuts=[
                    high_purity_study_layer_axis_cuts[layer]
                    for layer in high_purity_study_layers
                ],
                cuts_names=list(high_purity_study_layers),
            ),
        ],
        common_cats=StandardSelection(selected_categories),
    )
elif active_diagnostic_categories:
    category_selection = CartesianSelection(
        multicuts=[
            MultiCut(
                name="diagnostic_cutflow",
                cuts=[
                    category_cut_list[0]
                    for category_cut_list in active_diagnostic_categories.values()
                ],
                cuts_names=list(active_diagnostic_categories.keys()),
            ),
        ],
        common_cats=StandardSelection(selected_categories),
    )
else:
    category_selection = StandardSelection(selected_categories)


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
    categories=category_selection,
    weights={"common": {"inclusive": []}, "bysample": {}},
    weights_classes=[],
    variations={"weights": {"common": {"inclusive": []}}},
    variables=_variables_for_mode(category_mode, {
        **high_purity_study_variables,
        **high_purity_dedx_hit_variables,
        **signal_dedx_track_variables,
        **(fake_sideband_track_variables if enable_fake_sideband_histograms else {}),
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
        "nMuonPVetoTagProbePairZWindowPassNoFiducial": HistConf(
            [
                Axis(
                    coll="events",
                    field="nMuonPVetoTagProbePairZWindowPassNoFiducial",
                    bins=10,
                    start=0,
                    stop=10,
                    label="N(OS Z-window muon Pveto numerator before fiducial maps)",
                )
            ]
        ),
        "nMuonPVetoTagProbePairZWindowFiducialRejected": HistConf(
            [
                Axis(
                    coll="events",
                    field="nMuonPVetoTagProbePairZWindowFiducialRejected",
                    bins=10,
                    start=0,
                    stop=10,
                    label="N(OS Z-window muon Pveto numerator rejected by fiducial maps)",
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
        "nMuonPVetoTagProbePairSSZWindowPassNoFiducial": HistConf(
            [
                Axis(
                    coll="events",
                    field="nMuonPVetoTagProbePairSSZWindowPassNoFiducial",
                    bins=10,
                    start=0,
                    stop=10,
                    label="N(SS Z-window muon Pveto numerator before fiducial maps)",
                )
            ]
        ),
        "nMuonPVetoTagProbePairSSZWindowFiducialRejected": HistConf(
            [
                Axis(
                    coll="events",
                    field="nMuonPVetoTagProbePairSSZWindowFiducialRejected",
                    bins=10,
                    start=0,
                    stop=10,
                    label="N(SS Z-window muon Pveto numerator rejected by fiducial maps)",
                )
            ]
        ),
        "nElectronFiducialHotSpotsLoaded": HistConf(
            [
                Axis(
                    coll="events",
                    field="nElectronFiducialHotSpotsLoaded",
                    bins=100,
                    start=0,
                    stop=100,
                    label="N(electron fiducial-map hot spots loaded)",
                )
            ]
        ),
        "nMuonFiducialHotSpotsLoaded": HistConf(
            [
                Axis(
                    coll="events",
                    field="nMuonFiducialHotSpotsLoaded",
                    bins=100,
                    start=0,
                    stop=100,
                    label="N(muon fiducial-map hot spots loaded)",
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
        **tau_trigger_probability_variables,
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
    columns=sideband_event_columns,
    workflow_options={"skim_mode": "skim"} if skim_output else None,
    save_skimmed_files=skim_output,
)
