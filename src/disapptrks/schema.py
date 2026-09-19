"""Branch contract for the custom NanoAOD used by the analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


EVENT_ID_BRANCHES = frozenset({"run", "luminosityBlock", "event"})

ISOTRACK_BRANCHES = frozenset(
    {
        "nIsoTrack",
        "IsoTrack_pt",
        "IsoTrack_eta",
        "IsoTrack_phi",
        "IsoTrack_theta",
        "IsoTrack_charge",
        "IsoTrack_dxy",
        "IsoTrack_dz",
        "IsoTrack_hp_nValidHits",
        "IsoTrack_hp_nValidPixelHits",
        "IsoTrack_hp_trackerLayersWithMeasurement",
        "IsoTrack_missingInnerHits",
        "IsoTrack_missingMiddleHits",
        "IsoTrack_missingOuterHits",
        "IsoTrack_hp_nLostHitsOuter",
        "IsoTrack_hp_nLostTrackerHitsOuter",
        "IsoTrack_hp_nLostPixelHitsOuter",
        "IsoTrack_hp_nLostStripHitsOuter",
        "IsoTrack_hp_pixelLayersWithoutMeasurementOuter",
        "IsoTrack_hp_stripLayersWithoutMeasurementOuter",
        "IsoTrack_pfRelIso03_chg",
        "IsoTrack_caloEm",
        "IsoTrack_caloHad",
        "IsoTrack_isFiducialECALTrack",
    }
)

JET_BASE_BRANCHES = frozenset(
    {
        "nJet",
        "Jet_pt",
        "Jet_eta",
        "Jet_phi",
    }
)

JET_ID_BRANCHES = frozenset({"Jet_jetId"})

JET_ID_RECONSTRUCTION_BRANCHES = frozenset(
    {
        "Jet_neHEF",
        "Jet_neEmEF",
        "Jet_chHEF",
        "Jet_chEmEF",
        "Jet_muEF",
        "Jet_chMultiplicity",
        "Jet_neMultiplicity",
    }
)

JET_BRANCHES = JET_BASE_BRANCHES | JET_ID_BRANCHES

TAG_AND_PROBE_BRANCHES = frozenset(
    {
        "Electron_pt",
        "Electron_eta",
        "Electron_phi",
        "Electron_charge",
        "Electron_cutBased",
        "Muon_pt",
        "Muon_eta",
        "Muon_phi",
        "Muon_charge",
        "Muon_looseId",
        "Muon_tightId",
        "Muon_pfRelIso04_all",
        "HLT_IsoMu24",
        "Tau_pt",
        "Tau_eta",
        "Tau_phi",
        "Tau_charge",
        "Tau_decayMode",
        "Tau_idDecayModeNewDMs",
        "Tau_rawDeepTau2018v2p5VSe",
        "Tau_rawDeepTau2018v2p5VSjet",
        "Tau_rawDeepTau2018v2p5VSmu",
        "TrigObj_pt",
        "TrigObj_eta",
        "TrigObj_phi",
        "TrigObj_id",
        "TrigObj_filterBits",
    }
)

STORED_NO_MU_MET_BRANCHES = frozenset({"MetNoMu_pt", "MetNoMu_phi"})

# A downstream reconstruction can instead start from a MET collection and the
# muons/PF candidates used by that definition.  The current smoke file already
# stores MetNoMu, so the audit reports that route directly.
DERIVABLE_NO_MU_MET_INPUTS = (
    frozenset({"MET_pt", "MET_phi", "Muon_pt", "Muon_phi"}),
    frozenset({"PuppiMET_pt", "PuppiMET_phi", "Muon_pt", "Muon_phi"}),
)


def _has_no_mu_met(branches: frozenset[str]) -> bool:
    return STORED_NO_MU_MET_BRANCHES <= branches or any(
        inputs <= branches for inputs in DERIVABLE_NO_MU_MET_INPUTS
    )


def _missing_jet_branches(branches: frozenset[str]) -> frozenset[str]:
    missing = set(JET_BASE_BRANCHES - branches)
    has_stored_jet_id = JET_ID_BRANCHES <= branches
    has_reconstructable_jet_id = JET_ID_RECONSTRUCTION_BRANCHES <= branches
    if not (has_stored_jet_id or has_reconstructable_jet_id):
        missing.update(JET_ID_BRANCHES - branches)
        missing.update(JET_ID_RECONSTRUCTION_BRANCHES - branches)
    return frozenset(missing)


def required_branches(scope: str) -> frozenset[str]:
    base = EVENT_ID_BRANCHES | ISOTRACK_BRANCHES | JET_BASE_BRANCHES
    if scope == "search":
        return base
    if scope in ("backgrounds", "fiducial-maps"):
        return base | TAG_AND_PROBE_BRANCHES
    raise ValueError(f"unknown schema scope: {scope}")


@dataclass(frozen=True)
class SchemaReport:
    path: str
    branches: frozenset[str]
    missing_by_scope: dict[str, tuple[str, ...]]
    has_no_mu_met_inputs: bool

    def ready_for(self, scope: str) -> bool:
        return not self.missing_by_scope[scope] and self.has_no_mu_met_inputs

    def as_dict(self) -> dict:
        return {
            "path": self.path,
            "ready": {
                scope: self.ready_for(scope)
                for scope in self.missing_by_scope
            },
            "missing": self.missing_by_scope,
            "has_no_mu_met_inputs": self.has_no_mu_met_inputs,
            "stored_no_mu_met": STORED_NO_MU_MET_BRANCHES <= self.branches,
            "n_branches": len(self.branches),
        }


def audit_branches(branches: Iterable[str], path: str = "<memory>") -> SchemaReport:
    branch_set = frozenset(branches)
    missing = {}
    for scope in ("search", "backgrounds", "fiducial-maps"):
        missing_for_scope = set(required_branches(scope) - branch_set)
        missing_for_scope.update(_missing_jet_branches(branch_set))
        missing[scope] = tuple(sorted(missing_for_scope))
    return SchemaReport(
        path=path,
        branches=branch_set,
        missing_by_scope=missing,
        has_no_mu_met_inputs=_has_no_mu_met(branch_set),
    )


def audit_root_file(path: str | Path) -> SchemaReport:
    try:
        import uproot
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Schema auditing needs uproot; install with `pip install -e '.[analysis]'`."
        ) from exc

    path = Path(path)
    with uproot.open(path) as root_file:
        branches = root_file["Events"].keys()
    return audit_branches(branches, str(path))
