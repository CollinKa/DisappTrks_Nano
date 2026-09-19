"""Run-dependent trigger definitions used by the tau background estimate."""

from __future__ import annotations


ISO_MUON_REFERENCE_TRIGGER = "IsoMu24"

_TAU_CROSS_TRIGGERS = {
    "2022": "IsoMu24_eta2p1_LooseDeepTauPFTauHPS30_eta2p1_CrossL1",
    "2023": "IsoMu24_eta2p1_LooseDeepTauPFTauHPS30_eta2p1_CrossL1",
    "2024": "IsoMu24_eta2p1_LooseDeepTauPFTauHPS30_eta2p1_CrossL1",
    "2025": "IsoMu24_eta2p1_PNetTauhPFJet26_L2NN_eta2p3_CrossL1",
    "2026": "IsoMu24_eta2p1_PNetTauhPFJet26_L2NN_eta2p3_CrossL1",
}


def tau_cross_trigger_for_year(year: str) -> str:
    """Return the nominal unprescaled muon-tau cross-trigger for a Run-3 year."""

    calendar_year = str(year).split("_", 1)[0]
    try:
        return _TAU_CROSS_TRIGGERS[calendar_year]
    except KeyError as error:
        raise ValueError(
            f"no nominal tau cross-trigger is configured for year {year!r}"
        ) from error
