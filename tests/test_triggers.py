import pytest

from disapptrks.triggers import ISO_MUON_REFERENCE_TRIGGER, tau_cross_trigger_for_year


def test_tau_cross_trigger_uses_deeptau_for_2022_through_2024():
    expected = "IsoMu24_eta2p1_LooseDeepTauPFTauHPS30_eta2p1_CrossL1"
    assert tau_cross_trigger_for_year("2022_preEE") == expected
    assert tau_cross_trigger_for_year("2023_postBPix") == expected
    assert tau_cross_trigger_for_year("2024") == expected


def test_tau_cross_trigger_uses_pnet_for_2025_and_2026():
    expected = "IsoMu24_eta2p1_PNetTauhPFJet26_L2NN_eta2p3_CrossL1"
    assert tau_cross_trigger_for_year("2025") == expected
    assert tau_cross_trigger_for_year("2026") == expected


def test_tau_trigger_reference_is_isomu24():
    assert ISO_MUON_REFERENCE_TRIGGER == "IsoMu24"


def test_tau_cross_trigger_rejects_unconfigured_year():
    with pytest.raises(ValueError):
        tau_cross_trigger_for_year("2021")
