import pytest

ak = pytest.importorskip("awkward")

from cuts import _z_sideband_skim


def test_zmumu_sideband_skim_keeps_all_layer_bins_without_high_purity_cut():
    muons = [
        {"pt": 46.0, "eta": 0.2, "phi": 0.0, "charge": 1,
         "tightId": True, "pfRelIso04_all": 0.05},
        {"pt": 46.0, "eta": -0.2, "phi": 3.14159, "charge": -1,
         "tightId": True, "pfRelIso04_all": 0.05},
    ]
    broad_track = {
        "pt": 80.0, "eta": 0.5, "dxy": 0.1,
        "hp_trackerLayersWithMeasurement": 4,
        "hp_nValidTrackerHits": 4,
        "isHighPurityTrack": False,
    }
    events = ak.Array([
        {"event": 1, "HLT": {"IsoMu24": True}, "Muon": muons,
         "IsoTrack": [broad_track]},
        {"event": 2, "HLT": {"IsoMu24": True}, "Muon": muons,
         "IsoTrack": [{**broad_track, "dxy": 0.01}]},
        {"event": 3, "HLT": {"IsoMu24": True}, "Muon": muons,
         "IsoTrack": [{**broad_track, "hp_trackerLayersWithMeasurement": 5}]},
        {"event": 4, "HLT": {"IsoMu24": True}, "Muon": muons,
         "IsoTrack": [{**broad_track, "hp_trackerLayersWithMeasurement": 6}]},
    ])

    mask = _z_sideband_skim(events, {"control": "zmumu"})

    assert ak.to_list(mask) == [True, False, True, True]
