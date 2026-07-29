import pytest

ak = pytest.importorskip("awkward")

from disapptrks.selections import (
    build_lepton_veto_tag_probe_pairs,
    build_muon_veto_tag_probe_pairs,
    fiducial_map_probe_track_mask,
    muon_veto_probe_track_mask,
    select_random_fiducial_tag_probe_pair,
)


def test_lepton_pairs_keep_probe_coordinates_for_fiducial_maps():
    tags = ak.Array(
        [
            [
                {
                    "pt": 50.0,
                    "eta": 0.0,
                    "phi": 0.0,
                    "charge": 1,
                }
            ]
        ]
    )
    probes = ak.Array(
        [
            [
                {
                    "pt": 45.0,
                    "eta": 1.25,
                    "phi": -2.5,
                    "charge": -1,
                    "dRMinElectron": 0.2,
                    "dRMinVetoElectron": 0.1,
                    "dRMinMuon": 0.3,
                    "dRMinTauHad": 0.4,
                    "dRMinJet": 0.6,
                    "caloEnergy": 5.0,
                    "missingOuterHits": 3,
                    "hp_trackerLayersWithMeasurement": 4,
                }
            ]
        ]
    )

    pairs = build_lepton_veto_tag_probe_pairs(
        tags,
        probes,
        tag_mass=0.000511,
        probe_mass=0.000511,
    )

    assert ak.to_list(pairs.probe_pt) == [[45.0]]
    assert ak.to_list(pairs.probe_eta) == [[1.25]]
    assert ak.to_list(pairs.probe_phi) == [[-2.5]]
    assert ak.to_list(pairs.probe_passElectronVeto) == [[True]]
    assert ak.to_list(pairs.probe_passVetoElectronVeto) == [[False]]


def test_muon_pairs_keep_loose_muon_veto_separate_from_generic_veto():
    tags = ak.Array(
        [
            [
                {
                    "pt": 50.0,
                    "eta": 0.0,
                    "phi": 0.0,
                    "charge": 1,
                }
            ]
        ]
    )
    probes = ak.Array(
        [
            [
                {
                    "pt": 45.0,
                    "eta": 1.25,
                    "phi": -2.5,
                    "charge": -1,
                    "dRMinMuon": 0.1,
                    "dRMinLooseMuon": 0.3,
                    "caloEnergy": 5.0,
                    "missingOuterHits": 3,
                    "hp_trackerLayersWithMeasurement": 4,
                }
            ]
        ]
    )

    pairs = build_muon_veto_tag_probe_pairs(tags, probes)

    assert ak.to_list(pairs.probe_passMuonVeto) == [[False]]
    assert ak.to_list(pairs.probe_passLooseMuonVeto) == [[True]]
    assert ak.to_list(pairs.tag_index) == [[0]]
    assert ak.to_list(pairs.probe_index) == [[0]]


def test_fiducial_random_arbitration_is_one_pair_and_event_stable():
    pairs = ak.Array(
        [
            [
                {"tag_index": 0, "probe_index": 0, "probe_eta": 0.1},
                {"tag_index": 0, "probe_index": 1, "probe_eta": 0.2},
                {"tag_index": 1, "probe_index": 2, "probe_eta": 0.3},
            ],
            [],
            [
                {"tag_index": 0, "probe_index": 3, "probe_eta": 1.1},
                {"tag_index": 1, "probe_index": 4, "probe_eta": 1.2},
            ],
        ]
    )
    run = ak.Array([355100, 355100, 355101])
    lumi = ak.Array([10, 10, 20])
    event = ak.Array([1001, 1002, 2001])

    selected = select_random_fiducial_tag_probe_pair(
        pairs,
        run,
        lumi,
        event,
        stage="before",
        seed=20220723,
    )
    repeated = select_random_fiducial_tag_probe_pair(
        pairs,
        run,
        lumi,
        event,
        stage="before",
        seed=20220723,
    )

    assert ak.to_list(ak.num(selected)) == [1, 0, 1]
    assert ak.to_list(selected) == ak.to_list(repeated)

    order = [2, 0, 1]
    reordered = select_random_fiducial_tag_probe_pair(
        pairs[order],
        run[order],
        lumi[order],
        event[order],
        stage="before",
        seed=20220723,
    )
    selected_by_event = {
        event_id: record
        for event_id, record in zip(
            ak.to_list(event),
            ak.to_list(selected),
        )
    }
    reordered_by_event = {
        event_id: record
        for event_id, record in zip(
            ak.to_list(event[order]),
            ak.to_list(reordered),
        )
    }
    assert selected_by_event == reordered_by_event


def test_fiducial_map_probe_uses_legacy_old_hit_cuts():
    tracks = ak.Array(
        [
            [
                {
                    "pt": 45.0,
                    "eta": 0.6,
                    "phi": 1.0,
                    "inECALCrack": False,
                    "inDTWheelGap": False,
                    "inCSCTransition": False,
                    "inTOBCrack": False,
                    "isFiducialECALTrack": True,
                    "hp_nValidPixelHits": 3,
                    "hp_nValidHits": 7,
                    "missingInnerHits": 0,
                    "missingMiddleHits": 0,
                    "pfRelIso03_chg": 0.01,
                    "dxy": 0.01,
                    "dz": 0.1,
                    "hp_trackerLayersWithMeasurement": 4,
                    "dRMinJet": 0.6,
                    "dRMinElectron": 0.2,
                    "dRMinMuon": 0.2,
                    "dRMinTauHad": 0.2,
                    "caloEnergy": 5.0,
                }
            ]
        ]
    )

    assert ak.to_list(fiducial_map_probe_track_mask(tracks, flavor="muon")) == [
        [True]
    ]
    assert ak.to_list(muon_veto_probe_track_mask(tracks)) == [[False]]
