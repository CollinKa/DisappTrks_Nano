import sys
from pathlib import Path

import awkward as ak

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pocket_coffea"))

from workflow import (
    _dedx_hits_for_high_purity_tracks,
    _dedx_hits_grouped_by_track,
    _dedx_track_summaries,
)


def test_dedx_hits_are_linked_to_selected_isotrack_rows():
    events = ak.Array(
        [
            {
                "IsoTrackDeDxHit": [
                    {
                        "isoTrackIdx": 0,
                        "subdet": 1,
                        "layer": 2,
                        "isPixel": 1,
                        "passesStripShapeSelection": 1,
                        "pixelSize": 4,
                        "pixelSizeX": 2,
                        "pixelSizeY": 3,
                    },
                    {
                        "isoTrackIdx": 1,
                        "subdet": 5,
                        "layer": 3,
                        "isPixel": 0,
                        "passesStripShapeSelection": 0,
                        "pixelSize": -1,
                        "pixelSizeX": -1,
                        "pixelSizeY": -1,
                    },
                ]
            }
        ]
    )
    selected_tracks = ak.Array([[{"sourceIsoTrackIdx": 1}]])

    selected = _dedx_hits_for_high_purity_tracks(events, selected_tracks)

    assert ak.to_list(selected.isoTrackIdx) == [[1]]
    assert ak.to_list(selected.detectorLayer) == [[53]]
    assert ak.to_list(selected.pixelSize) == [[None]]
    assert ak.to_list(selected.stripPassesShapeSelection) == [[0]]


def test_dedx_track_summaries_are_computed_per_selected_track():
    events = ak.Array(
        [
            {
                "IsoTrackDeDxHit": [
                    {
                        "isoTrackIdx": 0,
                        "dEdx": 3.0,
                        "isPixel": 1,
                        "passesStripShapeSelection": 1,
                    },
                    {
                        "isoTrackIdx": 1,
                        "dEdx": 2.0,
                        "isPixel": 0,
                        "passesStripShapeSelection": 1,
                    },
                    {
                        "isoTrackIdx": 1,
                        "dEdx": 4.0,
                        "isPixel": 0,
                        "passesStripShapeSelection": 0,
                    },
                    {
                        "isoTrackIdx": 1,
                        "dEdx": 21.0,
                        "isPixel": 1,
                        "passesStripShapeSelection": 1,
                    },
                ]
            }
        ]
    )
    selected_tracks = ak.Array(
        [[{
            "sourceIsoTrackIdx": 1,
            "hp_nValidTrackerHits": 5,
            "hp_trackerLayersWithMeasurement": 5,
        }]]
    )

    grouped = _dedx_hits_grouped_by_track(events, selected_tracks)
    summaries = _dedx_track_summaries(selected_tracks, grouped)

    assert ak.to_list(summaries.nRetainedDeDxHits) == [[3]]
    assert ak.to_list(summaries.nRetainedDeDxHitsMinusLayers) == [[-2]]
    assert ak.to_list(summaries.dEdxMedian) == [[4.0]]
    assert ak.to_list(summaries.dEdxTruncatedMeanDropMaximum) == [[3.0]]
    assert ak.to_list(summaries.dEdxMaximum) == [[21.0]]
    assert ak.to_list(summaries.dEdxRange) == [[19.0]]
    assert ak.to_list(summaries.dEdxMaximumOverMedian) == [[5.25]]
    assert ak.to_list(summaries.nDeDxHitsAbove10) == [[1]]
    assert ak.to_list(summaries.nDeDxHitsAbove20) == [[1]]
    assert ak.to_list(summaries.nStripDeDxHits) == [[2]]
    assert ak.to_list(summaries.nStripShapeFailures) == [[1]]
    assert ak.to_list(summaries.stripShapeFailureFraction) == [[0.5]]
