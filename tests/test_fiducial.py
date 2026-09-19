import numpy as np
import pytest

from disapptrks.fiducial import (
    make_fiducial_map_from_outputs,
    plot_fiducial_map_payload,
    summarize_fiducial_map,
    write_fiducial_map_payload,
)


class FakeAxis:
    def __init__(self, name, edges):
        self.name = name
        self.edges = np.asarray(edges, dtype=float)


class FakeHist2D:
    def __init__(self, counts, eta_edges, phi_edges):
        self._counts = np.asarray(counts, dtype=float)
        self.axes = [
            FakeAxis("probe_eta", eta_edges),
            FakeAxis("probe_phi", phi_edges),
        ]

    def values(self, flow=False):
        return self._counts


def test_fiducial_hot_spot_is_identified():
    before = np.full((2, 2), 100.0)
    after = np.array([[1.0, 1.0], [1.0, 50.0]])
    edges = np.array([-1.0, 0.0, 1.0])

    summary = summarize_fiducial_map(
        before, after, edges, edges, threshold=1.0
    )

    assert len(summary.hot_spots) == 1
    assert summary.hot_spots[0].eta == 0.5
    assert summary.hot_spots[0].phi == 0.5


def test_fiducial_stddev_can_exclude_extreme_outlier():
    before = np.full((10, 10), 100.0)
    after = np.ones((10, 10))
    after[0, 1] = 20.0
    after[9, 9] = 500.0
    edges = np.arange(11.0)

    legacy = summarize_fiducial_map(before, after, edges, edges, threshold=2.0)
    robust = summarize_fiducial_map(
        before,
        after,
        edges,
        edges,
        threshold=2.0,
        stddev_exclude_top=1,
    )

    legacy_spots = {(hot_spot.eta, hot_spot.phi) for hot_spot in legacy.hot_spots}
    robust_spots = {(hot_spot.eta, hot_spot.phi) for hot_spot in robust.hot_spots}

    assert (0.5, 1.5) not in legacy_spots
    assert (0.5, 1.5) in robust_spots
    assert (9.5, 9.5) in robust_spots
    assert len(robust.stddev_excluded_bins) == 1
    assert robust.stddev_excluded_bins[0].eta == 9.5
    assert robust.stddev_excluded_bins[0].phi == 9.5
    assert robust.stddev_inefficiency < legacy.stddev_inefficiency


def test_make_fiducial_map_from_outputs_sums_2d_histograms():
    eta_edges = np.array([-1.0, 0.0, 1.0])
    phi_edges = np.array([-2.0, 0.0, 2.0])
    output = {
        "variables": {
            "electronFiducialBefore_eta_phi": FakeHist2D(
                [[100.0, 100.0], [100.0, 100.0]],
                eta_edges,
                phi_edges,
            ),
            "electronFiducialAfter_eta_phi": FakeHist2D(
                [[1.0, 1.0], [1.0, 50.0]],
                eta_edges,
                phi_edges,
            ),
        }
    }

    summary, before, after, returned_eta_edges, returned_phi_edges = (
        make_fiducial_map_from_outputs(
            [output],
            before_variable="electronFiducialBefore_eta_phi",
            after_variable="electronFiducialAfter_eta_phi",
            threshold=1.0,
        )
    )

    assert before.sum() == 400.0
    assert after.sum() == 53.0
    assert np.allclose(returned_eta_edges, eta_edges)
    assert np.allclose(returned_phi_edges, phi_edges)
    assert len(summary.hot_spots) == 1


def test_write_fiducial_map_payload_writes_json_and_npz(tmp_path):
    before = np.full((2, 2), 100.0)
    after = np.array([[1.0, 1.0], [1.0, 50.0]])
    edges = np.array([-1.0, 0.0, 1.0])
    summary = summarize_fiducial_map(before, after, edges, edges, threshold=1.0)
    output_json = tmp_path / "electron_fiducial_map.json"
    output_npz = tmp_path / "electron_fiducial_map.npz"

    write_fiducial_map_payload(
        summary,
        before=before,
        after=after,
        eta_edges=edges,
        phi_edges=edges,
        output_json=output_json,
        output_npz=output_npz,
        metadata={"flavor": "electron"},
    )

    assert output_json.exists()
    assert output_npz.exists()
    assert '"flavor": "electron"' in output_json.read_text()
    payload = np.load(output_npz)
    assert np.allclose(payload["inefficiency"], summary.inefficiency)


def test_plot_fiducial_map_payload_accepts_root56_colormap(tmp_path):
    pytest.importorskip("matplotlib")

    before = np.full((2, 2), 100.0)
    after = np.array([[1.0, 1.0], [1.0, 50.0]])
    edges = np.array([-1.0, 0.0, 1.0])
    summary = summarize_fiducial_map(before, after, edges, edges, threshold=1.0)
    output_json = tmp_path / "map.json"
    output_npz = tmp_path / "map.npz"
    write_fiducial_map_payload(
        summary,
        before=before,
        after=after,
        eta_edges=edges,
        phi_edges=edges,
        output_json=output_json,
        output_npz=output_npz,
    )

    written = plot_fiducial_map_payload(
        output_npz,
        output_prefix=tmp_path / "map",
        flavor="electron",
        json_path=output_json,
        formats=["png"],
        colormap="root56",
    )

    assert len(written) == 4
    assert all(path.exists() for path in written)
