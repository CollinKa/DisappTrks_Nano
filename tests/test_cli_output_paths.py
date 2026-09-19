from pathlib import Path
from types import SimpleNamespace

import pytest

from disapptrks.cli import (
    _period_float_values,
    _publish_output_command,
    _standard_coffea_files,
)
from disapptrks.datasets import OutputAlreadyExistsError


def test_standard_coffea_files_prefers_merged_output(tmp_path: Path):
    merged = tmp_path / "output_all.coffea"
    shard = tmp_path / "output_job_0.coffea"
    merged.touch()
    shard.touch()

    assert _standard_coffea_files(tmp_path) == [merged]


def test_standard_coffea_files_uses_sorted_job_shards(tmp_path: Path):
    second = tmp_path / "output_job_2.coffea"
    first = tmp_path / "output_job_1.coffea"
    second.touch()
    first.touch()

    assert _standard_coffea_files(tmp_path) == [first, second]


def test_standard_coffea_files_rejects_empty_directory(tmp_path: Path):
    with pytest.raises(SystemExit, match="no top-level .coffea files"):
        _standard_coffea_files(tmp_path)


def test_standard_coffea_files_rejects_ambiguous_outputs(tmp_path: Path):
    (tmp_path / "output_category_a.coffea").touch()
    (tmp_path / "output_category_b.coffea").touch()

    with pytest.raises(SystemExit, match="ambiguous .coffea outputs"):
        _standard_coffea_files(tmp_path)


def test_period_float_values_accepts_single_bare_value():
    assert _period_float_values(["0.9"], ["2022CD"], "--value") == {
        "2022CD": 0.9
    }


def test_period_float_values_maps_multiple_periods():
    assert _period_float_values(
        ["2022CD=0.9", "2022EFG=0.91"],
        ["2022CD", "2022EFG"],
        "--value",
    ) == {"2022CD": 0.9, "2022EFG": 0.91}


def test_period_float_values_rejects_missing_period():
    with pytest.raises(SystemExit, match="missing for: 2022EFG"):
        _period_float_values(
            ["2022CD=0.9"], ["2022CD", "2022EFG"], "--value"
        )


def _publish_output_args(tmp_path: Path, **overrides):
    defaults = dict(
        input_dir=tmp_path,
        period="2025",
        mode="fake_tracks/basic",
        eos_base="root://cmseos.fnal.gov//store/group/lpcdisapptrks/disapptrks_output",
        overwrite=False,
        suffix=None,
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_publish_output_command_uploads_when_absent(tmp_path, monkeypatch):
    calls = []

    def fake_publish_output_dir(local_dir, *, period, mode, eos_base, overwrite):
        calls.append((local_dir, period, mode, eos_base, overwrite))
        return f"{eos_base}/{period}/{mode}"

    monkeypatch.setattr(
        "disapptrks.cli.publish_output_dir", fake_publish_output_dir
    )

    result = _publish_output_command(_publish_output_args(tmp_path))

    assert result == 0
    assert calls == [
        (
            tmp_path,
            "2025",
            "fake_tracks/basic",
            "root://cmseos.fnal.gov//store/group/lpcdisapptrks/disapptrks_output",
            False,
        )
    ]


def test_publish_output_command_rejects_nonexistent_input_dir(tmp_path):
    with pytest.raises(SystemExit, match="is not a directory"):
        _publish_output_command(
            _publish_output_args(tmp_path, input_dir=tmp_path / "missing")
        )


def test_publish_output_command_reports_conflict_without_touching_eos(
    tmp_path, monkeypatch
):
    def fake_publish_output_dir(local_dir, *, period, mode, eos_base, overwrite):
        raise OutputAlreadyExistsError(f"{eos_base}/{period}/{mode}")

    monkeypatch.setattr(
        "disapptrks.cli.publish_output_dir", fake_publish_output_dir
    )

    with pytest.raises(SystemExit, match="--overwrite.*--suffix"):
        _publish_output_command(_publish_output_args(tmp_path))


def test_publish_output_command_overwrite_passes_through(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        "disapptrks.cli.publish_output_dir",
        lambda local_dir, **kwargs: calls.append(kwargs) or "dest",
    )

    _publish_output_command(_publish_output_args(tmp_path, overwrite=True))

    assert calls[0]["overwrite"] is True


def test_publish_output_command_suffix_avoids_original_path(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        "disapptrks.cli.publish_output_dir",
        lambda local_dir, **kwargs: calls.append(kwargs) or "dest",
    )

    _publish_output_command(_publish_output_args(tmp_path, suffix="dev"))

    assert calls[0]["mode"] == "fake_tracks/basic_dev"
    assert calls[0]["overwrite"] is False


def test_publish_output_command_rejects_overwrite_and_suffix_together(tmp_path):
    with pytest.raises(SystemExit, match="mutually exclusive"):
        _publish_output_command(
            _publish_output_args(tmp_path, overwrite=True, suffix="dev")
        )
