import sys
from types import SimpleNamespace

import pytest

from disapptrks.datasets import (
    build_dataset_definition,
    count_root_events,
    filter_latest_prod_versions,
    group_signal_files,
    group_osunano_files,
    is_allowed_osunano_path,
    osunano_area_and_top_dir,
    primary_dataset_from_path,
    publish_dataset_json,
    root_files_from_lines,
    run_year_era_from_path,
    signal_point_from_path,
    write_grouped_filelists,
)


def test_signal_files_are_grouped_by_directory_below_signal_sim():
    point_700 = "AMSB_Wino_M700GeV_ctau1000cm_TuneCP5_13p6TeV_madgraph-pythia8"
    point_800 = "AMSB_Wino_M800GeV_ctau1000cm_TuneCP5_13p6TeV_madgraph-pythia8"
    files = [
        f"root://cmseosmgm01.fnal.gov:1094//store/group/nano/dev/SignalSim/{point_700}/production/date/0000/file_2.root",
        f"root://cmseosmgm01.fnal.gov:1094//store/group/nano/dev/SignalSim/{point_800}/production/date/0000/file_1.root",
        f"root://cmseosmgm01.fnal.gov:1094//store/group/nano/dev/SignalSim/{point_700}/production/date/0000/file_1.root",
    ]

    assert signal_point_from_path(files[0]) == point_700
    assert group_signal_files(files) == {
        point_700: sorted((files[0], files[2])),
        point_800: [files[1]],
    }


def test_signal_point_returns_none_without_signal_sim_directory():
    assert signal_point_from_path("root://host//store/group/nano/file.root") is None


def test_signal_files_can_be_grouped_below_configurable_parent():
    point = "AMSB_Wino_M700GeV_ctau1000cm_TuneCP5_13p6TeV_madgraph-pythia8"
    path = (
        "root://cmseosmgm01.fnal.gov:1094//store/group/lpcdisapptrks/"
        f"nano/dev_v2/{point}/production/date/0000/file.root"
    )

    assert signal_point_from_path(path, marker="dev_v2") == point
    assert group_signal_files([path], marker="dev_v2") == {point: [path]}


def test_count_root_events_sums_tree_metadata_concurrently(monkeypatch):
    entries = {"file1.root": 10, "file2.root": 25, "file3.root": 7}

    class FakeRootFile:
        def __init__(self, path):
            self.path = path

        def __enter__(self):
            return {"Events": SimpleNamespace(num_entries=entries[self.path])}

        def __exit__(self, exc_type, exc, traceback):
            return False

    fake_uproot = SimpleNamespace(
        open=lambda path, timeout: FakeRootFile(path)
    )
    monkeypatch.setitem(sys.modules, "uproot", fake_uproot)
    reports = []

    total, counts = count_root_events(
        list(entries),
        max_workers=2,
        progress=lambda *report: reports.append(report),
    )

    assert total == 42
    assert counts == entries
    assert len(reports) == 3


def test_root_files_from_lines_filters_comments_and_non_root_files():
    files = root_files_from_lines(
        [
            "",
            "# comment",
            "root://cmseos.fnal.gov//store/a/file2.root",
            "root://cmseos.fnal.gov//store/a/not_root.txt",
            "root://cmseos.fnal.gov//store/a/file1.root",
        ]
    )

    assert files == [
        "root://cmseos.fnal.gov//store/a/file1.root",
        "root://cmseos.fnal.gov//store/a/file2.root",
    ]


def test_build_dataset_definition_includes_required_metadata():
    dataset = build_dataset_definition(
        dataset_name="Run2024G_Muon_OSUNano_EOS",
        files=["root://cmseos.fnal.gov//store/a/file.root"],
        sample="DATA_Muon",
        year="2024",
        era="G",
        primary_dataset="Muon",
        nevents="0",
    )

    entry = dataset["Run2024G_Muon_OSUNano_EOS"]
    assert entry["files"] == ["root://cmseos.fnal.gov//store/a/file.root"]
    assert entry["metadata"]["sample"] == "DATA_Muon"
    assert entry["metadata"]["year"] == "2024"
    assert entry["metadata"]["era"] == "G"
    assert entry["metadata"]["nevents"] == "0"
    assert entry["metadata"]["isMC"] == "False"


def test_osunano_path_classification():
    path = "root://cmseos.fnal.gov//store/group/lpcdisapptrks/nano/prod/Muon1_Run2023C_v3/nano_9.root"

    assert primary_dataset_from_path(path) == "Muon"
    assert run_year_era_from_path(path) == ("2023", "C")
    assert osunano_area_and_top_dir(path) == ("prod", "Muon1_Run2023C_v3")
    assert is_allowed_osunano_path(path)


def test_dev_v2_path_classification():
    path = "root://cmseos.fnal.gov//store/group/lpcdisapptrks/nano/dev_v2/Muon0/nested/Run2023C_nano.root"

    assert osunano_area_and_top_dir(path) == ("dev_v2", "Muon0")
    assert is_allowed_osunano_path(path)


def test_dev_v2_files_are_only_included_when_selected():
    standard = "root://cmseos.fnal.gov//store/group/lpcdisapptrks/nano/dev/Muon0/nested/Run2023C_nano.root"
    special = "root://cmseos.fnal.gov//store/group/lpcdisapptrks/nano/dev_v2/Muon0/nested/Run2023C_nano.root"

    standard_grouped = group_osunano_files([standard, special])
    special_grouped = group_osunano_files([standard, special], source_areas=("dev_v2",))

    assert standard_grouped[("Muon", "2023C")] == [standard]
    assert special_grouped[("Muon", "2023C")] == [special]


def test_filter_latest_prod_versions_is_only_for_explicit_override():
    files = [
        "root://cmseos.fnal.gov//store/group/lpcdisapptrks/nano/prod/Muon0_Run2023C_v1/nano_1.root",
        "root://cmseos.fnal.gov//store/group/lpcdisapptrks/nano/prod/Muon0_Run2023C_v4/nano_1.root",
        "root://cmseos.fnal.gov//store/group/lpcdisapptrks/nano/prod/Muon1_Run2023C_v2/nano_1.root",
        "root://cmseos.fnal.gov//store/group/lpcdisapptrks/nano/dev/Muon/nested/Run2023C_file.root",
    ]

    filtered = filter_latest_prod_versions(files)

    assert files[0] not in filtered
    assert files[1] in filtered
    assert files[2] in filtered
    assert files[3] in filtered


def test_group_osunano_files_keeps_all_versions_by_default():
    files = [
        "root://cmseos.fnal.gov//store/group/lpcdisapptrks/nano/prod/Muon0_Run2023C_v1/nano_1.root",
        "root://cmseos.fnal.gov//store/group/lpcdisapptrks/nano/prod/Muon0_Run2023C_v4/nano_1.root",
    ]

    grouped = group_osunano_files(files)

    assert grouped[("Muon", "2023C")] == files


def test_group_osunano_files_rejects_unlisted_and_case_mismatched_top_dirs():
    files = [
        "root://cmseos.fnal.gov//store/group/lpcdisapptrks/nano/prod/Muon0_Run2023C_v4/nano.root",
        "root://cmseos.fnal.gov//store/group/lpcdisapptrks/nano/prod/muon0_Run2023C_v4/nano.root",
        "root://cmseos.fnal.gov//store/group/lpcdisapptrks/nano/prod/Muon2_Run2023C_v4/nano.root",
    ]

    grouped = group_osunano_files(files)

    assert grouped[("Muon", "2023C")] == [files[0]]


def test_group_osunano_files_splits_primary_and_era_groups():
    files = [
        "root://cmseos.fnal.gov//store/group/lpcdisapptrks/nano/prod/Muon_Run2022C/nano.root",
        "root://cmseos.fnal.gov//store/group/lpcdisapptrks/nano/prod/Muon_Run2022F/nano.root",
        "root://cmseos.fnal.gov//store/group/lpcdisapptrks/nano/dev/EGamma0/a/Run2024G_nano.root",
        "root://cmseos.fnal.gov//store/group/lpcdisapptrks/nano/dev/EGamma0/a/Run2025C_nano.root",
    ]

    grouped = group_osunano_files(files)

    assert grouped[("Muon", "2022CD")] == [files[0]]
    assert grouped[("Muon", "2022EFG")] == [files[1]]
    assert grouped[("EGamma", "2024")] == [files[2]]
    assert grouped[("EGamma", "2025")] == [files[3]]


def test_write_grouped_filelists_can_also_write_dataset_jsons(tmp_path):
    grouped = {
        ("Muon", "2023C"): [
            "root://cmseos.fnal.gov//store/group/lpcdisapptrks/nano/prod/Muon0_Run2023C_v4/nano.root"
        ]
    }

    outputs = write_grouped_filelists(
        grouped,
        output_dir=tmp_path / "filelists",
        dataset_json_dir=tmp_path / "datasets",
    )

    entry = outputs["Muon_2023C"]
    assert entry["n_files"] == 1
    assert entry["year"] == "2023_preBPix"
    assert entry["sample"] == "DATA_Muon"
    assert entry["filelist"].read_text().count(".root") == 1
    dataset_json = entry["dataset_json"].read_text()
    assert "Run2023C_Muon_OSUNano_EOS" in dataset_json
    assert '"nano_version": 12' in dataset_json


def test_write_grouped_filelists_uses_2024_nano_v15_metadata(tmp_path):
    grouped = {
        ("Muon", "2024"): [
            "root://cmseos.fnal.gov//store/group/lpcdisapptrks/nano/prod/Muon0_Run2024G/nano.root"
        ]
    }

    outputs = write_grouped_filelists(
        grouped,
        output_dir=tmp_path / "filelists",
        dataset_json_dir=tmp_path / "datasets",
    )

    dataset_json = outputs["Muon_2024"]["dataset_json"].read_text()
    assert '"year": "2024"' in dataset_json
    assert '"nano_version": 15' in dataset_json


def test_write_grouped_filelists_adds_special_output_suffix(tmp_path):
    grouped = {
        ("Muon", "2023C"): [
            "root://cmseos.fnal.gov//store/group/lpcdisapptrks/nano/dev_v2/Muon0/Run2023C_nano.root"
        ]
    }

    outputs = write_grouped_filelists(
        grouped,
        output_dir=tmp_path / "filelists",
        dataset_json_dir=tmp_path / "datasets",
        output_suffix="OSUv2",
    )

    entry = outputs["Muon_2023C"]
    assert entry["filelist"].name == "Muon_2023C_OSUv2.txt"
    assert entry["dataset_json"].name == "eos_2023C_Muon_OSUv2.json"
    assert "Run2023C_Muon_OSUNano_EOS_OSUv2" in entry["dataset_json"].read_text()


def test_publish_dataset_json_uploads_when_absent(tmp_path, monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[0] == "xrdfs":
            return SimpleNamespace(returncode=1)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("disapptrks.datasets.subprocess.run", fake_run)

    local = tmp_path / "eos_2025_Muon_OSUv2.json"
    local.write_text("{}")

    destination = publish_dataset_json(
        local,
        name="eos_2025_Muon_OSUv2",
        eos_base="root://cmseos.fnal.gov//store/group/lpcdisapptrks/dataset_jsons",
    )

    assert destination == (
        "root://cmseos.fnal.gov//store/group/lpcdisapptrks/dataset_jsons/"
        "eos_2025_Muon_OSUv2.json"
    )
    assert calls[0][:2] == ["xrdfs", "root://cmseos.fnal.gov"]
    assert calls[-1][0] == "xrdcp"


def test_publish_dataset_json_refuses_to_overwrite_without_force(tmp_path, monkeypatch):
    def fake_run(cmd, **kwargs):
        if cmd[0] == "xrdfs":
            return SimpleNamespace(returncode=0)
        raise AssertionError("xrdcp should not run without --force")

    monkeypatch.setattr("disapptrks.datasets.subprocess.run", fake_run)

    local = tmp_path / "eos_2025_Muon_OSUv2.json"
    local.write_text("{}")

    with pytest.raises(FileExistsError):
        publish_dataset_json(
            local,
            name="eos_2025_Muon_OSUv2",
            eos_base="root://cmseos.fnal.gov//store/group/lpcdisapptrks/dataset_jsons",
        )


def test_publish_dataset_json_force_skips_existence_check(tmp_path, monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("disapptrks.datasets.subprocess.run", fake_run)

    local = tmp_path / "eos_2025_Muon_OSUv2.json"
    local.write_text("{}")

    publish_dataset_json(
        local,
        name="eos_2025_Muon_OSUv2",
        eos_base="root://cmseos.fnal.gov//store/group/lpcdisapptrks/dataset_jsons",
        force=True,
    )

    # Only the xrdcp call should have happened -- no existence check with force=True.
    assert len(calls) == 1
    assert calls[0][0] == "xrdcp"


def test_publish_output_dir_copies_files_without_double_nesting(tmp_path, monkeypatch):
    from disapptrks.datasets import publish_output_dir

    local_dir = tmp_path / "fake_tracks_smoke"
    local_dir.mkdir()
    (local_dir / "output_all.coffea").write_text("payload")
    nested = local_dir / "nested"
    nested.mkdir()
    (nested / "extra.txt").write_text("extra")

    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[0] == "xrdfs":
            return SimpleNamespace(returncode=1)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("disapptrks.datasets.subprocess.run", fake_run)

    destination = publish_output_dir(
        local_dir,
        period="2099_test",
        mode="fake_tracks_smoke",
        eos_base="root://cmseos.fnal.gov//store/user/mjoyce/claude_verify_scratch",
    )

    assert destination == (
        "root://cmseos.fnal.gov//store/user/mjoyce/claude_verify_scratch/"
        "2099_test/fake_tracks_smoke"
    )
    xrdcp_destinations = [cmd[-1] for cmd in calls if cmd[0] == "xrdcp"]
    assert destination + "/output_all.coffea" in xrdcp_destinations
    assert destination + "/nested/extra.txt" in xrdcp_destinations
    # Never a doubled path segment like .../fake_tracks_smoke/fake_tracks_smoke/...
    assert not any(d.count("fake_tracks_smoke") > 1 for d in xrdcp_destinations)
