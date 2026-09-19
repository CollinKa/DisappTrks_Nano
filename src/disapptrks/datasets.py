"""Helpers for building PocketCoffea dataset JSON files."""

from __future__ import annotations

import json
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable
from urllib.parse import urlsplit


def root_files_from_lines(lines: list[str]) -> list[str]:
    """Return sorted ROOT-file paths from command/filelist lines."""
    files = []
    for line in lines:
        path = line.strip()
        if not path or path.startswith("#"):
            continue
        if path.endswith(".root"):
            files.append(path)
    return sorted(files)


def list_eos_root_files(
    eos_path: str,
    *,
    xrootd: str = "root://cmseos.fnal.gov",
    recursive: bool = False,
) -> list[str]:
    """List ROOT files under an EOS path using ``xrdfs ls -u``."""
    command = ["xrdfs", xrootd, "ls", "-u"]
    if recursive:
        command.append("-R")
    command.append(eos_path)

    result = subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return root_files_from_lines(result.stdout.splitlines())


def count_root_events(
    files: list[str],
    *,
    tree_name: str = "Events",
    max_workers: int = 12,
    timeout: int = 120,
    progress: Callable[[int, int, str, int], None] | None = None,
) -> tuple[int, dict[str, int]]:
    """Count tree entries using only each ROOT file's metadata.

    Files are opened concurrently because XRootD connection latency dominates
    this operation.  No event branches are read.
    """

    if max_workers <= 0:
        raise ValueError("max_workers must be positive")

    try:
        import uproot
    except ImportError as exc:
        raise RuntimeError(
            "uproot is required for --count-events; install the analysis dependencies"
        ) from exc

    def count_one(path: str) -> tuple[str, int]:
        with uproot.open(path, timeout=timeout) as root_file:
            return path, int(root_file[tree_name].num_entries)

    counts: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(count_one, path): path for path in files}
        for completed, future in enumerate(as_completed(futures), 1):
            path = futures[future]
            try:
                _, count = future.result()
            except Exception as exc:
                raise RuntimeError(f"failed to count {tree_name} entries in {path}") from exc
            counts[path] = count
            if progress is not None:
                progress(completed, len(files), path, count)

    return sum(counts.values()), counts


def signal_point_from_path(path: str, *, marker: str = "SignalSim") -> str | None:
    """Return the signal-point directory immediately below ``marker``.

    This works for both ``/store/...`` paths and full XRootD URLs.  For example,
    ``.../SignalSim/AMSB_Wino_M700GeV_ctau1000cm_TuneCP5/...`` maps to
    ``AMSB_Wino_M700GeV_ctau1000cm_TuneCP5``.
    """

    url_path = urlsplit(path).path
    parts = PurePosixPath(url_path).parts
    try:
        marker_index = parts.index(marker)
    except ValueError:
        return None
    if marker_index + 1 >= len(parts):
        return None
    return parts[marker_index + 1]


def group_signal_files(
    files: list[str], *, marker: str = "SignalSim"
) -> dict[str, list[str]]:
    """Group ROOT files by their signal-point directory below ``marker``."""

    grouped: dict[str, list[str]] = {}
    for path in files:
        point = signal_point_from_path(path, marker=marker)
        if point is None:
            continue
        grouped.setdefault(point, []).append(path)
    return {point: sorted(paths) for point, paths in sorted(grouped.items())}


@dataclass(frozen=True)
class EraGroup:
    label: str
    years_eras: tuple[tuple[str, str], ...]
    metadata_year: str
    metadata_era: str
    nano_version: int


ERA_GROUPS = (
    EraGroup("2022CD", (("2022", "C"), ("2022", "D")), "2022_preEE", "CD", 12),
    EraGroup(
        "2022EFG",
        (("2022", "E"), ("2022", "F"), ("2022", "G")),
        "2022_postEE",
        "EFG",
        12,
    ),
    EraGroup("2023C", (("2023", "C"),), "2023_preBPix", "C", 12),
    EraGroup("2023D", (("2023", "D"),), "2023_postBPix", "D", 12),
    EraGroup(
        "2024",
        tuple(("2024", era) for era in "ABCDEFGHI"),
        "2024",
        "all",
        15,
    ),
    EraGroup(
        "2025",
        tuple(("2025", era) for era in "ABCDEFGHI"),
        "2025",
        "all",
        15,
    ),
    EraGroup(
        "2026",
        tuple(("2026", era) for era in ("ABCD")),
        "2026",
        "all",
        15,
    )
)

ERA_GROUP_BY_LABEL = {group.label: group for group in ERA_GROUPS}
YEAR_ERA_TO_GROUP = {
    year_era: group.label for group in ERA_GROUPS for year_era in group.years_eras
}
PRIMARY_DATASETS = ("Muon", "EGamma", "JetMET")
ALLOWED_DEV_DIRS = (
    "EGamma0",
    "EGamma1",
    "EGamma2",
    "EGamma22",
    "EGamma3",
    "EGamma4",
    "EGamma5",
    "JetMET",
    "JetMET0",
    "JetMET1",
    "JetMET2",
    "JetMET3",
    "Muon",
    "Muon0",
    "Muon1",
    "Muon3",
)
ALLOWED_PROD_DIRS = (
    "JetMET_Run2022C",
    "JetMET_Run2022D",
    "JetMET_Run2022E",
    "JetMET_Run2022F",
    "JetMET_Run2022G",
    "JetMET_Run2023C",
    "JetMET_Run2023C_v1",
    "JetMET_Run2023C_v2",
    "JetMET_Run2023C_v3",
    "JetMET_Run2023C_v4",
    "JetMET_Run2023D",
    "JetMET_Run2023D_v1",
    "JetMET_Run2023D_v2",
    "JetMET_Run2024C",
    "JetMET_Run2024D",
    "JetMET_Run2024E",
    "JetMET_Run2024F",
    "JetMET_Run2024G",
    "JetMET_Run2024H",
    "JetMET_Run2024I",
    "JetMET0_Run2023C_v1",
    "JetMET0_Run2023C_v2",
    "JetMET0_Run2023C_v3",
    "JetMET0_Run2023C_v4",
    "JetMET0_Run2023D_v1",
    "JetMET0_Run2023D_v2",
    "JetMET0_Run2024C",
    "JetMET0_Run2024D",
    "JetMET0_Run2024E",
    "JetMET0_Run2024F",
    "JetMET0_Run2024G",
    "JetMET0_Run2024H",
    "JetMET0_Run2024I",
    "JetMET1_Run2023C_v1",
    "JetMET1_Run2023C_v2",
    "JetMET1_Run2023C_v3",
    "JetMET1_Run2023D_v1",
    "JetMET1_Run2023D_v2",
    "JetMET1_Run2024C",
    "JetMET1_Run2024D",
    "JetMET1_Run2024E",
    "JetMET1_Run2024F",
    "Muon0_Run2023C_v1",
    "Muon0_Run2023C_v2",
    "Muon0_Run2023C_v3",
    "Muon0_Run2023C_v4",
    "Muon0_Run2023D_v1",
    "Muon0_Run2023D_v2",
    "Muon0_Run2024C",
    "Muon0_Run2024D",
    "Muon0_Run2024E",
    "Muon0_Run2024F",
    "Muon0_Run2024G",
    "Muon0_Run2024H",
    "Muon0_Run2024I",
    "Muon0_Run2024I_v2",
    "Muon1_Run2023C_v1",
    "Muon1_Run2023C_v2",
    "Muon1_Run2023C_v3",
    "Muon1_Run2023D_v1",
    "Muon1_Run2023D_v2",
    "Muon1_Run2024C",
    "Muon1_Run2024D",
    "Muon1_Run2024E",
    "Muon1_Run2024F",
    "Muon_Run2022C",
    "Muon_Run2022D",
    "Muon_Run2022F",
    "Muon_Run2022G",
)
RUN_RE = re.compile(r"Run(20\d{2})([A-Z])")
PROD_DATASET_DIR_RE = re.compile(
    r"^(?P<primary>Muon\d*|EGamma\d*|JetMET\d*)_Run(?P<year>20\d{2})(?P<era>[A-Z])(?:_v(?P<version>\d+))?$"
)


def primary_dataset_from_path(path: str) -> str | None:
    """Infer Muon/EGamma/JetMET from an OSUNano EOS path."""
    for part in Path(path).parts:
        if part.startswith("Muon"):
            return "Muon"
        if part.startswith("EGamma"):
            return "EGamma"
        if part.startswith("JetMET"):
            return "JetMET"
    return None


def run_year_era_from_path(path: str) -> tuple[str, str] | None:
    """Return the first ``(year, era)`` found in a RunYYYYX path or filename."""
    match = RUN_RE.search(path)
    if not match:
        return None
    return match.group(1), match.group(2)


def era_group_label_from_path(path: str) -> str | None:
    year_era = run_year_era_from_path(path)
    if year_era is None:
        return None
    return YEAR_ERA_TO_GROUP.get(year_era)


def osunano_area_and_top_dir(path: str) -> tuple[str, str] | None:
    """Return ``(area, top_dir)`` for files under an OSUNano EOS area."""
    parts = Path(path).parts
    for area in ("dev", "prod", "dev_v2"):
        if area in parts:
            index = parts.index(area)
            if index + 1 < len(parts):
                return area, parts[index + 1]
    return None


def is_allowed_osunano_path(
    path: str,
    *,
    allowed_dev_dirs: tuple[str, ...] = ALLOWED_DEV_DIRS,
    allowed_prod_dirs: tuple[str, ...] = ALLOWED_PROD_DIRS,
) -> bool:
    """Keep only files from the explicitly listed case-sensitive EOS dirs."""
    area_top = osunano_area_and_top_dir(path)
    if area_top is None:
        return False
    area, top_dir = area_top
    if top_dir.startswith("JetMET"):
        return True
    if area in ("dev", "dev_v2"):
        return top_dir in allowed_dev_dirs
    if area == "prod":
        return top_dir in allowed_prod_dirs
    return False


def _prod_dataset_component(path: str) -> str | None:
    parts = Path(path).parts
    if "prod" not in parts:
        return None
    for part in parts:
        if PROD_DATASET_DIR_RE.match(part):
            return part
    return None


def _prod_dataset_base_and_version(component: str) -> tuple[str, int]:
    match = PROD_DATASET_DIR_RE.match(component)
    if match is None:
        return component, 1
    version = int(match.group("version") or 1)
    base = re.sub(r"_v\d+$", "", component)
    return base, version


def filter_latest_prod_versions(files: list[str]) -> list[str]:
    """Drop older ``*_vN`` prod directories while keeping dev files untouched."""
    latest: dict[str, tuple[int, str]] = {}
    for path in files:
        component = _prod_dataset_component(path)
        if component is None:
            continue
        base, version = _prod_dataset_base_and_version(component)
        if base not in latest or version > latest[base][0]:
            latest[base] = (version, component)

    if not latest:
        return files

    allowed_components = {component for _, component in latest.values()}
    filtered = []
    for path in files:
        component = _prod_dataset_component(path)
        if component is None or component in allowed_components:
            filtered.append(path)
    return filtered


def group_osunano_files(
    files: list[str],
    *,
    primary_datasets: tuple[str, ...] = PRIMARY_DATASETS,
    group_labels: tuple[str, ...] = tuple(group.label for group in ERA_GROUPS),
    prod_version_policy: str = "all",
    source_areas: tuple[str, ...] = ("dev", "prod"),
) -> dict[tuple[str, str], list[str]]:
    """Group OSUNano ROOT files by ``(primary_dataset, era_group_label)``."""
    if prod_version_policy not in ("latest", "all"):
        raise ValueError("prod_version_policy must be 'latest' or 'all'")
    selected_files = []
    for path in files:
        area_top = osunano_area_and_top_dir(path)
        if (
            area_top is not None
            and area_top[0] in source_areas
            and is_allowed_osunano_path(path)
        ):
            selected_files.append(path)
    files = selected_files
    if prod_version_policy == "latest":
        files = filter_latest_prod_versions(files)

    grouped = {
        (primary, label): []
        for primary in primary_datasets
        for label in group_labels
    }
    for path in files:
        primary = primary_dataset_from_path(path)
        label = era_group_label_from_path(path)
        if primary in primary_datasets and label in group_labels:
            grouped[(primary, label)].append(path)

    return {key: sorted(value) for key, value in grouped.items()}


def write_filelist(files: list[str], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(f"{path}\n" for path in sorted(files)))


def scan_eos_bases_for_root_files(
    eos_paths: list[str],
    *,
    xrootd: str = "root://cmseos.fnal.gov",
) -> list[str]:
    files: list[str] = []
    for eos_path in eos_paths:
        files.extend(list_eos_root_files(eos_path, xrootd=xrootd, recursive=True))
    return sorted(set(files))


def write_grouped_filelists(
    grouped: dict[tuple[str, str], list[str]],
    *,
    output_dir: Path,
    dataset_json_dir: Path | None = None,
    nano_version: int | None = None,
    output_suffix: str = "",
) -> dict[str, dict[str, Path | int | str]]:
    """Write grouped filelists and optionally matching PocketCoffea JSONs."""
    filename_suffix = f"_{output_suffix}" if output_suffix else ""
    outputs: dict[str, dict[str, Path | int | str]] = {}
    for (primary, label), files in sorted(grouped.items()):
        key = f"{primary}_{label}"
        filelist_path = output_dir / f"{primary}_{label}{filename_suffix}.txt"
        write_filelist(files, filelist_path)

        group = ERA_GROUP_BY_LABEL[label]
        entry: dict[str, Path | int | str] = {
            "filelist": filelist_path,
            "n_files": len(files),
        }

        if dataset_json_dir is not None:
            dataset_name_suffix = f"_{output_suffix}" if output_suffix else ""
            dataset_name = f"Run{label}_{primary}_OSUNano_EOS{dataset_name_suffix}"
            sample = f"DATA_{primary}"
            dataset_path = dataset_json_dir / f"eos_{label}_{primary}{filename_suffix}.json"
            dataset = build_dataset_definition(
                dataset_name=dataset_name,
                files=files,
                sample=sample,
                year=group.metadata_year,
                era=group.metadata_era,
                primary_dataset=primary,
                is_mc=False,
                nevents="0",
                nano_version=group.nano_version if nano_version is None else nano_version,
            )
            write_dataset_definition(dataset, dataset_path)
            entry["dataset_json"] = dataset_path
            entry["dataset_name"] = dataset_name
            entry["sample"] = sample
            entry["year"] = group.metadata_year
            entry["era"] = group.metadata_era

        outputs[key] = entry
    return outputs


def build_dataset_definition(
    *,
    dataset_name: str,
    files: list[str],
    sample: str,
    year: str,
    era: str,
    primary_dataset: str,
    is_mc: bool = False,
    nevents: str = "0",
    nano_version: int = 15,
    extra_metadata: dict[str, str | int | bool] | None = None,
) -> dict:
    """Build a PocketCoffea dataset-definition dictionary."""
    metadata: dict[str, str | int | bool] = {
        "sample": sample,
        "year": year,
        "era": era,
        "isMC": str(is_mc),
        "primaryDataset": primary_dataset,
        "nevents": str(nevents),
        "nano_version": nano_version,
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    return {
        dataset_name: {
            "files": files,
            "metadata": metadata,
        }
    }


def write_dataset_definition(dataset: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dataset, indent=2) + "\n")


# Kept in sync with workflow.py's DATASET_JSON_EOS_BASE_DEFAULT -- this is the
# group's canonical location for shared dataset JSONs (see that module for the
# resolution side; this is the publish side used by `make-dataset-json --publish`).
DATASET_JSON_EOS_BASE_DEFAULT = (
    "root://cmseos.fnal.gov//store/group/lpcdisapptrks/dataset_jsons"
)


def publish_dataset_json(
    local_path: Path,
    *,
    name: str,
    eos_base: str = DATASET_JSON_EOS_BASE_DEFAULT,
    force: bool = False,
) -> str:
    """Copy a dataset JSON to the group's canonical shared EOS location.

    ``name`` is the canonical dataset name, without a directory or ``.json``
    suffix (e.g. ``eos_2023C_Muon`` or ``eos_2023C_Muon_OSUv2``). Refuses to
    overwrite an existing canonical file unless ``force`` is set, so a
    development/test regeneration can't silently clobber the group's current
    dataset. Returns the destination URL.
    """
    parsed = urlsplit(eos_base)
    server = f"{parsed.scheme}://{parsed.netloc}"
    remote_dir = "/" + parsed.path.lstrip("/")
    remote_path = f"{remote_dir}/{name}.json"
    destination = f"{eos_base}/{name}.json"

    if not force:
        check = subprocess.run(
            ["xrdfs", server, "stat", remote_path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if check.returncode == 0:
            raise FileExistsError(
                f"{destination} already exists; pass --force to overwrite the "
                "group's canonical dataset JSON"
            )

    subprocess.run(["xrdcp", "-f", str(local_path), destination], check=True)
    return destination


# Shared EOS space for PocketCoffea/postprocessing output, mirroring the
# dataset-JSON and fiducial-map shared spaces above.
OUTPUT_EOS_BASE_DEFAULT = (
    "root://cmseos.fnal.gov//store/group/lpcdisapptrks/disapptrks_output"
)


class OutputAlreadyExistsError(Exception):
    """Raised when the destination for publish_output_dir already exists on EOS.

    Callers (the CLI) decide how to resolve this -- overwrite or publish under a
    different path -- rather than this function silently picking one.
    """

    def __init__(self, destination: str):
        super().__init__(f"{destination} already exists")
        self.destination = destination


def publish_output_dir(
    local_dir: Path,
    *,
    period: str,
    mode: str,
    eos_base: str = OUTPUT_EOS_BASE_DEFAULT,
    overwrite: bool = False,
) -> str:
    """Copy a local output directory (e.g. analysis_output/<period>/<mode>) to the
    group's shared EOS output space, at <eos_base>/<period>/<mode>.

    Raises OutputAlreadyExistsError if that destination already exists and
    ``overwrite`` is not set -- this never decides on its own whether to replace
    good production output with a development/test run's output.
    """
    parsed = urlsplit(eos_base)
    server = f"{parsed.scheme}://{parsed.netloc}"
    remote_dir = f"{parsed.path.rstrip('/')}/{period}/{mode}"
    destination = f"{eos_base}/{period}/{mode}"

    if not overwrite:
        check = subprocess.run(
            ["xrdfs", server, "stat", remote_dir],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if check.returncode == 0:
            raise OutputAlreadyExistsError(destination)

    # xrdcp -r copies the source directory itself as a new subdirectory of the
    # destination (like `cp -r src dst/` creating `dst/src`), which would nest an
    # extra `<mode>/<mode>` level here. Copy each file individually instead, so
    # the destination is exactly <eos_base>/<period>/<mode>/<relative path>.
    for local_file in sorted(local_dir.rglob("*")):
        if not local_file.is_file():
            continue
        relative = local_file.relative_to(local_dir)
        subprocess.run(
            ["xrdcp", "-f", str(local_file), f"{destination}/{relative.as_posix()}"],
            check=True,
        )
    return destination
