"""FIDUCIAL_MAP_ERAS is spelled out in selections.py so that module stays
importable on Dask workers, where it is shipped by value and an intra-package
import would fail.  Keep it in step with the dataset definitions."""

from disapptrks.datasets import ERA_GROUPS
from disapptrks.selections import FIDUCIAL_MAP_ERAS


def test_fiducial_map_eras_match_era_groups():
    assert FIDUCIAL_MAP_ERAS == {
        group.metadata_year: group.label for group in ERA_GROUPS
    }
