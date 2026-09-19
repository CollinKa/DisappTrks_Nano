# Lepton fiducial maps

## What they are

The maps mark the `(eta, phi)` regions where the detector loses electrons or
muons -- dead ECAL towers, bad muon chambers and similar. They are built from
Z tag-and-probe: each bin holds a probe count before and after the lepton veto,
and their ratio is the lepton reconstruction inefficiency there. Bins more than
2 sigma above the mean inefficiency are published as *hot spots*, each with an
`eta`, `phi`, `radius` and `sigma`.

The search needs them because a real lepton crossing such a region is not
reconstructed as a lepton. Its track then has no matching lepton, survives the
lepton vetoes and looks like a disappearing track. Vetoing tracks inside hot
spots removes that background. The same maps are used for the Pveto estimate,
so signal and background have to use them identically or the two do not
combine.

## Resolution order

`_load_fiducial_hot_spots(flavor, year, era)` in
[`src/disapptrks/selections.py`](../src/disapptrks/selections.py):

1. `DISAPPTRKS_<ELECTRON|MUON>_FIDUCIAL_MAP_JSON` -- an explicit file.
2. `DISAPPTRKS_FIDUCIAL_MAP_DIR` -- a directory holding exactly
   `electron_fiducial_map.json` and `muon_fiducial_map.json`.
3. Otherwise, by era from the group's EOS space.

Use 1 and 2 only to test an unreleased map. The normal path is 3.

| dataset `year` | map period | file |
|---|---|---|
| `2022_preEE` | `2022CD` | `<flavor>_fiducial_map_2022CD_v2.json` |
| `2022_postEE` | `2022EFG` | `<flavor>_fiducial_map_2022EFG_v2.json` |
| `2023_preBPix` | `2023C` | `<flavor>_fiducial_map_2023C_v2.json` |
| `2023_postBPix` | `2023D` | `<flavor>_fiducial_map_2023D_v2.json` |

under `root://cmseos.fnal.gov//store/group/lpcdisapptrks/fiducialmaps`.
`FIDUCIAL_MAP_ERAS` is derived from `ERA_GROUPS` in `datasets.py`, so the labels
cannot drift from the dataset definitions.

## Always set `DISAPPTRKS_REQUIRE_FIDUCIAL_MAPS=1`

Without it, a map that cannot be resolved yields zero hot spots and the veto
quietly passes every track. The job still finishes and the yields look
plausible -- only high. Set it for anything whose numbers matter.

## Why xrootd and not /eos

LPC condor runs the payload container with `--contain`, binding only `/cvmfs`,
`/etc/hosts` and `/etc/grid-security`:

```
apptainer exec --pid --ipc --contain --bind /cvmfs --bind /etc/hosts \
  --bind /etc/grid-security --home <scratch>:/srv --pwd /srv <image> ...
```

So `/eos/uscms` is unreachable from a worker no matter what is bound on the
login node. The proxy is shipped as `/srv/x509up_u<uid>`, so xrootd works on
both. Verified on `cmswn2229`: the POSIX listing fails, the xrootd read returns
91 hot spots.

Reads are `lru_cache`d per flavor and era, so each worker fetches once.

## Verifying

```python
import os
os.environ["DISAPPTRKS_REQUIRE_FIDUCIAL_MAPS"] = "1"
from disapptrks.selections import _load_fiducial_hot_spots
len(_load_fiducial_hot_spots("electron", year="2022_preEE"))
```

Expected counts (electron / muon):

| period | electron | muon |
|---|---|---|
| 2022CD | 91 | 113 |
| 2022EFG | 91 | 124 |
| 2023C | 85 | 95 |
| 2023D | 80 | 95 |

Every published radius is `0.0707` -- half the diagonal of one 0.1 x 0.1 bin.

## Known mismatch with the disapptrks skills

The skills in [mlj5j/test-osu-analysis](https://github.com/mlj5j/test-osu-analysis)
(`disapptrks-lepton-backgrounds`, `disapptrks-signal-acceptance`) describe
`_fiducial_map_era` and `_eos_fiducial_hot_spots` in
`pocket_coffea/workflow.py`, with automatic per-era EOS resolution, as if they
already exist. **They are in no pushed branch of this repository.** As of
2026-09-19, `origin/MattDev` (`6c415fd`) reads JSON but only from a path you
configure by hand, with no era logic and no EOS access; no other branch has
even that.

Anyone following those skills against this repository before commit `48b4a9f`
would have set `DISAPPTRKS_*_FIDUCIAL_MAP_JSON`, had it silently ignored, and
run with zero hot spots. That is what happened to the first signal-acceptance
scan here.

This repository now implements the documented interface -- same function names,
env vars and precedence -- so the original can replace it when it lands.

## History

Before `db19d01` the loader opened a ROOT file,
`{flavor}FiducialMap_mc.root`, under an absolute path in a `CMSSW_15_0_10`
checkout. That path is not bound inside the container, the missing file
produced zero hot spots, and `search_track_mask` -- which does request both
lepton maps -- ran with no lepton fiducial veto. Any signal efficiency measured
before that commit is too high.
