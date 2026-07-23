# DisappTrks_Nano

This repository is the migration layer between the Run 3 disappearing-track
analysis and custom NanoAOD/PocketCoffea. The disappearing-track candidates are
the custom NanoAOD `IsoTrack` collection. Object matching, event-level angular
variables, and analysis selections are computed downstream in Python.

## Current status

- The `IsoTrack` branch contract is encoded in `disapptrks.schema`.
- The nominal search-track and event selections are implemented in
  `disapptrks.selections`.
- The legacy fiducial-map hot-spot algorithm is implemented in
  `disapptrks.fiducial`.
- `pocket_coffea` contains the first PocketCoffea workflow and local
  dataset configuration.

The current local smoke-test file is expected outside this repository as
`../nano_99.root`. Its dataset metadata is labeled as 2024F Muon data. Check
productions before running:

```bash
python3 -m pip install -e '.[analysis,test,pocket-coffea]'
disapptrks audit-schema ../nano_99.root
```

The audit exits with status 2 if the requested scope is incomplete.

Build a PocketCoffea dataset JSON directly from an EOS directory:

```bash
disapptrks make-dataset-json /store/user/YOUR_PATH \
  --recursive \
  --dataset-name Run2024G_Muon_OSUNano_EOS \
  --sample DATA_Muon \
  --year 2024 \
  --era G \
  --primary-dataset Muon \
  -o pocket_coffea/datasets/eos_2024G_muon.json
```

The command uses `xrdfs root://cmseos.fnal.gov ls -u` so the output JSON gets
full XRootD file URLs. If you already have a text filelist, pass it with
`--filelist` instead of an EOS path.

## PocketCoffea setup

Install this analysis and the recommended PocketCoffea release in one
environment:

```bash
python3 -m pip install -e '.[analysis,test,pocket-coffea]'
```

The `pocket-coffea` extra pins `pocket-coffea==0.9.12`, matching the current
PocketCoffea release line and its Coffea/Awkward dependency stack. If you are
developing PocketCoffea itself from a sibling checkout, install that checkout
instead of the extra:

```bash
python3 -m pip install -e ../PocketCoffea
python3 -m pip install -e '.[analysis,test]'
```

Once a file passes the schema audit:

```bash
cd pocket_coffea
pocket-coffea run --cfg config.py --test -lf 1 -lc 1 -c 50000 -e iterative -ps -o output_test_nano99
```

## LPC Dask submission

The preferred LPC path is PocketCoffea's Dask executor with the local LPC
executor from `pocket_coffea/executors_lpc.py`.  This follows the working
`displaced_leptons` setup and avoids maintaining one custom Condor job wrapper
per file slice.

From the repository root on `cmslpc`, run the bootstrap once.  This creates the
`./shell` Apptainer/Singularity entrypoint and a `.env` Python environment with
`lpcjobqueue`.

```bash
./setup_lpc.sh
./shell
```

Inside `./shell`, confirm that the active Python and worker-only packages are
coming from the container venv:

```bash
which python
python -c "import disapptrks; print(disapptrks.__file__)"
python -c "import lpcjobqueue; print(lpcjobqueue.__file__)"
```

The expected Python is `/srv/.env/bin/python`.  If `disapptrks` is missing,
install this package from the repository root inside `./shell`:

```bash
python -m pip install '.[analysis]'
```

Run from `pocket_coffea` with the Python module entrypoint.  This avoids the LPC
container quirk where the `pocket-coffea` executable can point at a Python that
does not see `lpcjobqueue`.

```bash
cd pocket_coffea
DISAPPTRKS_DATASET_JSON=datasets/eos_2023_Muon.json \
python -m pocket_coffea.scripts.runner run \
  --cfg config.py \
  --outputdir analysis_output/2023_muon_smoke \
  --executor dask@lpc \
  --executor-custom-setup executors_lpc.py \
  --custom-run-options run_options_lpc_dask.yaml \
  --limit-files 1 \
  --limit-chunks 1
```

`run_options_lpc_dask.yaml` is intentionally a small smoke-test profile:
`scaleout: 2`, `queue: microcentury`, `local-virtualenv: true`, and
`worker-python: /srv/.env/bin/python`.  Condor logs are written under
`$HOME/pocketcoffea_dask_logs/<output-tag>/condor_log` by default so they are
visible to the LPC schedd outside the container.

After the smoke test starts workers successfully, increase worker count from the
command line, for example:

```bash
DISAPPTRKS_DATASET_JSON=datasets/eos_2023_Muon.json \
python -m pocket_coffea.scripts.runner run \
  --cfg config.py \
  --outputdir analysis_output/2023_muon_pveto \
  --executor dask@lpc \
  --executor-custom-setup executors_lpc.py \
  --custom-run-options run_options_lpc_dask.yaml \
  --scaleout 60 \
  --queue workday
```

If workers fail with a missing local module, check the worker `.err` files in
the log directory above.  The config registers the local analysis modules with
`cloudpickle.register_pickle_by_value(...)` so Dask does not have to import
`cuts.py`, `workflow.py`, or `disapptrks.selections` from exactly the same path
on every worker.

## Tau Pveto validation

The tau-pveto modes are selected with `DISAPPTRKS_CATEGORY_MODE`:

- `tau_mu_pveto` for Muon datasets
- `tau_ele_pveto` for EGamma datasets

For short local/iterative checks inside the same `./shell` environment, run from
`pocket_coffea`:

```bash
scripts/run_tau_pveto_smokes.sh
```

This runs one file and one chunk for 2022/2023 Muon and EGamma datasets.  To run
only a couple of cases:

```bash
CASES="2023C_mu 2023D_eg" LIMIT_FILES=1 LIMIT_CHUNKS=1 \
  scripts/run_tau_pveto_smokes.sh
```

The smoke script sets `DISAPPTRKS_ALLOW_MISSING_JET_VETO_MAP=1` by default so it
can test the tau-pveto workflow without local JME payloads.  For a
production-like validation with real jet-veto-map payloads, override it:

```bash
DISAPPTRKS_ALLOW_MISSING_JET_VETO_MAP=0 scripts/run_tau_pveto_smokes.sh
```

After the smoke outputs exist under `analysis_output/tau_pveto_smoke`, write the
cutflow and Pveto LaTeX tables:

```bash
scripts/make_tau_pveto_tables.sh 2022
scripts/make_tau_pveto_tables.sh 2023
```

This writes `tau_mu_cutflow.tex`, `tau_mu_pveto.tex`, `tau_ele_cutflow.tex`,
`tau_ele_pveto.tex`, and `tau_pveto_combined.tex` under
`tables/tau_pveto/<year>/`.

## LPC manual Condor fallback

If Dask/lpcjobqueue is unavailable, the v2-style wrapper in
`pocket_coffea/scripts` remains as a fallback.  The Condor worker runs in a
sandbox, so it does not depend on `/uscms` or `/uscms_data` paths being mounted
inside the job.

Build a relocatable Python target directory once from the repository root:

```bash
python3 -m pip install --target pocket_coffea/python_env '.[analysis,pocket-coffea]'
```

If the target environment already exists and EOS reads fail with
`No module named 'XRootD'`, add the XRootD Python bindings to it:

```bash
python3 -m pip install --target pocket_coffea/python_env --upgrade xrootd
```

Then submit from `pocket_coffea`:

```bash
cd pocket_coffea
FILES_PER_JOB=5 CHUNKSIZE=50000 \
  scripts/submit_lpc_dataset.sh datasets/eos_2023C_muon.json 2023C_muon_pveto DATA_Muon 2023
```

The wrapper transfers `config.py`, `cuts.py`, `workflow.py`, `../src`, the
dataset JSON, your X509 proxy, and `python_env` into the Condor sandbox.  Each
job makes a per-job dataset JSON, runs PocketCoffea iterative over its file
slice, and returns outputs under `analysis_output/<tag>/`.

After the jobs finish, merge the per-job outputs:

```bash
pocket-coffea merge-outputs analysis_output/2023C_muon_pveto/*.coffea \
  -o analysis_output/2023C_muon_pveto/output_Run2023C_Muon_OSUNano_EOS.coffea
```

Summarize the current muon-veto tag-and-probe prototype from a PocketCoffea
output file:

```bash
disapptrks summarize-pveto output_test_nano99/output_Run2024F_Muon_OSUNano_local.coffea
```

By default this reports
`muon_veto_zwindow_pass / muon_veto_zwindow` with a binomial statistical
uncertainty.

To apply the same-sign subtraction used to estimate the flat non-Z background
under the opposite-sign Z peak:

```bash
disapptrks summarize-pveto output_test_nano99/output_Run2024F_Muon_OSUNano_local.coffea --ss-subtract
```

This reports
`(muon_veto_zwindow_pass - muon_veto_ss_zwindow_pass) / (muon_veto_zwindow - muon_veto_ss_zwindow)`.

## Physics-equivalence caveat

For data, the observed `IsoTrack` missing-hit values can be used directly.
Exact MC equivalence still requires deciding how to reproduce the stochastic
hit-drop and TOB-drop treatment from the legacy OSUT3 workflow. The electron
and muon inefficiency maps are analysis inputs applied downstream; they are not
assumed to be embedded in a separate track table.

See [MIGRATION.md](MIGRATION.md) for the source-to-PocketCoffea mapping and
validation sequence.

For collaborator-facing details on PocketCoffea category modes, where to edit
cuts, and a step-by-step muon Pveto workflow, see
[docs/pocket_coffea_workflows.md](docs/pocket_coffea_workflows.md).

Run unit tests with:

```bash
python3 -m pip install -e '.[test]'
pytest
```
