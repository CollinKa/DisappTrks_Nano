# PocketCoffea Workflow Guide

This note is the maintainer map for the Run-3 PocketCoffea migration.  It is
meant for collaborators who need to understand what a mode does, where the cuts
live, and how to add or change selections without chasing the whole repository.

## Code Map

The main analysis pieces are split by responsibility:

| File | Purpose |
| --- | --- |
| `src/disapptrks/selections.py` | Physics/object logic on Awkward arrays. Put reusable masks, tag definitions, pair builders, and AN-style cutflow helpers here. |
| `pocket_coffea/workflow.py` | Event content built during processing. This constructs derived collections such as `MuonTag`, `IsoTrackCandidate`, tag-probe pairs, and per-event counts. |
| `pocket_coffea/cuts.py` | PocketCoffea `Cut` objects. These are usually thin wrappers around event-level counts or booleans already produced by the workflow. |
| `pocket_coffea/config.py` | Dataset filtering, category-mode routing, categories, variables, and histograms. Select modes here by `DISAPPTRKS_CATEGORY_MODE`. |
| `src/disapptrks/tables.py` | Cutflow/Pveto table formatting from PocketCoffea outputs. |
| `src/disapptrks/lepton_backgrounds.py` | Lepton-background estimates from `Pveto`, `Poffline`, and `Pmiss` inputs. |
| `src/disapptrks/cli.py` | Command-line entrypoints for dataset JSONs, fiducial maps, tables, and background estimates. |

As a rule of thumb: if the change is a physics definition, start in
`selections.py`; if the change is a named PocketCoffea category, start in
`cuts.py` and `config.py`; if the change needs a new output object/count, add it
in `workflow.py`.

## Common Selection Sequence

The standard search-like selections are exposed in AN language:

| Category | Definition |
| --- | --- |
| `basic_selection` | Event-level BasicSelection: no-muon MET, leading jet, tight-lepton-veto jet ID, dijet dphi, and jet-MET dphi. MET triggers are handled as preselection/skims. |
| `isolated_track_selection` | `basic_selection` plus at least one `IsoTrackIsolated`, corresponding to the AN isolated-track selection. |
| `candidate_track_selection` | Isolated track plus electron, muon, and hadronic tau vetoes. |
| `disappearing_track_selection` | Candidate track plus calorimeter energy and missing outer hit requirements. |
| `search` | Alias for the full disappearing-track category. |

The reusable AN-style helpers are in `src/disapptrks/selections.py`:

- `basic_event_selection_mask`
- `isolated_track_selection_mask`
- `candidate_track_selection_mask`
- `disappearing_track_selection_mask`
- corresponding `*_cutflow_masks` helpers for cumulative cutflow rows

The workflow builds the collections and counts used by the categories:

- `IsoTrackIsolated` and `nIsoTrackIsolated`
- `IsoTrackCandidate` and `nIsoTrackCandidate`
- `IsoTrackSearch` and `nIsoTrackSearch`

## Running Modes

The active mode is selected with:

```bash
DISAPPTRKS_CATEGORY_MODE=<mode>
```

Supported modes are:

| Mode | Typical dataset | What it builds |
| --- | --- | --- |
| `muon_pveto` | `DATA_Muon` | Muon tag-probe pairs, muon `Pveto` categories, muon Table-16 diagnostics, optional muon lepton-background control categories. |
| `electron_pveto` | `DATA_EGamma` | Electron tag-probe pairs and electron `Pveto` categories/diagnostics. |
| `tau_mu_pveto` | `DATA_Muon` | Tau-veto measurement with muon tags and low-`MT` tags. |
| `tau_ele_pveto` | `DATA_EGamma` | Tau-veto measurement with electron tags and low-`MT` tags. |
| `fiducial_maps` | `DATA_Muon` or `DATA_EGamma` | Before/after eta-phi histograms used to make electron and muon fiducial-map JSON/NPZ files. |
| `fake_tracks` | `DATA_JetMET`, `DATA_MET`, `DATA_Muon`, or `DATA_EGamma` | Fake-track control regions. Use `DISAPPTRKS_FAKE_TRACK_CONTROL=basic`, `zmumu`, or `zee`. |
| `muon_backgrounds` | `DATA_Muon` | Combined muon plus tau-mu categories. Heavy mode; useful for postprocessing inputs, but can be expensive. |
| `egamma_backgrounds` | `DATA_EGamma` | Combined electron plus tau-ele categories. Heavy mode; useful for postprocessing inputs, but can be expensive. |
| `all` | Diagnostic only | Builds every category; generally too heavy for production. |

Skim triggers are inferred from the mode and sample in `config.py`. For example,
`muon_pveto` applies the SingleMuon skim, while `electron_pveto` applies the
SingleElectron/EGamma skim.

## Muon Pveto Workflow

Use `DISAPPTRKS_CATEGORY_MODE=muon_pveto` on Muon datasets, for example
`datasets/eos_2022CD_Muon.json`.

### Inputs

Required inputs:

- Muon NanoAOD dataset JSON with metadata `sample: DATA_Muon`
- golden JSON payloads
- jet veto map payloads
- optional electron and muon fiducial-map JSON files

Fiducial-map JSONs are passed either separately:

```bash
DISAPPTRKS_ELECTRON_FIDUCIAL_MAP_JSON=/path/to/electron_fiducial_map.json
DISAPPTRKS_MUON_FIDUCIAL_MAP_JSON=/path/to/muon_fiducial_map.json
```

or as a directory:

```bash
DISAPPTRKS_FIDUCIAL_MAP_DIR=/path/to/fiducial_maps
```

When using the directory form, the code looks for:

```text
electron_fiducial_map.json
muon_fiducial_map.json
```

Both maps are used for lepton fiducial selections in muon Pveto.

### Event And Object Setup

During `apply_object_preselection`, the workflow:

1. Adds muon trigger-match helper fields with `add_muon_derived_fields`.
2. Adds isolated-track derived fields with `add_isotrack_derived_fields`.
   This includes track-crack flags, calorimeter energy, and minimum `dR` values
   to jets, electrons, muons, loose muons, and hadronic taus.
3. Adds event-level quantities with `add_event_derived_fields`, including
   no-muon MET and leading-jet kinematics.
4. Builds common track collections:
   - `IsoTrackProbe`
   - `IsoTrackIsolated`
   - `IsoTrackCandidate`
   - `IsoTrackSearch`

### Muon Tag Definition

Muon tags are built with `muon_tag_mask`, which follows the tag progression in
`muon_tag_progression_masks`:

- `pt > 26 GeV`
- `|eta| < 2.1`
- `tightId`
- `pfRelIso04_all < 0.15`
- matched to the `IsoMu24` trigger object

The selected collection is stored as `MuonTag`.

### Probe Track Definition

Muon Pveto uses `MuonVetoProbeTrack`, built with
`muon_veto_probe_track_mask`. This is the tag-probe denominator track with the
measured muon veto intentionally left open. It keeps the other probe-track
requirements, including:

- isolated-track-style track quality
- `caloEnergy < 10 GeV`
- electron veto
- hadronic tau veto

The muon veto is measured on top of this denominator.

### Pair Building

The workflow builds all `MuonTag x MuonVetoProbeTrack` pairs with
`build_muon_veto_tag_probe_pairs`. Each pair stores:

- invariant mass
- opposite-sign and same-sign flags
- probe kinematics
- probe layer bin
- whether the probe passes the muon veto
- whether the probe passes the muon Pveto numerator before fiducial maps

The Z-window selection uses `|m(tag, probe) - mZ| < 10 GeV`.

### Fiducial Maps In Muon Pveto

Muon Pveto applies electron and muon fiducial-map hot spots to the Pveto
numerator. The workflow loads hot spots with:

- `DISAPPTRKS_ELECTRON_FIDUCIAL_MAP_JSON`
- `DISAPPTRKS_MUON_FIDUCIAL_MAP_JSON`
- or `DISAPPTRKS_FIDUCIAL_MAP_DIR`

The numerator mask is:

```text
muon Pveto pair pass mask AND probe is outside all lepton fiducial hot spots
```

If no fiducial-map path is set, no hot spots are applied.

### Output Categories

The most important muon Pveto categories are:

| Category | Meaning |
| --- | --- |
| `muon_veto_tag` | At least one selected muon tag. |
| `muon_veto_probe` | At least one muon tag and one probe track. |
| `muon_veto_zwindow` | OS tag-probe pairs in the Z window. |
| `muon_veto_zwindow_pass` | OS Z-window pairs passing the plain muon veto. |
| `muon_pveto_zwindow_pass` | OS Z-window pairs passing the full muon Pveto numerator, including missing outer hits and fiducial maps. |
| `muon_veto_ss_zwindow` | SS Z-window control pairs. |
| `muon_pveto_ss_zwindow_pass` | SS Z-window control pairs passing the full numerator. |
| `muon_pveto_zwindow_pass_NLayers4` | Layer-specific numerator for exactly 4 layers. |
| `muon_pveto_zwindow_pass_NLayers5` | Layer-specific numerator for exactly 5 layers. |
| `muon_pveto_zwindow_pass_NLayers6plus` | Layer-specific numerator for 6 or more layers. |

Table-16 diagnostic categories are prefixed with `muon_table16_`.

### Poffline And Pmiss Controls

The lepton-background control categories are disabled by default because they
increase the category load. Enable them with:

```bash
DISAPPTRKS_ENABLE_LEPTON_BACKGROUND_CATEGORIES=1
```

For muon mode, this adds per-layer categories:

- `muon_background_control_{layer}`
- `muon_background_offline_{layer}`
- `muon_background_trigger_{layer}`

These feed the postprocessing estimate:

- `Poffline = N(muon_background_offline) / N(muon_background_control)`
- `Pmiss = N(muon_background_trigger) / N(muon_background_offline)`
- `Pveto` comes from the muon Pveto tag-probe pair categories

Postprocess with:

```bash
disapptrks estimate-lepton-background \
  --mode muon \
  --run-period 2022CD \
  --output-json tables/muon_background_2022CD.json \
  --output-tex tables/muon_background_2022CD.tex \
  analysis_output/2022CD_muon_pveto/output_*.coffea
```

### Example Dask@LPC Command

Run from `DisappTrks_Nano/pocket_coffea` inside the LPC `./shell` environment:

```bash
DISAPPTRKS_CATEGORY_MODE=muon_pveto \
DISAPPTRKS_ENABLE_LEPTON_BACKGROUND_CATEGORIES=1 \
DISAPPTRKS_DATASET_JSON=datasets/eos_2022CD_Muon.json \
DISAPPTRKS_FIDUCIAL_MAP_DIR=/path/to/fiducial_maps/2022CD \
python -m pocket_coffea.scripts.runner run \
  --cfg config.py \
  --outputdir analysis_output/2022CD_muon_pveto \
  --executor dask@lpc \
  --executor-custom-setup executors_lpc.py \
  --custom-run-options run_options_lpc_dask.yaml \
  --scaleout 60 \
  --queue workday
```

For a smoke test, add:

```bash
--limit-files 1 --limit-chunks 1 --scaleout 2 --queue microcentury
```

## Adding Or Modifying A Cut

Use this checklist when changing selections:

1. Add or edit the boolean mask in `src/disapptrks/selections.py`.
2. If it should be counted as a PocketCoffea category, make sure
   `workflow.py` stores an event-level count or boolean.
3. Add a `Cut` in `pocket_coffea/cuts.py`.
4. Add the category to the right `selected_categories` branch in
   `pocket_coffea/config.py`.
5. Add a histogram variable in `config.py` if the count should be stored.
6. Add or update table labels in `src/disapptrks/tables.py` if the result is
   used in a cutflow or AN table.
7. Run a one-file smoke test before submitting a full LPC job.

Keep AN terminology in public names when possible. If the Nano implementation
needs a technical name, put the AN name in the wrapper or category so the
relationship remains obvious.

## Useful Environment Variables

| Variable | Meaning |
| --- | --- |
| `DISAPPTRKS_DATASET_JSON` | Dataset JSON to run. |
| `DISAPPTRKS_DATASET_SAMPLE` | Optional sample override, e.g. `DATA_Muon`. Usually inferred from metadata. |
| `DISAPPTRKS_DATASET_YEAR` | Optional year override, e.g. `2022_preEE`. Usually inferred from metadata. |
| `DISAPPTRKS_CATEGORY_MODE` | Workflow/category mode. Default is `muon_pveto`. |
| `DISAPPTRKS_ENABLE_LEPTON_BACKGROUND_CATEGORIES` | Adds Poffline/Pmiss control categories. |
| `DISAPPTRKS_ENABLE_SEARCH_DIAGNOSTICS` | Adds detailed search/cutflow diagnostic categories. |
| `DISAPPTRKS_FAKE_TRACK_CONTROL` | Fake-track control choice: `basic`, `zmumu`, or `zee`. |
| `DISAPPTRKS_FIDUCIAL_MAP_DIR` | Directory containing `electron_fiducial_map.json` and `muon_fiducial_map.json`. |
| `DISAPPTRKS_ELECTRON_FIDUCIAL_MAP_JSON` | Explicit electron fiducial-map JSON path. |
| `DISAPPTRKS_MUON_FIDUCIAL_MAP_JSON` | Explicit muon fiducial-map JSON path. |
| `DISAPPTRKS_JET_VETO_MAP_DIR` | Directory containing JME jet-veto-map payloads. |
| `DISAPPTRKS_ALLOW_MISSING_JET_VETO_MAP` | Set only for non-production diagnostics when jet-veto-map payloads are unavailable. |
