# PocketCoffea Workflow Guide

This note is the maintainer map for the Run-3 PocketCoffea migration.  It is
meant for collaborators who need to understand what a mode does, where the cuts
live, and how to add or change selections without chasing the whole repository.
For a compact context packet to hand to another Codex session, see
`docs/codex_handoff_background_estimates.md`.

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
| `muon_pmiss_poffline` | `DATA_Muon` | Muon `Poffline` and `Pmiss` control categories only. No Pveto pair categories. |
| `electron_pmiss_poffline` | `DATA_EGamma` | Electron `Poffline` and `Pmiss` control categories only. No Pveto pair categories. |
| `tau_mu_pmiss_poffline` | `DATA_Muon` | Legacy-equivalent single-muon-triggered tau control used for `Nctrl`, `Poffline`, and `Pmiss`; it does not require a low-`MT` muon. |
| `tau_ele_pmiss_poffline` | `DATA_EGamma` | Compatibility/diagnostic tau control; it is not used by the final tau estimator. |
| `tau_pmiss_poffline` | `DATA_Muon` | Tau normalization selected by the year-dependent IsoMu24+tau cross-trigger; supplies `Nctrl`, `Poffline`, and the tau modified-MET spectrum. |
| `tau_trigger_probability` | Muon data containing the cross-trigger path | Optional AN diagnostic for the muon+tau-trigger normalization factor. |
| `fiducial_maps` | `DATA_Muon` or `DATA_EGamma` | Before/after eta-phi histograms used to make electron and muon fiducial-map JSON/NPZ files. |
| `fake_tracks` | `DATA_JetMET`, `DATA_MET`, `DATA_Muon`, or `DATA_EGamma` | Fake-track control regions. Use `DISAPPTRKS_FAKE_TRACK_CONTROL=basic`, `zmumu`, or `zee`. |
| `high_purity_study` | `DATA_Muon` or `DATA_EGamma` | Lightweight Z-sideband comparison of track-quality inputs before and after the `highPurity` bit. |
| `z_sideband_skim` | `DATA_Muon` or `DATA_EGamma` | Writes reusable ROOT skims with an inclusive raw-Nano Z control preselection and a broad ≥4-layer d0-sideband track, without cutting high-purity inputs. |
| `signal_acceptance` | Signal MC | Full signal-region event and disappearing-track selection before and after the nominal `highPurity` requirement in every layer bin. |
| `muon_backgrounds` | `DATA_Muon` | Combined muon plus tau-mu categories. Heavy mode; useful for postprocessing inputs, but can be expensive. |
| `egamma_backgrounds` | `DATA_EGamma` | Combined electron plus tau-ele categories. Heavy mode; useful for postprocessing inputs, but can be expensive. |
| `all` | Diagnostic only | Builds every category; generally too heavy for production. |

Skim triggers are inferred from the mode and sample in `config.py`. For example,
`muon_pveto` applies the SingleMuon skim, while `electron_pveto` applies the
SingleElectron/EGamma skim.

### High-purity input study

This mode keeps the nominal Z control-region and fake-track sideband selection,
but deliberately forms one track collection before the `isHighPurityTrack`
requirement. A native PocketCoffea `CartesianSelection` applies the
before/with-high-purity and requested-layer axes to that collection. Each input
feature is booked once and filled in those track-level categories. The default
is the four-layer bin. An inclusive raw-Nano Z-pair plus broad ≥4-layer
d0-sideband skim is applied first, so expensive derived-object construction
runs only on potentially relevant events; the exact downstream selection is
unchanged.

```bash
DISAPPTRKS_CATEGORY_MODE=high_purity_study \
DISAPPTRKS_FAKE_TRACK_CONTROL=zmumu \
DISAPPTRKS_ENABLE_HIGH_PURITY_DEDX_HISTOGRAMS=1 \
DISAPPTRKS_DATASET_JSON=datasets/eos_2025_Muon.json \
DISAPPTRKS_DATASET_SAMPLE=DATA_Muon \
DISAPPTRKS_DATASET_YEAR=2025 \
scripts/run_lpc_dask.sh --scaleout 50 --skip-bad-files
```

Use `zee` with the EGamma dataset for the electron control. To study additional
bins, set (for example)
`DISAPPTRKS_HIGH_PURITY_STUDY_LAYERS=NLayers4,NLayers5,NLayers6plus`.
After the job finishes, make one multipage PDF per requested layer bin:

```bash
disapptrks plot-high-purity-study output_all.coffea \
  --control zmumu --sample DATA_Muon --title-prefix 2025
```

With `DISAPPTRKS_ENABLE_HIGH_PURITY_DEDX_HISTOGRAMS=1`, the same plotting
command also writes `zmumu_high_purity_dedx_hits_NLayers4.pdf` (or the
corresponding `zee` file).  The hit rows are linked through `isoTrackIdx` and
restricted to hits belonging to selected high-purity sideband tracks. The
per-hit pages show only the with-high-purity population because MiniAOD retains
`DeDxHitInfo` only for tracks satisfying its retention selection; a nominal
before/after overlay would therefore not measure the effect of the high-purity
requirement. One-dimensional pages cover the raw `IsoTrackDeDxHit` fields.
Two-dimensional pages show detector occupancy and hit charge, path length,
dE/dx, local position, hit type, strip-shape status,
and pixel cluster sizes versus the encoded detector layer.  Pixel cluster-size
plots exclude strip rows, whose source value is the `-1` sentinel. Track-aligned
pages summarize the retained-hit count and its difference from the
measured-layer count, median dE/dx, a mean after dropping the maximum hit,
maximum, standard deviation, range, maximum-to-median ratio, counts above 10
and 20 MeV/mm, and the number and fraction of strip hits failing the
strip-shape selection. The option defaults to off so ordinary high-purity jobs
do not read or histogram the much larger per-hit table.

### Signal acceptance of the high-purity requirement

The `signal_acceptance` mode applies the MET trigger, data-quality machinery
(with the golden JSON automatically skipped for MC), full basic event
selection, and full disappearing-track selection. It creates paired final
categories that are identical except that the nominal category requires the
`highPurity` bit in every layer bin. The 4-, 5-, and six-or-more-layer rows
therefore each measure the corresponding signal-efficiency change.

Run with both electron and muon fiducial maps required so the tested selection
matches the production search selection:

```bash
DISAPPTRKS_CATEGORY_MODE=signal_acceptance \
DISAPPTRKS_ENABLE_SIGNAL_DEDX_HISTOGRAMS=1 \
DISAPPTRKS_OUTPUT_VARIANT=chargino700_high_purity \
DISAPPTRKS_DATASET_JSON=datasets/local_chargino_700.json \
DISAPPTRKS_DATASET_SAMPLE=SIGNAL_Chargino \
DISAPPTRKS_DATASET_YEAR=2022_postEE \
DISAPPTRKS_REQUIRE_FIDUCIAL_MAPS=1 \
DISAPPTRKS_ELECTRON_FIDUCIAL_MAP_JSON=data/fiducial_maps/electron_fiducial_map_2022EFG_v2.json \
DISAPPTRKS_MUON_FIDUCIAL_MAP_JSON=data/fiducial_maps/muon_fiducial_map_2022EFG_v2.json \
python -m pocket_coffea.scripts.runner run \
  --cfg config.py \
  --outputdir analysis_output/2022EFG/signal_acceptance/chargino700_high_purity \
  --executor iterative
```

Then print the event-level retention:

```bash
disapptrks summarize-signal-high-purity \
  analysis_output/2022EFG/signal_acceptance/chargino700_high_purity/output_all.coffea \
  --sample SIGNAL_Chargino \
  --full-cutflow
```

When `DISAPPTRKS_ENABLE_SIGNAL_DEDX_HISTOGRAMS=1`, the workflow also stores
the compact per-track dE/dx summaries for signal candidates passing the final
event selection, all disappearing-track requirements, the requested fiducial
maps, and `highPurity`. It does not histogram the much larger per-hit table.
The default layer bins are 4, 5, and six-or-more layers; override them with
`DISAPPTRKS_SIGNAL_DEDX_LAYERS` if needed. Make the PDFs with:

```bash
disapptrks plot-signal-dedx-study \
  analysis_output/2022EFG/signal_acceptance/chargino700_high_purity/output_all.coffea \
  --sample SIGNAL_Chargino \
  --title-prefix "2022EFG chargino"
```

The detailed output contains two parallel cumulative columns for every basic
event and disappearing-track cut. The explicit `highPurity track ID` row is
the only definition that differs between them; subsequent rows retain that
difference. Use `--layers` to restrict the printed comparison to selected bins.

The cutflow uses PocketCoffea's native categorization and event counting. The
event and pre-layer rows are `StandardSelection` categories. After the layer
split, `CartesianSelection` combines independent high-purity, layer-bin, and
cumulative-stage `MultiCut` axes. This follows PocketCoffea's recommended
pattern for products larger than the 64-mask `PackedSelection` limit and keeps
the standard `cutflow`, `sumw`, and `sumw2` outputs.

### Reusable Z-sideband ROOT skims

The skim requires an inclusive raw-Nano `zmumu` or `zee` Z control preselection and at least one raw
IsoTrack with `pt > 55 GeV`, `abs(eta) < 2.1`, four measured tracker layers,
and `0.05 <= abs(dxy) < 0.50 cm`. It deliberately does not require
`highPurity`, hit quality, missing hits, inactive layers, chi-squared,
isolation, calorimeter energy, overlap vetoes, or fiducial maps. Those remain
available for unbiased downstream study.

Muon/electron trigger-object matching and the exact tag multiplicity are
intentionally deferred to the downstream analysis because the matching fields
are constructed after PocketCoffea's raw skim stage. The skim therefore uses
at least one opposite-sign Z-mass pair among tight isolated raw leptons; this is
inclusive with respect to the final Z control.

For a small LPC-visible test destination:

```bash
DISAPPTRKS_CATEGORY_MODE=z_sideband_skim \
DISAPPTRKS_FAKE_TRACK_CONTROL=zmumu \
DISAPPTRKS_SKIM_OUTPUT=skim/zmumu \
DISAPPTRKS_OUTPUT_VARIANT=zmumu \
DISAPPTRKS_DATASET_JSON=datasets/eos_2025_Muon_OSUv2.json \
DISAPPTRKS_DATASET_SAMPLE=DATA_Muon \
DISAPPTRKS_DATASET_YEAR=2025 \
python -m pocket_coffea.scripts.runner run \
  --cfg config.py \
  --outputdir analysis_output/2025/z_sideband_skim/zmumu \
  --executor condor \
  --executor-custom-setup lpc_condor_executor.py \
  --custom-run-options run_options_lpc_z_sideband_skim.yaml
```

Use the direct Condor executor for skim production, not Dask. For full
production, prefer an XRootD EOS destination for `DISAPPTRKS_SKIM_OUTPUT`.
PocketCoffea writes one ROOT file per independently retryable processed chunk
and exits before object preselection, categories, or histograms are evaluated.
If the LPC schedd is unavailable from the submission environment, the local
fallback `run_options_lpc_skim.yaml` can be used with `run_lpc_dask.sh`; use a
fresh output directory when retrying so a failed run's partial files are not
mixed with the new production.

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

Leg-specific jobs only require the map for the leg being measured:
`tau_ele_pveto` and `tau_ele_pmiss_poffline` require the electron map;
`tau_mu_pveto` and `tau_mu_pmiss_poffline` require the muon map. Combined modes
or directory-based production submissions may still provide both.

The `fake_tracks` mode is different: legacy fake-track channels inherited both
`cutTrkFiducialElectron` and `cutTrkFiducialMuon` from the generic track
selection, in addition to the ECAL fiducial flag. To match that behavior, Nano
fake-track candidates apply both the electron and muon fiducial hot-spot maps.
For production fake-track jobs with `DISAPPTRKS_REQUIRE_FIDUCIAL_MAPS=1`, pass
both map JSONs or use `DISAPPTRKS_FIDUCIAL_MAP_DIR`.

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

When building the JSON maps with `disapptrks make-fiducial-map`, hot spots are
identified with a sigma threshold. The legacy/default behavior uses:

```bash
--threshold 2.0
```

For eras with a pathological eta-phi bin that inflates the standard deviation,
the map builder can exclude the highest-inefficiency occupied bins from the
stddev calculation only:

```bash
--stddev-exclude-top 1
```

The excluded bin is still tested and reported as a hot spot; it just does not
set the width used to identify the other bins. The JSON records the excluded
bins under `stddev_excluded_bins`.

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

For production checks, set:

```bash
DISAPPTRKS_REQUIRE_FIDUCIAL_MAPS=1
```

This makes the job fail if either the electron or muon fiducial-map JSON is not
configured, not readable by the worker, or contains no hot spots. Muon Pveto
outputs also include diagnostic variables:

- `nElectronFiducialHotSpotsLoaded`
- `nMuonFiducialHotSpotsLoaded`
- `nMuonPVetoTagProbePairZWindowPassNoFiducial`
- `nMuonPVetoTagProbePairZWindowFiducialRejected`
- `nMuonPVetoTagProbePairSSZWindowPassNoFiducial`
- `nMuonPVetoTagProbePairSSZWindowFiducialRejected`

Use these to verify that the maps were loaded and that they actually overlap
the Z-window numerator probes.

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

For production, run the lepton-background control categories in their own
control-only modes. For muons:

```bash
DISAPPTRKS_CATEGORY_MODE=muon_pmiss_poffline
```

This builds the same muon tag collection used by `muon_pveto`, but it only
selects the per-layer control categories:

- `muon_background_control_{layer}`
- `muon_background_offline_{layer}`
- `muon_background_trigger_{layer}`

The analogous modes are `electron_pmiss_poffline`, `tau_mu_pmiss_poffline`,
and `tau_ele_pmiss_poffline`.

These feed the postprocessing estimate:

- `Pveto` comes from the matching lepton Pveto tag-probe pair categories.
- `Poffline` and `Pmiss` normally come from the legacy-style MET histogram
  integration stored by the `*_pmiss_poffline` output.
- The scalar categories `background_control`, `background_offline`, and
  `background_trigger` remain available as fallback/debugging counts.
- `N_lepton = N_ctrl * Pveto * Poffline * Pmiss / epsilon_trig^lepton`.
  `epsilon_trig^lepton` is the separate legacy trigger-efficiency divisor, not
  the same quantity as `Pmiss`.
- The current Nano single-tau control is selected with the muon+tau cross
  trigger, so derive and pass the dissertation `P(tau)` correction using the
  dedicated trigger-probability mode:

```bash
DISAPPTRKS_CATEGORY_MODE=tau_trigger_probability \
DISAPPTRKS_DATASET_JSON=datasets/eos_2022CD_Muon.json \
DISAPPTRKS_DATASET_SAMPLE=DATA_Muon \
DISAPPTRKS_DATASET_YEAR=2022_preEE \
scripts/run_lpc_dask.sh --scaleout 200 --skip-bad-files
```

Then extract the factor:

```bash
disapptrks extract-tau-trigger-probability \
  --sample DATA_Muon \
  --output-json tables/tau_trigger_probability_2022CD.json \
  analysis_output/2022CD_tau_trigger_probability_dask/output_*.coffea
```

Following dissertation Equation 7.7, the extractor calculates
`P(tau) = P(muon+tau)/P(muon) = N(cross)/N(muon)`, where the cross-trigger and
single-muon-trigger counts are each measured independently (not one
conditioned on the other) over the same common muon/tau eta-leg baseline. The
extractor prints the resulting `--tau-probability` and
`--tau-probability-error` arguments, and stores the underlying counts in
JSON. This correction is required when the tau control region is selected by
the muon+tau cross trigger. The cross-trigger is `IsoMu24+LooseDeepTau30` for
2022--2024 and `IsoMu24+PNetTau26+L2NN` for 2025--2026.

You can still add these categories to a full Pveto job with
`DISAPPTRKS_ENABLE_LEPTON_BACKGROUND_CATEGORIES=1`, but this is heavier. It also
makes it easier to accidentally include duplicate Poffline/Pmiss histograms when
combining outputs. By default, Pveto modes no longer write the
`n<Prefix>Background...` histograms; use the dedicated `*_pmiss_poffline` modes
for production Poffline/Pmiss.

Each `category_mode`'s Table-16/Pveto/Poffline-Pmiss diagnostic-cutflow family
(`muon_table16_categories`, `electron_pveto_diagnostic_categories`,
`tau_pveto_diagnostic_categories`, `tau_background_diagnostic_categories`,
`fake_z_control_diagnostic_categories`) is a cumulative cutflow -- one `Cut` per
stage -- so `config.py` builds it as a single `MultiCut` axis inside a
`CartesianSelection` (`active_diagnostic_categories`, per
pocketcoffea-conventions' recommended pattern for a cumulative-stage axis)
instead of merging it flat into the mode's `StandardSelection`. Each `MultiCut`
axis gets its own `PackedSelection` budget independent of the rest of that
mode's categories, so turning on `DISAPPTRKS_ENABLE_PVETO_DIAGNOSTICS` for a
single-flavor Pveto/Poffline-Pmiss mode no longer competes with that mode's
other categories for the Coffea `PackedSelection` 64-mask cap. This does not
extend to `muon_backgrounds`/`egamma_backgrounds`/`all`, which still combine
multiple category families (and, for `all`, multiple diagnostic families at
once) into one flat `StandardSelection` -- keep those reserved for debugging,
per the invariant above.

Pveto outputs contain the four counters used to reproduce the legacy
`calculateTriggerEfficiencyFile()` epsilon divisor:

- `n<Prefix>TriggerEffProbesPT55`
- `n<Prefix>TriggerEffProbesSSPT55`
- `n<Prefix>TriggerEffProbesFiringTrigger`
- `n<Prefix>TriggerEffSSProbesFiringTrigger`

The `estimate-lepton-background` command calculates
`epsilon_trig^lepton = (passes_OS - passes_SS) / (total_OS - total_SS)` from
these counters and prints `trigger_efficiency_method=legacy-tag-probe`.
Layer rows use the suffixed counters, e.g.
`n<Prefix>TriggerEffProbesPT55_NLayers4`; `combinedBins` uses the unsuffixed
combined counters.
Manual `--trigger-efficiency` and `--trigger-efficiency-error` are explicit
overrides for comparisons/debugging.

After this change, rerun the relevant Pveto job if the `Pveto` pair counts
changed. The `*_pmiss_poffline` jobs also need to be rerun if they were produced
before the MET-integration histograms were added.

New `*_pmiss_poffline` outputs also contain the histograms needed to duplicate
the legacy integration:

- ordinary no-muon MET after the tag/control-track selection
- ordinary no-muon MET after the MET trigger
- `deltaPhi(leading jet, lepton-removed MET)` versus lepton-removed MET

The `estimate-lepton-background` command uses these histograms automatically
when they are present. It builds the MET-trigger turn-on from ordinary no-muon
MET, weights the lepton-removed MET versus delta-phi distribution, and
integrates the region passing `--met-cut` and `--phi-cut`, matching the legacy
`printPpassMetTriggers()` approach. If the histograms are absent, it falls back
to the older scalar cutflow ratios and prints `met_method=cutflow-ratio`.

After this change, rerun the `*_pmiss_poffline` jobs before comparing to the
AN. Existing Pveto outputs can still be reused.

For nominal AN comparisons, the postprocessor should print:

```text
trigger_efficiency_method=legacy-tag-probe
met_method=hist-integrated
```

If `--trigger-efficiency` was supplied, the method prints `manual` instead. If
it prints `trigger_efficiency_method=default`, rerun the relevant `*_pveto`
job with current code. If it prints `met_method=cutflow-ratio`, rerun the
relevant `*_pmiss_poffline` job with current code.

If older Pveto outputs already contain duplicate `n<Prefix>Background...`
histograms and are passed together with dedicated `*_pmiss_poffline` outputs,
the postprocessor now prefers the dedicated Pmiss/Poffline outputs and prints a
message saying how many files are used for those factors. This avoids summing
the same Poffline/Pmiss histograms twice.

Postprocess with:

```bash
disapptrks estimate-lepton-background \
  --mode muon \
  --run-period 2022CD \
  --output-json tables/muon_background_2022CD.json \
  --output-tex tables/muon_background_2022CD.tex \
  analysis_output/2022CD_muon_pveto/output_*.coffea \
  analysis_output/2022CD_muon_pmiss_poffline/output_*.coffea
```

For the tau background, run the muon and electron legs on `DATA_Muon` and
`DATA_EGamma`, respectively. In both `*_pmiss_poffline` modes the selected
The normalization uses the Muon-dataset muon+tau cross-trigger skim without
requiring a low-`MT` offline muon. `Nctrl`, `Poffline`, and `Pmiss` are formed from a reconstructed
hadronic tau matched to the isolated track. The tau is treated as invisible in
the modified-MET calculation.

Combine both legs in postprocessing. The effective trigger efficiency is
required explicitly until its per-leg treatment is finalized:

```bash
disapptrks estimate-tau-background \
  --run-period 2022CD \
  --output-json tables/tau_background_2022CD.json \
  --output-tex tables/tau_background_2022CD.tex \
  --trigger-efficiency 0.90 \
  --trigger-efficiency-error 0.006 \
  --tau-probability-files \
    analysis_output/2022CD_tau_trigger_probability/output_*.coffea \
  --tau-control-files \
    analysis_output/2022CD_tau_pmiss_poffline/output_*.coffea \
  --tau-mu-files \
    analysis_output/2022CD_tau_mu_pveto/output_*.coffea \
  --tau-ele-files \
    analysis_output/2022CD_tau_ele_pveto/output_*.coffea
```

This command combines:

- `Pveto` OS/SS pairs from both tag-and-probe legs;
- `Nctrl`, `Poffline`, and `Pmiss` from the cross-triggered Muon-data tau control;
- the explicitly supplied effective trigger efficiency.

The defaults assume `DATA_Muon` and `DATA_EGamma` sample keys.

### Example Dask@LPC Command

Run from `DisappTrks_Nano/pocket_coffea` inside the LPC `./shell` environment:

```bash
DISAPPTRKS_CATEGORY_MODE=muon_pveto \
DISAPPTRKS_ENABLE_PVETO_DIAGNOSTICS=0 \
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

Run Poffline/Pmiss separately with:

```bash
DISAPPTRKS_CATEGORY_MODE=muon_pmiss_poffline \
DISAPPTRKS_ENABLE_PVETO_DIAGNOSTICS=0 \
DISAPPTRKS_DATASET_JSON=datasets/eos_2022CD_Muon.json \
python -m pocket_coffea.scripts.runner run \
  --cfg config.py \
  --outputdir analysis_output/2022CD_muon_pmiss_poffline \
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
| `DISAPPTRKS_ENABLE_LEPTON_BACKGROUND_CATEGORIES` | Adds Poffline/Pmiss control categories to a Pveto mode. Prefer the dedicated `*_pmiss_poffline` modes for production. |
| `DISAPPTRKS_ENABLE_PVETO_DIAGNOSTICS` | Adds detailed Pveto cutflow diagnostic categories (their own `CartesianSelection` `MultiCut` axis, so they no longer compete with the mode's other categories for the `PackedSelection` cap). Leave off for production Pmiss/Poffline runs unless you need the diagnostic tables. |
| `DISAPPTRKS_ENABLE_SEARCH_DIAGNOSTICS` | Adds detailed search/cutflow diagnostic categories. |
| `DISAPPTRKS_FAKE_TRACK_CONTROL` | Fake-track control choice: `basic`, `zmumu`, or `zee`. |
| `DISAPPTRKS_FIDUCIAL_MAP_DIR` | Directory containing `electron_fiducial_map.json` and `muon_fiducial_map.json`. |
| `DISAPPTRKS_ELECTRON_FIDUCIAL_MAP_JSON` | Explicit electron fiducial-map JSON path. |
| `DISAPPTRKS_MUON_FIDUCIAL_MAP_JSON` | Explicit muon fiducial-map JSON path. |
| `DISAPPTRKS_ENABLE_FAKE_SIDEBAND_HISTOGRAMS` | Set to `0` for production fake-track jobs to skip exploratory sideband hit-pattern and dE/dx histograms and event manifests while retaining estimate counts and transfer-factor fits. Defaults to `1`. |
| `DISAPPTRKS_FAKE_TRACK_REQUIRE_DEDX_CUT` | Set to `0` to drop the fake-track dE/dx max-over-median cut (see `PROBE_TRACK_DEDX_MAX_OVER_MEDIAN` in `selections.py`) for an AN cross-check. Defaults to `1`. |
| `DISAPPTRKS_LEPTON_BACKGROUND_REQUIRE_DEDX_CUT` | Set to `0` to drop the high-purity/dE/dx cut on the muon/electron/tau Pveto probe-track selections and the Poffline/Pmiss control-track selection (`muon_pveto`, `electron_pveto`, `tau_mu_pveto`, `tau_ele_pveto`, `muon_pmiss_poffline`, `electron_pmiss_poffline`, `tau_mu_pmiss_poffline`, `tau_ele_pmiss_poffline`, `tau_pmiss_poffline`) for an AN cross-check. Defaults to `1`. These control tracks select the same kind of track the signal selection targets, so the background estimate's track quality matches it. Shares `PROBE_TRACK_DEDX_MAX_OVER_MEDIAN` with the fake-track cut, but looks it up per track (`probe_track_dedx_mask`) rather than per caller-chosen layer bin (`dedx_max_over_median_mask`), since `MuonVetoProbeTrack`/etc. are one mixed-NLayers collection built at `layer="combinedBins"`, not one collection per bin (the Poffline/Pmiss control track is looped per exact layer bin already, so either lookup would work there, but it uses the same shared per-track helper for consistency). A track whose own measured layer count has no working point (`NLayers6plus` and up -- no working point is planned for `combinedBins`/`NLayers6plus`) gets the high-purity requirement only; NLayers4/NLayers5 tracks get cut wherever they appear. Does not affect `fiducial_map_probe_track_mask`, which is deliberately independent of track quality. |
| `DISAPPTRKS_HIGH_PURITY_STUDY_LAYERS` | Comma-separated layer bins for `high_purity_study`; defaults to `NLayers4`. |
| `DISAPPTRKS_ENABLE_HIGH_PURITY_DEDX_HISTOGRAMS` | Set to `1` to add linked `IsoTrackDeDxHit` one- and two-dimensional histograms to `high_purity_study`; defaults to `0`. |
| `DISAPPTRKS_SKIM_OUTPUT` | Required output directory for `z_sideband_skim`; may be a worker-visible local path or XRootD EOS URL. |
| `DISAPPTRKS_JET_VETO_MAP_DIR` | Directory containing JME jet-veto-map payloads. |
| `DISAPPTRKS_ALLOW_MISSING_JET_VETO_MAP` | Set only for non-production diagnostics when jet-veto-map payloads are unavailable. |
