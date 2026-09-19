# Codex Memory: DisappTrks Nano Background Estimates

Use this as project memory for a fresh Codex session helping with the
DisappTrks Nano/PocketCoffea background-estimate migration.

## Working Context

- Parent workspace: `DisappTrks/`
- Legacy analysis repo: `DisappTrks/`
- Nano migration repo: `DisappTrks_Nano/`
- Primary task: develop and debug the Run-3 PocketCoffea implementation of
  lepton background estimates: `Pveto`, `Poffline`, `Pmiss`, fiducial maps,
  and related cutflows.
- The physics reference is the AN plus the legacy DisappTrks code, especially:
  - `DisappTrks/BackgroundEstimation/python/`
  - `DisappTrks/BackgroundEstimation/test/bkgdEstimate_2022.py`
  - `DisappTrks/StandardAnalysis/python/EventSelections.py`
  - `DisappTrks/StandardAnalysis/python/Cuts.py`

## Current Code Organization

- `DisappTrks_Nano/src/disapptrks/selections.py`
  contains reusable physics selections and pair builders.
- `DisappTrks_Nano/pocket_coffea/workflow.py`
  constructs derived collections, tag-probe pairs, and event-level counts.
- `DisappTrks_Nano/pocket_coffea/cuts.py`
  defines PocketCoffea `Cut` wrappers.
- `DisappTrks_Nano/pocket_coffea/config.py`
  routes category modes and histogram variables.
- `DisappTrks_Nano/src/disapptrks/lepton_backgrounds.py`
  computes final `N_lepton = N_ctrl * Pveto * Poffline * Pmiss /
  epsilon_trig^lepton`, with optional `N_ctrl` prescale.
- `DisappTrks_Nano/src/disapptrks/cli.py`
  exposes commands for fiducial maps, Pveto tables, and estimates.
- `DisappTrks_Nano/docs/pocket_coffea_workflows.md`
  is the main collaborator-facing workflow guide.
- `DisappTrks_Nano/docs/codex_handoff_background_estimates.md`
  is the main context packet for another Codex session.

## Current Status

The current working baseline has:

- fiducial-map production and loading for electron and muon veto hot spots;
- dedicated `*_pveto` jobs for `Pveto` and the legacy epsilon counters;
- dedicated `*_pmiss_poffline` jobs for legacy-style `Poffline`/`Pmiss`
  histograms;
- postprocessing that combines the two outputs and calculates the legacy
  epsilon divisor from Pveto tag-probe counters. `Pmiss` separately handles the
  MET-trigger turn-on.

Joyce reports that the latest 2022CD electron `Poffline`/`Pmiss` values look
much closer to the AN after the current fixes. Continue from this baseline.

## Implemented Modes

Production-oriented modes:

- `fiducial_maps`
- `muon_pveto`
- `electron_pveto`
- `tau_mu_pveto`
- `tau_ele_pveto`
- `muon_pmiss_poffline`
- `electron_pmiss_poffline`
- `tau_mu_pmiss_poffline`
- `tau_ele_pmiss_poffline`
- `tau_trigger_probability` optional legacy/AN diagnostic only
- `fake_tracks`

Heavy/diagnostic modes:

- `muon_backgrounds`
- `egamma_backgrounds`
- `all`

Avoid the heavy modes for production unless debugging. They can exhaust Coffea
`PackedSelection` slots.

## Important Recent Decisions

### Pmiss/Poffline Split

`Poffline` and `Pmiss` were split into dedicated control-only modes:

- `muon_pmiss_poffline`
- `electron_pmiss_poffline`
- `tau_mu_pmiss_poffline`
- `tau_ele_pmiss_poffline`

These avoid overloading PocketCoffea by not building full Pveto pair category
sets.

### Fiducial Map Loading

Pveto jobs can load fiducial maps using:

```bash
DISAPPTRKS_ELECTRON_FIDUCIAL_MAP_JSON=/path/to/electron_fiducial_map.json
DISAPPTRKS_MUON_FIDUCIAL_MAP_JSON=/path/to/muon_fiducial_map.json
```

For leg-specific tau workflows, only the active leg's map is required:
`tau_ele_*` uses the electron map and `tau_mu_*` uses the muon map.
For fake-track workflows, provide both maps: legacy fake-track selections used
both electron and muon fiducial vetoes plus the ECAL fiducial flag.

or:

```bash
DISAPPTRKS_FIDUCIAL_MAP_DIR=/path/to/fiducial_maps
```

The directory form expects:

```text
electron_fiducial_map.json
muon_fiducial_map.json
```

Production validation should use:

```bash
DISAPPTRKS_REQUIRE_FIDUCIAL_MAPS=1
```

### Fiducial Map Hot Spots

Default hot-spot threshold:

```bash
--threshold 2.0
```

Optional robust stddev:

```bash
--stddev-exclude-top 1
```

This excludes the highest-inefficiency occupied bin only from the stddev
calculation. The bin is still tested and can still be reported as a hot spot.

### Pveto Formula

The 2022/2023 legacy scripts use the histogram branch in
`LeptonBkgdEstimate.printPpassVetoTagProbe()` because
`_useHistogramsForPpassVeto=True` by default. That branch uses:

```text
Pveto = (N_pass_OS - N_pass_SS) / (N_total_OS - N_total_SS)
```

The older non-histogram fallback branch uses the electron/muon two-lepton
denominator:

```text
Pveto = N_pass / (2*N_total - N_pass)
```

Reference:

```text
DisappTrks/BackgroundEstimation/python/bkgdEstimate.py
LeptonBkgdEstimate.printPpassVetoTagProbe()
```

Nano implements the histogram-branch formula in:

```text
DisappTrks_Nano/src/disapptrks/lepton_backgrounds.py
```

### Poffline/Pmiss Track Selection

The near-lepton control-track mask lives in:

```text
DisappTrks_Nano/pocket_coffea/workflow.py::_lepton_background_track_mask
```

It currently requires:

- track `pt > 55 GeV`
- requested layer bin
- standard isolated-track quality
- `dR(track, jet) > 0.5`
- no missing-outer-hit requirement
- muon control: `caloEnergy < 10 GeV`
- electron control: no calorimeter-energy cut
- electron match: `0 <= dRMinElectron < 0.1`
- muon match: `0 <= dRMinMuon < 0.1`
- tau match: `0 <= dRMinTauHad < 0.1`

The `dR(track, jet) > 0.5` requirement was added to match the legacy
`ElectronTagPt55`/`MuonTagPt55` channels, which include `isoTrkCuts`.

## Known Issues And Gotchas

- `PackedSelection` slot exhaustion means too many categories are enabled.
  Use focused modes and disable diagnostics.
- Missing histogram-key errors mean `config.py` selected a variable that
  `workflow.py` did not create in that mode.
- Electron 2022CD values previously disagreed with AN Table 28. Fixes made:
  - electron/muon `Pveto` formula restored to the legacy histogram-branch
    direct OS-minus-SS ratio;
  - Poffline/Pmiss control-track mask now includes `dR(track, jet) > 0.5`.
- After changing the Poffline/Pmiss track mask, rerun the
  `*_pmiss_poffline` jobs. The `Pveto` formula change only requires rerunning
  postprocessing.
- After adding the legacy-style MET integration histograms, rerun the
  `*_pmiss_poffline` jobs again. The postprocessor prints
  `met_method=hist-integrated` when it found and used the new histograms.
- Older Pveto outputs can contain duplicate `n<Prefix>Background...`
  Poffline/Pmiss histograms. When such a Pveto output is combined with a
  dedicated `*_pmiss_poffline` output, the postprocessor now prefers the
  dedicated background output and ignores the duplicate Pveto-side background
  histograms for Poffline/Pmiss.
- New focused Pveto jobs no longer write Poffline/Pmiss background histograms by
  default. They are only added to Pveto outputs when
  `DISAPPTRKS_ENABLE_LEPTON_BACKGROUND_CATEGORIES=1` is explicitly set.
- Legacy 2022 electron estimates use a `MET lumi / EGamma lumi` prescale
  factor. Nano postprocessing exposes this as `--control-prescale`; the default
  is `1.0`.
- Omit `--tau-probability` for the current Nano tau background estimate:
  `tau_mu` uses the single-muon-trigger control region and `tau_ele` uses the
  single-electron-trigger control region. The optional
  `tau_trigger_probability` mode is only for a legacy/AN comparison involving
  muon+tau-trigger normalization.
- The final tau estimate combines the `tau_mu` and `tau_ele` legs before
  forming ratios. Use `disapptrks estimate-tau-background` with both
  `--tau-mu-files` and `--tau-ele-files`; it sums raw Pveto pair counts,
  raw Poffline/Pmiss components, raw epsilon components, and `N_ctrl` first.
  Do not combine finished per-leg `N_tau` estimates.
- The Pveto-output `n<Prefix>TriggerEff...` counters are used to reproduce the
  legacy `calculateTriggerEfficiencyFile()` epsilon divisor. Do not confuse
  this with `Pmiss`: `Pmiss` is the MET-trigger turn-on probability from
  Pmiss/Poffline histograms. Epsilon is layer-specific for `NLayers4`,
  `NLayers5`, and `NLayers6plus`; `combinedBins` uses the unsuffixed combined
  counters. `--trigger-efficiency` and `--trigger-efficiency-error` are manual
  overrides.
- Legacy `Pmiss` can use trigger-efficiency files via
  `useFilesForTriggerEfficiency()`, rather than a simple event-level MET HLT
  bit. Nano now duplicates this with histogram integration when the new
  Pmiss/Poffline histograms are available.
- For AN comparisons, the expected postprocessing diagnostics are
  `trigger_efficiency_method=legacy-tag-probe` and
  `met_method=hist-integrated`. `default` epsilon or `cutflow-ratio` MET means
  the input files are stale or incomplete for the nominal estimate.
- Do not pass `--trigger-efficiency` for normal production. That option is a
  manual override. Omitting it lets the extractor use the counters from the
  `*_pveto` output.

## Useful Validation Commands

```bash
python -m py_compile DisappTrks_Nano/pocket_coffea/config.py DisappTrks_Nano/pocket_coffea/workflow.py
python -m py_compile DisappTrks_Nano/src/disapptrks/lepton_backgrounds.py
python -m pytest DisappTrks_Nano/tests/test_lepton_backgrounds.py
git -C DisappTrks_Nano diff --check
```

Always check the worktree before editing:

```bash
git -C DisappTrks_Nano status --short
```

There may be unrelated dirty files. Do not revert unrelated local changes.
