# Codex Skills: DisappTrks Nano Background Estimate Work

Use this as operating guidance for a Codex session working on the
DisappTrks Nano/PocketCoffea background estimates.

## How To Approach Changes

1. Read the Nano implementation first.
2. Cross-check the legacy DisappTrks code for physics definitions.
3. Keep AN terminology in user-facing names and docs.
4. Make small, scoped edits.
5. Validate with compile checks, focused tests, and a one-file PocketCoffea
   smoke test when possible.

Prefer `rg` for search:

```bash
rg -n "pattern" DisappTrks_Nano DisappTrks
```

Use `apply_patch` for manual file edits. Do not revert unrelated dirty files.

## Where To Put New Code

- New or changed physics mask:
  `DisappTrks_Nano/src/disapptrks/selections.py`
- New derived collection, pair collection, or event count:
  `DisappTrks_Nano/pocket_coffea/workflow.py`
- New PocketCoffea category:
  `DisappTrks_Nano/pocket_coffea/cuts.py`
  and `DisappTrks_Nano/pocket_coffea/config.py`
- New histogram variable:
  `DisappTrks_Nano/pocket_coffea/config.py`
- New table or estimate formula:
  `DisappTrks_Nano/src/disapptrks/tables.py`
  or `DisappTrks_Nano/src/disapptrks/lepton_backgrounds.py`
- New CLI behavior:
  `DisappTrks_Nano/src/disapptrks/cli.py`
- New workflow documentation:
  `DisappTrks_Nano/docs/pocket_coffea_workflows.md`

## Legacy Reference Search Patterns

For electron backgrounds:

```bash
rg -n "ElectronTagPt55|ZtoEleProbeTrk|printPpassVetoTagProbe|printPpassMetCut|printPpassMetTriggers" DisappTrks/BackgroundEstimation DisappTrks/StandardAnalysis
```

For muon backgrounds:

```bash
rg -n "MuonTagPt55|ZtoMuProbeTrk|printPpassVetoTagProbe|printPpassMetCut|printPpassMetTriggers" DisappTrks/BackgroundEstimation DisappTrks/StandardAnalysis
```

For tau backgrounds:

```bash
rg -n "ZtoTau|TauTag|tau.*pveto|TauTagProbeSelections" DisappTrks/BackgroundEstimation DisappTrks/StandardAnalysis
```

For common track selections:

```bash
rg -n "isoTrkCuts|candTrkCuts|disTrkCuts|cutTrkPt55|cutTrkJetDeltaPhi|cutTrkNMissOut" DisappTrks/StandardAnalysis/python
```

## PocketCoffea Mode Development

When adding a new mode:

1. Add the mode to `_skim_cuts_for_mode()` in `pocket_coffea/config.py`.
2. Add a `selected_categories` branch in `pocket_coffea/config.py`.
3. Add minimal histogram prefixes in `_variables_for_mode()`.
4. Make `workflow.py::_mode_enabled()` or mode checks build only the required
   objects and counts.
5. Add documentation in `docs/pocket_coffea_workflows.md`.

Keep production modes narrow. If a mode only needs Poffline/Pmiss, do not build
full Pveto tag-probe pair categories.

## Cutflow Debugging Pattern

If a category is missing or zero:

1. Find its `Cut` in `pocket_coffea/cuts.py`.
2. Identify the event field the cut reads, usually `nSomething`.
3. Check where that field is created in `pocket_coffea/workflow.py`.
4. Check whether the active `DISAPPTRKS_CATEGORY_MODE` builds that object.
5. Check whether `config.py::_variables_for_mode()` selects histograms whose
   fields exist in that mode.

Common failure:

```text
ValueError: key "nSomeField" does not exist
```

Fix by either:

- creating `events["nSomeField"]` in that mode; or
- removing that histogram from the minimal variable set for the mode.

## PackedSelection Debugging Pattern

Common failure:

```text
RuntimeError: Exhausted all slots in PackedSelection
```

Likely cause: too many categories/cuts selected at once.

Recommended response:

- Use focused modes such as `electron_pveto` or
  `electron_pmiss_poffline`.
- Disable diagnostic category sets:

```bash
DISAPPTRKS_ENABLE_PVETO_DIAGNOSTICS=0
DISAPPTRKS_ENABLE_SEARCH_DIAGNOSTICS=0
```

- Avoid `all`, `muon_backgrounds`, and `egamma_backgrounds` for production.

## Lepton Background Postprocessing

The normal final step is:

```bash
disapptrks estimate-lepton-background \
  --mode electron \
  --run-period 2022CD \
  --output-json tables/electron_background_2022CD.json \
  --output-tex tables/electron_background_2022CD.tex \
  pocket_coffea/analysis_output/2022CD_electron_pveto/output_*.coffea \
  pocket_coffea/analysis_output/2022CD_electron_pmiss_poffline/output_*.coffea
```

Use `--mode muon` for muons. For the final tau background, use the dedicated
combined command instead of making separate tau-muon and tau-electron final
estimates:

```bash
disapptrks estimate-tau-background \
  --run-period 2022CD \
  --output-json tables/tau_background_2022CD.json \
  --output-tex tables/tau_background_2022CD.tex \
  --tau-mu-files \
    pocket_coffea/analysis_output/2022CD_tau_mu_pveto/output_*.coffea \
    pocket_coffea/analysis_output/2022CD_tau_mu_pmiss_poffline/output_*.coffea \
  --tau-ele-files \
    pocket_coffea/analysis_output/2022CD_tau_ele_pveto/output_*.coffea \
    pocket_coffea/analysis_output/2022CD_tau_ele_pmiss_poffline/output_*.coffea
```

That command sums the raw tau-muon and tau-electron ingredients before forming
`Pveto`, `Poffline`, `Pmiss`, `epsilon_trig^tau`, and final `N_tau`.
Only pass `--trigger-efficiency` and `--trigger-efficiency-error` for an
explicit manual-override comparison.

Remember:

- `Pveto` comes from tag-probe pair counts.
- `Poffline` and `Pmiss` normally come from the legacy-style MET histograms in
  the `*_pmiss_poffline` output. The scalar background
  offline/control/trigger counts are a fallback and diagnostic cross-check.
- New `*_pmiss_poffline` outputs include MET-shape histograms. The extractor
  uses them automatically for the legacy-style trigger-turn-on integration and
  prints `met_method=hist-integrated`.
- If the extractor prints `met_method=cutflow-ratio`, rerun the
  `*_pmiss_poffline` job with current code.
- If older Pveto outputs also contain `n<Prefix>Background...` histograms, do
  not sum them with the dedicated Pmiss/Poffline histograms. The extractor now
  prefers dedicated background-only outputs when both kinds are passed.
- Current 2022/2023 histogram-based `Pveto` uses the direct OS-minus-SS ratio.
  The electron/muon two-lepton denominator is only for the older non-histogram
  fallback branch in legacy.
- Use the Pveto tag-probe trigger-matching counters for the separate legacy
  epsilon divisor. The MET-trigger turn-on probability is handled separately by
  `Pmiss`. `--trigger-efficiency` is only a manual override.
- Use `--control-prescale` for the legacy MET/lepton-dataset luminosity or
  prescale factor when it is not unity.
- For tau, combine `tau_mu` and `tau_ele` with `estimate-tau-background`; do
  not add finished per-leg `N_tau` values.
- Omit `--tau-probability` for the current Nano tau background estimate because
  the tau legs use single-muon and single-electron control triggers. Use
  `tau_trigger_probability` only for an explicit legacy/AN muon+tau-trigger
  normalization comparison.

Expected nominal diagnostics:

```text
trigger_efficiency_method=legacy-tag-probe
met_method=hist-integrated
```

If those are not printed, debug the inputs before tuning physics cuts.

## Minimum Validation Before Handing Back

For Python-only changes:

```bash
python -m py_compile <changed files>
git -C DisappTrks_Nano diff --check
```

For lepton-background formula changes:

```bash
python -m pytest DisappTrks_Nano/tests/test_lepton_backgrounds.py
```

For fiducial-map changes:

```bash
python -m pytest DisappTrks_Nano/tests/test_fiducial.py
```

For PocketCoffea workflow/config changes, recommend a smoke test on LPC:

```bash
--limit-files 1 --limit-chunks 1 --scaleout 2 --queue microcentury
```

## Documentation Discipline

When a selection or formula changes, update at least one of:

- `docs/pocket_coffea_workflows.md`
- `docs/codex_handoff_background_estimates.md`
- this skills file, if the working procedure changes

Do not leave new mode names, environment variables, or AN-convention changes
only in code.
