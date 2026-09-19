# Codex Handoff: Background Estimate Development

This note is intended to be handed to another Codex session before continuing
development of the Run-3 DisappTrks PocketCoffea background-estimate workflow.
It summarizes the code structure, conventions, current modes, and known
debugging points.

Companion files:

- `docs/codex_memory_background_estimates.md`: compact project memory/state.
- `docs/codex_skills_background_estimates.md`: operating guidance for a Codex
  session working on this code.

## Current Handoff Status

The Nano lepton-background path now implements the pieces needed for the
AN-style estimate:

- fiducial-map production and loading for electron and muon veto hot spots
- focused `*_pveto` modes for `Pveto` and the legacy epsilon counters
- focused `*_pmiss_poffline` modes for `Poffline`/`Pmiss` control histograms
- postprocessing that uses the legacy-style histogram integrations when those
  histograms are available

Joyce's latest 2022CD electron `Poffline`/`Pmiss` check looked much closer to
the AN after the most recent fixes. The next developer should treat the current
implementation as the working baseline and continue validating against the AN
tables and the legacy code.

For a complete result, postprocess both the Pveto output and the
Pmiss/Poffline output together. Do not pass `--trigger-efficiency` unless doing
an explicit manual-override comparison. The expected successful diagnostics are:

```text
trigger_efficiency_method=legacy-tag-probe
met_method=hist-integrated
```

## Repository Context

There are two related codebases under the same parent directory:

- `DisappTrks/`: the legacy CMSSW/OSUT3 DisappTrks analysis code.
- `DisappTrks_Nano/`: the newer custom NanoAOD plus PocketCoffea migration.

When implementing or debugging the Nano workflow, use the legacy code as the
physics reference, especially:

- `DisappTrks/BackgroundEstimation/python/`
- `DisappTrks/BackgroundEstimation/test/bkgdEstimate_2022.py`
- `DisappTrks/StandardAnalysis/python/EventSelections.py`
- `DisappTrks/StandardAnalysis/python/Cuts.py`

The Nano code should keep AN terminology in public names where possible:
`Pveto`, `Poffline`, `Pmiss`, `basic_selection`,
`isolated_track_selection`, `candidate_track_selection`, and
`disappearing_track_selection`.

## Main Nano Files

| File | Role |
| --- | --- |
| `src/disapptrks/selections.py` | Physics definitions on Awkward arrays: object masks, tag definitions, probe-track masks, pair builders, AN-style cutflow masks. |
| `pocket_coffea/workflow.py` | Builds derived event/object collections and event-level counters during PocketCoffea processing. |
| `pocket_coffea/cuts.py` | PocketCoffea `Cut` objects, usually thin wrappers around counts or booleans stored by `workflow.py`. |
| `pocket_coffea/config.py` | Dataset filtering, category mode routing, category selection, histogram variable selection. |
| `src/disapptrks/lepton_backgrounds.py` | Computes final lepton background estimates from `Pveto`, `Poffline`, and `Pmiss` inputs. |
| `src/disapptrks/tables.py` | Cutflow and Pveto table extraction/formatting. |
| `src/disapptrks/cli.py` | CLI commands for fiducial maps, Pveto tables, fake-track tables, and lepton-background estimates. |
| `docs/pocket_coffea_workflows.md` | Collaborator-facing workflow guide. Keep it updated when modes or commands change. |

Rule of thumb:

- Put reusable physics masks in `selections.py`.
- Put derived collections and counters in `workflow.py`.
- Put named PocketCoffea categories in `cuts.py` and `config.py`.
- Put postprocessing formulas in `lepton_backgrounds.py` or `tables.py`.

## Current Production Modes

The PocketCoffea mode is selected with:

```bash
DISAPPTRKS_CATEGORY_MODE=<mode>
```

Important modes:

| Mode | Dataset | Purpose |
| --- | --- | --- |
| `fiducial_maps` | `DATA_Muon` or `DATA_EGamma` | Produce before/after eta-phi histograms used to build electron and muon fiducial-map JSON/NPZ files. |
| `muon_pveto` | `DATA_Muon` | Muon `Pveto` tag-probe pairs and categories. |
| `electron_pveto` | `DATA_EGamma` | Electron `Pveto` tag-probe pairs and categories. |
| `tau_mu_pveto` | `DATA_Muon` | Tau `Pveto` measurement using muon low-`MT` tags. |
| `tau_ele_pveto` | `DATA_EGamma` | Tau `Pveto` measurement using electron low-`MT` tags. |
| `muon_pmiss_poffline` | `DATA_Muon` | Muon `Poffline` and `Pmiss` control categories only. |
| `electron_pmiss_poffline` | `DATA_EGamma` | Electron `Poffline` and `Pmiss` control categories only. |
| `tau_mu_pmiss_poffline` | `DATA_Muon` | Legacy-equivalent single-muon-triggered tau control for `Nctrl`, `Poffline`, and `Pmiss`; no low-`MT` muon is required. |
| `tau_ele_pmiss_poffline` | `DATA_EGamma` | Compatibility/diagnostic only; excluded from the final tau normalization. |
| `tau_pmiss_poffline` | `DATA_Muon` | AN-style tau normalization using the muon+tau cross-trigger; no low-`MT` offline muon is required. |
| `tau_trigger_probability` | `DATA_Tau` or tau-trigger dataset | Optional legacy/AN diagnostic counts for a muon+tau-trigger normalization. Not used by the current Nano tau_mu/tau_ele single-lepton-trigger control regions. |
| `fake_tracks` | `DATA_JetMET`, `DATA_MET`, `DATA_Muon`, or `DATA_EGamma` | Fake-track estimate control regions. |

Avoid using `muon_backgrounds`, `egamma_backgrounds`, or `all` for production
unless explicitly debugging. They can overload PocketCoffea/Coffea because they
select too many categories and can exhaust `PackedSelection` slots.

## Fiducial Maps

The fiducial-map job stores:

- `ElectronFiducialBefore`
- `ElectronFiducialAfter`
- `MuonFiducialBefore`
- `MuonFiducialAfter`

The JSON/NPZ maps are built from the `.coffea` output with:

```bash
disapptrks make-fiducial-map \
  --flavor electron \
  --output-json /path/to/electron_fiducial_map.json \
  /path/to/fiducial_map_output/output_*.coffea
```

or `--flavor muon`.

Hot spots use the default threshold:

```bash
--threshold 2.0
```

For eras with one pathological high-inefficiency bin inflating the stddev, use:

```bash
--stddev-exclude-top 1
```

This removes the most extreme occupied bin only from the stddev calculation.
The bin is still tested and reported as a hot spot.

Pveto jobs load maps with either explicit paths:

```bash
DISAPPTRKS_ELECTRON_FIDUCIAL_MAP_JSON=/path/to/electron_fiducial_map.json
DISAPPTRKS_MUON_FIDUCIAL_MAP_JSON=/path/to/muon_fiducial_map.json
```

Leg-specific tau jobs only require the fiducial map for the active leg:
`tau_ele_pveto` and `tau_ele_pmiss_poffline` use the electron map, while
`tau_mu_pveto` and `tau_mu_pmiss_poffline` use the muon map. The directory form
can still be used when both files are available.

Fake-track jobs are not leg-specific. Legacy fake-track selections inherited
both `cutTrkFiducialElectron` and `cutTrkFiducialMuon` from `isoTrkCuts`, in
addition to `cutTrkFiducialECAL`. Nano therefore applies both electron and muon
hot-spot maps to fake-track candidates. For production `fake_tracks` jobs with
`DISAPPTRKS_REQUIRE_FIDUCIAL_MAPS=1`, provide both map JSONs or use
`DISAPPTRKS_FIDUCIAL_MAP_DIR`.

or a directory:

```bash
DISAPPTRKS_FIDUCIAL_MAP_DIR=/path/to/fiducial_maps
```

The directory must contain exactly:

```text
electron_fiducial_map.json
muon_fiducial_map.json
```

For production validation, set:

```bash
DISAPPTRKS_REQUIRE_FIDUCIAL_MAPS=1
```

This makes the job fail if the maps cannot be loaded or contain no hot spots.

## Lepton Background Formula

The AN-style estimate is:

```text
N_lepton = N_ctrl * Pveto * Poffline * Pmiss / epsilon_trig^lepton
```

In Nano postprocessing this is handled by:

```text
src/disapptrks/lepton_backgrounds.py
```

Definitions:

- `N_ctrl`: control yield category, e.g.
  `electron_background_control_NLayers4`.
- `Poffline`: ratio of offline-MET-passing control events to control events.
- `Pmiss`: ratio of MET-trigger-passing control events to offline-MET-passing
  control events.
- `Pveto`: probability from the tag-probe pair categories.
- `epsilon_trig^lepton`: separate trigger-efficiency divisor from the legacy
  `calculateTriggerEfficiencyFile()` path. Nano calculates it from the Pveto
  output counters `n<Prefix>TriggerEffProbesPT55`,
  `n<Prefix>TriggerEffProbesSSPT55`,
  `n<Prefix>TriggerEffProbesFiringTrigger`, and
  `n<Prefix>TriggerEffSSProbesFiringTrigger`, using the same OS-minus-SS
  formula as legacy. This is separate from `Pmiss`, which handles the
  MET-trigger turn-on. For `NLayers4`, `NLayers5`, and `NLayers6plus`, Nano
  uses the suffixed layer-specific counters; for `combinedBins`, it uses the
  unsuffixed combined counters.

`N_ctrl` can also be scaled with `--control-prescale`. This matches the legacy
MET/lepton-dataset luminosity or prescale correction. The default is `1.0`.
For the AN-style Nano tau background, supply the era-specific `--tau-probability`
correction because the normalization uses the muon+tau HLT path. The supporting
trigger ratio can be measured with
`DISAPPTRKS_CATEGORY_MODE=tau_trigger_probability`, then extracted with
`disapptrks extract-tau-trigger-probability`.

The final tau estimate combines the tau-muon and tau-electron legs for
`P_veto`, while using only the cross-triggered `tau_pmiss_poffline` control for the
normalization and MET probabilities. Use `disapptrks estimate-tau-background`,
not two separate `estimate-lepton-background --mode tau_*` final tables:

- `Pveto` OS/SS tag-probe pair counts.
- `N_ctrl` from `tau_pmiss_poffline`.
- `Poffline` and `Pmiss` components from `tau_pmiss_poffline`.
- `epsilon_trig^tau` tag-probe trigger-efficiency numerator/denominator
  components.

Then it forms the ratios and final `N_tau`.

Example:

```bash
disapptrks estimate-tau-background \
  --run-period 2022CD \
  --output-json tables/tau_background_2022CD.json \
  --output-tex tables/tau_background_2022CD.tex \
  --tau-control-files \
    analysis_output/2022CD_tau_pmiss_poffline/output_*.coffea \
  --tau-mu-files \
    analysis_output/2022CD_tau_mu_pveto/output_*.coffea \
  --tau-ele-files \
    analysis_output/2022CD_tau_ele_pveto/output_*.coffea
```

The defaults are `--tau-mu-sample DATA_Muon` and
`--tau-ele-sample DATA_EGamma`.

Current category naming:

```text
<mode>_background_control_{layer}
<mode>_background_offline_{layer}
<mode>_background_trigger_{layer}
```

where `<mode>` is `muon`, `electron`, `tau_mu`, or `tau_ele`, and `{layer}` is
`NLayers4`, `NLayers5`, `NLayers6plus`, or `combinedBins`.

## Important Legacy Convention For Pveto

For the 2022/2023 background scripts, the legacy code uses the histogram-based
branch for `Pveto`:

```text
Pveto = (N_pass_OS - N_pass_SS) / (N_total_OS - N_total_SS)
```

The electron/muon two-lepton denominator,
`N_pass / (2*N_total - N_pass)`, appears only in the older non-histogram
fallback branch.

Reference:

```text
DisappTrks/BackgroundEstimation/python/bkgdEstimate.py
LeptonBkgdEstimate.printPpassVetoTagProbe()
```

The relevant legacy branch is:

```python
if (self._flavor == "electron" or self._flavor == "muon") and not self._useHistogramsForPpassVeto:
    eff = scaledPasses / (2.0 * total - scaledPasses)
else:
    eff = scaledPasses / total
```

For the 2022/2023 scripts, `_useHistogramsForPpassVeto=True` by default, so
electron, muon, and tau all use the direct histogram-branch ratio after
same-sign subtraction:

```text
Pveto = (N_pass_OS - N_pass_SS) / (N_total_OS - N_total_SS)
```

The two-lepton denominator is only for the older non-histogram fallback branch.

Nano implements this in `src/disapptrks/lepton_backgrounds.py` via
`pveto_count_from_pair_counts(...)`.

## Poffline/Pmiss Control Selection

The control-only modes build the relevant tag collection and count events with
near-lepton tracks. The core helper is:

```text
pocket_coffea/workflow.py::_lepton_background_track_mask
```

Current behavior:

- track `pt > 55 GeV`
- requested layer bin
- standard isolated-track quality
- `dR(track, jet) > 0.5`
- no missing-outer-hit requirement
- for muons: `caloEnergy < 10 GeV`
- for electrons: no calorimeter-energy cut
- electron control: `0 <= dRMinElectron < 0.1`
- muon control: `0 <= dRMinMuon < 0.1`
- tau control: `0 <= dRMinTauHad < 0.1`

The `dR(track, jet) > 0.5` requirement is important. It was added because the
legacy `ElectronTagPt55` and `MuonTagPt55` control channels include
`isoTrkCuts`, which include `cutTrkJetDeltaPhi`.

`Poffline` and `Pmiss` are counted in:

```text
pocket_coffea/workflow.py::_store_lepton_background_controls
```

The control denominator mirrors the legacy `ElectronTagPt55`, `MuonTagPt55`,
and `TauTagPt55` event-side baseline:

- event quality: golden JSON, MET filters, jet-veto map
- at least one tag
- at least one jet with `pt > 110 GeV`, `|eta| < 2.4`, and tight-lepton-veto ID
- maximum dijet delta phi `< 2.5`

It intentionally does not include ordinary `MetNoMu > 120` or ordinary
`deltaPhi(MetNoMu, leading jet) > 0.5`; the legacy code applies the
lepton-removed MET and delta-phi requirements through the Poffline/Pmiss
histogram integrals.

The offline numerator adds:

- MET-no-mu-minus-selected-lepton `pt >= 120 GeV`
- leading-jet delta phi to that MET direction `>= 0.5`

The trigger event is:

```text
offline_event AND _met_trigger_mask(events)
```

New outputs from the `*_pmiss_poffline` modes also store the histograms needed
to reproduce the legacy integration:

```text
n<Prefix>BackgroundMetNoMuPt_{layer}
n<Prefix>BackgroundMetNoMuPtTrig_{layer}
n<Prefix>BackgroundMetMinusOnePt_{layer}
n<Prefix>BackgroundMetMinusOnePtTrig_{layer}
n<Prefix>BackgroundDeltaPhiMetJetLeadingVsMetMinusOnePt_{layer}
```

where `<Prefix>` is `Muon`, `Electron`, `TauMu`, or `TauEle`.

The postprocessor uses these histograms automatically when present. It builds a
MET-trigger turn-on from ordinary no-muon MET, weights the 2D lepton-removed MET
versus delta-phi histogram, and integrates the region above `--met-cut` and
`--phi-cut`. This is the nominal path for AN comparisons. If the histograms are
missing, it falls back to the older scalar cutflow ratios and prints
`met_method=cutflow-ratio`.

After changing this code, rerun the relevant `*_pmiss_poffline` jobs. Existing
Pveto outputs can be reused for `Pveto`, but do not use their
`n<Prefix>Background...` histograms for Poffline/Pmiss if those histograms are
also present in the dedicated `*_pmiss_poffline` output. The postprocessor now
protects against this by preferring background-histogram outputs that do not
also contain Pveto tag-probe pair histograms.

Pveto outputs contain `n<Prefix>TriggerEff...` counters used to reproduce the
legacy `calculateTriggerEfficiencyFile()` epsilon divisor. This is not the same
quantity as `Pmiss`: `Pmiss` is the MET-trigger turn-on probability from the
Pmiss/Poffline histograms.

Quick output sanity check:

```bash
python - <<'PY'
from coffea.util import load
out = load("pocket_coffea/analysis_output/2022CD_electron_pmiss_poffline/output_all.coffea")
for key in sorted(str(k) for k in out.get("variables", {}).keys()):
    if "BackgroundMet" in key or "BackgroundDeltaPhi" in key:
        print(key)
PY
```

## Typical Commands

Electron Pveto:

```bash
cd DisappTrks_Nano/pocket_coffea
DISAPPTRKS_CATEGORY_MODE=electron_pveto \
DISAPPTRKS_ENABLE_PVETO_DIAGNOSTICS=0 \
DISAPPTRKS_REQUIRE_FIDUCIAL_MAPS=1 \
DISAPPTRKS_DATASET_JSON=datasets/eos_2022CD_EGamma.json \
DISAPPTRKS_FIDUCIAL_MAP_DIR=/path/to/fiducial_maps/2022CD \
python -m pocket_coffea.scripts.runner run \
  --cfg config.py \
  --outputdir analysis_output/2022CD_electron_pveto \
  --executor dask@lpc \
  --executor-custom-setup executors_lpc.py \
  --custom-run-options run_options_lpc_dask.yaml \
  --scaleout 60 \
  --queue workday
```

Electron Poffline/Pmiss:

```bash
cd DisappTrks_Nano/pocket_coffea
DISAPPTRKS_CATEGORY_MODE=electron_pmiss_poffline \
DISAPPTRKS_ENABLE_PVETO_DIAGNOSTICS=0 \
DISAPPTRKS_DATASET_JSON=datasets/eos_2022CD_EGamma.json \
python -m pocket_coffea.scripts.runner run \
  --cfg config.py \
  --outputdir analysis_output/2022CD_electron_pmiss_poffline \
  --executor dask@lpc \
  --executor-custom-setup executors_lpc.py \
  --custom-run-options run_options_lpc_dask.yaml \
  --scaleout 60 \
  --queue workday
```

Postprocess electron background:

```bash
cd DisappTrks_Nano
disapptrks estimate-lepton-background \
  --mode electron \
  --run-period 2022CD \
  --output-json tables/electron_background_2022CD.json \
  --output-tex tables/electron_background_2022CD.tex \
  pocket_coffea/analysis_output/2022CD_electron_pveto/output_*.coffea \
  pocket_coffea/analysis_output/2022CD_electron_pmiss_poffline/output_*.coffea
```

For a smoke test, add this to the runner command:

```bash
--limit-files 1 --limit-chunks 1 --scaleout 2 --queue microcentury
```

## Debugging Checklist

If a job fails with:

```text
RuntimeError: Exhausted all slots in PackedSelection
```

then the selected category set is too large. Use a focused mode such as
`electron_pveto` or `electron_pmiss_poffline`, and keep diagnostics disabled:

```bash
DISAPPTRKS_ENABLE_PVETO_DIAGNOSTICS=0
DISAPPTRKS_ENABLE_SEARCH_DIAGNOSTICS=0
```

If histogram filling fails with a missing key like:

```text
ValueError: key "nElectronFiducialHotSpotsLoaded" does not exist
```

then `config.py` selected a histogram variable that `workflow.py` did not
create in that mode. Fix either the variable filtering in `_variables_for_mode`
or create the event field in `apply_object_preselection`/`count_objects`.

If the fiducial map seems to have no effect:

- Run with `DISAPPTRKS_REQUIRE_FIDUCIAL_MAPS=1`.
- Check `nElectronFiducialHotSpotsLoaded` and
  `nMuonFiducialHotSpotsLoaded`.
- For muon Pveto, compare:
  `nMuonPVetoTagProbePairZWindowPassNoFiducial` and
  `nMuonPVetoTagProbePairZWindowFiducialRejected`.

If electron 2022CD numbers do not match AN Table 28:

- Check the `Pveto` convention first:
  2022/2023 should use the direct histogram-branch OS-minus-SS ratio.
- Check that the Poffline/Pmiss control track includes `dR(track, jet) > 0.5`.
- Check the legacy prescale/luminosity factor:
  `MET lumi / EGamma lumi` is used in legacy 2022 electron estimates.
- Check whether legacy `Pmiss` used trigger-efficiency files rather than a
  direct MET-HLT event bit.
- Compare the exact electron tag definition:
  Nano uses `Electron.cutBased >= 4`, dxy/dz barrel/endcap cuts, `pt > 35`,
  `|eta| < 2.1`, and the single-electron HLT mask.

## Development Safety

The working tree may contain unrelated local changes. Before editing, run:

```bash
git -C DisappTrks_Nano status --short
```

Do not revert unrelated files. Keep edits tightly scoped, and use:

```bash
python -m py_compile <changed .py files>
python -m pytest DisappTrks_Nano/tests/test_lepton_backgrounds.py
git -C DisappTrks_Nano diff --check
```

for quick validation.

## Current Validation Items

The Nano implementation now has the full plumbing for fiducial maps, `Pveto`,
`Poffline`, and `Pmiss`. The remaining work is validation and polishing rather
than inventing missing infrastructure:

- Compare all layer bins and combined bins to the AN tables after rerunning
  both the relevant `*_pveto` and `*_pmiss_poffline` jobs with current code.
- Confirm whether a non-unity legacy `MET lumi / lepton lumi` prescale should
  be passed with `--control-prescale` for each run period and channel.
- Do not pass `--tau-probability` for the current Nano tau background estimate
  unless intentionally doing a legacy/AN muon+tau-trigger-normalization
  comparison.
- Keep checking that postprocessing reports
  `trigger_efficiency_method=legacy-tag-probe`
  and `met_method=hist-integrated`; otherwise the inputs were produced with
  older code or the wrong mode.
- If a remaining discrepancy appears, compare the exact legacy channel in
  `DisappTrks/BackgroundEstimation/python/*TagProbeSelections.py` and
  `DisappTrks/BackgroundEstimation/python/bkgdEstimate.py` before changing
  Nano logic.
- The exact Run-3 electron veto object used for fiducial maps should remain
  under review: Nano currently uses `Electron.cutBased >= 1` for veto electrons.
- The exact Run-3 tau object working point should remain under review:
  `hadronic_tau_veto_object_mask` uses DeepTau 2018v2p5 raw thresholds and
  `idDecayModeNewDMs`.
