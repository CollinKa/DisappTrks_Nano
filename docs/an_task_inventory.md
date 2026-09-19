# AN-24-155 analysis task inventory

Source: `Disappearing_Tracks_Run_3_Analysis (4).pdf`, dated 2026-05-29.

This is a clean inventory of work required to complete the analysis documented in the AN and its intended 2022–2026 extension. The PDF contains a mature 2022–2023 baseline plus some later-period material, but the target scope is 2022, 2023, 2024, 2025, and 2026 using custom NanoAOD inputs. It does **not** import statuses from the previous dashboard. A task is marked complete only when an explicit deliverable and its validation evidence exist.

## Current progress mapping (2026-08-04)

The status mapping below records the collaboration update given on 2026-08-04. The legacy 2022–2023 implementation and its results are accepted as the reference for the custom-NanoAOD migration. The targeted validation consists of electron, muon, and tau P_veto comparisons and the cutflow for one representative signal-MC sample. “Validated against legacy” means these checks produced comparable results within reason; it does not mean every signal point was compared or that final 2022–2026 estimates and uncertainties are frozen.

| Task ID | Status | Current evidence / interpretation | Remaining work |
|---|---|---|---|
| SCOPE-01 | partially complete | The target analysis scope is 2022–2026. Legacy 2022–2023 results are the accepted migration reference. | Freeze detailed run-period groupings, luminosities, and dataset coverage for 2024–2026. |
| SCOPE-02 | complete | The later-year content is intentional: the analysis is intended to include 2024, 2025, and 2026 after migration to custom NanoAOD. | Represent later years as target scope rather than treating the 2024 row as anomalous. |
| PROD-02 | in progress, nearly complete | Nearly all data have been processed into custom NanoAOD and the output files are available. | Identify the missing production units, finish or formally waive them, and run completeness/integrity checks before marking complete. |
| PROD-06 | not yet demonstrated | Availability of NanoAOD files is established, but a complete schema/branch validation report was not reported. | Validate required branches across all produced data files. |
| PROD-08 | not yet demonstrated | Production is nearly complete, but duplicate-event, bad-file, and full event-coverage checks were not reported. | Produce the integrity and coverage report. |
| SEL-01 | complete / accepted | Physics-object definitions and overlap rules have been validated in the NanoAOD implementation. | Preserve the comparison artifacts and configuration provenance. |
| SEL-02 | complete / accepted | Event cleaning and missing-momentum filters have received final validation. | Preserve the validation artifact and frozen configuration reference. |
| SEL-03 | complete / accepted | The basic event selection has received final validation. | Preserve the supporting cutflows/distributions. |
| SEL-04 | complete / accepted | The isolated-track selection has received final validation. | Preserve the supporting cutflows/distributions. |
| SEL-05 | complete / accepted | The candidate-track lepton and jet separation vetoes have received final validation. | Preserve the supporting cutflows/distributions. |
| SEL-06 | complete / accepted | The disappearing-track requirements have received final validation. | Preserve the supporting cutflows/distributions. |
| SEL-07 | complete / accepted | The 4, 5, and >=6 tracker-layer categorization has received final validation. | Preserve the bin-validation evidence. |
| SEL-08 | complete / accepted | Electron-reconstruction fiducial maps and the distinct ECAL veto maps are trusted as good. | Preserve the frozen map artifacts/versions and their provenance for dashboard evidence. |
| SEL-09 | complete / accepted | Muon fiducial maps are trusted as good. | Preserve the frozen map artifact/version and its provenance for dashboard evidence. |
| SEL-10 | complete / accepted | Tracker/jet-veto maps are trusted as good. | Preserve the frozen map artifact/version and its provenance for dashboard evidence. |
| SEL-13 | complete for migration validation | One representative signal-MC cutflow reproduced the legacy result within reason using NanoAOD. | Preserve the comparison and code versions. Broader signal-grid validation remains under SIG-08/SIG-09. |
| TRIG-04 | in progress | The current P_veto jobs now also calculate the lepton control-trigger efficiency epsilon_trigger^l. | Finish every required period/flavor/layer instance, validate the efficiencies against the accepted reference or an independent check, and freeze the outputs. |
| BGLEP-03 | validated against legacy; rerun in progress | NanoAOD code produced electron, muon, and tau P_veto values comparable to the trusted legacy result. The P_veto jobs are now being rerun and also calculate epsilon_trigger^l. | Finish the reruns for every required period/flavor/layer bin, record the P_veto and trigger-efficiency validation, and freeze the outputs. |
| BGLEP-05 | in progress | P_offline / pTmiss-related jobs are being rerun as inputs to the lepton background estimates. | Finish all required period/flavor/layer instances, validate against the legacy reference, and freeze outputs. |
| BGLEP-06 | in progress | P_trigger is included in the current lepton-background jobs. | Finish all required period/flavor/layer instances, validate against the legacy reference, and freeze outputs. |
| BGLEP-08 | in progress | Constituent lepton-background jobs are being rerun. | Combine final N_ctrl, P_veto, P_offline, P_trigger, and epsilon_trigger inputs for electron, muon, and tau estimates with propagated uncertainties. |
| BGFAKE-01 | validated against legacy for the first estimation stage | The first step of the fake-track estimation in NanoAOD gives results comparable to the trusted legacy code. | Attach the exact artifact/comparison and identify whether this establishes the control yield, transfer-factor input, or both. |
| BGFAKE-02 | partial / status to disambiguate | “First step” may include the d_xy sideband or transfer-factor setup, but the update is not specific enough to mark the transfer factor complete. | Map the completed output to BGFAKE-02 or BGFAKE-03 and finish the other component. |
| BGFAKE-03 | partial / status to disambiguate | Same evidence as BGFAKE-02. | Record whether the completed first step is the control yield. |
| BGFAKE-04 | not yet complete | Only the first step of the fake-track method was reported as reproduced. | Complete the transfer factor, control yield, nominal estimate, uncertainty propagation, and validation. |

### Current summary

- **Target analysis scope:** 2022–2026 using custom NanoAOD files.
- **Accepted migration reference:** legacy 2022–2023 results.
- **Migration validation achieved:** comparable NanoAOD electron, muon, and tau P_veto values; a comparable cutflow for one representative signal-MC sample; and comparable output for the first fake-track-estimation stage.
- **Production:** custom data NanoAOD is available and nearly complete, but completeness and integrity certification remain open.
- **Accepted detector inputs:** electron, muon, ECAL, and tracker/jet-veto maps.
- **Selection validation complete:** event cleaning, the full search-selection chain, and layer categorization. Migration cutflow validation used one representative signal-MC sample.
- **Active work:** rerunning P_veto jobs (which now also calculate epsilon_trigger^l) and P_offline/pTmiss jobs (including P_trigger) toward final lepton background estimates.
- **Not implied by this update:** final lepton or fake-track estimates, closure tests, systematic uncertainties, statistical-model readiness, or unblinding readiness.

## Scope and task dimensions

- Mature baseline currently documented in the AN: 2022 and 2023 data, 62.5 fb^-1.
- Target analysis scope: 2022, 2023, 2024, 2025, and 2026.
- Baseline run periods used in the current result: 2022 CD, 2022 EFG, 2023 C, and 2023 D. Later-year groupings remain to be frozen.
- Search bins: tracker layers with measurement = 4, 5, and >=6.
- Signal models: wino-like and higgsino-like chargino scenarios over mass and lifetime grids.
- Instrumental backgrounds: electrons, muons, tau/single-hadron tracks, and fake tracks.
- The dashboard should distinguish `analysis task`, `deliverable`, and `validation/review gate`. Producing a number is not the same as validating and approving it.

## 0. Scope, configuration, and reproducibility

| ID | Task | Completion evidence | Depends on |
|---|---|---|---|
| SCOPE-01 | Freeze the 2022–2026 analysis scope and run-period grouping | Approved list of periods, luminosities, datasets, and layer bins | — |
| SCOPE-02 | Record 2024–2026 as the intended extension beyond the mature 2022–2023 AN baseline | Consistent dashboard scope and AN plan covering all target years | SCOPE-01 |
| SCOPE-03 | Freeze the dataset manifest for data, signal MC, and background MC | Versioned manifest containing dataset names, processing versions, event counts, and file locations | SCOPE-01 |
| SCOPE-04 | Freeze the trigger menu by run period | Versioned mapping of L1/HLT paths and run applicability | SCOPE-01 |
| SCOPE-05 | Freeze object definitions and all four cumulative selections (basic, isolated-track, candidate-track, disappearing-track) | Versioned configuration matching AN tables | SCOPE-01 |
| SCOPE-06 | Freeze signal mass/lifetime grids and production-mode definitions for wino and higgsino models | Versioned signal grid with cross sections and branching assumptions | SCOPE-03 |
| SCOPE-07 | Establish reproducible software/environment provenance | Commit, release/container, correction versions, and executable run instructions recorded | SCOPE-03, SCOPE-05 |
| SCOPE-08 | Define artifact naming and storage conventions | Documented paths for histograms, tables, plots, datacards, logs, and validation reports | SCOPE-07 |

## 1. Samples and production

Each production/validation task below expands over its applicable datasets and run periods.

| ID | Task | Completion evidence | Depends on |
|---|---|---|---|
| PROD-01 | Validate certified data luminosities and good-run selections | Luminosity totals reproduce the AN period totals and sum to 62.5 fb^-1 | SCOPE-03 |
| PROD-02 | Produce/locate data ntuples for all required primary datasets | Complete job/output manifest with no unexplained missing or duplicate files | SCOPE-03, SCOPE-07 |
| PROD-03 | Produce/locate wino-like signal ntuples across the frozen grid | Complete job/output manifest and event-count reconciliation | SCOPE-06, SCOPE-07 |
| PROD-04 | Produce/locate higgsino-like signal ntuples across the frozen grid | Complete job/output manifest and event-count reconciliation | SCOPE-06, SCOPE-07 |
| PROD-05 | Produce/locate simulated background samples used in validation and closure tests | Complete job/output manifest and event-count reconciliation | SCOPE-03, SCOPE-07 |
| PROD-06 | Validate ROOT schemas and required branches | Automated schema report passes for every sample family | PROD-02–PROD-05 |
| PROD-07 | Validate normalization metadata (generated events, weights, cross sections, luminosities) | Normalization reconciliation report with no unexplained discrepancies | PROD-03–PROD-05 |
| PROD-08 | Validate sample overlap, duplicate events, bad files, and event coverage | Automated integrity report passes | PROD-02–PROD-05 |
| PROD-09 | Produce canonical analysis histograms/cutflows from frozen inputs | Versioned artifacts for all periods, samples, systematic variations, and bins | PROD-06–PROD-08, SCOPE-05 |

## 2. Trigger studies

| ID | Task | Completion evidence | Depends on |
|---|---|---|---|
| TRIG-01 | Validate the trigger-path selection and applicability in each run period | Per-period trigger audit and approved configuration | SCOPE-04, PROD-02 |
| TRIG-02 | Measure inclusive missing-momentum trigger turn-ons in data | Efficiency curves, ratio panels, and numerical working-point efficiencies | TRIG-01, PROD-09 |
| TRIG-03 | Validate trigger efficiency on simulation and derive signal trigger scale factors | Per-period scale factors with statistical/systematic uncertainties | TRIG-02, PROD-03, PROD-04 |
| TRIG-04 | Measure the charged-lepton control-trigger efficiency by period/flavor/bin | Final epsilon_trigger^l values replacing any flat Run-2 assumption | TRIG-01, PROD-09 |
| TRIG-05 | Measure the single-tau-trigger probability/correction | Validated P(tau) derived from muon and muon+tau trigger samples | TRIG-01, PROD-09 |
| TRIG-06 | Review trigger plots, thresholds, plateau choices, and time stability | Signed-off trigger validation record | TRIG-02–TRIG-05 |

**Explicit open item in the AN:** Section 5.1 says a flat Run-2 value of 0.84 is currently used for part of the lepton trigger efficiency and that Run-3 dataset-specific studies are ongoing. This must be closed before final background estimates.

## 3. Event selection and detector-quality vetoes

| ID | Task | Completion evidence | Depends on |
|---|---|---|---|
| SEL-01 | Validate physics-object definitions and overlap rules | Object-level distributions and implementation/AN cross-check | SCOPE-05, PROD-09 |
| SEL-02 | Validate event cleaning and missing-momentum filters | Per-period filter audit and signal-efficiency check | PROD-09 |
| SEL-03 | Validate basic event selection | Data/MC cutflow and key kinematic distributions | SEL-01, SEL-02 |
| SEL-04 | Validate isolated-track selection | Data/MC and signal cutflows/distributions | SEL-03 |
| SEL-05 | Validate candidate-track lepton and jet separation vetoes | Veto efficiency and control-distribution review | SEL-04 |
| SEL-06 | Validate disappearing-track requirements (E_calo and missing outer hits) | Signal/background distributions and optimization/sign-off | SEL-05 |
| SEL-07 | Validate the 4, 5, and >=6 layer categorization | Mutually exclusive/exhaustive bin checks and yields | SEL-06 |
| SEL-08 | Produce and validate electron-reconstruction fiducial maps and ECAL veto maps for every period | Frozen maps, construction inputs, and map-review plots | PROD-02, SEL-04 |
| SEL-09 | Produce and validate muon-system fiducial veto maps for every period | Frozen maps, construction inputs, and map-review plots | PROD-02, SEL-04 |
| SEL-10 | Produce and validate tracker/jet-veto maps for every period | Frozen maps, construction inputs, and map-review plots | PROD-02, SEL-04 |
| SEL-11 | Validate fiducial-map independence from the blinded search region | Documented control-region construction and no signal-region inspection | SEL-08–SEL-10 |
| SEL-12 | Freeze all maps and selection code before unblinding | Checksums/version tags and approval record | SEL-01–SEL-11 |
| SEL-13 | Reproduce final cutflows for representative signals and all data periods | Approved cutflow tables with old/new implementation comparison | SEL-12, PROD-09 |

## 4. Charged-lepton backgrounds

The tasks BGLEP-02 through BGLEP-08 expand over flavor (`electron`, `muon`, `tau/single-hadron`), run period, and layer bin unless the method explicitly uses a combined layer category.

| ID | Task | Completion evidence | Depends on |
|---|---|---|---|
| BGLEP-01 | Freeze tag-and-probe and lepton control-region definitions | Approved selections for electron, muon, electron-tag tau, and muon-tag tau samples | SEL-12, TRIG-01 |
| BGLEP-02 | Measure lepton control yield N_ctrl^l | Yield tables with statistical uncertainties and trigger-efficiency correction | BGLEP-01, TRIG-04 |
| BGLEP-03 | Measure P_veto | Opposite-sign/same-sign-subtracted numerator and denominator counts plus final probabilities | BGLEP-01 |
| BGLEP-04 | Validate same-sign subtraction and non-Drell–Yan/fake contamination | Closure/purity study and assigned uncertainty | BGLEP-03 |
| BGLEP-05 | Measure P_offline | Turn-on/control plots and per-period/bin probabilities | BGLEP-01 |
| BGLEP-06 | Measure P_trigger | Trigger-matching counts and per-period/bin probabilities | BGLEP-01, TRIG-04 |
| BGLEP-07 | Apply tau-trigger scaling | Per-period correction with propagated uncertainty | TRIG-05, BGLEP-02 |
| BGLEP-08 | Calculate nominal electron, muon, and tau background estimates | Final per-period/layer tables with complete uncertainty propagation | BGLEP-02–BGLEP-07 |
| BGLEP-09 | Perform charged-lepton closure tests in simulation/control data | Closure plots/tables for every flavor and bin; discrepancies resolved or covered | BGLEP-08, PROD-05 |
| BGLEP-10 | Evaluate charged-lepton method systematics | Approved list and covariance/correlation model | BGLEP-04–BGLEP-09 |
| BGLEP-11 | Review and freeze charged-lepton estimates | Background-group sign-off and immutable input artifact for results | BGLEP-08–BGLEP-10 |

**Explicit open item in the AN:** Section 5.1 calls the electron, muon, and tau estimates “ongoing” and labels the electron result preliminary. Tables 28–30 therefore cannot by themselves be treated as final completion evidence.

## 5. Fake-track background

Tasks expand over run period and layer bin where applicable.

| ID | Task | Completion evidence | Depends on |
|---|---|---|---|
| BGFAKE-01 | Freeze fake-track control regions and d_xy signal/sideband definitions | Approved selections and blinding-safe implementation | SEL-12 |
| BGFAKE-02 | Measure the d_xy transfer factor zeta | Fit inputs, fit-quality plots, parameters, and uncertainties for each period/bin | BGFAKE-01, PROD-09 |
| BGFAKE-03 | Measure fake-track control yields | Yield tables with statistical uncertainties | BGFAKE-01 |
| BGFAKE-04 | Calculate nominal fake-track estimates | Final per-period/layer estimates with propagated uncertainties | BGFAKE-02, BGFAKE-03 |
| BGFAKE-05 | Validate the alternate Z->mumu estimate | Independent estimates and compatibility test | BGFAKE-01–BGFAKE-04 |
| BGFAKE-06 | Test dependence on the d_xy sideband definition | Alternate-sideband results and assigned systematic | BGFAKE-02–BGFAKE-04 |
| BGFAKE-07 | Validate fit model/range and zero-mean assumption | Alternative-fit study and goodness-of-fit results | BGFAKE-02 |
| BGFAKE-08 | Perform fake-track closure in simulation | Closure results per layer bin and period-equivalent selection | BGFAKE-04, PROD-05 |
| BGFAKE-09 | Perform the 2022G/background closure test described in Appendix E | Approved closure table/plot and discrepancy treatment | BGFAKE-04 |
| BGFAKE-10 | Evaluate fake-track systematic uncertainties | Final systematic magnitudes and correlation model | BGFAKE-05–BGFAKE-09 |
| BGFAKE-11 | Review and freeze fake-track estimates | Background-group sign-off and immutable input artifact for results | BGFAKE-04, BGFAKE-10 |

## 6. Signal corrections and efficiencies

Tasks expand over run period, signal model, mass/lifetime point, and layer bin as applicable.

| ID | Task | Completion evidence | Depends on |
|---|---|---|---|
| SIG-01 | Derive/apply pileup weights | Validated pileup distributions and correction provenance | PROD-01, PROD-03, PROD-04 |
| SIG-02 | Fit missing-middle-hit correction | Fit output, uncertainty, and pre/post-correction validation | PROD-09 |
| SIG-03 | Fit missing-outer-hit corrections (pre-TOB, TOB drop, post-TOB) | Fit output, covariance, and pre/post-correction validation | PROD-09 |
| SIG-04 | Validate missing-hit correction algorithms and application | Reproduction of AN distributions/tables and code-level checks | SIG-02, SIG-03 |
| SIG-05 | Apply signal trigger scale factors | Corrected efficiencies and uncertainty variations in all signal points | TRIG-03 |
| SIG-06 | Derive ISR reweighting from Z->mumu data/MC | Per-period pT weights, validation plots, and uncertainty | PROD-09 |
| SIG-07 | Apply ISR reweighting to every signal point | Corrected signal artifacts and varied templates | SIG-06, PROD-03, PROD-04 |
| SIG-08 | Compute final signal acceptance x efficiency | Per-period/model/grid/bin yield tables before and after corrections | SIG-01, SIG-04, SIG-05, SIG-07, SEL-12 |
| SIG-09 | Validate interpolation/reweighting across generated lifetimes and masses | Closure at generated points and interpolation uncertainty | SIG-08 |
| SIG-10 | Review and freeze corrected signal yields | Signal-group sign-off and immutable result inputs | SIG-08, SIG-09 |

## 7. Systematic uncertainties

| ID | Task | Completion evidence | Depends on |
|---|---|---|---|
| SYST-01 | Finalize background statistical and systematic uncertainty decomposition | Per-background/bin values with nuisance type specified | BGLEP-11, BGFAKE-11 |
| SYST-02 | Evaluate integrated-luminosity uncertainty and correlations | Approved per-year components/correlation scheme | PROD-01 |
| SYST-03 | Evaluate pileup uncertainty | Up/down signal yields for all signal bins | SIG-01, SIG-08 |
| SYST-04 | Evaluate trigger-efficiency uncertainty | Up/down signal yields for all signal bins | SIG-05, SIG-08 |
| SYST-05 | Evaluate ISR uncertainty | Up/down signal yields for all signal bins | SIG-07, SIG-08 |
| SYST-06 | Evaluate missing-inner-hit modeling uncertainty | Control-region efficiencies and signal variations | SIG-08 |
| SYST-07 | Evaluate missing-middle-hit modeling uncertainty | Control-region efficiencies and signal variations | SIG-02, SIG-08 |
| SYST-08 | Evaluate accidental missing-outer-hit modeling uncertainty | Efficiency calculation and signal variations | SIG-03, SIG-08 |
| SYST-09 | Evaluate jet energy scale/resolution and unclustered-energy uncertainties | Up/down signal yields for all signal bins | SIG-08 |
| SYST-10 | Evaluate ECAL energy-deposit modeling uncertainty | Data/MC control-region efficiency comparison and signal variations | SIG-08 |
| SYST-11 | Finalize electron- and muon-veto scale factors and uncertainties | Updated Table-45-equivalent values and signal variations | BGLEP-01, SIG-08 |
| SYST-12 | Evaluate track-reconstruction efficiency uncertainty | POG inputs, period mapping, and signal variations | SIG-08 |
| SYST-13 | Evaluate finite simulated-sample statistical uncertainty | Per-bin statistical nuisances/templates | SIG-08 |
| SYST-14 | Build and review the full nuisance correlation model | Machine-readable correlation map across periods, bins, backgrounds, and signals | SYST-01–SYST-13 |
| SYST-15 | Produce final systematic-summary tables for both signal models | Approved tables for every period and layer bin | SYST-14 |

**Explicit open item in the AN:** Section 7.2.9/Table 45 says application of lepton-veto scale factors to signal simulation is still in progress and values will be updated.

## 8. Statistical model and expected results

| ID | Task | Completion evidence | Depends on |
|---|---|---|---|
| STAT-01 | Build the per-period, per-layer-bin datacards/workspaces | Validated model files containing signal, four background components, and observations masked/blinded | BGLEP-11, BGFAKE-11, SIG-10, SYST-14 |
| STAT-02 | Validate nuisance parameter types and correlations | Datacard audit: gamma constraints for control-sample statistics and log-normal/appropriate constraints elsewhere | STAT-01 |
| STAT-03 | Validate nominal yields against frozen source tables | Automated yield reconciliation passes exactly within defined precision | STAT-01 |
| STAT-04 | Run model sanity checks | Fit diagnostics, nuisance pulls/impacts on Asimov data, and numerical stability checks | STAT-02, STAT-03 |
| STAT-05 | Produce expected limits for wino-like signals | 1D cross-section limits at benchmark lifetimes and full mass-lifetime surface | STAT-04 |
| STAT-06 | Produce expected limits for higgsino-like signals | 1D cross-section limits at benchmark lifetimes and full mass-lifetime surface | STAT-04 |
| STAT-07 | Produce expected mass-lifetime exclusion contours | Approved wino and higgsino contours with +/-1 sigma and +/-2 sigma bands | STAT-05, STAT-06 |
| STAT-08 | Combine Run 3 expected results with Run 2 datacards | Validated combined model and compatibility/reproduction checks | STAT-04, approved Run-2 inputs |
| STAT-09 | Produce expected Run 2 + Run 3 exclusion contours | Approved combined wino and higgsino plots | STAT-08 |
| STAT-10 | Perform independent statistical-model review | Reproducibility/sign-off record; stated exclusions reproduce from frozen inputs | STAT-01–STAT-09 |

## 9. Blinding, unblinding, and observed results

| ID | Task | Completion evidence | Depends on |
|---|---|---|---|
| BLIND-01 | Document and enforce the blinding scheme | Code/config guard plus written definition of hidden observations | SCOPE-05 |
| BLIND-02 | Complete the pre-unblinding checklist | All selections/maps/backgrounds/signal/systematics/statistical model frozen and signed off | SEL-12, BGLEP-11, BGFAKE-11, SIG-10, SYST-15, STAT-10 |
| BLIND-03 | Obtain collaboration approval to unblind | Recorded authorization | BLIND-02 |
| BLIND-04 | Unblind and record observed counts without post-hoc changes | Immutable observation record and audit trail | BLIND-03 |
| BLIND-05 | Run predefined data-quality/anomaly checks on observed candidates | Candidate-event validation report following the frozen protocol | BLIND-04 |
| BLIND-06 | Populate observed yields in the final result table | Completed Table-50-equivalent observation column | BLIND-05 |
| BLIND-07 | Produce observed wino and higgsino limits/contours | Approved observed limit plots with expected bands | BLIND-06, STAT-10 |
| BLIND-08 | Produce observed Run 2 + Run 3 combinations | Approved combined observed results | BLIND-07, STAT-08 |
| BLIND-09 | Freeze final numerical results and interpretation statements | Versioned result package and sign-off | BLIND-06–BLIND-08 |

**Explicit open item in the AN:** Table 50 has dashes in every observation cell; the analysis is still blinded. The text currently mixes expected-only figures with observed-limit wording and an exclusion statement, so it requires consistency review before approval.

## 10. Documentation and review

| ID | Task | Completion evidence | Depends on |
|---|---|---|---|
| DOC-01 | Update sample, trigger, object, and selection sections from frozen configurations | AN text/tables match executable configuration and manifests | SCOPE-03–SCOPE-06, SEL-12 |
| DOC-02 | Update all background tables and plots with frozen estimates | AN numbers reproduce from frozen background artifacts | BGLEP-11, BGFAKE-11 |
| DOC-03 | Update signal-correction and systematic sections | AN numbers reproduce from frozen signal/systematic artifacts | SIG-10, SYST-15 |
| DOC-04 | Update the results section with the validated statistical model | Expected text/plots/tables reproduce from frozen results | STAT-10 |
| DOC-05 | After authorization, update observed-result content | Observation table, observed plots, and interpretation are consistent | BLIND-09 |
| DOC-06 | Resolve internal inconsistencies, typos, stale captions, and cross-references | Editorial/technical consistency checklist passes | DOC-01–DOC-05 |
| DOC-07 | Archive machine-readable numbers for every final table and plot | Complete auxiliary-data package with provenance | DOC-02–DOC-05 |
| DOC-08 | Complete internal analysis review and address comments | All comment threads resolved with response log | DOC-01–DOC-07 |
| DOC-09 | Prepare final approval/paper/public-result material as required | Approved final documentation package | DOC-08 |

## Recommended dashboard hierarchy

Use four levels rather than creating hundreds of unrelated flat rows:

1. **Workstream** — production, trigger, selection, lepton background, fake background, signal, systematics, statistics, unblinding, documentation.
2. **Task** — the IDs above.
3. **Task instance** — expansion by period, flavor/model, layer bin, and sometimes signal grid point.
4. **Deliverable/check** — artifact URL/path, validation result, reviewer, and approval date.

Minimum task fields should be: ID, title, workstream, scope dimensions, owner, status, priority, dependencies, completion criteria, artifact links, validation state, reviewer, review date, blocker, and last update. Status should be computed from required deliverables/checks where possible rather than entered as a subjective percentage.

## Immediate critical path from the current AN

1. Freeze/confirm scope, especially the anomalous 2024 entry.
2. Replace the flat Run-2 lepton trigger efficiency with Run-3 measurements.
3. Finalize and validate electron, muon, and tau background estimates.
4. Finish application of lepton-veto scale factors to signal simulation.
5. Freeze all signal and background systematic inputs and correlations.
6. Rebuild/validate expected results from the frozen model.
7. Pass the pre-unblinding review, unblind, and generate observed results.
8. Update the AN so all tables, captions, claims, and plots agree with the final artifacts.
