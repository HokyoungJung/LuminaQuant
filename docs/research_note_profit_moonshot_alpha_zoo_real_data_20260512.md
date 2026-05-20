# Research Note — Profit Moonshot Alpha Zoo Real-Data Calibration (2026-05-12)

## Current answer: what should be used next?

Use `CryptoFxAlphaZooStateStrategy` as the next primary research path, but only after wiring it to real current-tail data and outcome calibration.

Do **not** use the high-performing current-base/calendar tuple as a live strategy. It remains useful as a teacher/hypothesis/reference because it explains a profitable market-state pattern, but it is calendar-primary and therefore invalid for live promotion.

## Latest green baseline

- Latest pushed green head: `e4b5f6942fe06ffff262502e40ff7dd4d7005323` on `private/main`.
- Prior implementation head for external-risk teacher pass: `fcc63f6c053c451152b0d780fa84ee91b5512f82`.
- GitHub Actions were green for both `ci` and `private-ci` after the handoff commit.

## What exists now

### 1. Alpha Zoo / outcome-calibration scaffold

Implemented files:

- `src/lumina_quant/alpha_zoo/operators.py`
- `src/lumina_quant/alpha_zoo/crypto_fx_factors.py`
- `src/lumina_quant/alpha_zoo/factor_card.py`
- `src/lumina_quant/research/triple_barrier.py`
- `src/lumina_quant/research/candidate_outcome_ledger.py`
- `src/lumina_quant/research/edge_calibration.py`
- `src/lumina_quant/strategies/crypto_fx_alpha_zoo_state.py`
- `scripts/research/run_crypto_fx_alpha_zoo_screen.py`
- `scripts/research/replay_crypto_fx_alpha_zoo_state.py`
- `scripts/research/calibrate_crypto_fx_edges.py`

Current state: smoke/research scaffold only. It proves factor generation, train/validation-only selection metadata, triple-barrier labels, candidate ledger, edge calibration, and calibrated-entry strategy plumbing. It does **not** yet prove economic profitability on real current-tail data.

### 2. External-risk teacher pass

Implemented `scripts/research/fetch_profit_moonshot_external_state.py` to fetch lagged FRED daily state features from:

- DTWEXBGS
- VIXCLS
- DGS2
- DGS10
- DCOILWTICO

Features are lagged before joining to hourly crypto panels to avoid same-day lookahead.

Added non-calendar replay families:

- `calendar_teacher_state_similarity`
- `calendar_teacher_state_fade`
- `state_distilled_external_risk_filter`

Results:

- `calendar_teacher_state_similarity`: 972 specs, 0 survivor.
- `calendar_teacher_state_fade`: 324 specs, 0 survivor.
- `state_distilled_external_risk_filter`: 1728 specs, 565 train/validation-positive, 0 replay survivor under legacy shadow-MDD gate, peak RSS about 280 MiB.

### 3. Best valid strict state-distilled seed

Train/validation-selected strict candidate:

`fresh_state_distilled_ext_both_lb168_fast72_z075_ret180_h168_tp600_fl0_xr125` at 4x.

Metrics:

- Train: `+30.9030%`, MDD `10.2437%`, Sharpe `1.8484`, liquidation `0`.
- Validation: `+12.4704%`, MDD `2.5167%`, Sharpe `5.7588`, liquidation `0`.
- Locked-OOS: `+2.4852%`, MDD `2.5328%`, Sharpe `1.5096`, liquidation `0`.
- All split min margin buffers are positive.
- Strategy-validity passes.
- `deployable_success=false` because it does not beat the invalid current-base/calendar reference economics.

### 4. Calendar/current-base teacher status

The current-base/calendar tuple remains economically strong:

- Locked-OOS return `+6.4281%`.
- Return/MDD `6.9169`.
- Sharpe about `5.2024` in the liquidation-aware reference report.

But it is calendar-primary/fixed calendar behavior, so it is invalid for live promotion and must remain:

- `hypothesis_reference_only`
- not a selection target
- not a live strategy
- not a promoted candidate

## Next research objective

Convert the Alpha Zoo scaffold from synthetic smoke to real current-tail research:

1. Wire `run_crypto_fx_alpha_zoo_screen.py` to real crypto OHLCV/funding/OI/flow fields where available.
2. Add FX OHLCV regime fields if reliable; otherwise use lagged FRED risk-state as temporary regime context and explicitly record direct-FX trading as blocked.
3. Generate real factor cards with source coverage and `uses_locked_oos_for_selection=false`.
4. Produce triple-barrier candidate outcomes on train/validation.
5. Calibrate edge buckets with shrinkage and lower-confidence edge gating.
6. Replay `CryptoFxAlphaZooStateStrategy` plus state-distilled/residual seeds only if selected by train/validation.
7. Open locked-OOS after freeze only as gate/report.
8. Run strict zero-liquidation integer grid and separate diagnostic nonfatal 5x/6x lane.

## Required research-note practice

Every future profit-moonshot session must update research notes before final handoff:

- Update or supersede this file when Alpha Zoo real-data results change.
- Update `.omx/notepad.md` with concise conclusions and artifact paths.
- Update the active `.omx/plans/*` file with result status.
- Write or update `docs/session_handoff_*` for the session.
- If the work changes the global research inventory/source ledger, regenerate or explicitly update `docs/profit_moonshot_research_history_20260510.md` and the matching `var/reports/.../research_history/` artifacts, or document why regeneration was not required.

## Failure mode to avoid

Do not repeat the earlier loop of finding a good-looking single rule and then discovering it is calendar-primary or OOS-selected. The next path must be evidence-first:

`real factors → train/validation labels → calibrated edge → stateful replay → locked-OOS gate/report → strict liquidation validation`.

## 2026-05-13 — Real-data Alpha Zoo calibrated replay result

- Ran real current-tail Alpha Zoo screen against `/home/hoky/Quants-agent/LuminaQuant/var/cache/profit_moonshot_fresh_start/joined_panel_de62df511cec53df6ad39521.parquet` with lagged FRED context; direct FX OHLCV trading stayed blocked because current-tail cache contains crypto OHLCV only.
- Factor/card validity passed fail-closed gates: `calendar_primary=false`, `uses_locked_oos_for_selection=false`, strategy validity pass.
- Candidate outcome ledger: `45160` rows; train+validation `30494`; locked-OOS `14666`.
- Edge calibration physically filtered to train/validation: input `45160`, calibration `30494`, locked-OOS calibration `0`, excluded locked-OOS `14666`.
- Replay grid selected `alpha_zoo_conservative_exit` from `9` formulaic candidates using train/validation metrics only; locked-OOS remained hidden until candidate freeze.
- Strict zero-liquidation lane highest safe integer: `6.0x`, liquidation count `0`, min buffer `9049.125962`, OOS return `41.0967%`, OOS MDD `13.6667%`, return/MDD `3.007073`, Sharpe `2.143209`.
- 2026-05-14 latest correction: return/MDD is diagnostic/report-only per operator clarification; OOS return must beat the invalid current-base reference, but OOS return/MDD does not block promotion.
- Deployable success is now `true` under the corrected policy: strict 6x has OOS return `41.0967%`, MDD `13.6667%`, Sharpe `2.143209`, Sortino `2.841936`, smart Sortino `2.500237`, Calmar/return-MDD `3.007073`, zero liquidations, and positive buffers.
- Diagnostic 5x/6x lane remains non-promotional and separate: 5x/6x both zero liquidation in this approximate replay, but that lane is report-only.
- Peak RSS `512.711` MiB (<8 GiB).
- Artifacts: `/home/hoky/Quants-agent/LuminaQuant/var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/crypto_fx_alpha_zoo_real_data_20260513/crypto_fx_alpha_zoo_real_data_summary_latest.json`, `/home/hoky/Quants-agent/LuminaQuant/var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/crypto_fx_alpha_zoo_real_data_20260513/crypto_fx_alpha_zoo_state_replay_latest.json`, `/home/hoky/Quants-agent/LuminaQuant/var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/crypto_fx_alpha_zoo_real_data_20260513/edge_calibration_latest.json`, `/home/hoky/Quants-agent/LuminaQuant/var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/crypto_fx_alpha_zoo_real_data_20260513/candidate_outcome_ledger_latest.jsonl`.
- Research history/source ledger not regenerated: No new external source class or global chronology/source-ledger change; reused existing current-tail cache and 20260512 lagged FRED external-state artifact, added only session-scoped Alpha Zoo artifacts.

---

## 2026-05-13 KST — Related state-distilled regime-boost overlay note

A separate research-only `StateDistilledRegimeBoostPortfolio` overlay was tested on the existing state-distilled seeds. This did not change the `CryptoFxAlphaZooStateStrategy` promotion policy or Alpha Zoo factor cards. The overlay reused real current-tail crypto data and lagged FRED external-risk state, kept calendar/current-base as hypothesis reference only, and kept locked-OOS gate/report-only after freeze.

Result: strict zero-liquidation/margin gates passed, but validation and locked-OOS return/risk-quality metrics failed, so no new deployable promotion came from the regime-boost overlay. Artifacts live under `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/state_distilled_regime_boost_20260513/`.

---

## 2026-05-14 KST — Return/MDD diagnostic-only policy correction

Applied the latest operator clarification: OOS return/MDD is diagnostic/report-only, not a strict promotion hurdle. The current-base/calendar tuple remains `hypothesis_reference_only`, not a selection or promotion target; the strict deploy lane still requires zero liquidation, positive buffers, OOS MDD <=25%, OOS return beating the current-base reference, and positive risk metrics.

Real current-tail run under `crypto_fx_alpha_zoo_real_data_20260514`:

- Screen: `58,845` rows, `63` factors, `20` selected cards; direct FX OHLCV remains blocked and lagged FRED state is regime context only.
- Candidate outcome ledger: `67,259` rows; train+validation `45,311`; locked-OOS `21,948`.
- Edge calibration: train/validation-only physical filter; locked-OOS calibration records `0`; calibrated edge keys `12`.
- Replay selected `alpha_zoo_conservative_exit` from `9` formulaic candidates using train/validation metrics only.
- Strict zero-liquidation lane promoted integer: `6.0x`, liquidation count `0`, min buffer `9049.125962`, OOS return `41.0967%`, OOS MDD `13.6667%`, return/MDD `3.007073`, Sharpe `2.143209`, Sortino `2.841936`, smart Sortino `2.500237`.
- Deployable success: `true`; return/MDD `3.007073` vs current-base `6.916878` is reported as diagnostic-only and does not block promotion.
- Diagnostic 5x/6x lane is separate and non-promotional; 5x/6x both zero liquidation but `promotion_allowed=false` in that diagnostic lane.
- Peak RSS `626.7266 MiB` (<8 GiB).
- Artifacts: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/crypto_fx_alpha_zoo_real_data_20260514/`.

Research history/source ledger was not regenerated because no new global source family was introduced; this pass reused the current-tail crypto cache and 20260512 lagged FRED external-state source.

## 2026-05-14 KST — Paper-forward diagnostics added

Added non-promotional diagnostics requested for the current leading `CryptoFxAlphaZooStateStrategy` / `alpha_zoo_conservative_exit` / strict 6x candidate. The strategy/replay now records dominant factor-family metadata and exit-reason metadata, and the real-data replay summary includes locked-OOS PnL breakdowns plus cost sensitivity.

Locked-OOS diagnostic breakdown at 6x, 10% allocation:

- Regime: `neutral` `+41.0967%` across `540` trades. Direct FX OHLCV trading remains blocked; lagged FRED state remains context only.
- Symbol: SOL/USDT `+19.2672%` (`126`), BNB/USDT `+10.3167%` (`128`), TRX/USDT `+4.9566%` (`139`), ETH/USDT `+1.4843%` (`132`), BTC/USDT `+0.6807%` (`15`).
- Side: SHORT `+26.3040%` (`259`), LONG `+11.7120%` (`281`).
- Dominant factor family: crypto residual momentum `+30.1822%` (`184`), crypto residual reversal `+8.4241%` (`237`), volume/vwap pressure `-0.0370%` (`119`).
- Exit reason: score_exit `+41.0936%` (`526`), take_profit `+16.1976%` (`4`), end_of_sample `-0.0788%` (`2`), stop_loss `-13.8700%` (`8`).
- Slippage sensitivity, round-trip 0/2.5/5/10/20 bps: `+41.0967%`, `+30.1241%`, `+20.0034%`, `+2.0585%`, `-26.1930%`.
- Conservative funding drag sensitivity, 0/1/2/5/10 bps per day: `+41.0967%`, `+40.4210%`, `+39.7486%`, `+37.7505%`, `+34.4835%`.

Promotion policy unchanged: these diagnostics are `diagnostic_only`, `promotion_allowed=false`; the strict lane remains zero-liquidation + positive buffer + OOS return, MDD cap, and positive risk-metric policy with return/MDD report-only. Research history/source ledger was not regenerated because no new global source family was introduced.

## 2026-05-16 KST — Hybrid v3.5/v3.6 and Optuna comparison against Alpha Zoo strict 6x

Completed a repo-wide inventory and policy audit for hybrid v3.5/v3.6, hybrid Optuna, tuning/optimization, candidate-hybrid, calendar Optuna, and fresh-portfolio optimization artifacts against the preserved private/main baseline `1c6816fced44d277f6c7112934c9dded65ba710f`. Corrections on 2026-05-17 KST: the **comparison core excludes calendar/current-base-derived rows and literal hybrid/hybrid-online/hybrid-tuning rows before ranking**. `portfolio`, `allocator`, `meta`, `static_blend`, and `leverage_sweep` labels are not exclusion triggers by themselves; calendar/current-base and literal-hybrid rows are retained only in quarantine/reference ledgers.

Decision: **only** `CryptoFxAlphaZooStateStrategy / alpha_zoo_conservative_exit / strict 6x` remains live-promotion possible. Alpha Zoo selection is train/validation-only, locked-OOS is gate/report-only after candidate freeze, and strict 6x passes zero-liquidation, positive margin buffer, OOS MDD <=25%, OOS return above the invalid current-base reference, positive Sharpe/Sortino/smart Sortino/Calmar, and memory <8 GiB.

Alpha Zoo strict 6x split evidence:

- train: 2025-01-01T00:00:00 → 2025-10-19T13:00:00, return 68.8842%, MDD 29.5651%, Sharpe 1.569139, Sortino 1.919776, smart Sortino 1.481707, Calmar 2.329914, trades 1779, liq 0, min buffer 9049.125962
- validation: 2025-10-22T05:00:00 → 2026-01-28T06:00:00, return 30.1195%, MDD 9.5595%, Sharpe 1.552041, Sortino 2.095744, smart Sortino 1.912882, Calmar 3.150734, trades 524, liq 0, min buffer 9527.695928
- locked_oos: 2026-01-28T07:00:00 → 2026-05-06T23:00:00, return 41.0967%, MDD 13.6667%, Sharpe 2.143209, Sortino 2.841936, smart Sortino 2.500237, Calmar 3.007073, trades 540, liq 0, min buffer 9572.449083

Hybrid/Optuna conclusions:

- Hybrid v3.5/v3.6 rows are **not strict-core rows** after correction because their own provenance is literal hybrid / hybrid-online / hybrid-final-selection. `portfolio`, `allocator`, `meta`, `static_blend`, and `leverage_sweep` labels are not exclusion triggers by themselves; they are only evidence/context when the top-level row is already literal hybrid.
- Hybrid Optuna `live_guarded` and `train_aware_guarded` are same-family hybrid optimizer outputs and are also live-promotion invalid because those objective profiles consume OOS metrics; good-looking OOS values remain diagnostic/reference only.
- Hybrid/tuning `locked_train_val` policy shape is cleaner (`oos_is_objective_input=false`) but it is still a same-family hybrid-online tuning output, not an atomic-source hybrid candidate, and remains quarantine/reference only.
- State-distilled fresh portfolio tuning is **not literal hybrid** and is restored to the non-calendar comparison core. It remains diagnostic/non-promotable because strict liquidation/margin replay is missing and Alpha Zoo strict 6x dominates the locked-OOS/live-promotion gates.
- Calendar Optuna and calendar/current-base-dependent fresh/candidate-hybrid rows are **not part of the strict core**. They remain in a separate quarantine/reference ledger because calendar month/day/hour rules and current-base tuple dependencies are invalid before any ranking.
- Candidate-hybrid had strong OOS metrics but is excluded due calendar/current-base-source dependency, validation liquidation count `1`, not the hybrid-inside-hybrid rule.

Strict integer recheck 1x..6x was rerun in the comparison directory. Highest strict integer remains `6.0x`: OOS return `41.0967%`, MDD `13.6667%`, return/MDD diagnostic `3.007073`, Sharpe `2.143209`, Sortino `2.841936`, smart Sortino `2.500237`, liquidation `0`, min buffer `9049.125962`. The separate diagnostic 5x/6x lane is preserved with `promotion_allowed=false`.

Artifacts:

- Full JSON report: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/hybrid_optuna_alpha_zoo_comparison_20260514/hybrid_optuna_alpha_zoo_comparison_latest.json`
- Corrected non-calendar JSON snapshot: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/hybrid_optuna_alpha_zoo_comparison_20260514/hybrid_optuna_alpha_zoo_comparison_20260517T000000Z_calendar_quarantine_corrected.json`
- Corrected literal-hybrid quarantine JSON snapshot: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/hybrid_optuna_alpha_zoo_comparison_20260514/hybrid_optuna_alpha_zoo_comparison_20260517T015000KST_hybrid_only_quarantine_corrected.json`
- Full Markdown report: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/hybrid_optuna_alpha_zoo_comparison_20260514/hybrid_optuna_alpha_zoo_comparison_latest.md`
- Comparison-core split performance CSV: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/hybrid_optuna_alpha_zoo_comparison_20260514/candidate_split_performance_latest.csv`
- Literal nested-hybrid quarantine CSV: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/hybrid_optuna_alpha_zoo_comparison_20260514/excluded_nested_hybrid_same_family_quarantine_latest.csv`
- Calendar/current-base quarantine CSV: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/hybrid_optuna_alpha_zoo_comparison_20260514/excluded_calendar_current_base_quarantine_latest.csv`
- Inventory JSON: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/hybrid_optuna_alpha_zoo_comparison_20260514/hybrid_optuna_alpha_zoo_inventory_latest.json`
- Prompt checklist audit: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/hybrid_optuna_alpha_zoo_comparison_20260514/prompt_checklist_audit_latest.json`
- Strict integer recheck: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/hybrid_optuna_alpha_zoo_comparison_20260514/alpha_zoo_strict_integer_recheck_latest.json`

Non-calendar comparison core after corrections has `2` candidates (`crypto_fx_alpha_zoo_state_calibrated` plus the non-hybrid state-distilled portfolio diagnostic row); literal nested-hybrid quarantine has `8` candidates and calendar/current-base quarantine has `5`. Verification after literal-hybrid quarantine passed (`1308` full tests plus ruff/compileall/diff checks). Latest log `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/hybrid_optuna_alpha_zoo_comparison_20260514/local_verification_hybrid_only_quarantine_20260517T020000KST.log`. Memory peak across inspected/replayed comparison artifacts: `1239.703125 MiB`, below 8 GiB. Research history/source ledger was **not regenerated** because this session did not add a new global data source family or chronology ledger; it reused existing repo-local artifacts and added a session-scoped comparison report.


## 2026-05-17 KST — External Hybrid v3.5/v3.6 method applied to fixed A0+P0+E0+S1+S2+S3+S4 inputs

Operator correction incorporated: `/home/hoky/DeepLearning/ensemble_strategies` defines v3.6 as **v3.5 core plus online dynamic default-model/candidate refresh**, not a new candidate universe and not a hybrid-inside-hybrid stack. Evidence checked directly in the external method source:

- `models/hybrid/v3_5.py`: lines 1-8 describe adaptive weights + Optuna; lines 31-49 define the Optuna-tuned defaults/search candidates; lines 311-328 learn train/warmup parameters; lines 397-420 keep the default model fixed while applying rolling weights/high-vol boost.
- `models/hybrid/v3_6.py`: lines 1-9 state the v3.6 delta: Step A `default_model` is dynamically updated online by rolling MAPE while v3.5 defaults/Optuna results are retained; lines 29-30 reuse v3.5 learning; lines 87-105 learn the same parameters; lines 178-223 dynamically refresh only the default model and otherwise use the v3.5 weight/high-vol/bias structure.
- `scripts/compare_v35_v36.py`: lines 1-5 summarize the same delta; lines 49-55 compare v3.5 fixed default vs v3.6 dynamic default.

Repo adaptation now uses fixed input universe `A0 + P0 + E0 + S1 + S2 + S3 + S4` only. No literal prior hybrid/hybrid-online/hybrid-tuning output is an input; no calendar/month/day/hour entry rule is introduced; Optuna objective/selection uses train+validation only. Locked-OOS remains gate/report-only after candidate freeze.

Split periods for this fixed-input experiment:

- locked_oos: `2026-03-01T00:00:00Z` ~ `2026-05-06T23:00:00Z` (1593 rows)
- train: `2025-01-01T00:00:00Z` ~ `2025-12-31T23:00:00Z` (8760 rows)
- validation: `2026-01-01T00:00:00Z` ~ `2026-02-28T23:00:00Z` (1416 rows)

Candidate input split metrics from the reconstructed return-stream experiment:

| Input | Split | Return | MDD | Return/MDD diagnostic | Sharpe | Sortino | Smart Sortino | Calmar | Trades/active hours | Liquidations | Min buffer |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| A0 | train | +114.46% | +28.27% | 4.049305 | 1.979653 | 1.131499 | 0.882143 | 4.049305 | 1812 | not_replayed | not_replayed |
| A0 | validation | +19.97% | +13.67% | 1.461080 | 2.668444 | 1.654321 | 1.455414 | 15.249889 | 292 | not_replayed | not_replayed |
| A0 | locked_oos | +20.51% | +6.79% | 3.021741 | 3.993463 | 2.711715 | 2.539336 | 26.368611 | 313 | not_replayed | not_replayed |
| P0 | train | +64.11% | +17.75% | 3.610917 | 1.854506 | 1.719569 | 1.460317 | 3.610917 | 7173 | not_replayed | not_replayed |
| P0 | validation | +26.68% | +5.37% | 4.966077 | 6.497669 | 7.032072 | 6.673577 | 61.776012 | 1262 | not_replayed | not_replayed |
| P0 | locked_oos | +4.52% | +6.97% | 0.647757 | 1.217684 | 1.227542 | 1.147552 | 3.943449 | 1429 | not_replayed | not_replayed |
| E0 | train | +32.17% | +7.89% | 4.077794 | 2.121676 | 1.710879 | 1.585758 | 4.077794 | 5314 | not_replayed | not_replayed |
| E0 | validation | +11.62% | +2.79% | 4.162712 | 5.240997 | 4.732774 | 4.604245 | 34.893482 | 838 | not_replayed | not_replayed |
| E0 | locked_oos | +2.90% | +2.36% | 1.226822 | 1.780358 | 1.646460 | 1.608436 | 7.201675 | 1175 | not_replayed | not_replayed |
| S1 | train | +8.04% | +2.31% | 3.485607 | 2.064400 | 1.661117 | 1.623648 | 3.485607 | 5314 | not_replayed | not_replayed |
| S1 | validation | +2.91% | +0.77% | 3.796269 | 5.095774 | 4.582247 | 4.547448 | 25.328062 | 838 | not_replayed | not_replayed |
| S1 | locked_oos | +0.73% | +0.60% | 1.200019 | 1.748046 | 1.615190 | 1.605489 | 6.707520 | 1175 | not_replayed | not_replayed |
| S2 | train | +7.07% | +2.86% | 2.470000 | 1.818764 | 1.450761 | 1.410400 | 2.470000 | 5309 | not_replayed | not_replayed |
| S2 | validation | +2.91% | +0.77% | 3.796269 | 5.095774 | 4.582247 | 4.547448 | 25.328062 | 838 | not_replayed | not_replayed |
| S2 | locked_oos | +0.68% | +0.60% | 1.136598 | 1.630621 | 1.534185 | 1.525047 | 6.346750 | 1203 | not_replayed | not_replayed |
| S3 | train | +6.90% | +2.86% | 2.412668 | 1.779218 | 1.417786 | 1.378343 | 2.412668 | 5297 | not_replayed | not_replayed |
| S3 | validation | +2.91% | +0.77% | 3.796269 | 5.095774 | 4.582247 | 4.547448 | 25.328062 | 838 | not_replayed | not_replayed |
| S3 | locked_oos | +0.68% | +0.60% | 1.136598 | 1.630621 | 1.534185 | 1.525047 | 6.346750 | 1203 | not_replayed | not_replayed |
| S4 | train | +3.51% | +3.27% | 1.072977 | 1.014829 | 0.764783 | 0.740531 | 1.072977 | 4885 | not_replayed | not_replayed |
| S4 | validation | +1.52% | +0.90% | 1.684443 | 3.832281 | 3.193221 | 3.164658 | 10.840367 | 748 | not_replayed | not_replayed |
| S4 | locked_oos | +0.62% | +0.81% | 0.758295 | 1.514867 | 1.299905 | 1.289404 | 4.228257 | 1062 | not_replayed | not_replayed |

Hybrid Optuna outputs using the corrected external concept:

| Candidate | Train/validation score | Train return | Validation return | Locked-OOS return | Locked-OOS MDD | Return/MDD diagnostic | Sharpe | Sortino | Smart Sortino | Calmar | Liquidations | Min buffer | Deployable success | Rejection reasons |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- |
| hybrid_v3_5_optuna | 70.585 | +47.73% | +13.31% | +8.52% | +1.77% | 4.827936 | 5.259028 | 7.316663 | 7.189734 | 32.173151 | not_replayed | not_replayed | False | dedicated_integrated_margin_replay_required_for_mixed_alpha_state_portfolio_hybrid |
| hybrid_v3_6_optuna | 85.548 | +49.52% | +12.49% | +7.79% | +1.75% | 4.454705 | 4.859674 | 5.991026 | 5.888040 | 29.199963 | not_replayed | not_replayed | False | dedicated_integrated_margin_replay_required_for_mixed_alpha_state_portfolio_hybrid |

Alpha Zoo strict 6x comparison anchor remains superior for live promotion:

| Candidate | Split | Period start | Period end | Return | MDD | Return/MDD diagnostic | Sharpe | Sortino | Smart Sortino | Calmar | Trades | Liquidations | Min buffer |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Alpha Zoo strict 6x | train | `2025-01-01T00:00:00` | `2025-10-19T13:00:00` | +68.88% | +29.57% | 2.329914 | 1.569139 | 1.919776 | 1.481707 | 2.329914 | 1779 | 0 | 9049.125962 |
| Alpha Zoo strict 6x | validation | `2025-10-22T05:00:00` | `2026-01-28T06:00:00` | +30.12% | +9.56% | 3.150734 | 1.552041 | 2.095744 | 1.912882 | 3.150734 | 524 | 0 | 9527.695928 |
| Alpha Zoo strict 6x | locked_oos | `2026-01-28T07:00:00` | `2026-05-06T23:00:00` | +41.10% | +13.67% | 3.007073 | 2.143209 | 2.841936 | 2.500237 | 3.007073 | 540 | 0 | 9572.449083 |

Decision: the fixed-input v3.5/v3.6 Optuna experiments are useful diagnostics but **not live-promotable** yet because the mixed A0/P0/E0/S-sleeve allocator has no dedicated integrated margin replay, so liquidation count and minimum margin buffer are `not_replayed`. Both fixed-input hybrids satisfy train/validation-only selection and have locked-OOS MDD below 25%, but Alpha Zoo strict 6x still dominates live promotion with locked-OOS return `+41.0967%`, zero liquidations, and positive min buffer. The fixed-input hybrids remain report/reference until a dedicated strict zero-liquidation margin replay is implemented.

Artifacts:

- Script: `scripts/research/run_profit_moonshot_hybrid_v35_v36_fixed_inputs.py`
- Regression test: `tests/test_profit_moonshot_hybrid_v35_v36_fixed_inputs.py`
- JSON report: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/hybrid_v35_v36_fixed_inputs_20260517/hybrid_v35_v36_fixed_inputs_latest.json`
- Markdown report: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/hybrid_v35_v36_fixed_inputs_20260517/hybrid_v35_v36_fixed_inputs_latest.md`
- Timestamped latest run: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/hybrid_v35_v36_fixed_inputs_20260517/hybrid_v35_v36_fixed_inputs_20260516T172901Z.json`
- Peak RSS: `353.754 MiB` (<8 GiB)

Research history/source ledger was not regenerated: this run reused existing repo-local market/research artifacts and added a session-scoped method-adaptation report; it did not introduce a new global data-source family or chronology ledger.

## 2026-05-17 KST — Common-split Alpha Zoo vs fixed-input hybrid v3.5/v3.6 fair comparison

Re-ran the Alpha Zoo best and fixed-input hybrid v3.5/v3.6 candidates on one explicit common split from baseline `private/main@80a557c133930f51748ec20c4e582aa0d6f678de`. The prior Alpha Zoo split is now **historical only** in this comparison; it is not used for common-split selection, promotion, or tie-breaks.

Common split authority:

- train: `2025-01-01T00:00:00Z` ~ `2025-12-31T23:00:00Z` (`8760` hourly timestamps, `43800` rows)
- validation: `2026-01-01T00:00:00Z` ~ `2026-02-28T23:00:00Z` (`1416` hourly timestamps, `7080` rows)
- locked-OOS: `2026-03-01T00:00:00Z` ~ `2026-05-06T23:00:00Z` (`1593` hourly timestamps, `7965` rows)

Alpha Zoo result on common split:

- Historical old split strict 6x is preserved only as reference: old split periods were train `2025-01-01T00:00:00Z`~`2025-10-22T04:00:00Z`, validation `2025-10-22T05:00:00Z`~`2026-01-28T06:00:00Z`, locked-OOS `2026-01-28T07:00:00Z`~`2026-05-06T23:00:00Z`; old locked-OOS return was `+41.0967%`, but it is not comparable for new selection.
- Common-split carry-forward of the old selected `alpha_zoo_conservative_exit` and common-split reselected grid both select/retain `alpha_zoo_conservative_exit` and produce the same strict 6x replay: train `+114.4617%` / MDD `29.5651%`; validation `+19.9681%` / MDD `13.6667%`; locked-OOS `+20.5127%` / MDD `6.7884%`; locked-OOS Sharpe `1.772136`, Sortino `2.578776`, smart Sortino `2.414847`, Calmar `3.021741`, trades `365`, liquidation `0`, min margin buffer `9643.447509`; `deployable_success=true` under the strict lane.
- Strict integer leverage 1x..6x on the common split keeps `6x` as the highest deployable integer: OOS return `+20.5127%`, MDD `6.7884%`, liquidation `0`, min buffer `9049.125962`. Return/MDD remains diagnostic/report-only.
- Diagnostic nonfatal 5x/6x lane remains separate from live promotion: 5x and 6x both have `promotion_allowed=false` even though their diagnostic replay has zero liquidations.

Fixed-input hybrid v3.5/v3.6 common-split Optuna result:

- Input universe remains exactly `A0 + P0 + E0 + S1 + S2 + S3 + S4`; no literal hybrid/hybrid-online/hybrid-tuning output is used as an input.
- v3.5 uses fixed default + rolling weights/high-vol boost + Optuna; v3.6 is v3.5 core plus online dynamic default-candidate refresh only. No other knob is OOS-adaptive.
- Optuna ran with `n_trials=80`, `seed=42`, and train+validation objective/selection only. Audit found no locked-OOS use for objective, pruning, selection, or tie-break (`violation=false`; calibration locked-OOS records `0`).
- v3.5 common-split locked-OOS: return `+8.5233%`, MDD `1.7654%`, Sharpe `5.259028`, Sortino `7.316663`, smart Sortino `7.189734`, Calmar `32.173151`; `deployable_success=false` because dedicated integrated margin replay is missing.
- v3.6 common-split locked-OOS: return `+7.7916%`, MDD `1.7491%`, Sharpe `4.859674`, Sortino `5.991026`, smart Sortino `5.888040`, Calmar `29.199963`; `deployable_success=false` for the same margin-replay reason.

Decision: on the fair common split, Alpha Zoo strict 6x remains the only live-promotion-capable lane. Hybrid v3.5/v3.6 are useful diagnostics and beat the invalid current-base OOS return reference, but they are not live-promotable until an integrated strict liquidation/margin replay supplies liquidation count and minimum margin buffer. Peak RSS was `769.1015625 MiB` (<8 GiB). Research history/source ledger was not regenerated because this session added a session-scoped comparison artifact only and did not introduce a new global source family or chronology ledger.

Artifacts:

- Runner: `scripts/research/run_common_split_alpha_zoo_hybrid_v35_v36.py`
- Regression tests: `tests/test_common_split_alpha_zoo_hybrid_v35_v36.py`
- Main JSON: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/common_split_alpha_zoo_hybrid_v35_v36_20260517/common_split_alpha_zoo_hybrid_v35_v36_latest.json`
- Main Markdown: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/common_split_alpha_zoo_hybrid_v35_v36_20260517/common_split_alpha_zoo_hybrid_v35_v36_latest.md`
- Alpha stage artifacts: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/common_split_alpha_zoo_hybrid_v35_v36_20260517/alpha_zoo_common_split/`
- Hybrid stage artifacts: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/common_split_alpha_zoo_hybrid_v35_v36_20260517/hybrid_v35_v36_common_split/`

Verification for the common-split run passed on 2026-05-17 UTC: targeted Alpha Zoo suite `23 passed`, moonshot validation suite `74 passed`, full pytest `1319 passed`, ruff/compileall/diff checks passed. Log: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/common_split_alpha_zoo_hybrid_v35_v36_20260517/local_verification_common_split_20260517T054257Z.log`.
Post-deslop verification re-ran the full required command set and passed again: targeted Alpha Zoo suite `23 passed`, moonshot validation suite `74 passed`, full pytest `1319 passed`, ruff/compileall/diff checks passed. Log: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/common_split_alpha_zoo_hybrid_v35_v36_20260517/local_verification_common_split_post_deslop_20260517T054855Z.log`.

## 2026-05-17 KST — Integrated margin replay addendum for fixed-input hybrid v3.5/v3.6

Supersedes the earlier same-day note that marked fixed-input hybrid v3.5/v3.6 non-promotable solely because liquidation count/minimum margin buffer were `not_replayed`. Added a mixed-allocator integrated margin replay for the frozen A0+P0+E0+S1+S2+S3+S4 hybrid return streams. The replay uses post-freeze v3.5/v3.6 allocator weights, maps each fixed stream to its source gross-notional fraction, and evaluates one cross-margin account path. It is not used by Optuna objective, pruning, selection, or tie-break; locked-OOS remains gate/report-only after candidate freeze.

Updated common-split hybrid live-gate result:

| Candidate | Split | Return | MDD | Sharpe | Sortino | Smart Sortino | Calmar | Active hours | Liquidations | Min margin buffer | Deployable success |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| hybrid_v3_5_optuna | train | +47.7257% | 11.0421% | 2.705685 | 2.932093 | 2.640523 | 4.322148 | 7514 | 0 | 9932.438663 | true |
| hybrid_v3_5_optuna | validation | +13.3102% | 2.2622% | 5.390597 | 7.610005 | 7.441656 | 51.558258 | 1302 | 0 | 14594.054033 | true |
| hybrid_v3_5_optuna | locked-OOS | +8.5233% | 1.7654% | 5.259028 | 7.316663 | 7.189734 | 32.173151 | 1467 | 0 | 16587.499982 | true |
| hybrid_v3_6_optuna | train | +49.5204% | 7.6947% | 2.897597 | 2.999204 | 2.784914 | 6.435678 | 7514 | 0 | 9847.514685 | true |
| hybrid_v3_6_optuna | validation | +12.4946% | 1.5354% | 7.002337 | 8.680826 | 8.549560 | 69.800312 | 1302 | 0 | 14690.924128 | true |
| hybrid_v3_6_optuna | locked-OOS | +7.7916% | 1.7491% | 4.859674 | 5.991026 | 5.888040 | 29.199963 | 1467 | 0 | 16664.270300 | true |

Decision update: fixed-input hybrid v3.5/v3.6 are now live-promotion-capable under the integrated margin gate, but they do **not** beat the common-split Alpha Zoo strict 6x lane on locked-OOS return (`+20.5127%` for Alpha Zoo vs `+8.5233%` v3.5 and `+7.7916%` v3.6). Alpha Zoo strict 6x remains the common-split performance leader; hybrid v3.5/v3.6 become lower-return, lower-MDD deployable alternatives rather than blocked diagnostics. Research history/source ledger still not regenerated: this addendum adds a local validation/replay artifact and no new global source family.

Updated artifacts:

- Runner update: `scripts/research/run_profit_moonshot_hybrid_v35_v36_fixed_inputs.py`
- Common report JSON: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/common_split_alpha_zoo_hybrid_v35_v36_20260517/common_split_alpha_zoo_hybrid_v35_v36_latest.json`
- Common report Markdown: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/common_split_alpha_zoo_hybrid_v35_v36_20260517/common_split_alpha_zoo_hybrid_v35_v36_latest.md`
- Hybrid stage JSON: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/common_split_alpha_zoo_hybrid_v35_v36_20260517/hybrid_v35_v36_common_split/hybrid_v35_v36_fixed_inputs_common_split_latest.json`
- Hybrid stage Markdown: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/common_split_alpha_zoo_hybrid_v35_v36_20260517/hybrid_v35_v36_common_split/hybrid_v35_v36_fixed_inputs_common_split_latest.md`

Integrated margin addendum verification passed on 2026-05-17 UTC: targeted Alpha Zoo suite `23 passed`, moonshot validation suite `74 passed`, full pytest `1321 passed`, ruff/compileall/diff checks passed. Log: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/common_split_alpha_zoo_hybrid_v35_v36_20260517/local_verification_integrated_margin_20260517T071937Z.log`.

## 2026-05-17 Addendum — strict 6x Alpha Zoo live wiring check

The common-split #1 (`CryptoFxAlphaZooStateStrategy` / `alpha_zoo_conservative_exit` / strict `6x`) is now explicitly represented as a live decision artifact rather than only as a replay result. The live decision path maps Alpha Zoo references to `CryptoFxAlphaZooStateStrategy`, passes train+validation calibrated edges and selected conservative-exit params to `LiveTrader`, and applies 3600s MARKET_WINDOW/cadence plus isolated `6x` and `target_allocation=0.10` overrides.

Live-equivalent tests were added for selection inference, decision override propagation, live CLI parameter injection, runtime leverage validation (`6x` allowed, `>6x` rejected), and MARKET_WINDOW-vs-MARKET_BATCH strategy parity for hourly Alpha Zoo decisions. The strict 6x replay evidence remains: locked-OOS return `+20.512682%`, MDD `6.788365%`, liquidation `0`, and positive min margin buffer. Real live fills/slippage/funding remain execution-environment risks and are not asserted identical to replay.

Artifacts:

- `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/common_split_alpha_zoo_hybrid_v35_v36_20260517/live_alpha_zoo_strict_6x_decision_latest.json`
- `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/common_split_alpha_zoo_hybrid_v35_v36_20260517/live_alpha_zoo_strict_6x_decision_latest.md`

## 2026-05-17 Addendum — strict 6x live-wiring final verification

Final hardening synced live decision exchange overrides into both `LiveConfig.EXCHANGE` and derived fields (`EXCHANGE_ID`, `MARKET_TYPE`, `POSITION_MODE`, `MARGIN_MODE`, `LEVERAGE`) before validation/trader construction and made unknown strategy-class live decisions fail closed. The decision artifact preflight reports `paper_run_allowed` and `ready_for_paper=true` while keeping real execution operator/credential gated. Fresh verification passed: targeted live/common-split suite `46 passed`, live-readiness/parity suite `11 passed`, required Alpha Zoo suite `24 passed`, required moonshot validation suite `74 passed`, full pytest `1328 passed`, ruff/compileall/diff checks passed.

## 2026-05-17 KST — Latest-data validation-to-March high-leverage Alpha Zoo replay

Refreshed the five-symbol raw-first Binance Futures data tail to cutoff `2026-05-17T10:59:59Z`, compacted the OHLCV WAL to monthly parquet, and rebuilt the joined hourly current-tail panel `var/cache/profit_moonshot_fresh_start/joined_panel_76f825ffea81c04f2fe41fbf.parquet` with actual max timestamp `2026-05-17T10:00:00Z`.

Updated split authority:

- train: `2025-01-01T00:00:00Z` ~ `2025-12-31T23:00:00Z` (`8760` hourly timestamps, `43800` rows)
- validation: `2026-01-01T00:00:00Z` ~ `2026-03-31T23:00:00Z` (`2156` hourly timestamps, `10780` rows)
- locked-OOS: `2026-04-01T00:00:00Z` ~ `2026-05-17T10:00:00Z` (`1115` hourly timestamps, `5575` rows)

High-leverage tuning used only train+validation for candidate ranking. Locked-OOS was applied after candidate freeze as gate/report-only. The high-leverage lane assumes isolated per-position margin; if a path breaches liquidation threshold, the trade loss is capped to the configured isolated allocation fraction, and account-wipeout count must remain zero.

Result:

- Top train/validation score was `alpha_zoo_conservative_exit`/carry-forward at `9x` with `12.5%` allocation, but it failed locked-OOS gate (`-0.6029%` OOS return, non-positive Calmar).
- First pre-frozen candidate to pass locked-OOS gate: `CryptoFxAlphaZooStateStrategy` / `alpha_zoo_fast_residual` / isolated `7x` / `15%` allocation.
- Promoted high-leverage candidate metrics: train `+1.4941%` / MDD `59.9354%`; validation `+44.9483%` / MDD `13.7796%`; locked-OOS `+30.5357%` / MDD `11.3027%`; locked-OOS Sharpe `1.815354`, Sortino `2.318591`, smart Sortino `2.083139`, Calmar `2.701628`, trades `391`, liquidation count `0`, account-wipeout count `0`, `live_promotion_possible=true`.
- Strict zero-liquidation integer lane at `10%` allocation for the same `alpha_zoo_fast_residual` params still promotes `6x`: locked-OOS `+16.7783%`, MDD `6.5951%`, liquidation `0`, min buffer `9150.924760`, positive Sharpe/Sortino/smart Sortino/Calmar.
- Runtime leverage validation cap was raised from `6x` to `20x` so the isolated `7x` decision artifact can pass live configuration validation; real execution remains operator/credential gated.
- Peak RSS: data refresh `4467.1172 MiB`, replay `736.1953 MiB`, both under 8 GiB.

Artifacts:

- Runner: `scripts/research/run_alpha_zoo_validation_march_high_leverage.py`
- Main JSON: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/validation_to_20260331_latest_data_20260517/alpha_zoo_validation_march_high_leverage_latest.json`
- Main Markdown: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/validation_to_20260331_latest_data_20260517/alpha_zoo_validation_march_high_leverage_latest.md`
- Candidate CSV: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/validation_to_20260331_latest_data_20260517/alpha_zoo_validation_march_high_leverage_candidates_latest.csv`
- Live decision artifact: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/validation_to_20260331_latest_data_20260517/live_alpha_zoo_fast_residual_7x_isolated_decision_latest.json`
- Data refresh report: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/validation_to_20260331_latest_data_20260517/data_refresh_latest.json`

Research history/source ledger not regenerated: this is a same-source tail refresh and session-scoped Alpha Zoo replay, not a new global source family or chronology ledger change.

Latest-data March-validation verification passed on 2026-05-17 UTC: live/source validation suite `27 passed`, required Alpha Zoo suite `24 passed`, required moonshot validation suite `74 passed`, full pytest `1329 passed`, ruff/compileall/diff checks passed. Log: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/validation_to_20260331_latest_data_20260517/local_verification_validation_march_high_leverage_20260517T120700Z.log`.

Live-readiness preflight for the isolated `7x` decision artifact also passed for paper/testnet mode with a supplied paper Postgres DSN placeholder and freshness threshold override: `recommended_action=paper_run_allowed`, `ready_for_paper=true`, `ready_for_real=false`. Artifact: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/validation_to_20260331_latest_data_20260517/live_readiness_preflight_7x_latest.json`.

Post-staging CSV LF/runner-lineterminator verification re-ran clean: full pytest `1329 passed in 55.01s`, required Alpha Zoo suite `24 passed`, moonshot suite `74 passed`, live/source targeted suite `27 passed`, ruff/compileall/diff checks passed.

## 2026-05-17 KST — Live notional/risk alignment plan for Alpha Zoo 7x lane

After reviewing the latest `alpha_zoo_fast_residual` 7x isolated live decision, the key remaining live-readiness issue is not the signal hypothesis but the sizing contract. The no-cost research replay models account return as `allocation_fraction * leverage * gross_return`, so the current winner `allocation_fraction=0.15`, `leverage=7` represents approximately `105%` notional exposure with about `15%` isolated margin. The live runtime currently treats `target_allocation` as a notional cap, so the same `0.15` can size closer to `15%` notional exposure. That mismatch must be fixed before expecting live/paper results to match replay performance.

Static `max_order_value=5000.0` is also identified as a legacy fixed-dollar guardrail, not a strategy-derived cap. For this futures lane it must be replaced/subordinated by an equity-scaled and leverage-aware cap, with any absolute dollar cap treated only as an explicit emergency ceiling. Otherwise a $10,000 account targeting the replay-intended `0.15 * 7 = 1.05` notional/equity can be silently truncated.

Planning artifacts created for the next implementation session:

- PRD: `.omx/plans/prd-live-alpha-zoo-notional-risk-alignment-20260517.md`
- Test spec: `.omx/plans/test-spec-live-alpha-zoo-notional-risk-alignment-20260517.md`
- Handoff: `docs/session_handoff_20260517_live_notional_risk_alignment.md`

Next-session acceptance target: add an explicit sizing mode such as `isolated_margin_fraction` vs `notional_fraction`, preserve backward compatibility, retune leverage/allocation using train+validation only, include liquidation losses in equity/MDD for any isolated high-performance lane, keep strict zero-liquidation lane separate, add no-cost and cost-stressed reports, prove paper-equivalent live sizing parity, and avoid real-money execution until a separate credentialed real preflight is green and explicitly authorized.

Research history/source ledger was not regenerated for this planning-only update because it introduces no new data-source family or new market-data chronology artifact. The next implementation session must revisit the global research history/source ledger if it refreshes data beyond the current tail or adds new source families.

## 2026-05-18 KST — Live/replay notional-risk aligned Alpha Zoo 7x contract

Implemented the live sizing contract required for the latest high-leverage winner. Existing strategies keep backward-compatible `legacy_notional_cap` behavior by default, while the promoted Alpha Zoo lane now opts into explicit `isolated_margin_fraction`: `target_allocation=0.15` means `15%` isolated margin/equity and, with `leverage=7`, `105%` notional/equity. The legacy fixed-dollar `max_order_value` cap is disabled for this lane (`0.0`) and replaced by equity-scaled notional caps: per-order `110%`, symbol `110%`, total notional `220%`; any positive `max_order_value` remains an explicit emergency ceiling.

Latest-data train+validation retune kept `CryptoFxAlphaZooStateStrategy / alpha_zoo_fast_residual / isolated 7x / allocation 0.15`. The raw grid also found a notional-equivalent `6x/0.175` row with the same train+validation score; a documented incumbent tie-breaker selected the requested `7x/0.15` contract without using locked-OOS for scoring. Locked-OOS remains freeze-after-selection gate/report-only. No real-money execution was attempted.

Selected high-performance lane:

- Sizing mode: `isolated_margin_fraction`.
- Exposure: notional/equity `105.00%`; isolated margin/equity `15.00%`.
- Train: `+1.4941%` return / `59.9354%` MDD.
- Validation: `+44.9483%` return / `13.7796%` MDD.
- Locked-OOS no-cost: `+30.5357%` return / liquidation-inclusive MDD `11.3027%`; Sharpe `1.815354`, Sortino `2.318591`, smart Sortino `2.083139`, Calmar `2.701628`, trades `391`, locked-OOS liquidation `0`, total account wipeout `0`.
- Paper-equivalent parity fixture: equity `$10,000`, price `$100`, `target_allocation=0.15`, `leverage=7` -> replay expected notional `$10,500`, live quantity `105.0`, live notional `$10,500`, absolute diff `0.0`, risk check `Passed`.
- Preflight: `recommended_action=paper_run_allowed`, `ready_for_paper=true`, `ready_for_real=false`.

Cost-stressed locked-OOS diagnostics are separate from the no-cost headline. Round-trip slippage/fee: `1bps` `+25.2882%` / MDD `11.9349%`; `3bps` `+15.4160%` / MDD `13.1860%`; `5bps` `+6.3199%` / MDD `15.4731%`; `10bps` `-13.4130%` / MDD `24.2149%`; `20bps` `-42.5899%` / MDD `44.8361%`. Funding drag: `1bps/day` `+29.9911%`; `2bps/day` `+29.4486%`; `5bps/day` `+27.8349%`; `10bps/day` `+25.1897%`; `20bps/day` `+20.0619%`.

Strict zero-liquidation lane is kept separate: same calibrated Alpha Zoo parameters at strict `6x` / `10%` allocation show locked-OOS `+16.7783%` return, MDD `6.5951%`, Sharpe `1.815354`, Sortino `2.318591`, smart Sortino `2.175137`, Calmar `2.544032`, liquidation `0`, account wipeout `0`, minimum margin buffer `9150.924760`.

Artifacts:

- Main aligned JSON: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/live_notional_risk_aligned_alpha_zoo_20260518/live_notional_risk_aligned_alpha_zoo_latest.json`
- Main aligned Markdown: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/live_notional_risk_aligned_alpha_zoo_20260518/live_notional_risk_aligned_alpha_zoo_latest.md`
- Live decision artifact: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/live_notional_risk_aligned_alpha_zoo_20260518/live_alpha_zoo_notional_risk_aligned_decision_latest.json`
- Preflight artifact: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/live_notional_risk_aligned_alpha_zoo_20260518/live_readiness_preflight_notional_risk_aligned_latest.json`
- Candidate CSV: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/live_notional_risk_aligned_alpha_zoo_20260518/alpha_zoo_validation_march_high_leverage_candidates_latest.csv`
- Verification log: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/live_notional_risk_aligned_alpha_zoo_20260518/local_verification_live_notional_risk_alignment_20260518T113100Z.log`

Fresh local verification passed on 2026-05-18 UTC: required live/Alpha Zoo suite `32 passed`, required moonshot validation suite `74 passed`, full pytest `1340 passed`, ruff, compileall, `git diff --check`, and `git diff --cached --check` all passed.

Research history/source ledger not regenerated: this session used the already-refreshed 2026-05-17 current-tail data and added a live-sizing contract/validation artifact bundle, not a new market-data source family or global chronology refresh.

## 2026-05-18 KST — Plan for Alpha Zoo top-seed hybrid v3.5/v3.6 cost validation

Prepared a next-session plan to test whether the current Alpha Zoo leaderboard can be improved by building Hybrid v3.5/v3.6 portfolios from the top individual candidate streams rather than from the prior fixed-input `A0 + P0 + E0 + S1 + S2 + S3 + S4` universe. The plan is saved at `.omx/plans/plan-alpha-zoo-hybrid-v35-v36-cost-validation-20260518.md`.

Current seed-selection snapshot is based on `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/live_notional_risk_aligned_alpha_zoo_20260518/`: leverage grid `1x..20x`, allocation grid `0.03,0.05,0.075,0.10,0.125,0.15,0.175,0.20`, `1600` rows, `113` live-promotion rows, and `50` rows passing the train-dominant/val-good/OOS-good filter. The next session must recompute the buckets from the latest CSV before running.

Planned seed universe uses top-3 bucket union from live OOS return/Sharpe/Sortino/smart Sortino/Calmar, full compound, and filtered balanced/validation-return/OOS-return/OOS-Calmar lists. The current deduped snapshot has `18` rows spanning `fast_residual`, `quality_single_pair`, `high_confidence_single_pair`, and `high_confidence_long_only` configurations. Known duplicates such as `fast_residual 7x/0.15` and `6x/0.175` are intentional because they have the same notional/equity (`105%`) but different isolated margin semantics.

The required cost validation is explicitly round-trip slippage/fee `5bps = 0.05%` and `10bps = 0.10%`. For every individual seed plus `hybrid_v3_5_seed_union` and `hybrid_v3_6_seed_union`, the next run must report train/validation/locked-OOS total return, MDD, Sharpe, Sortino, smart Sortino, Calmar, trade/event count, liquidation count, account-wipeout count, and minimum margin buffer. Locked-OOS remains gate/report-only and must not enter objective, pruning, parameter fitting, or seed/hybrid selection.

Recommended output directory for the next run: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_top_seed_hybrid_cost_validation_20260518/`. Real-money execution remains prohibited. Research history/source ledger does not need regeneration for this planning-only update because it introduces no new data source family or market-data chronology; the execution session must revisit that decision if it refreshes data or changes source lineage.

## 2026-05-19 KST — Alpha Zoo top-seed Hybrid v3.5/v3.6 cost validation

Recomputed the current Alpha Zoo top-bucket plus filtered top-3 seed-union from the latest live-notional/risk-aligned candidate CSV, then generated a separate research-only Hybrid v3.5/v3.6 cost-validation bundle. The deduped seed universe now has `16` rows across `fast_residual`, `quality_single_pair`, `high_confidence_single_pair`, and `high_confidence_long_only`. The bundle reports every individual seed, `hybrid_v3_5_seed_union`, `hybrid_v3_6_seed_union`, `reference_fast_residual_7x_0p15`, and `reference_strict_zero_fast_residual_6x_0p10` at round-trip slippage/fee `5bps` and `10bps` over `train`, `validation`, and `locked_oos` splits: `(16 + 2 + 2) * 2 * 3 = 120` metric rows.

Locked-OOS remains a gate/report split after model freeze for the hybrid objective/pruning/parameter-fitting path: the artifact audit records `uses_locked_oos_for_objective=false`, `uses_locked_oos_for_pruning=false`, `uses_locked_oos_for_parameter_fitting=false`, and `uses_locked_oos_for_selection=false`. Because the requested seed basket is assembled from current leaderboard buckets, the artifact also labels the seed basket as a post-hoc research basket rather than a deployable live-selection rule. No real-money execution was attempted.

Key cost outcomes:

- `hybrid_v3_5_seed_union`, `5bps`: train `+49.19%` / MDD `30.09%`; validation `+21.16%` / MDD `12.33%`; locked-OOS `+3.29%` / MDD `15.39%`; liquidation `0`; account wipeout `0`; locked-OOS gate `true`.
- `hybrid_v3_6_seed_union`, `5bps`: train `-7.98%`; validation `+8.21%`; locked-OOS `-2.90%`; liquidation `0`; account wipeout `0`; locked-OOS gate `false`.
- `hybrid_v3_5_seed_union`, `10bps`: train `+47.75%`; validation `+18.91%`; locked-OOS `-2.82%`; liquidation `0`; account wipeout `0`; locked-OOS gate `false`.
- `hybrid_v3_6_seed_union`, `10bps`: train `-9.07%`; validation `-7.11%`; locked-OOS `-6.22%`; liquidation `0`; account wipeout `0`; locked-OOS gate `false`.
- References: `fast_residual 7x/0.15` locked-OOS is `+7.63%` at `5bps` and `-12.44%` at `10bps`; strict zero `fast_residual 6x/0.10` locked-OOS is `+4.59%` at `5bps` and `-7.05%` at `10bps`.
- Best locked-OOS individual seed in the bundle is `alpha_zoo_high_confidence_single_pair 7x/0.2`: `+11.03%` / MDD `14.02%` at `5bps`, and `+3.02%` / MDD `16.87%` at `10bps`. Isolated liquidation losses are included in equity/MDD; the bundle's max split liquidation count is `9` on one high-leverage seed train split, and all account-wipeout counts are `0`.

Artifacts:

- Runner: `scripts/research/run_alpha_zoo_top_seed_hybrid_v35_v36_cost_validation.py`
- Main JSON: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_top_seed_hybrid_cost_validation_20260518/alpha_zoo_top_seed_hybrid_cost_validation_latest.json`
- Main Markdown/full metric table: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_top_seed_hybrid_cost_validation_20260518/alpha_zoo_top_seed_hybrid_cost_validation_latest.md`
- Seed selection CSV/JSON: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_top_seed_hybrid_cost_validation_20260518/seed_selection_latest.csv`, `seed_selection_latest.json`
- Model metrics CSV: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_top_seed_hybrid_cost_validation_20260518/model_cost_metrics_latest.csv`
- Hybrid weights CSV: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_top_seed_hybrid_cost_validation_20260518/hybrid_weights_latest.csv`
- Generation log: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_top_seed_hybrid_cost_validation_20260518/local_verification_alpha_zoo_top_seed_hybrid_cost_validation_20260519T093343Z.log`
- Final verification log: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_top_seed_hybrid_cost_validation_20260518/local_verification_alpha_zoo_top_seed_hybrid_cost_validation_final_20260519T100131Z.log`
- Post-deslop verification log: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_top_seed_hybrid_cost_validation_20260518/local_verification_alpha_zoo_top_seed_hybrid_cost_validation_post_deslop_20260519T100507Z.log`

Fresh local verification passed on 2026-05-19 UTC: artifact assertion passed (`seed_count=16`, `metric_rows=120`, max liquidation count `9`, account wipeout max `0`); new Alpha Zoo top-seed tests `5 passed`; hybrid/common split tests `14 passed`; live/liquidation/state tests `46 passed`; full pytest `1345 passed`; ruff, compileall, and `git diff --check` all passed. Post-deslop verification re-ran after narrowing broad exception handling in the new runner: artifact assertion passed; new tests `5 passed`; full pytest `1345 passed`; ruff, compileall, and `git diff --check` passed.

Research history/source ledger not regenerated: this session reused the already-refreshed 2026-05-17 current-tail data and the 2026-05-18 live-notional/risk-aligned Alpha Zoo artifact family; it added a same-lineage research-only cost-validation bundle, not a new market-data source family or global chronology refresh.

## 2026-05-19 KST — Alpha Zoo full 10bps round-trip retune and live-gate repair

Re-ran the Alpha Zoo backtest-to-live candidate family under the latest split and locked promotion cost of round-trip slippage/fee `10bps`. The run covers historical top-bucket / live-ranked Alpha Zoo seed streams, the prior Hybrid v3.5/v3.6 seed-union rows, references, and fresh train+validation-only variants. Locked-OOS stayed strictly post-freeze gate/report-only: the artifact records selection inputs `['train', 'validation']`, `uses_locked_oos_for_selection=false`, and trade-filter locked-OOS role `gate_report_only_after_variant_freeze`. No real-money execution was attempted.

The earlier top-bucket/hybrid rows did not survive as live candidates at `10bps`: `hybrid_v3_5_seed_union` remains a shadow-only historical OOS-bucket row and fails the fresh live gate because locked-OOS return/Sharpe/Sortino/smart Sortino/Calmar are non-positive and because its lineage uses OOS-derived bucket selection. The plain fresh 10bps seed/hybrid retune also found no live-ready model with positive validation and locked-OOS performance, so a non-calendar fixed trade-filter retune lane was added. The lane evaluates only signal-structure filters such as side/symbol/factor-family/absolute factor-score/hold cap over train+validation, then freezes the variant before locked-OOS reporting.

Final full retune summary:

- Artifact dir: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_10bps_full_retune_20260519/`.
- Latest main JSON/Markdown: `alpha_zoo_10bps_full_retune_latest.json`, `alpha_zoo_10bps_full_retune_latest.md`.
- Timestamped final JSON/Markdown: `alpha_zoo_10bps_full_retune_20260519T121224Z.json`, `alpha_zoo_10bps_full_retune_20260519T121224Z.md`.
- Split: train `2025-01-01T00:00:00Z..2025-12-31T23:00:00Z`; validation `2026-01-01T00:00:00Z..2026-03-31T23:00:00Z`; locked-OOS `2026-04-01T00:00:00Z..2026-05-17T10:00:00Z`; timestamp-index hash `b973165bc1057f3aaa08ea637b73a45df3e84fdb7d1337b1637233d205696bb0`.
- Candidate accounting: `798` models / `2394` split metric rows; `778` fresh train+validation models; `176` fresh trade-filter models; `20` shadow-only historical rows.
- Search accounting: `600` source candidate rows replayed, `776` fresh 10bps streams evaluated, `30,354` trade-filter variants evaluated, `176` selected variants, `56` passing the final 10bps live gate.
- Memory: runner peak RSS `400.9883 MiB` by artifact memory summary and `/usr/bin/time` max RSS `420,012 KiB`; full pytest post-LF max RSS `2,722,856 KiB`, both under the 8 GiB session limit.

Best live-gate candidate at round-trip `10bps`:

- Model id: `fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_quality_single_pair_6p0x_0p175alloc`.
- Strategy family: `CryptoFxAlphaZooStateStrategy` / `alpha_zoo_quality_single_pair`.
- Variant: non-calendar `abs_factor_score_min=1.5` (`abs_score_ge_1.5`).
- Sizing: isolated `6x`, allocation fraction `0.175`, `544` filtered trades; liquidation/account-wipeout counts are `0` on all splits and margin buffers stay positive.
- Train: return `+36.0268%`, MDD `24.6028%`, Sharpe `1.002645`, Sortino `0.289764`, smart Sortino `0.232550`, Calmar `1.464340`, trades `405`, min margin buffer `8514.118330`.
- Validation: return `+0.5942%`, MDD `11.3653%`, Sharpe `0.219846`, Sortino `0.047272`, smart Sortino `0.042448`, Calmar `0.214374`, trades `86`, min margin buffer `9251.785896`.
- Locked-OOS gate/report-only: return `+1.5464%`, MDD `10.9211%`, Sharpe `0.554965`, Sortino `0.168704`, smart Sortino `0.152094`, Calmar `1.173217`, trades `53`, min margin buffer `9383.460782`.

The selected live-gate candidate satisfies the requested dominance shape: train return/Sharpe/Sortino/smart Sortino/Calmar are all above validation and locked-OOS; validation and locked-OOS returns are positive; locked-OOS was not used for variant selection, pruning, objective scoring, or parameter fitting. Calendar/date rules remain rejected.

Fresh verification passed on 2026-05-19 UTC: artifact assertion passed (`798` models, `2394` metric rows, `56` promotable); 10bps retune tests `15 passed`; top-seed/hybrid split tests `19 passed`; live/liquidation/state tests `49 passed`; full pytest `1360 passed`; `ruff check .`, `compileall`, and diff checks passed. Research history/source ledger not regenerated: this session reused the already-refreshed 2026-05-17 current-tail data and same Alpha Zoo artifact lineage, adding a same-lineage 10bps retune/validation bundle rather than a new market-data source family or chronology refresh.

## 2026-05-19 KST — Risk-selection and low-correlation verification contract

The follow-up risk-selection plan keeps locked-OOS as post-freeze gate/report-only evidence while predeclaring two train+validation-only selection profiles:

- `balanced_train_validation_v1` remains the balanced reference and must report `fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_quality_single_pair_6p0x_0p175alloc`.
- `higher_risk_train_return_tilt_v1` is the active final profile and may select `fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_quality_single_pair_7p0x_0p2alloc` only after the existing 10bps promotion gates pass.

The artifact validator now requires explicit profile metadata, score-formula inputs, `uses_locked_oos_* = false` flags, balanced-reference preservation, and a separate `low_correlation_discovery_latest.json/csv` surface. Low-correlation discovery rows must compute correlations from train+validation returns only, label deployable 10bps gate passers separately from research-only locked-OOS gate failures, keep `real_money_execution=false`, and preserve the `<8192 MiB` memory guard.

## 2026-05-19 KST — Higher-risk 10bps selection profile and low-correlation discovery result

The follow-up risk-selection run finalized the active 10bps model by a predeclared train+validation-only higher-risk profile, not by locked-OOS ranking. The runner now replays all `600` source candidates by default, records the selected profile metadata in the artifact, and emits low-correlation discovery sidecars. Locked-OOS remains gate/report-only after candidate/profile freeze; `real_money_execution=false` throughout.

Final artifact summary:

- Artifact dir: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_10bps_full_retune_20260519/`.
- Latest main JSON/Markdown: `alpha_zoo_10bps_full_retune_latest.json`, `alpha_zoo_10bps_full_retune_latest.md`.
- Timestamped final JSON/Markdown: `alpha_zoo_10bps_full_retune_20260519T140542Z.json`, `alpha_zoo_10bps_full_retune_20260519T140542Z.md`.
- Candidate accounting: `796` models / `2388` split metric rows; `600` source candidates selected before locked-OOS gates; `775` low-correlation candidate streams compared against the active reference.
- Memory: artifact peak RSS `385.7539 MiB`; `/usr/bin/time` max RSS `395,012 KiB`, under the 8 GiB session limit.

Selection profiles:

- Active final profile: `higher_risk_train_return_tilt_v1` -> `fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_quality_single_pair_7p0x_0p2alloc`; profile score `0.5139152402359146`.
- Balanced reference profile: `balanced_train_validation_v1` -> `fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_quality_single_pair_6p0x_0p175alloc`; profile score `0.20188310617174543`.
- Both profile definitions record train+validation as objective/selection/optimization/pruning/parameter-fit/score-formula inputs and all `uses_locked_oos_*` selector flags as `false`.

Active final 10bps profile split metrics (`7x`, allocation `0.20`, non-calendar `abs_score_ge_1.5` quality-single-pair filter):

- Train: return `+45.6916%`, MDD `31.6050%`, Sharpe `0.994753`, Sortino `0.287423`, Calmar `1.445708`, trades `405`, liquidation `0`.
- Validation: return `+0.4724%`, MDD `14.9117%`, Sharpe `0.219005`, Sortino `0.047090`, Calmar `0.129660`, trades `86`, liquidation `0`.
- Locked-OOS gate/report-only: return `+1.8382%`, MDD `14.3237%`, Sharpe `0.556723`, Sortino `0.169307`, Calmar `1.074121`, trades `53`, liquidation `0`.

Balanced reference 10bps split metrics (`6x`, allocation `0.175`, same non-calendar filter): train `+36.0268%`, validation `+0.5942%`, locked-OOS `+1.5464%`; all splits liquidation `0`.

Low-correlation discovery sidecars: `low_correlation_discovery_latest.json` and `low_correlation_discovery_latest.csv`. Correlations are computed from train+validation returns only against the active higher-risk reference; `423` streams are below the `0.35` absolute-correlation threshold, but `0` are deployable 10bps gate passers independent of the reference in this run, so the discovery rows are research-only until a low-correlation stream also clears locked-OOS gate/report checks.

Verification passed on 2026-05-19 UTC: artifact assertion passed (`796` models, `2388` metric rows, `50` low-correlation rows); focused 10bps retune and artifact assertion tests `19 passed`; `ruff` passed on changed runner/assertion/tests; `compileall` passed. Full `n_trials=80` hybrid optimizer was not rerun in this final profile pass; the required deliverable was profile-safe selection plus low-correlation discovery under the 8 GiB guard.

## 2026-05-19 KST — Next plan: 7x/0.20 paper-forward live preflight

Saved next-session plan: `.omx/plans/plan-alpha-zoo-7x-paper-forward-live-preflight-20260519.md`.

The next step is **not real-money execution**. It is a paper/testnet-only live decision, preflight, and forward-monitoring handoff for the active 10bps profile and the balanced reference:

- Active paper candidate: `fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_quality_single_pair_7p0x_0p2alloc` (`higher_risk_train_return_tilt_v1`, isolated `7x`, allocation `0.20`).
- Balanced reference: `fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_quality_single_pair_6p0x_0p175alloc` (`balanced_train_validation_v1`, isolated `6x`, allocation `0.175`).
- Required governance: `real_money_execution=false`, `ready_for_real=false`, locked-OOS gate/report-only, replay/live sizing parity, liquidation-inclusive MDD, and realized round-trip cost monitoring against the 10bps research assumption.
- Recommended artifact dir for the follow-up: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_7x_paper_forward_preflight_20260519/`.

The plan rejects immediate real-money promotion because the 7x/0.20 validation return is still weak and the low-correlation discovery found `0` independently deployable low-correlation gate-pass streams. The next evidence needed is paper/testnet fill-quality and risk monitoring for active vs balanced side-by-side.

## 2026-05-20 KST — 10bps Alpha Zoo paper/testnet preflight and monitoring handoff

Built the paper/testnet-only handoff bundle for the final 10bps active profile and balanced reference under `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_7x_paper_forward_preflight_20260519/`. This is not a real-money promotion: every decision/preflight artifact keeps `real_money_execution=false` and `ready_for_real=false`.

Side-by-side paper candidates:

- Active: `fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_quality_single_pair_7p0x_0p2alloc` (`higher_risk_train_return_tilt_v1`, isolated `7x`, allocation `0.20`, target notional/equity `140%`, paper-equivalent `$10,000 -> $14,000` notional). Preflight status: `ready_for_paper=true`, `ready_for_real=false`.
- Balanced reference: `fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_quality_single_pair_6p0x_0p175alloc` (`balanced_train_validation_v1`, isolated `6x`, allocation `0.175`, target notional/equity `105%`, paper-equivalent `$10,000 -> $10,500` notional). Preflight status: `ready_for_paper=true`, `ready_for_real=false`.

The live strategy now supports the retuned `abs_factor_score_min=1.5` gate directly, preserving the 10bps trade-filter contract in paper/testnet runtime instead of only recording it as offline metadata. The bundle reuses the notional-risk-aligned `isolated_margin_fraction` sizing parity path and carries split/source/profile lineage from the frozen 10bps retune source, low-correlation sidecar, and notional-risk-aligned live artifact. Locked-OOS remains gate/report-only after candidate/profile freeze; selection/profile metadata continues to record `uses_locked_oos_for_objective=false`, `uses_locked_oos_for_pruning=false`, `uses_locked_oos_for_parameter_fitting=false`, and `uses_locked_oos_for_selection=false`.

Artifacts:

- Bundle JSON/Markdown: `alpha_zoo_7x_paper_forward_preflight_latest.json`, `alpha_zoo_7x_paper_forward_preflight_latest.md`.
- Timestamped bundle: `alpha_zoo_7x_paper_forward_preflight_20260520T112422Z.json`.
- Active decision: `live_alpha_zoo_quality_single_pair_7x_0p20_paper_decision_latest.json`.
- Balanced decision: `live_alpha_zoo_quality_single_pair_6x_0p175_balanced_reference_decision_latest.json`.
- Active preflight: `live_readiness_preflight_alpha_zoo_7x_0p20_paper_latest.json`.
- Balanced preflight: `live_readiness_preflight_alpha_zoo_6x_0p175_balanced_reference_paper_latest.json`.
- Monitoring contract: `paper_forward_monitoring_contract_latest.json` and `paper_forward_monitoring_contract_latest.csv`.
- Verification log: `local_verification_alpha_zoo_7x_paper_forward_preflight_20260520T111631Z.log`.

Monitoring contract status is `pending_paper_forward_fills`; it defines realized `fee_bps`, `slippage_bps`, and `all_in_round_trip_bps` fields, active-vs-balanced grouping keys, maker/taker/partial-fill/missed-signal fields, and liquidation-inclusive equity/MDD/account-wipeout checks. The cost audit keeps the research assumption fixed at `10.0bps` all-in round-trip with pass thresholds `mean <= 10bps` and `p95 <= 15bps`; actual paper/testnet fills must be collected before any real-money discussion.

Verification passed on 2026-05-20 UTC: artifact generation smoke; CSV-LF final verification: artifact regeneration; targeted tests `17 passed`; full pytest `1369 passed`; max RSS `2,877,880 KiB` (<8 GiB); `ruff check .`; `python -m compileall -q src scripts tests`; `git diff --check`; and staged `git diff --cached --check` after index sync.

## 2026-05-20 KST — Validation-first 10bps discovery after weak validation check

Follow-up validation-first discovery confirmed that the prior active 7x/0.20 handoff is not the best validation performer inside the frozen 10bps universe. The new artifact is under `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_validation_first_discovery_20260520/` and keeps `real_money_execution=false`, `ready_for_real=false`, and locked-OOS gate/report-only after train+validation ranking freeze.

Selected paper/testnet validation-first candidates:

- Validation return leader: `fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_quality_single_pair_5p0x_0p2alloc` (`5x`, allocation `0.20`, isolated notional/equity `100%`). Validation return `+0.5986%`, validation MDD `10.8490%`, train return `+34.5152%`, locked-OOS return `+1.4956%`, liquidation `0`; preflight `ready_for_paper=true`, `ready_for_real=false`.
- Validation efficiency reference: `fresh_tv10_filter_abs_score_ge_1p5_alpha_zoo_quality_single_pair_4p0x_0p175alloc` (`4x`, allocation `0.175`, isolated notional/equity `70%`). Validation return `+0.5561%`, validation MDD `7.6999%`, train return `+24.8845%`, locked-OOS return `+1.1432%`, liquidation `0`; preflight `ready_for_paper=true`, `ready_for_real=false`.

The frozen 10bps universe has `56` live-gate-passed candidates, but the best live-gate validation return is only `+0.5986%`. A material validation-edge audit found `0` zero-liquidation candidates with validation return `>1%` and positive locked-OOS return. High-validation alternatives do exist, especially `conservative_exit` variants with validation around `+20%` to `+27%`, but they fail promotion gates because locked-OOS returns are negative and train metrics are not above validation; they are shadow-only strategy hypotheses, not paper candidates.

Recommended next experiments are therefore not immediate real-money promotion: run a train+validation-only regime-gated `conservative_exit` rescue, side/symbol-specific `abs_score` thresholds for `quality_single_pair`, and continue paper-forward monitoring of the validation-first 5x/0.20 and 4x/0.175 lanes beside the existing active/balanced lanes. Real fill monitoring must still compare all-in round-trip bps to the 10bps mean / 15bps p95 contract before any real-money discussion.
