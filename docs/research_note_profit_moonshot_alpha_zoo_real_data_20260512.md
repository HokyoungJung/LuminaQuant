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
