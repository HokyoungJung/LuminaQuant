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
- Deployable success remains `false`: OOS return beats the invalid current-base reference at strict 6x, but return/MDD `3.007073` is below current-base reference `6.916878`.
- Diagnostic 5x/6x lane is non-promotional and separate: 5x/6x both zero liquidation in this approximate replay, promotion_allowed=false.
- Peak RSS `512.711` MiB (<8 GiB).
- Artifacts: `/home/hoky/Quants-agent/LuminaQuant/var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/crypto_fx_alpha_zoo_real_data_20260513/crypto_fx_alpha_zoo_real_data_summary_latest.json`, `/home/hoky/Quants-agent/LuminaQuant/var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/crypto_fx_alpha_zoo_real_data_20260513/crypto_fx_alpha_zoo_state_replay_latest.json`, `/home/hoky/Quants-agent/LuminaQuant/var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/crypto_fx_alpha_zoo_real_data_20260513/edge_calibration_latest.json`, `/home/hoky/Quants-agent/LuminaQuant/var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/crypto_fx_alpha_zoo_real_data_20260513/candidate_outcome_ledger_latest.jsonl`.
- Research history/source ledger not regenerated: No new external source class or global chronology/source-ledger change; reused existing current-tail cache and 20260512 lagged FRED external-state artifact, added only session-scoped Alpha Zoo artifacts.
