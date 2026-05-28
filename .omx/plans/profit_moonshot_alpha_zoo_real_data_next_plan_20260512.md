# Profit Moonshot Next Plan — Alpha Zoo Real-Data Calibration (2026-05-12)

## What to use now

Use `CryptoFxAlphaZooStateStrategy` as the next primary research path.

Do **not** use the old current-base/calendar tuple as a live candidate or selection target. It remains a `hypothesis_reference_only` teacher/reference because its locked-OOS economics are strong but its calendar-primary/fixed month behavior is invalid for live promotion.

Use these existing components:

- Alpha Zoo factor layer: `src/lumina_quant/alpha_zoo/`
- Outcome/calibration layer: `src/lumina_quant/research/triple_barrier.py`, `candidate_outcome_ledger.py`, `edge_calibration.py`
- Strategy scaffold: `src/lumina_quant/strategies/crypto_fx_alpha_zoo_state.py`
- Research scripts:
  - `scripts/research/run_crypto_fx_alpha_zoo_screen.py`
  - `scripts/research/replay_crypto_fx_alpha_zoo_state.py`
  - `scripts/research/calibrate_crypto_fx_edges.py`
- State-distilled baseline/seed: `state_distilled_leadership_unwind` and `state_distilled_external_risk_filter`

## Current status

- Latest pushed green head: `fcc63f6c053c451152b0d780fa84ee91b5512f82` on `private/main`.
- GitHub Actions green:
  - private-ci: `https://github.com/hoky1227/Quants-agent/actions/runs/25738751636`
  - ci: `https://github.com/hoky1227/Quants-agent/actions/runs/25738751722`
- Calendar/current-base reference remains economically strong but invalid:
  - locked-OOS return `+6.4281%`
  - return/MDD `6.9169`
  - role: `hypothesis_reference_only`, not selector/live strategy
- Best valid state-distilled external-risk strict row at 4x:
  - `fresh_state_distilled_ext_both_lb168_fast72_z075_ret180_h168_tp600_fl0_xr125`
  - train `+30.9030%`, validation `+12.4704%`, locked-OOS `+2.4852%`
  - liquidation count `0/0/0`, min margin buffers positive
  - strategy-valid, but `deployable_success=false` because it does not beat current-base reference economics
- Crypto/FX Alpha Zoo foundation exists but is only a smoke/research scaffold. It has not yet been run on real full current-tail economics.

## Non-negotiable constraints

1. No calendar/month/day/hour entry rules in valid candidate families.
2. Invalid current-base/calendar tuple is teacher/reference only; never a selection target.
3. Train/validation-only selection. Locked-OOS opens only after candidate freeze as gate/report-only.
4. Factor cards and strategy-validity metadata must fail closed on calendar-primary or OOS-selected candidates.
5. Strict deploy lane requires:
   - train/validation/OOS liquidation count `0`
   - every split min margin buffer `> 0`
   - OOS MDD `<= 25%`
   - OOS return beats baseline reference; return/MDD is diagnostic/report-only
6. Diagnostic nonfatal 5x/6x lane must remain separate and non-promotional.
7. Memory must stay below 8 GiB.

## Next session TODO

### A. Wire Alpha Zoo to real data

- Inspect current current-tail crypto panel/data loaders used by profit moonshot replays.
- Build a real-data input adapter for `run_crypto_fx_alpha_zoo_screen.py` instead of synthetic smoke data.
- Include crypto OHLCV/funding/OI/flow fields when available.
- Include FX/regime features if reliable data exists; otherwise use FRED lagged risk-state as a temporary regime filter and record that FX direct trading is still blocked.

### B. Real train/validation factor screen

- Run Alpha Zoo screen on real current-tail data.
- Persist factor cards with:
  - `calendar_primary=false`
  - `uses_locked_oos_for_selection=false`
  - source/data coverage metadata
  - train/validation IC/quantile/triple-barrier stats
- Do not rank by locked-OOS.

### C. Triple-barrier candidate ledger

- Convert strategy/factor entries into triple-barrier outcomes.
- Append candidate ledger rows with realized net PnL, stop/take/time exit, MAE/MFE, cost/funding/spread fields where available.
- Bucket by `strategy × side × asset × regime × volatility/factor_bucket`.

### D. Edge calibration

- Calibrate only on train/validation.
- Require positive lower-confidence edge.
- Use shrinkage for sparse buckets.
- Block/downsize excessive tail-loss buckets.

### E. Stateful replay and portfolio comparison

Replay a narrow, interpretable set:

1. `CryptoFxAlphaZooStateStrategy` real-data calibrated variant.
2. `state_distilled_external_risk_filter` best valid seed.
3. Residual-pair/state-distilled combo only if selected by train/validation.

Freeze candidates on train/validation, then open locked-OOS only for report/gate.

### F. Liquidation-aware validation

- Run integer leverage grid `1x..6x`.
- Report:
  - strict zero-liquidation best leverage
  - diagnostic nonfatal 5x/6x performance
  - liquidation count/event drawdown/equity loss/recovery separately
- Do not promote any row with strict-lane liquidation count `>0` or min margin buffer `<=0`.

### G. Verification and handoff

Run at minimum:

```bash
uv run --extra dev pytest tests/test_crypto_fx_alpha_zoo.py tests/test_triple_barrier_labeler.py tests/test_edge_calibration.py tests/test_crypto_fx_alpha_zoo_state_strategy.py -q
uv run --extra dev pytest tests/test_profit_moonshot_fresh_start_replay.py tests/test_profit_moonshot_liquidation_aware_validation.py tests/test_profit_moonshot_live_final_selection.py tests/test_profit_moonshot_pass_under_8gb_validator.py -q
uv run --extra dev pytest -q
uv run --extra dev ruff check .
uv run --extra dev python -m compileall -q src scripts tests
git diff --check
git diff --cached --check
```

Then write/update:

- `.omx/notepad.md`
- `.omx/plans/profit_moonshot_alpha_zoo_real_data_next_plan_20260512.md` or successor
- `docs/research_note/research_note.md` or successor
- `docs/session_handoff_*alpha_zoo*_20260512.md` or successor
- `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/crypto_fx_alpha_zoo_*` or successor

## Stop condition

Stop only when one of these is true:

1. A valid strict deployable candidate beats the invalid calendar reference on OOS return with zero liquidations, positive min buffers, OOS MDD <=25%, and positive Sharpe/Sortino/smart Sortino/Calmar; return/MDD remains diagnostic/report-only; or
2. Real-data Alpha Zoo path is exhausted with reproducible artifacts showing why no live promotion is allowed, plus the next best research seed is documented.


## Research-note update requirement

Every future session must update the active research notes before final handoff. At minimum, update `.omx/notepad.md`, the active `.omx/plans/*` file, and `docs/research_note/research_note.md` or its successor. If the session changes the global source inventory or research chronology, regenerate/update `docs/research_note/research_history.md` plus the matching `var/reports/profit_moonshot_20260501/research_history/` artifacts, or explicitly state why regeneration was not required.

## 2026-05-13 — Real-data Alpha Zoo calibrated replay result

- Ran real current-tail Alpha Zoo screen against `/home/hoky/Quants-agent/LuminaQuant/var/cache/profit_moonshot_fresh_start/joined_panel_de62df511cec53df6ad39521.parquet` with lagged FRED context; direct FX OHLCV trading stayed blocked because current-tail cache contains crypto OHLCV only.
- Factor/card validity passed fail-closed gates: `calendar_primary=false`, `uses_locked_oos_for_selection=false`, strategy validity pass.
- Candidate outcome ledger: `45160` rows; train+validation `30494`; locked-OOS `14666`.
- Edge calibration physically filtered to train/validation: input `45160`, calibration `30494`, locked-OOS calibration `0`, excluded locked-OOS `14666`.
- Replay grid selected `alpha_zoo_conservative_exit` from `9` formulaic candidates using train/validation metrics only; locked-OOS remained hidden until candidate freeze.
- Strict zero-liquidation lane highest safe integer: `6.0x`, liquidation count `0`, min buffer `9049.125962`, OOS return `41.0967%`, OOS MDD `13.6667%`, return/MDD `3.007073`, Sharpe `2.143209`.
- Deployable success is `true` under the latest operator correction: OOS return beats the invalid current-base reference at strict 6x, while return/MDD `3.007073` vs `6.916878` is diagnostic-only.
- Diagnostic 5x/6x lane is non-promotional and separate: 5x/6x both zero liquidation in this approximate replay, promotion_allowed=false.
- Peak RSS `512.711` MiB (<8 GiB).
- Artifacts: `/home/hoky/Quants-agent/LuminaQuant/var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/crypto_fx_alpha_zoo_real_data_20260513/crypto_fx_alpha_zoo_real_data_summary_latest.json`, `/home/hoky/Quants-agent/LuminaQuant/var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/crypto_fx_alpha_zoo_real_data_20260513/crypto_fx_alpha_zoo_state_replay_latest.json`, `/home/hoky/Quants-agent/LuminaQuant/var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/crypto_fx_alpha_zoo_real_data_20260513/edge_calibration_latest.json`, `/home/hoky/Quants-agent/LuminaQuant/var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/crypto_fx_alpha_zoo_real_data_20260513/candidate_outcome_ledger_latest.jsonl`.
- Research history/source ledger not regenerated: No new external source class or global chronology/source-ledger change; reused existing current-tail cache and 20260512 lagged FRED external-state artifact, added only session-scoped Alpha Zoo artifacts.

## 2026-05-14 — Return/MDD-diagnostic policy result

- Applied latest operator correction: OOS return/MDD is diagnostic-only; OOS return remains the current-base reference gate.
- Regenerated real-data artifacts under `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/crypto_fx_alpha_zoo_real_data_20260514/` using `--strict-real-data` and explicit output paths.
- Screen rows `58,845`; factor count `63`; selected cards `20`.
- Candidate ledger `67,259` rows; train+validation `45,311`; locked-OOS `21,948`.
- Edge calibration records `45,311`; locked-OOS calibration `0`; calibrated edge keys `12`.
- Replay selected `alpha_zoo_conservative_exit` from `9` formulaic train/validation candidates.
- Strict 6x lane: OOS return `41.0967%`, MDD `13.6667%`, return/MDD `3.007073`, Sharpe `2.143209`, Sortino `2.841936`, smart Sortino `2.500237`, liquidation `0`, min buffer `9049.125962`.
- Decision: `deployable_success=true`; return/MDD `3.007073` is reported versus current-base `6.916878` but is not a gate.
- Diagnostic 5x/6x lane remains non-promotional and separate.
- Peak RSS `626.7266 MiB`; source ledger not regenerated because no new global source family was introduced.

## 2026-05-14 — Paper-forward diagnostics follow-up

- Added non-promotional paper-forward diagnostics to the Alpha Zoo replay/summary artifacts for the current train/validation-selected `alpha_zoo_conservative_exit` strict 6x candidate.
- New diagnostics report locked-OOS PnL by regime, symbol, side, dominant factor family, and exit reason, plus round-trip slippage and conservative funding-cost sensitivity.
- Locked-OOS diagnostic highlights at 6x/10% allocation: regime `neutral` +41.0967% (540 trades); symbols SOL +19.2672%, BNB +10.3167%, TRX +4.9566%, ETH +1.4843%, BTC +0.6807%; side SHORT +26.3040%, LONG +11.7120%; factor families residual momentum +30.1822%, residual reversal +8.4241%, volume/vwap pressure -0.0370%; exit reasons score_exit +41.0936%, take_profit +16.1976%, stop_loss -13.8700%.
- Slippage sensitivity: 0/2.5/5/10/20 bps round-trip -> locked-OOS +41.0967%/+30.1241%/+20.0034%/+2.0585%/-26.1930%.
- Funding drag sensitivity: 0/1/2/5/10 bps per day -> locked-OOS +41.0967%/+40.4210%/+39.7486%/+37.7505%/+34.4835%.
- These diagnostics do not change the promotion gate; strict live promotion remains only through the strict zero-liquidation lane, with return/MDD diagnostic-only.

## 2026-05-16 KST — Hybrid/Optuna comparison execution result

Status: initial artifact generation and verification completed; 2026-05-17 corrections applied because calendar/current-base rows and nested-hybrid/same-family rows must be excluded from the strict core universe, not merely marked invalid.

- Baseline preserved: `private/main` `1c6816fced44d277f6c7112934c9dded65ba710f`.
- Comparison artifact directory: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/hybrid_optuna_alpha_zoo_comparison_20260514/`.
- Strict core candidate rows after calendar/current-base plus nested-hybrid/same-family exclusion: `1`; calendar/current-base quarantined rows: `5`; nested-hybrid/same-family quarantined rows: `9`.
- Live promotion possible: `crypto_fx_alpha_zoo_state_calibrated`.
- OOS-objective/ranking/selection invalid rows remain diagnostic only; nested hybrid/portfolio/allocator/leverage-sweep/static-blend/same-family tuning outputs are no longer strict-core candidates even when train/validation selection metadata is clean.
- Calendar/current-base rows: `5`, excluded from strict core and stored only in quarantine/reference ledger.
- Nested-hybrid/same-family rows: `9`, excluded from strict core and stored only in quarantine/reference ledger; no hybrid/Optuna/tuning row is live-promotable.
- Strict 1x..6x recheck confirms Alpha Zoo strict `6.0x` as highest zero-liquidation deployable integer; diagnostic 5x/6x remains non-promotional.
- Max observed peak RSS: `1239.703125 MiB` (<8 GiB).
- Research history/source ledger: not regenerated; no new source family/global chronology change.


## 2026-05-17 KST — Fixed-input hybrid v3.5/v3.6 method adaptation result

- External method concept verified from `/home/hoky/DeepLearning/ensemble_strategies`: v3.6 = v3.5 core plus online dynamic default-candidate refresh; v3.5 Optuna knobs/weight-ratio/high-vol boost/max-weight remain learned/frozen before locked-OOS.
- Implemented fixed-input experiment for `A0 + P0 + E0 + S1 + S2 + S3 + S4` in `scripts/research/run_profit_moonshot_hybrid_v35_v36_fixed_inputs.py`.
- Generated report artifacts under `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/hybrid_v35_v36_fixed_inputs_20260517/`.
- Result: v3.5 locked-OOS +8.5233% / MDD 1.7654%; v3.6 locked-OOS +7.7916% / MDD 1.7491%; both train/validation-only and locked-OOS report-only, but both live-promotion false because mixed hybrid margin replay is not available. Alpha Zoo strict 6x remains live-promotion anchor (+41.0967% OOS, zero liquidations, positive buffers).
- Research history/source ledger not regenerated because no new global source family was introduced.

## 2026-05-17 — Common-split comparison closeout

- Implemented additive common-split runner `scripts/research/run_common_split_alpha_zoo_hybrid_v35_v36.py` and regression tests `tests/test_common_split_alpha_zoo_hybrid_v35_v36.py`.
- Common split is now the comparison authority: train `2025-01-01T00:00:00Z`~`2025-12-31T23:00:00Z`, validation `2026-01-01T00:00:00Z`~`2026-02-28T23:00:00Z`, locked-OOS `2026-03-01T00:00:00Z`~`2026-05-06T23:00:00Z`.
- Old Alpha Zoo split is historical-only. Common-split carry-forward and reselected Alpha Zoo both land on `alpha_zoo_conservative_exit` strict 6x, OOS `+20.5127%`, MDD `6.7884%`, Sharpe `1.772136`, Sortino `2.578776`, smart Sortino `2.414847`, Calmar `3.021741`, liquidation `0`, positive buffers, deployable success true.
- Fixed-input hybrid v3.5/v3.6 Optuna used train+validation only and did not use locked-OOS for objective/pruning/selection. v3.5 OOS `+8.5233%`, v3.6 OOS `+7.7916%`; both remain non-promotable because integrated margin replay is missing.
- Strict integer 1x..6x common-split recheck keeps 6x as highest strict deployable; diagnostic 5x/6x remains separate/non-promotional.
- Main artifacts live under `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/common_split_alpha_zoo_hybrid_v35_v36_20260517/`.
- Research history/source ledger not regenerated: no new global source family or chronology ledger change.
