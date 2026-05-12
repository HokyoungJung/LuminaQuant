# Session Handoff — Crypto/FX Alpha Zoo State Foundation (2026-05-12)

## Summary

Implemented the first `CryptoFxAlphaZooStateStrategy` foundation requested by the latest Alpha Zoo / Vibe-Trading / outcome-calibration direction. The change turns the next profit-moonshot path from single-rule mining into a deterministic research-factory scaffold:

1. Alpha191-style crypto/FX factor operators and 63 bounded factor specs.
2. Train/validation-only factor screening with locked-OOS report-only metadata.
3. Factor cards with fail-closed strategy-validity metadata.
4. Triple-barrier outcome labels for candidate entries.
5. Candidate outcome ledger and shrinkage-based edge calibration.
6. Crypto-only strategy that uses FX as USD/JPY risk-regime filter and requires calibrated positive edge before entry.

No live promotion was made. The generated report is a smoke/plumbing artifact only; real train/validation/OOS performance remains a follow-up task.

## Changed files

### Added

- `src/lumina_quant/alpha_zoo/__init__.py`
- `src/lumina_quant/alpha_zoo/operators.py`
- `src/lumina_quant/alpha_zoo/crypto_fx_factors.py`
- `src/lumina_quant/alpha_zoo/factor_card.py`
- `src/lumina_quant/research/__init__.py`
- `src/lumina_quant/research/triple_barrier.py`
- `src/lumina_quant/research/candidate_outcome_ledger.py`
- `src/lumina_quant/research/edge_calibration.py`
- `src/lumina_quant/strategies/crypto_fx_alpha_zoo_state.py`
- `scripts/research/run_crypto_fx_alpha_zoo_screen.py`
- `scripts/research/replay_crypto_fx_alpha_zoo_state.py`
- `scripts/research/calibrate_crypto_fx_edges.py`
- `tests/test_crypto_fx_alpha_zoo.py`
- `tests/test_triple_barrier_labeler.py`
- `tests/test_edge_calibration.py`
- `tests/test_crypto_fx_alpha_zoo_state_strategy.py`

### Modified

- `src/lumina_quant/strategies/registry.py`

### Artifacts

- `.omx/plans/profit_moonshot_crypto_fx_alpha_zoo_state_plan_20260512.md`
- `docs/session_handoff_20260512_crypto_fx_alpha_zoo_state.md`
- `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/crypto_fx_alpha_zoo_state_20260512/`

## Validity and selection controls

- Calendar field guard: valid Alpha Zoo factors reject month/day/hour/session fields.
- `screen_factor_frame()` ranks factors using only `train` and `validation` splits.
- Locked-OOS metrics are written as report-only stats and never enter `selection_score`.
- `FactorCard.strategy_validity` fail-closes if OOS selection, calendar fields, or missing source refs are detected.
- `CryptoFxAlphaZooStateStrategy.strategy_validity` records `calendar_primary=false` and `locked_oos_role=gate_report_only`.
- Strategy entry requires calibrated lower-confidence edge unless disabled for controlled tests/smoke.

## Local verification

- Smoke scripts passed:
  - `run_crypto_fx_alpha_zoo_screen.py`
  - `replay_crypto_fx_alpha_zoo_state.py`
  - `calibrate_crypto_fx_edges.py`
- Targeted pytest: `14 passed in 0.66s`.
- Focused pytest: `20 passed in 1.34s`.
- Full pytest: `1289 passed in 288.64s`.
- Ruff: `All checks passed!`.
- Compileall: passed.
- Hardcoded parameter audit: `new=0`, `baselined=567`.
- `git diff --check`: passed.

## Report snapshot

- `factor_count`: 63
- `screen_row_count`: 720 synthetic smoke rows
- `smoke_signal_count`: 21
- `calibration_bucket_count`: 2
- `decision`: `research_scaffold_only_no_live_promotion`
- `uses_locked_oos_for_selection`: false
- `calendar_primary`: false

## Next recommended work

1. Run the Alpha Zoo screen on real current-tail crypto OHLCV/funding plus FX OHLCV data.
2. Convert strategy/factor entries to triple-barrier outcomes and append them to the candidate ledger.
3. Calibrate edges by `strategy × side × asset × regime × volatility/factor bucket` using train/validation only.
4. Replay the calibrated strategy plus state-distilled leadership/unwind and residual-pair sleeves.
5. Freeze candidates on train/validation, then open locked-OOS as gate/report only.
6. Run strict zero-liquidation integer leverage grid and diagnostic nonfatal 5x/6x lanes separately.

## Risk

This PR proves infrastructure, not alpha. The next session should not treat smoke outputs as economic evidence or live promotion evidence.
