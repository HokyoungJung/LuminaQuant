# Profit Moonshot Crypto/FX Alpha Zoo State Plan — 2026-05-12

## Decision

Build a research-factory foundation instead of another hand-tuned moonshot rule:

`CryptoFxAlphaZooStateStrategy = Alpha191-style formula factors + state-distilled crypto residuals + FX USD/JPY regime filter + triple-barrier outcome ledger + calibrated edge gate`.

This is **research scaffold only**. It does not promote a live strategy because no real train/validation/OOS live-equivalent replay has yet beaten the invalid current-base reference under strict zero-liquidation rules.

## Hard constraints preserved

- No calendar/month/day/hour entry rules in valid candidate families.
- Train/validation-only selection; locked-OOS remains gate/report-only.
- Factor cards fail closed when `calendar_primary`, calendar fields, missing source refs, or OOS selection are detected.
- Strict live promotion is separate from diagnostic/nonfatal liquidation analysis.
- Strategy entries fail closed unless calibrated lower-confidence edge is positive, unless explicitly disabled for tests/smoke.
- FX v0 is a crypto risk-regime filter, not a directly traded sleeve.

## Implemented v0 scope

- `src/lumina_quant/alpha_zoo/`: bounded operator/factor/card layer with 63 crypto/FX factors.
- `src/lumina_quant/research/`: triple-barrier labeler, candidate outcome ledger, edge calibration with shrinkage/tail-loss decisions.
- `src/lumina_quant/strategies/crypto_fx_alpha_zoo_state.py`: crypto-only state strategy using residual momentum/reversal, volume/VWAP proxy, breakout failure, trend efficiency, FX regime adjustment, and calibration gate.
- Research scripts:
  - `scripts/research/run_crypto_fx_alpha_zoo_screen.py`
  - `scripts/research/replay_crypto_fx_alpha_zoo_state.py`
  - `scripts/research/calibrate_crypto_fx_edges.py`
- Registry: `crypto_fx_alpha_zoo_state` is live opt-in with small optuna trial cap.

## Artifacts

- Report directory: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/crypto_fx_alpha_zoo_state_20260512/`
- Summary: `crypto_fx_alpha_zoo_state_v0_summary_latest.json` / `.md`
- Smoke factor screen: `crypto_fx_alpha_zoo_screen_latest.json` / `.md`
- Smoke replay: `crypto_fx_alpha_zoo_state_replay_smoke_latest.json`
- Calibration smoke: `edge_calibration_smoke_latest.json`

Smoke result: 63 factors, 720 synthetic rows, 21 strategy smoke signals, 2 calibration buckets. This proves plumbing and gates, not profitability.

## Verification completed locally

- Smoke scripts: screen, replay, and calibration completed successfully.
- Targeted pytest: `14 passed`.
- Focused pytest with registry/factory coverage: `20 passed`.
- Full pytest: `1289 passed in 288.64s`.
- `uv run --extra dev ruff check .`: passed.
- `uv run --extra dev python -m compileall -q scripts tests src`: passed.
- Hardcoded parameter audit: `new=0`, `baselined=567`.
- `git diff --check`: passed.

## Next execution lane

1. Feed real crypto OHLCV/funding and FX OHLCV into the Alpha Zoo screen.
2. Generate candidate outcome ledger via triple-barrier labels from real candidate entries.
3. Calibrate edge buckets using train/validation only; block buckets with non-positive lower-confidence edge or excessive tail loss.
4. Run stateful live-equivalent replay for `CryptoFxAlphaZooStateStrategy` plus state-distilled leadership/unwind and residual-pair sleeves.
5. Open locked-OOS only after train/validation freeze; report/gate only.
6. Run integer leverage grid 1x..6x with strict zero-liquidation promotion lane and separate diagnostic 5x/6x nonfatal lane.
