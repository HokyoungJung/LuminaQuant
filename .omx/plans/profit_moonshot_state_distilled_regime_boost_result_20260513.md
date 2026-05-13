# Profit Moonshot State-Distilled Regime Boost Result — 2026-05-13

Implemented the approved `StateDistilledRegimeBoostPortfolio` research lane. Outcome: research artifact complete, strict live promotion false.

- Train/validation-only selection: yes.
- Locked-OOS gate/report-only after freeze: yes.
- Calendar/current-base teacher used for promotion: no.
- Booster leverage cap: configurable up to 25x; real effective max 4.5x after long-term volatility targeting.
- Strict lane: zero liquidation and positive buffers, but validation/OOS economics and risk-quality metrics failed.
- Memory: 292.85 MiB peak RSS in artifact, `/usr/bin/time -v` max RSS 299880 KiB.
- Main artifacts: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/state_distilled_regime_boost_20260513/`.

## Verification evidence

- Artifact assertion passed against latest summary/freeze/gate: generated `2026-05-13T14:04:49Z`, freeze hash `68db1c473bf43778ccdaba7c2e78ab4a754f71dde2557643fa4267b73d8b3535`.
- `uv run --extra dev pytest tests/test_profit_moonshot_regime_boost_portfolio.py -q` → 7 passed.
- `uv run --extra dev pytest tests/test_crypto_fx_alpha_zoo.py tests/test_triple_barrier_labeler.py tests/test_edge_calibration.py tests/test_crypto_fx_alpha_zoo_state_strategy.py -q` → 20 passed.
- `uv run --extra dev pytest tests/test_profit_moonshot_fresh_start_replay.py tests/test_profit_moonshot_liquidation_aware_validation.py tests/test_profit_moonshot_live_final_selection.py tests/test_profit_moonshot_pass_under_8gb_validator.py -q` → 74 passed.
- `uv run --extra dev pytest -q` → 1304 passed.
- `uv run --extra dev ruff check .` → pass.
- `uv run --extra dev python -m compileall -q src scripts tests` → pass.
- `git diff --check` / `git diff --cached --check` → pass.
