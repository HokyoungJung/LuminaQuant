# Cost-Aware Framework Model Assumptions

- Execution **never** fills on bar close; basis is `next_open` or `mid`.
- Portfolio construction and penalties use current-bar information only; next bar is used only for execution simulation.
- Missing OHLCV rows are dropped deterministically during panel build.
- Liquidity metrics use trailing windows only (no lookahead):
  - `adv`: rolling mean of volume
  - `adtv`: rolling mean of close*volume
  - `sigma`: rolling std of close-to-close returns
- Impact model proxy:
  - `impact_bps = k_strategy * sigma * sqrt(|order_notional| / ADTV) * 10_000`
- Participation cap is enforced each bar with configurable carry/drop behavior.
- Sensitivity and capacity diagnostics are deterministic functions of simulated costs and participation.
- Calibration compares realized vs predicted **impact** bps (not total slippage) and reports error reduction.
