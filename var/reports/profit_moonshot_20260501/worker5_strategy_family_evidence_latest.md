# Profit moonshot worker-5 strategy-family evidence

Generated: `2026-05-01T08:12:31.804594Z`

## Implemented families

- `PanicReboundMeanReversionStrategy` — long-only post-liquidation rebound confirmation with hard stop, take-profit, trailing, and time-stop controls.
- `SessionFilteredPairCarryStrategy` — BNB/TRX pair z-score wrapper with UTC session and expected-move gates before fresh entries.

## Candidate-generation evidence

- Bounded dry-run universe: `80` candidates.
- New profit-reboot candidates: `6`.
  - `panic_rebound_mr_5m_fast_confirm_18_0.018`
  - `panic_rebound_mr_5m_volume_strict_32_0.025`
  - `panic_rebound_mr_15m_slow_confirm_24_0.030`
  - `session_filtered_pair_carry_1h_bnbtrx_overlap_2.2_0.50`
  - `session_filtered_pair_carry_1h_bnbtrx_strict_2.6_0.65`
  - `session_filtered_pair_carry_4h_bnbtrx_asia_us_1.8_0.45`

## Bounded smoke backtests

- PASS: `LQ__OPTIMIZATION__STRATEGY=PanicReboundMeanReversionStrategy uv run python scripts/minimum_viable_run.py --days 45` — legacy CSV synthetic no-infra backtest completed; final_equity=10000.0000 trade_count=0.
- PASS: `LQ__OPTIMIZATION__STRATEGY=SessionFilteredPairCarryStrategy uv run python scripts/minimum_viable_run.py --days 45` — legacy CSV synthetic no-infra backtest completed; final_equity=10000.0000 trade_count=0.

## Promotion note

No new family is promoted as high-return alpha from synthetic smoke only; live-equivalent exact-window promotion still requires raw-first train/validation evidence.
