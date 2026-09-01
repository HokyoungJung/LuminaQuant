# Profit moonshot H3/H4/H5 handoff — 2026-05-09

## Status
- Baseline preserved from prior green `private/main` commit `333f8c7931e75ea3b828f7f7edc714ab66857c10` before H3/H4/H5 edits.
- Remaining hypotheses from `profit_moonshot_next_hypotheses_20260508` were processed without forcing standalone non-calendar promotion.
- H3 implemented calendar-conditioned vetoes using residual z, funding conflict, market-extreme, and flow-exhaustion guards.
- H4 implemented day-of-month plus entry-hour calendar sub-window variants.
- H5 implemented an explicit TRX/ETH calendar spread state machine with two simultaneous legs, volume caps, fees/slippage, stop/take/max-hold, and equity reporting.
- Locked-OOS remains report-only / gate-only; selection remains train/validation-only.

## Bounded H3/H4/H5 replay
Artifacts: `var/reports/profit_moonshot_20260501/current_tail_20260508/h3_h4_h5_calendar_conditioned/`

Command:
```bash
uv run --extra dev python scripts/research/replay_profit_moonshot_fresh_start.py \
  --output-dir var/reports/profit_moonshot_20260501/current_tail_20260508/h3_h4_h5_calendar_conditioned \
  --spec-name-contains fresh_calendar_trx_veto,fresh_calendar_trx_daywin,fresh_calendar_spread \
  --panel-cache-dir var/cache/profit_moonshot_fresh_start
```

Results:
- Specs evaluated: `80` (`64` calendar_rotation H3/H4 variants, `16` calendar_spread H5 variants).
- Replay survivors: `0`; success candidates: `0`.
- Train/validation-positive single sleeves: `54`.
- Peak RSS: `250.707 MiB` (`/usr/bin/time` max RSS `256,724 KiB`), below the <8GiB guard.
- Best replay row: `fresh_calendar_trx_veto_rz10_sweakest_thr180_h168`, train `+2.0672%`, validation `+1.9644%`, locked-OOS `+0.9737%`, MDD `+0.1512%`, Sharpe `5.3905`; failed `oos_return_beats_incumbent` because it did not exceed `1.2181%`.
- Best H4 day-window row: `fresh_calendar_trx_daywin_mid_postfund_sethusdt_thr180_h168`, train `+0.3934%`, validation `+1.2762%`, locked-OOS `+0.6495%`; failed OOS return and trip sufficiency.
- Best H5 spread row: `fresh_calendar_spread_trx_eth_hr50_thr180_h168_tp240`, train `-0.6100%`, validation `-0.1996%`, locked-OOS `+0.0832%`; failed train/validation and OOS gates.

## Portfolio follow-up
Artifacts: `var/reports/profit_moonshot_20260501/current_tail_20260508/h3_h4_h5_calendar_conditioned/portfolio/`

Command:
```bash
uv run --extra dev python scripts/research/tune_profit_moonshot_fresh_portfolio.py \
  --candidate-csv var/reports/profit_moonshot_20260501/current_tail_20260508/h3_h4_h5_calendar_conditioned/fresh_start_overhaul_replay_candidates.csv \
  --output-dir var/reports/profit_moonshot_20260501/current_tail_20260508/h3_h4_h5_calendar_conditioned/portfolio \
  --top-n 18 \
  --calendar-neighborhood-reps 8 \
  --max-sleeves 4 \
  --max-combos-per-size 6000
```

Results:
- Candidate sleeves considered: `18`; portfolio specs evaluated: `20,145`.
- Improved/promoted candidates: `0`.
- Peak RSS: `389.789 MiB` (`/usr/bin/time` max RSS `399,144 KiB`), below the <8GiB guard.
- Selected by validation: additive 4-sleeve veto portfolio, train `+12.1482%`, validation `+13.4822%`, locked-OOS `+2.2229%`, MDD `+0.7026%`, Sharpe `3.2651`; promotion `diagnostic_not_promoted`; failed `oos_return_risk_beats_current_champion` and `oos_mdd_beats_shadow`.
- Diagnostic best OOS: locked-OOS `+3.7780%`, MDD `+0.5981%`, Sharpe `5.2862`; promotion `diagnostic_not_promoted`; failed MDD / return-risk gates.

## Decision
- No H3/H4/H5 candidate is improved under the required gates.
- High-return locked-OOS diagnostics are retained only as quarantined research evidence.
- Current champion remains unchanged; do not promote H3/H4/H5 outputs.

## Verification
- Targeted regression suite: `uv run --extra dev pytest -q tests/test_profit_moonshot_fresh_start_replay.py tests/test_profit_moonshot_fresh_portfolio_tuning.py` -> `21 passed`.
- Focused ruff on touched replay/tests -> passed.
- Full suite: `/usr/bin/time -v uv run --extra dev pytest -q` -> `1201 passed in 311.72s` (wall `4:49.25`, max RSS `2,705,420 KiB`).
- Repo lint: `uv run --extra dev ruff check .` -> passed.
- Compileall: `python3 -m compileall -q src scripts tests` -> passed.
- Whitespace check: `git diff --check` -> passed.

## Remaining risk
- H3 veto portfolios raise train/validation and locked-OOS return when combined, but drawdown remains too high versus the shadow-MDD/current-champion return-risk gates.
- H5 spread construction is behavior-locked and bounded, but its first grid is not profitable on train/validation.
