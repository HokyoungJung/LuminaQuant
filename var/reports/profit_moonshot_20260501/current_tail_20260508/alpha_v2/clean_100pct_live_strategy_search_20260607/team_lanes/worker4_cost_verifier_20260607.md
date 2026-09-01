# Worker-4 cost/turnover/capacity verifier — materialized leader artifact

- Generated UTC: `2026-06-07T07:11:42Z`
- Team task: `6`
- Ultragoal: `G006-independent-cost-turnover-theory-ver`

## Verdict

- Report-only audit: `pass_with_gaps_documented`
- Real money: `fail`
- Small sleeve: `fail`

## Required cost grid

10 / 15 / 20 bps round-trip. Current evidence is strongest at 10 bps; 15 bps is missing for final promotion and 20 bps strict-no-leak stress is diagnostic with high tail drawdown.

## Required production telemetry

RPT, turnover, all-in fees/spread/slippage, BBO availability, timeout, partial-fill, cancel and reject telemetry, plus capacity/liquidity proxy.
