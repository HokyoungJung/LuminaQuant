# Session handoff — profit moonshot liquidation-aware next step — 2026-05-10

## Current repository state
- Latest pushed evidence commit: `02f4520cf906f48089b8852c2651a0f1e4bd0c1c` on `private/main`.
- GitHub Actions green for that commit:
  - `private-ci` run `25602465256` success.
  - `ci` run `25602465252` success.
- Integer-leverage audit and validator are complete; no new candidate is officially promoted yet.

## Best current interpretation
The correct next candidate is not a new standalone alpha. It is the existing current-base sleeve tuple at **integer `5x` leverage**, if and only if a liquidation-aware replay proves it cannot liquidate under realistic assumptions.

Current base `2.3427x`:
- OOS return `+6.8582%`, MDD `0.8198%`, return/MDD `8.3659`, monthly `+3.0883%`, Sharpe `5.6537`, Sortino `7.3961`, smart Sortino `7.1536`, Calmar `53.7350`.

Forced current-base `5x`:
- OOS return `+14.6371%`, MDD `1.6919%`, return/MDD `8.6514`, monthly `+6.4641%`, Sharpe `5.7215`, Sortino `7.4828`, smart Sortino `6.9764`, Calmar `66.2284`.

Conclusion: if liquidation risk is zero and margin buffer is safe, `5x` is the best known performance candidate.

## Why this is not yet deployable
The current portfolio tuner linearly scales equity curves. It does not yet model:
- exchange liquidation prices,
- intrabar high/low breach checks,
- maintenance margin,
- cross/isolated margin accounting,
- minimum margin buffer / margin ratio,
- forced-liquidation loss handling.

So the next session must implement or run a liquidation-aware validation before promotion.

## New-session prompt saved in repo
Use: `docs/next_session_prompt_profit_moonshot_liquidation_aware_20260510.md`

## Must preserve
- Selection remains train/validation only.
- Locked-OOS remains report-only/gate-only.
- Memory guard remains `<8 GiB`.
- Do not claim `5x` is deployed/promoted until liquidation count is `0` and margin buffer is positive on train/validation/OOS.
