# Task 4 scientist report - market-neutral cross-sectional relative strength

[OBJECTIVE] Evaluate whether existing LuminaQuant profit-moonshot artifacts support a new **market-neutral cross-sectional relative-strength** lane that can honestly beat the current champion under train/validation-only selection, locked-OOS report/gate-only, <8 GiB RSS, and no new dependencies.

[DATA] Primary evidence: `/home/hoky/Quants-agent/LuminaQuant/.omx/team/profit-moonshot-alpha-56afab4e/worktrees/worker-3/var/reports/profit_moonshot_20260501/current_tail_20260508/all_family_expansion/fresh_start_overhaul_replay_candidates.csv` with 6,805 replay candidate rows from the 2026-05-08 all-family expansion. Support inventory: `/home/hoky/Quants-agent/LuminaQuant/.omx/team/profit-moonshot-alpha-56afab4e/worktrees/worker-3/var/reports/profit_moonshot_20260501/current_tail_20260508/continuation/support_inventory_latest.json` with 5 symbols; per-symbol row count range 709,829-2,705,288; taker-flow coverage 3/5; liquidation coverage 0/5.

[DATA] Current champion comparison threshold from team context: locked-OOS return `+1.2181%`, OOS MDD `0.1662%`, OOS Sharpe `6.7264`; locked-OOS is report-only/gate-only, never selection input.

## Family summary

| family | specs n | train+val positive n | survivors n | success n | mean train return 95% CI | mean val return 95% CI | mean OOS return 95% CI | OOS > champion n |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `cross_momentum` | 80 | 0 | 0 | 0 | -2.0382% [-2.2539%, -1.8224%] | -0.3343% [-0.3676%, -0.3009%] | -0.3012% [-0.3330%, -0.2695%] | 0 |
| `cross_sectional_sharpe_rank` | 9 | 0 | 0 | 0 | -2.7069% [-2.7605%, -2.6533%] | -0.3586% [-0.3842%, -0.3329%] | -0.4353% [-0.4617%, -0.4089%] | 0 |
| `cross_sectional_sharpe_reversal` | 48 | 0 | 0 | 0 | -3.3021% [-3.6928%, -2.9114%] | -0.5400% [-0.6069%, -0.4731%] | -0.5147% [-0.5714%, -0.4579%] | 0 |
| `calendar_rotation` | 4392 | 2413 | 300 | 300 | +0.3021% [+0.2467%, +0.3576%] | +0.4524% [+0.4218%, +0.4831%] | +0.1767% [+0.1558%, +0.1976%] | 27 |

[FINDING] Existing cross-sectional relative-strength families are not promotable as currently expressed.
[STAT:n] Target families (`cross_momentum`, `cross_sectional_sharpe_rank`, `cross_sectional_sharpe_reversal`) cover 137 specs; 0 success candidates; 0 replay survivors; 0 train+validation-positive specs.
[STAT:ci] Combined target-family mean validation return is -0.4080%; 95% CI [-0.4424%, -0.3735%].
[STAT:effect_size] Best target locked-OOS return is below champion: -0.0915% vs champion `+1.2181%`.

[FINDING] The strongest existing relative-strength implementation is structurally different from the requested lane: it is a one-position cross-sectional selector, not a simultaneous market-neutral spread/basket.
[STAT:n] Existing target rows use 137 single-spec rows, while explicit simultaneous spread logic is only present in calendar H5 (`fresh_calendar_spread`) rather than target relative-strength families.
[STAT:effect_size] Best validation rows by family: cross_momentum: `fresh_xs_mom_lb48_z175_h96` train -0.6374%, val -0.1286%, locked-OOS -0.1050%, OOS MDD 0.1116%, OOS Sharpe -5.286, round trips train/val/oos 125/22/21, success=False; cross_sectional_sharpe_rank: `fresh_cross_sharpe_rank_lb24_r5` train -2.6247%, val -0.3258%, locked-OOS -0.4751%, OOS MDD 0.4981%, OOS Sharpe -14.647, round trips train/val/oos 572/80/89, success=False; cross_sectional_sharpe_reversal: `fresh_cross_sharpe_reversal_lb72_r3_h48` train -1.7160%, val -0.2580%, locked-OOS -0.2958%, OOS MDD 0.2958%, OOS Sharpe -23.523, round trips train/val/oos 435/71/81, success=False.

[FINDING] A follow-up is feasible only as a **true neutral construction**, not as another broad all-hours standalone rank sweep.
[STAT:n] Universe breadth is only 5 symbols, so each rebalance can form at most a small top-vs-bottom basket; current support has 709,829+ rows on all symbols, enough for price/return ranks, but cross-sectional degrees of freedom are limited.
[STAT:ci] The calendar reference has 300 successes out of 4392 specs and mean OOS return +0.1767% with 95% CI [+0.1558%, +0.1976%], which is materially better than target-family mean OOS CIs in this artifact.

## Minimal implementable follow-up spec

1. Add a narrow family such as `cross_sectional_relative_strength_spread` to `scripts/research/replay_profit_moonshot_fresh_start.py` rather than expanding existing one-position `cross_momentum`/Sharpe selectors.
2. At each rebalance, rank symbols by residualized return/Sharpe over 24-72h, then open simultaneous long-top / short-bottom legs with equal dollar risk and explicit net exposure tolerance (`abs(long_notional-short_notional) <= 5% gross`).
3. Keep the grid tiny: lookback `{24,48,72}`, hold `{12,24,48}`, rank spread floor `{0.08,0.12}`, gross allocation no larger than current 0.8%, max 1 long + 1 short initially.
4. Add gates before replay survivor status: positive train and validation, train/validation MDD no worse than current single-sleeve calendar reference, locked-OOS return `>1.2181%`, locked-OOS MDD `<=0.1662%`, zero liquidations, and no OOS-based ranking.
5. Tests: extend `tests/test_profit_moonshot_fresh_start_replay.py` with deterministic pair-open/close, dollar-neutral accounting, and a regression that no OOS-ranked/MDD-failed row becomes `success_candidate`.

[LIMITATION] This report is artifact-driven; it does not run a new heavy replay. The inference that a true neutral construction is the only plausible follow-up comes from existing one-position target-family failures and the separate H5 spread-state-machine precedent.
[LIMITATION] The cross-sectional universe is small (`n=5` symbols), so statistical confidence for cross-sectional ranking is inherently weaker than in equity universes; any implementation should remain a bounded probe, not a broad moonshot sweep.
[LIMITATION] Taker-flow coverage is incomplete (3/5) and liquidation coverage is zero (0/5), so this lane should rely on price/return/funding/risk controls only unless data collection changes.

Figure: `/home/hoky/Quants-agent/LuminaQuant/.omx/team/profit-moonshot-alpha-56afab4e/worktrees/worker-3/.omx/scientist/figures/task4_cross_sectional_family_returns.svg` (stdlib SVG fallback saved because matplotlib is unavailable: No module named 'matplotlib').

## Verification evidence

- Parsed CSV with Python REPL/stdlib `csv.DictReader`; no raw DataFrames emitted.
- Visualization saved as stdlib SVG fallback because matplotlib is unavailable in this worker environment; no package installation performed.
- Artifact inputs were read-only; report/figure are the only intended worktree outputs.
