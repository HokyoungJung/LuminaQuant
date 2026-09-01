# Session handoff — performance-first switch override applied

## Current checkpoint
- repo: `/home/hoky/Quants-agent/LuminaQuant`
- branch: `private-main`
- rerun baseline commit: `ed5c179`
- current date context: `2026-04-16`

## What changed after the validated reboot rerun
The full reboot rerun had left the live default on `balanced_overlay_mode` because the
mixed/calm promotion gate still blocked hybrid when validation performance lagged balanced.

This follow-up changed the mixed/calm gate to a **performance-first override**:
- if hybrid is healthy,
- beats balanced on refreshed OOS by a large enough margin,
- beats pair tactical on refreshed OOS,
- carries a materially better OOS drawdown profile,
- and still has strong absolute validation metrics,

then `hybrid_guarded_mode` can become the live default even if balanced still wins on validation.

Pair remains tactical-only.

## Current validated result
Canonical artifacts:
- `var/reports/exact_window_backtests/followup_status/portfolio_incumbent_autoresearch_grouped/current_switch_validation_current/refreshed_operating_switch_current/portfolio_operating_switch_latest.md`
- `var/reports/exact_window_backtests/followup_status/portfolio_incumbent_autoresearch_grouped/final_master_scoreboard_current/portfolio_master_scoreboard_latest.md`

Current state:
- favored_group: `mixed`
- confidence: `0.0`
- trend: `bullish`
- breadth: `broad`
- volatility: `calm`
- pair_liquidity_state: `normal`

Current live default:
- **mode:** `hybrid_guarded_mode`
- allocation: `{"hybrid_online_portfolio": 1.0}`

Current reboot-lane performance snapshot:
- hybrid_guarded_mode:
  - OOS return: `+0.6868%`
  - OOS sharpe: `3.2370`
  - OOS max DD: `0.2573%`
  - val return: `+6.5372%`
  - val sharpe: `3.2857`
- balanced_overlay_mode:
  - OOS return: `+0.1091%`
  - OOS sharpe: `0.4828`
  - OOS max DD: `0.5162%`
  - val return: `+8.3078%`
  - val sharpe: `4.1120`
- pair_tactical_mode:
  - OOS return: `+0.2892%`
  - OOS sharpe: `3.2765`
  - tactical-only

Interpretation:
- pair is still tactical-only
- hybrid now wins the diversified live policy lane because the OOS edge is decisive enough
- balanced remains the smaller-overlay backup

## Memory / execution constraints preserved
- heavy runs were still sequential only
- low-memory env remained pinned (`POLARS_MAX_THREADS=1`, `RAYON_NUM_THREADS=1`, etc.)
- no article `batch_01~44` rerun
- no user python/uv heavy process remained alive after completion

## Code changes in this follow-up
- `scripts/research/write_portfolio_operating_switch.py`
  - added performance-first override for mixed/calm promotion
- `tests/unit/test_portfolio_operating_switch.py`
  - added regression coverage for:
    - decisive OOS edge -> hybrid promotion
    - insufficient OOS edge -> keep balanced

## Recommended next goal
If continuing from here, the next useful work is **not** another blind reboot rerun.
Instead, focus on one of:
1. tighter profit-aware switch policy experiments around the mixed/calm override threshold
2. lightweight sensitivity testing around the new override margins
3. operator safeguards for switching back to balanced if hybrid loses the OOS edge

