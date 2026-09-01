# Performance-first coverage-adjusted retune — 2026-04-17

## Goal
- re-tune the mixed/calm `hybrid_guarded_mode` performance-first override using the
  **coverage-adjusted replay** as the primary path check
- keep `pair_tactical_mode` **tactical-only**
- require agreement across:
  1. current one-shot switch
  2. one-shot threshold frontier
  3. coverage-adjusted replay

## Final threshold choice
- min OOS return edge: `+0.5000%`
- min OOS sharpe edge: `2.5000`
- min hybrid val return: `+6.0000%`
- min hybrid val sharpe: `3.0000`

These replace the earlier, looser profile:
- previous return edge `+0.4000%`
- previous sharpe edge `2.0000`
- previous val return `+5.0000%`
- previous val sharpe `3.0000`

## Why this profile
### 1) One-shot switch still promotes hybrid
Current reboot-validation one-shot state remains:
- favored_group `mixed`
- trend `bullish`
- breadth `broad`
- volatility `calm`
- pair_liquidity `normal`
- live default `hybrid_guarded_mode`

Current hybrid vs balanced edge:
- OOS return edge: `+0.5777%`
- OOS sharpe edge: `+2.7542`
- hybrid val return / sharpe: `+6.5372%` / `3.2857`

So the tuned rounded profile still clears the live gate with visible headroom.

### 2) Frontier still supports the tuned profile
Default frontier sweep still reports:
- passing combinations: `300 / 900`
- max passing return-edge threshold on the default grid: `+0.5000%`
- max passing sharpe-edge threshold on the default grid: `2.5000`
- max passing val-return threshold on the default grid: `+6.0000%`
- max passing val-sharpe threshold on the default grid: `3.0000`

A finer-grained scratch sweep showed that even tighter combinations can still pass
today's exact one-shot numbers (`+0.5500%`, `2.7500`, `+6.5000%`, `3.2500`), but
those thresholds leave only razor-thin margin versus the current one-shot metrics.
They were rejected as too brittle for routine artifact refreshes.

### 3) Coverage-adjusted replay stays unchanged
Under the tuned profile, coverage-adjusted replay remains:
- OOS return: `+0.6839%`
- sharpe: `3.4091`
- max DD: `0.2406%`
- mode counts: `{"balanced_overlay_mode": 14, "core_mode": 9, "defensive_overlay_mode": 3, "hybrid_guarded_mode": 5, "risk_off_mode": 3}`

Strict replay remains weaker because the unresolved early OOS distortion is still the
pair-liquidity coverage gap:
- strict replay OOS return: `+0.3400%`
- strict replay sharpe: `1.6644`
- coverage gap days: `18`

## Interpretation
- coverage-adjusted replay supports tightening the override without reducing the
  replayed path quality
- the tuned rounded profile is materially stricter than the old policy
- the profile intentionally stops short of the absolute fine-grid frontier because
  one-shot headroom gets too thin near the exact boundary
- `pair_tactical_mode` remains benchmark / tactical-only and is still excluded from
  default promotion

## Practical conclusion
- keep `hybrid_guarded_mode` as the current mixed/calm live default
- keep `balanced_overlay_mode` as the smaller-overlay backup
- keep `pair_tactical_mode` tactical-only
- use the tuned rounded profile as the new canonical performance-first override
