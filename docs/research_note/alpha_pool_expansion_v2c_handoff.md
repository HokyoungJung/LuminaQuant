# Alpha Pool Expansion v2c — Data-PC Handoff (2026-07-09)

Third wave, on top of `alpha_pool_expansion_v2_handoff.md` (PR #41) and
`alpha_pool_expansion_v2b_handoff.md` (PR #42). Same evaluation contract:
**two-tier gate** (pool admission: net-20bps edge > 0 AND DSR > 0 after N_eff AND
incremental orthogonal `factor_ic` > 0 / live promotion: unchanged (a)-(f)),
**NO-SURVIVORSHIP full reporting**, cost grid **10/15/20/30bps on
`backtest_cost_realistic.yaml` MANDATORY**, `emit_candidate_overfit_stats=ON` via
`research.yaml`, admit+merge through `selection.apply_selection_reject_and_dedup`
with strict `robust_score_params`. All lanes `research_only`, 0% real.

## The 10 v2c lanes

| # | Lane | Axis / anchor | Family | Route |
|---|------|---------------|--------|-------|
| 1 | `CrossSectionalNearLowRecoveryStrategy` | 52wk-LOW capitulation-recovery (rebound + argmin recency, residualized on the near-high nearness statistic) | cross_sectional | allow_multi_asset=True |
| 2 | `CrossSectionalTimeUnderWaterStrategy` | Drawdown-DURATION (time since running peak, depth-residualized) — a pure time transform | cross_sectional | allow_multi_asset=True |
| 3 | `CrossSectionalPriceDelayPremiumStrategy` | Hou-Moskowitz D1 price-delay share (new `price_delay.py`; volume-invariance = illiquidity-alias rejection built in) | cross_sectional | allow_multi_asset=True |
| 4 | `InformationDiscretenessMomentumStrategy` | Da-Gurun-Warachka frog-in-the-pan sign census conditioning momentum (new `information_discreteness.py`) | cross_sectional | allow_multi_asset=True |
| 5 | `CrossSectionalIntermediateEchoMomentumStrategy` | Novy-Marx echo as the two-window spread z(t-12..t-7wk) − z(t-6..t-2wk); pure-echo cells ship as decomposition controls | cross_sectional | allow_multi_asset=True |
| 6 | `IdiosyncraticSkewInnovationStrategy` | d(skew)/dt of beta-hedged residuals, non-overlapping 30v30 (Chen-Hong-Stein) — complement of the lottery LEVEL axis | cross_sectional | allow_multi_asset=True |
| 7 | `CrossSectionalOffSessionTugOfWarStrategy` | TradFi-perp cash-anchor tug-of-war (per-UTC-day 1h-bar cash vs off-session decomposition; LPS/Akbas transplant) | cross_sectional (1h) | allow_multi_asset=True |
| 8 | `SilentVolumeShockResolutionStrategy` | Easley-O'Hara information arrival: quiet-price volume shock ARMS, entry on lagged resolution sign | time_series_momentum | per-symbol |
| 9 | `RoundNumberBarrierStrategy` | Osler round-number barrier, ex-ante-frozen grid + placebo-grid falsifier unit test; HIGHEST data-mining prior in the wave | mean_reversion | per-symbol |
| 10 | Family meta-momentum allocator tilt (`quality_gated_allocation.py`, `family_momentum_window` default 0=OFF) | Factor momentum at family granularity, bounded ±30% tilt on M2/MR1 manifest weights | allocator (offline) | manifest route |

## Evaluation notes
- Lane 10's three-arm measurement (base M2 / MR1 turnover-tilt / family tilt) doubles
  as the FIRST live test of the never-backtested M2/MR1 allocator spine — run all
  three arms on identical windows and report the deltas.
- Lane 9 (round numbers): the placebo-grid falsifier must ALSO be run on real data
  (half-shifted grid) — a live signal on the placebo grid falsifies the lane outright.
- Lane 7 evaluates on 1h bars (the only non-1d lane this wave).
- Per-symbol lanes 8/9: RPT-per-split >= 10bps mandatory.
- Full mechanism/gate details: the lane test files are the executable spec;
  design provenance in the wave-3 lane document (research corpus).

Cycle running total: PR #41 (9 lanes) + PR #42 (14 lanes) + this PR (10 lanes) =
**33 new research lanes**, all theory-anchored, build-gated, deterministic-tested.
Report ALL of them — a silently missing candidate is a survivorship defect.
