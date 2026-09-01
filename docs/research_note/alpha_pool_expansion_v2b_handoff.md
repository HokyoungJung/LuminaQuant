# Alpha Pool Expansion v2b — Data-PC Handoff (2026-07-09)

Second breadth wave on top of `alpha_pool_expansion_v2_handoff.md` (wave-1, PR #41).
Same evaluation contract: **two-tier gate** (pool admission: net-20bps edge > 0 AND
DSR > 0 after N_eff AND incremental orthogonal `factor_ic` > 0 vs existing sleeves /
live promotion: unchanged (a)-(f)), **NO-SURVIVORSHIP full reporting** of every
candidate including rejects, cost grid **10/15/20/30bps on
`configs/profiles/backtest_cost_realistic.yaml` (sqrt-impact + funding) MANDATORY
for admission**, `emit_candidate_overfit_stats=ON` via `configs/profiles/research.yaml`,
per-fold admit AND final merge routed through `selection.apply_selection_reject_and_dedup`
with strict `robust_score_params`. All 14 sleeves are `research_only`, 0% real allocation.

## The 14 v2b lanes (all build-gated at author time vs their real incumbents)

| # | strategy_class | Axis / anchor | Family | Route |
|---|----------------|---------------|--------|-------|
| 1 | `TrendGatedResidualMomentumStrategy` | BHM 2011 residual momentum on BTC-purged returns; ADF NON-stationarity anti-gate (exact complement of `StationarityGatedResidualReversion`) | cross_sectional | allow_multi_asset=True |
| 2 | `CrossSectionalCapitalGainsOverhangStrategy` | Grinblatt-Han 2005 disposition / capital-gains overhang (volume-weighted cost basis; new `grinblatt_han_reference_price`) | cross_sectional | allow_multi_asset=True |
| 3 | `CrossSectionalDownsideBetaAsymmetryStrategy` | Ang-Chen-Xing 2006 downside-beta premium (beta_minus − beta_plus vs BTC; Zhang 2021 JBF crypto prior) | cross_sectional | allow_multi_asset=True |
| 4 | `SystematicCoskewnessPremiumStrategy` | Harvey-Siddique 2000 co-skewness premium (beta-residualized third co-moment) | cross_sectional | allow_multi_asset=True |
| 5 | `DownsideTailRiskPremiumStrategy` | ES_5% lower-tail level, vol-neutralized (Zhang 2021; Dobrynskaya 2024) — SHAPE-vs-LEVEL distinct from Hill tail-index | cross_sectional | allow_multi_asset=True |
| 6 | `LongRunOverreactionReversalStrategy` | De Bondt-Thaler long-run overreaction, 3-6mo formation + skip-month, liquid majors | cross_sectional | allow_multi_asset=True |
| 7 | `CrossSectionalSeasonalPersistenceStrategy` | Heston-Sadka same-week-of-quarter RELATIVE persistence, time+XS demeaned (provably not calendar timing) | cross_sectional | allow_multi_asset=True |
| 8 | `CrossSectionalCloseLocationAccumulationStrategy` | CLV/Chaikin money-flow accumulation DOUBLY residualized on momentum + nearness (new `cross_sectional_residualize`) | cross_sectional | allow_multi_asset=True |
| 9 | `CrossSectionalRegressionTrendQualityStrategy` | Signed R² of log-price OLS (smooth-trend continuation; distinct from Kaufman-ER axis) | cross_sectional | allow_multi_asset=True |
| 10 | `CrossSectionalPathConvexityStrategy` | Orthonormal quadratic curvature of log price; ZERO first-order momentum loading by construction | cross_sectional | allow_multi_asset=True |
| 11 | `PriceVolumeCorrContinuationStrategy` | Ying-Karpoff volume-confirmed continuation (per-symbol weekly gate) | time_series_momentum | per-symbol |
| 12 | `SpreadStressLiquidityReversionStrategy` | Nagel 2012 liquidity-provision reversal gated on Corwin-Schultz HL-spread z-stress episodes (new `corwin_schultz_spread`); HIGH prior of death | mean_reversion | per-symbol |
| 13 | `MomentumCrashDynamicScalingOverlayStrategy` | Daniel-Moskowitz 2016 continuous crash throttle (bear × rebound + BSC child-own-vol), de-risk-only wrapper | overlay | TSMOM child |
| 14 | `AvgCorrelationCrashGuardOverlayStrategy` | Engle-Kelly equicorrelation z-spike crash guard (Pollet-Wilson 2010), de-risk-only wrapper | overlay | TSMOM child |

New pure indicators shipped: `reference_price.py` (GH cost basis), `hl_spread.py`
(Corwin-Schultz), `cross_sectional_residualize.py`, `comoment.py` (semibeta +
coskewness), `log_price_regression.py` (orthonormal basis / trend quality /
convexity). All pure Python/numpy, no scipy/sklearn/statsmodels.

## Evaluation notes (per-lane falsifying measurements)

- Weekly XS lanes (1,2,5,7,8,9,10): falsified by net-20bps XS long-short Sharpe <= 0
  OR incremental `factor_ic` <= 0 vs the named nearest incumbent. Expected NULL for
  the lower-prior lanes (7, 12) is stated in-module.
- Monthly lanes (3,4,6): cost-trivial by construction; falsified by `factor_ic` <= 0
  vs BAB/idio-vol/semivariance (3,4) or the short-horizon reversion family (6).
- Per-symbol lanes (11,12): RPT-per-split >= 10bps mandatory (the graveyard #13/#14
  gate); 12 additionally falsified if the spread-stress episodes do not flip the
  majors' daily-momentum sign (verdict-9 pre-registration).
- Overlays (13,14): judged on MDD/Calmar/vol reduction of the wrapped TSMOM book at
  equal or better net return — NEVER on Sharpe-lift-over-best-sleeve.
- Full per-lane build-gate details and fixtures: the lane test files
  (`tests/test_<lane>_alpha_sleeves.py` / overlay tests) are the executable spec.

Combined with wave-1 (PR #41: 5 leaves + MR1/MR2/X1 + E1), this cycle ships
**23 new research lanes**. Selection is the data-PC's job; report ALL of them.
