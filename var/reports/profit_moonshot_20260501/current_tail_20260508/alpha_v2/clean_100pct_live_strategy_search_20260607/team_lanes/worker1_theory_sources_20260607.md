# Worker-1 Backup — External Theory / Source Evidence

Scope: supporting `.omx/ultragoal` G002/G006/G007 with source-backed theory constraints.

## Accepted theory families

1. **Time-series momentum / trend following**
   - Source: Moskowitz, Ooi, Pedersen, “Time Series Momentum,” AQR. https://www.aqr.com/insights/research/journal-article/time-series-momentum
   - Use: own-past-return/trend families across diversified assets are plausible.
   - Constraint: no arbitrary symbol/date rules; use lagged features only.

2. **Volatility-managed/risk-scaled exposure**
   - Source: Moreira & Muir, “Volatility Managed Portfolios,” NBER WP 22208. https://www.nber.org/papers/w22208
   - Use: lagged realized volatility/risk scaling is theoretically defensible.
   - Constraint: vol estimate must be lagged; real-time implementation caveats require stress checks.

3. **Execution-cost-aware implementation**
   - Source: Frazzini, Israel, Moskowitz, “Trading Costs,” SSRN. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3229719
   - Use: deployment claims must model turnover, impact, and capacity.
   - Constraint: 10/15/20bps stress and paper fill telemetry are mandatory.

4. **Overfit / multiple-testing control**
   - Source: Bailey et al., “Pseudo-Mathematics and Financial Charlatanism,” SSRN. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2308659
   - Use: trial counts, degrees of freedom, and selected-winner bias must be disclosed.
   - Constraint: 100%+ headline cannot be a search target or OOS selector.

## Rejected theory usage

- Hardcoded calendar/symbol exceptions without market rationale.
- Parameter values justified only by historical locked OOS performance.
- Any train/validation process that is adjusted after locked OOS observation.

## Conclusion

The only source-backed path to a 100%+ reporting-label candidate is a pre-registered trend/volatility/cost-aware family evaluated once under clean train/validation selection. Sources do not justify promoting existing OOS-inspired artifacts to real money.
