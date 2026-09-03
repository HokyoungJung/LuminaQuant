# Final strategy evaluation — 2026-09-04

## Verdict

No strategy passed the preregistered validation gate. No portfolio or execution promotion was created.

## Coverage
- Registry coarse screen: 162/162
- Coarse status: {'excluded': 14, 'fail': 17, 'pass': 120, 'resource_excluded': 11}
- Rigorous cells: 66/66 complete
- Validation finalists: 0
- Portfolio: `skip_insufficient_survivors`
- Tick validation: `skip_no_finalists`

## Detailed shortlist

| Strategy | Validation return | Validation Sharpe | Locked-OOS return | Trades |
|---|---:|---:|---:|---:|
| PriceVolumeCorrContinuationStrategy | -0.371% | -340.315 | -0.195% | 60 |
| RebalancingPremiumHarvestStrategy | -0.141% | -5.809 | -0.947% | 206 |
| DisagreementGatedEnsembleStrategy | -23.148% | -32.435 | -18.901% | 1517 |
| BitcoinBuyHoldStrategy | 0.111% | -2.476 | -0.130% | 11 |
| MomentumCrashDynamicScalingOverlayStrategy | -0.342% | -10.034 | -0.162% | 42 |
| CompressionBreakoutContinuationStrategy | -0.194% | -6.295 | -0.207% | 73 |
| CrossSectionalPathConvexityStrategy | 0.295% | -4.541 | -0.419% | 68 |
| FalseBreakoutReversalStrategy | -0.395% | -10.427 | -0.333% | 1055 |
| CrossSectionalIntermediateEchoMomentumStrategy | -0.104% | -3.330 | -0.415% | 99 |
| LowTurnoverTrendPersistenceStrategy | -0.429% | -10.325 | -0.173% | 42 |
| TrendGatedResidualMomentumStrategy | -0.157% | -335.213 | -0.270% | 27 |

## Data and safety
- Official archive refresh: 228 symbol-days, 63,825,045 raw rows, 19,699,181 derived 1-second rows.
- TONUSDT post-delivery archive gaps remain fail-closed; no synthetic rows were created.
- Selection uses validation only; locked OOS is report-only.
- Order routing remains disabled.
