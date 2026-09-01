# Blend probe: TopCap incumbent vs TrendEfficiency

- Window: `2026-05-16T00:00:00Z -> 2026-06-13T09:25:00Z`
- Symbols: `BTC/USDT, ETH/USDT, BNB/USDT, SOL/USDT, TRX/USDT`
- Incumbent candidate: `topcap_tsmom_1h_exec_tightstop_tp_16_4_0.015`
- New candidate: `trend_efficiency_momentum_1h_teff_lo_20_20`
- Per-step return correlation: `0.2545`
- Incumbent standalone: return `2.3985%`, trades `157`
- TrendEfficiency standalone: return `0.1487%`, trades `55`

| Trend weight | Blended return | Blended MDD |
|---:|---:|---:|
| 5% | 2.2865% | 1.5177% |
| 10% | 2.1745% | 1.4395% |
| 20% | 1.9502% | 1.2829% |
| 30% | 1.7257% | 1.1260% |
| 50% | 1.2761% | 0.8118% |

Interpretation: TrendEfficiency is not a return enhancer versus the current TopCap candidate; it is only a potential small risk-sleeve/diversifier if drawdown reduction is more valuable than absolute return.
