# Interim Hurdle Comparison

- incumbent clean candidate: `relaxed_efficiency:hybrid_v3_5`
- incumbent OOS comp / MDD: `156.03%` / `19.75%`

| Policy | Cost | OOS comp | MDD | Gap vs incumbent comp | Stress verdict |
| --- | --- | ---: | ---: | ---: | --- |
| `best_single` | `10bps` | `54.56%` | `30.63%` | `-101.48%p` | LOSES |
| `cash_gated_top3` | `10bps` | `-20.64%` | `39.81%` | `-176.68%p` | LOSES |
| `top3_equal` | `10bps` | `-20.64%` | `39.81%` | `-176.68%p` | LOSES |
| `top5_equal` | `10bps` | `-1.00%` | `20.49%` | `-157.04%p` | LOSES |
| `best_single` | `20bps` | `27.10%` | `43.63%` | `-128.93%p` | LOSES |
| `cash_gated_top3` | `20bps` | `-8.46%` | `27.37%` | `-164.49%p` | LOSES |
| `top3_equal` | `20bps` | `-8.46%` | `27.37%` | `-164.49%p` | LOSES |
| `top5_equal` | `20bps` | `-0.09%` | `23.68%` | `-156.13%p` | LOSES |
