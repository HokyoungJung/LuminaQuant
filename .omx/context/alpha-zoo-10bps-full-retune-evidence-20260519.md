# Alpha Zoo 10bps full retune evidence appendix

Generated: 2026-05-19T11:08:21.118797Z
Scope: task 5 / worker-3 evidence-only lane. This appendix intentionally avoids runner/core/test changes and leaves final performance metrics blank until the retune runner emits measured artifacts.

## Cost-source evidence

- Primary promotion cost: **10bps round-trip all-in** (`round_trip_cost_bps=10`, `cost_model=round_trip_all_in`).
- Lower-cost rows, if produced, are diagnostic only and must not override the 10bps promotion gate.
- Binance USDⓈ-M Futures fee page: https://www.binance.com/en/fee/futureFee — Official fee surface reference; account-specific VIP/tier rates must be resolved at execution time, not hardcoded from stale memory.
- Binance Academy: bid-ask spread and slippage explained: https://www.binance.com/en/academy/articles/bid-ask-spread-and-slippage-explained — Public explanation that spread/slippage are real trading costs beyond visible fees and that large orders can walk the book.
- Binance Academy glossary: slippage: https://www.binance.com/en/academy/glossary/slippage — Public mitigation guidance: limit orders, order splitting, liquid pairs, and order-book depth checks.

## Diagnostic-only depth snapshot schema

The leader-gathered Binance USD-M public order-book snapshot is context only, not a training or selection input. The final evidence artifact should preserve:

- `diagnostic_only=true`
- `not_backtest_input=true`
- `observed_at_utc`
- `symbols`
- `notional_usdt_levels`
- per-symbol `spread_bps_approx`
- per-symbol `one_way_depth_slippage_bps_by_notional_approx`
- `book_source_url_or_endpoint`

Leader snapshot summary to carry forward:

- BTCUSDT: spread approximately `0.013bps`, depth slip approximately `0.007bps` for 10k USDT notional.
- ETHUSDT: spread approximately `0.047bps`, depth slip approximately `0.024bps` for 10k USDT notional.
- SOLUSDT: spread approximately `1.185bps`, depth slip approximately `0.59bps` to `1.03bps` across sampled notionals.

## Memory guard requirements

Required final artifact fields:

```json
{
  "memory_policy": {"limit_mb": 8192},
  "guard_status": "pass|fail",
  "pass_fail_reason": "peak_rss_mb=<value> < memory_policy.limit_mb=8192, or explicit failure reason",
  "peak_rss_mb": "<measured value>",
  "measured_rss_log_path": "var/reports/.../evidence_alpha_zoo_10bps_full_retune_rss.time.log"
}
```

Completion gate:

- `peak_rss_mb < 8192` (MiB-compatible field name retained for artifact contract compatibility).
- `guard_status=pass` only after measured RSS evidence exists.
- `pass_fail_reason` must say why the run passed or failed; pending placeholders are not final pass evidence.
- Heavy retune work should run sequentially; no parallel heavy Optuna/backtest batches in one process.

## Research-only and locked-OOS controls

- `real_money_execution_attempted=false` for every artifact in this bundle.
- Locked-OOS is gate/report-only after candidate freeze.
- Forbidden locked-OOS uses: objective, selection, pruning, parameter fitting, strategy-variant choice, and execution-cost calibration.
- Diagnostic depth/cost evidence must not select model variants or tune thresholds.

## Final doc update placeholders

Fill these only after the runner emits final metrics:

| Field | Placeholder |
| --- | --- |
| Top 10bps model | TBD after runner metrics |
| Train metrics | TBD after runner metrics |
| Validation metrics | TBD after runner metrics |
| Locked-OOS metrics | TBD after runner metrics |
| Memory peak RSS | TBD after measured RSS log |
| Live-readiness result | TBD; honestly report `no-live-ready` if gates fail |

## Generated artifact stubs

- `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_10bps_full_retune_20260519/execution_cost_evidence_latest.json`
- `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_10bps_full_retune_20260519/evidence_memory_guard_inputs_latest.json`
