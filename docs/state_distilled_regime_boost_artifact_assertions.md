# State-Distilled Regime Boost artifact assertions

Use this checklist for `StateDistilledRegimeBoostPortfolio` smoke runs and CI artifact reviews. The runner is research-only; promotion claims must come from frozen train/validation selection plus a later locked-OOS gate.

## Mandatory summary assertions

```python
assert summary["strategy_validity"]["calendar_primary"] is False
assert summary["selection"]["uses_locked_oos_for_selection"] is False
assert summary["selection"]["locked_oos_metrics_visible_during_selection"] is False
assert summary["selection"]["candidate_freeze_before_locked_oos_gate"] is True
assert (
    summary["selection"]["freeze_artifact_hash"]
    == summary["locked_oos_gate"]["freeze_artifact_hash"]
)
assert summary["selection"]["evaluated_count"] <= min(
    summary["selection"]["configured_grid_limit"],
    256,
    summary["selection"]["product_space_size"],
)
assert summary["booster"]["max_effective_leverage"] <= 25.0
assert summary["memory"]["peak_rss_bytes"] < 8 * 1024**3
```

If `summary["strict_lane"]["promoted_success"]` is true, additionally require:

```python
strict = summary["strict_lane"]
assert strict["locked_oos"]["max_drawdown"] <= 0.25
assert strict["liquidation_count_total"] == 0
assert strict["min_margin_buffer"] > 0
assert strict["locked_oos"]["sharpe"] > 0
assert strict["locked_oos"]["sortino"] > 0
assert strict["locked_oos"]["smart_sortino"] > 0
assert strict["locked_oos"]["calmar"] > 0
```

## Required provenance fields

The freeze artifact must include selected candidate ids, full selected config, train/validation ledger hash, input artifact hashes, code SHA plus dirty-tree marker, grid/search-space metadata, `selection_score_fields`, and `frozen_at`. Record the freeze SHA-256 in a separate sidecar/manifest, not inside the hashed freeze payload.

The locked-OOS gate artifact must record `locked_oos_opened_at`, the freeze artifact path/hash from the sidecar, and byte-identical selected params. Locked-OOS metrics are report/gate-only and must not appear in selection score inputs.

## Neutral-pair and lane rules

- Neutral pair universe, pair choice, hedge ratio, dispersion trigger, and overlay weights are fit/frozen from lagged train/validation features only.
- An OOS-only-good pair is never selectable.
- Strict and diagnostic lanes stay separate: diagnostic liquidations/recoveries may be reported, but cannot set `promoted_success`.
- Return/MDD ratio remains diagnostic-only; strict promotion uses zero liquidations, positive margin buffer, OOS MDD <=25%, and positive risk-quality metrics.
