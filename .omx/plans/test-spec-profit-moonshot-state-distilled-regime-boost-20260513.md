# Test Spec — Profit Moonshot State-Distilled Regime Boost Portfolio — 2026-05-13

## Unit/regression tests

Add `tests/test_profit_moonshot_regime_boost_portfolio.py` or equivalent coverage for:

1. **Config-driven rules**
   - Regime thresholds, side multipliers, leverage map, booster gates, allocation weights, pair overlay weights, and selection score weights are supplied by config dataclasses/dicts and serialized.
   - Test fails if calendar/month/day/hour fields are present in factor/selection provenance.

2. **Train/validation-only selection**
   - A poisoned locked-OOS candidate must not win if its train/validation score is worse.
   - Selection rows expose `uses_locked_oos_for_selection=false` and `locked_oos_metrics_visible_during_selection=false`.
   - Frozen config/candidate exists before locked-OOS replay/gate fields.

3. **Dynamic leverage and 25x cap**
   - Low long-term volatility plus high confidence can request a higher booster leverage, capped at 25x.
   - High long-term volatility, high short-term vol ratio, stress regime, or weak confidence downshifts/disables booster.
   - Effective leverage is asset-specific, parameterized, and never exceeds max configured cap.

4. **Strict vs diagnostic liquidation lanes**
   - Strict promotion fails closed when liquidation count >0 or min margin buffer <=0 in train, validation, or locked-OOS.
   - Diagnostic lane may report nonfatal liquidation/wipeout-equity drawdown/recovery but remains non-promotable.

5. **Neutral pair overlay**
   - In neutral high-dispersion regime, pair overlay creates offsetting long/short notional within tolerance.
   - Stress/high-vol regimes disable or reduce overlay according to config.

6. **Artifact and memory payloads**
   - Summary JSON includes selected params, factor/strategy validity card, selection provenance, freeze/gate flags, strict/diagnostic lane separation, memory summary, and report paths.

## Real-data smoke/research run

Run the research runner against current-tail real crypto/FRED data with a bounded grid and `/usr/bin/time -v`. Required artifact assertions:

```python
assert summary['strategy_validity']['calendar_primary'] is False
assert summary['selection']['uses_locked_oos_for_selection'] is False
assert summary['selection']['locked_oos_metrics_visible_during_selection'] is False
assert summary['selection']['candidate_freeze_before_locked_oos_gate'] is True
assert summary['booster']['max_effective_leverage'] <= 25.0
assert summary['memory']['peak_rss_bytes'] < 8 * 1024 ** 3
```

## Required verification commands before final report

```bash
uv run --extra dev pytest tests/test_profit_moonshot_regime_boost_portfolio.py -q
uv run --extra dev pytest tests/test_profit_moonshot_fresh_start_replay.py tests/test_profit_moonshot_liquidation_aware_validation.py tests/test_profit_moonshot_live_final_selection.py tests/test_profit_moonshot_pass_under_8gb_validator.py -q
uv run --extra dev pytest -q
uv run --extra dev ruff check .
uv run --extra dev python -m compileall -q src scripts tests
git diff --check
git diff --cached --check
```

## CI gate

After Lore commit and push to `private/main`, verify GitHub Actions `ci` and `private-ci` are green for the pushed commit.

## Critic iteration hardening tests

7. **Immutable freeze provenance**
   - Freeze artifact records selected candidate ids, full selected config, input artifact hashes, train/validation ledger hash, code SHA/dirty marker, `frozen_at`, and grid/search-space metadata.
   - A separate sidecar/manifest records the freeze artifact SHA-256 hash; the hash is not embedded inside the payload being hashed.
   - Locked-OOS gate artifact records `locked_oos_opened_at`, references the exact freeze artifact path/hash, and selected params are byte-identical to the frozen config.

8. **Bounded-grid / validation-overfit guard**
   - Default grid cap is 64; hard maximum is 256.
   - Oversized product spaces are deterministically capped with evaluated/skipped counts and search-space hash.
   - OOS metrics cannot influence pruning/ranking; a poisoned OOS-only win must lose when train/validation is weaker.

9. **Neutral-pair leakage guard**
   - Pair universe, pair choice, hedge ratio, dispersion trigger, and overlay weights are fit/frozen from train/validation lagged features only.
   - A poisoned OOS-only-good neutral pair cannot be selected.
   - Pair feature provenance includes as-of/lag statement.

10. **Strict OOS MDD gate**
   - Strict promotion fails when locked-OOS MDD exceeds 25%, even if other metrics are positive.
   - Return/MDD ratio remains diagnostic-only and is not used as a hard promotion gate.

Add the following artifact assertion to the real-data smoke gate:

```python
strict = summary['strict_lane']
if strict['promoted_success']:
    assert strict['locked_oos']['max_drawdown'] <= 0.25
    assert strict['liquidation_count_total'] == 0
    assert strict['min_margin_buffer'] > 0
assert summary['selection']['freeze_artifact_hash'] == summary['locked_oos_gate']['freeze_artifact_hash']
assert summary['selection']['evaluated_count'] <= min(summary['selection']['configured_grid_limit'], 256, summary['selection']['product_space_size'])
```
