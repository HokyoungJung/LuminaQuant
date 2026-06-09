# Current-search alpha overlay + fold backend acceleration — 2026-06-09

## What changed

- Added optional Rust-backed alpha-fold symbol simulation backend: `native/rust_alpha_fold` + `lumina_quant.alpha_zoo.native_alpha_fold_backend`.
- `run_alpha_zoo_clean_new_alpha_discovery.py` now uses cached OHLCV arrays and `--simulation-backend auto|python|rust`; `auto` resolved to Rust locally.
- Added cached train/validation/locked-OOS split masks and a fast finalize path. Parity is tested against the prior `broad69.finalize_candidate` metrics.
- Added new pre-registered leaf alpha `indicator_vwap_kalman_pullback_continuation`: Kalman trend + VWAP/Bollinger pullback + ATR distance/realized-volatility gates. It is a 30m+ theory-plausible trend re-entry leaf, not a post-OOS hard-coded rule.
- Added existing-candidate reuse selector research script. It ranks already-evaluated candidate rows with train/validation-only scores, then attaches locked-OOS report metrics. Because it was designed after historical OOS review, every output is fresh-forward-required and non-promotable.

## Speed evidence

Evaluator command passed on 2026-06-09:

```text
.venv/bin/python -m pytest tests/test_alpha_zoo_clean_new_alpha_discovery.py -q
.venv/bin/python scripts/benchmark_alpha_zoo_fold_backend.py --quick
```

Latest evaluator evidence:

- clean discovery tests: `24 passed`
- Rust backend: `resolved_backend=rust`
- symbol simulation parity: `true`, max return abs diff `0.0`
- symbol simulation speedup: `2.55x`
- fast finalize parity: `true`
- split/finalize speedup: `16.41x`

A broader 8-family/6-symbol/2-timeframe/2-leverage overlay was intentionally terminated after `4:54` and peak RSS about `5.0GB`; that configuration is still too heavy for quick iteration. The practical fast probe path should use family/symbol/timeframe/leverage subsets first.

## Existing + reflected alpha overlay smoke

Artifact: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/current_search_existing_alpha_overlay_fast_20260609/clean_new_alpha_discovery_latest.md`

Scope: 4 symbols (`BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT`), `1h`, 3 folds, leverage `2`, robust train/validation selector, 8 families including the newly added VWAP/Kalman pullback family.

Result at default `10bps` round-trip cost:

- OOS compounded: `+4.16%`
- annualized approx: `+17.73%`
- positive folds: `3/3`
- max OOS MDD: `1.45%`
- selected families: `cross_asset_lead_lag_momentum` once, `cross_sectional_vol_adjusted_momentum` twice
- new `indicator_vwap_kalman_pullback_continuation` appeared in retained candidate rows but was not selected by the train/validation robust selector.

Interpretation: useful smoke improvement versus the earlier weak probes, but it is too small and too narrow to claim deployability or 100%+ annualized potential.

## New alpha standalone smoke

Artifact: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/current_search_vwap_kalman_pullback_smoke_20260609/clean_new_alpha_discovery_latest.md`

Scope: same 4 symbols/`1h`/3 folds/leverage `2`, only `indicator_vwap_kalman_pullback_continuation`.

Result at default `10bps` round-trip cost:

- OOS compounded: `+0.61%`
- annualized approx: `+2.46%`
- positive folds: `2/3`
- monthly equity MDD: `1.11%`
- max OOS MDD: `2.10%`

Interpretation: theory-plausible and now available as a candidate input, but standalone edge is weak. Do not promote.

## Existing-candidate reuse selector

Artifact: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/existing_candidate_reuse_selector_20260609/existing_candidate_reuse_selector_latest.md`

- `robust_top1`: OOS comp `+22.14%`, annualized approx `+27.12%`, positive folds `6/10`, monthly equity MDD `3.10%`.
- `robust_top2_equal`: OOS comp `+20.02%`, annualized approx `+24.48%`, positive folds `7/10`, monthly equity MDD `3.67%`.
- `robust_diverse3_equal`: near-flat/weak.

Interpretation: consistent with the robust selector diagnostic; still post-failure reuse research, not live-clean.

## Decision

- Real/shadow allocation remains `0%`.
- No result here passes the requested live-quality 100%+ annualized clean target.
- Keep the new Rust fold backend and fast finalize path; they materially improve iteration throughput without changing clean-OOS semantics.
- Next useful branch: run smaller pre-registered family sweeps and only escalate to 10-fold/full-universe after a family passes current 3-fold smoke under `10/15/20bps`, RPT/turnover, and live-fill telemetry gates.
