# Portfolio Optimizer Hot-Path Microbenchmark — 2026-05-08

## Purpose

Validate the shared `StreamCache` optimization for the portfolio stream IO/loop/allocation hot path without committing large generated reports.

## Command shape

Synthetic run under `uv run python` with:

- 80 candidates;
- 600 points per split (`val` + `oos`);
- 40 repeated portfolio stream builds per split;
- compare `build_portfolio_stream(..., cache=None)` versus one shared `StreamCache` reused across repeated builds.

Raw JSON artifact: `.omx/plans/portfolio_optimizer_perf_20260508.json`.

## Result

- Uncached repeated normalization: `17.079857766000032s`
- Cached shared-core path: `1.7218335879999813s`
- Speedup: `9.919575204616242x`
- Checksums match: `true`
- Peak RSS: `109727744` bytes
- Under 8 GiB: `true`

## Interpretation

The shared core removes repeated timestamp normalization/allocation across clustering, fit metrics, and report stream generation while preserving deterministic portfolio stream values.
