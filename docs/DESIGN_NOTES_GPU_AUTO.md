# Design Notes: GPU Auto/Fallback + Deterministic Polars Pipelines

## Goal

Keep runtime behavior deterministic while allowing opportunistic GPU acceleration.

## Compute Resolution Contract

`compute_engine.resolve_compute_engine(mode, device, verbose)` resolves execution mode from
explicit arguments or environment (`LQ_GPU_MODE`, `LQ_GPU_DEVICE`, `LQ_GPU_VERBOSE`).

Resolution policy:

1. `cpu`
   - Never probes GPU.
   - Always uses CPU engine.
2. `gpu` / `forced-gpu`
   - Runs a GPU smoke probe.
   - Raises `GPUNotAvailableError` if probe fails.
3. `auto`
   - Runs the same probe.
   - Uses GPU on success, otherwise logs reason and falls back to CPU.

## Safety Invariants

- **No silent downgrade in forced mode**.
- **Auto mode must always return a valid engine**.
- **Verbose mode includes requested mode, resolved mode, and fallback reason**.

## Deterministic Data Processing Rules

To keep CPU/GPU parity predictable in weekly chunk pipelines:

- use explicit bucket math (`timestamp_ms // bucket_ms`) for resampling
- sort timestamps before `first/last` aggregations
- avoid `group_by_dynamic` and Python UDFs in core aggregation paths
- keep idempotent output writes keyed by deterministic constraints

## State Idempotency Coupling

Postgres state writes use deterministic conflict keys (`run_id`, `dedupe_key`, `fingerprint`, etc.)
and `ON CONFLICT DO UPDATE` to prevent duplicate semantic rows during retries/replays.

## Operational Guidance

- default to `LQ_GPU_MODE=auto` for production-like local runs
- use `LQ_GPU_MODE=forced-gpu` only when GPU availability is guaranteed
- run determinism tests after any pipeline/grouping change
