# Design Notes: GPU Auto/Fallback + Deterministic Polars Pipelines

## Goal

Keep runtime behavior deterministic while allowing opportunistic GPU acceleration.

## Compute Resolution Contract

`compute_engine.resolve_compute_engine(mode, device, verbose)` resolves execution mode from
explicit arguments or environment (`LQ_GPU_MODE`, `LQ_GPU_DEVICE`, `LQ_GPU_VRAM_GB`, `LQ_GPU_VERBOSE`).

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

Current project default is **GPU-first**:
- runtime config defaults to `execution.gpu_mode=gpu`
- `select_engine()` defaults to `LQ_GPU_MODE=gpu` when no explicit mode/env override is provided
- generic non-GPU CI lanes explicitly override `LQ_GPU_MODE=cpu`

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

- default to `LQ_GPU_MODE=gpu` for GPU-first local runs
- use `LQ_GPU_MODE=auto` when you want opportunistic fallback semantics on shared/mixed hardware
- use `LQ_GPU_MODE=forced-gpu` only when GPU availability is guaranteed
- install GPU runtime extras before enabling GPU mode:
  - `uv sync --extra gpu` (includes `cudf-polars-cu12` and `nvidia-nvjitlink-cu12`)
- run determinism tests after any pipeline/grouping change

## CI Design

The CI workflow uses a two-layer GPU strategy:

1. **GPU contract job (always runs on standard Ubuntu runners)**
   - installs the GPU extras
   - runs `tests/test_compute_engine.py`
   - runs `tests/test_verify_polars_gpu_runtime_script.py`
   - executes `scripts/ci/verify_polars_gpu_runtime.py` in skip-safe mode
   - validates that the codebase, dependency graph, and skip semantics stay healthy even when no GPU hardware is attached

2. **Strict GPU runtime smoke (runs only when a GPU runner is configured)**
   - enabled by repository variable `LQ_GPU_CI_RUNS_ON_JSON`
   - expects a JSON string or JSON label array for `runs-on`
   - example for a self-hosted runner:
     - `["self-hosted", "linux", "x64", "gpu"]`
   - optional guard:
     - `LQ_GPU_CI_REQUIRED=true`
   - executes `scripts/ci/verify_polars_gpu_runtime.py --require-gpu --mode forced-gpu`
   - uses a strict `polars.GPUEngine` query path and fails if the runtime cannot execute on GPU

This split keeps the default CI green on ordinary hosted runners while still supporting true Polars GPU validation on dedicated NVIDIA-backed runners.
