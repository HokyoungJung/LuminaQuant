# Stack Lift Divergences — Phase 0.5

**Generated:** 2026-06-10
**Branch:** `refactor/overhaul`
**Plan section:** §3 Phase 0.5.2 — "Blocking compatibility resolution"

This document records every dependency gap or constraint change required to
move from the Phase 0 stack (Python 3.11.15) to the Phase 0.5 target stack
(Python >=3.14). Per plan §3 Phase 0.5.2, any dep that lags 3.14 gets the
highest 3.14-compatible pin recorded here with a revisit trigger.

---

## Python Interpreter

| Field | Phase 0 (old) | Phase 0.5 (new) |
|-------|---------------|-----------------|
| Version | 3.11.15 (CPython, Clang 22.1.3) | 3.14.4 (CPython) |
| `requires-python` | `>=3.11,<3.14` | `>=3.14` |

**Change:** Dropped the `<3.14` cap. The `>=3.11` floor was also removed since
we are committing to 3.14 as the minimum for all later phases.

---

## Rust Toolchain

| Field | Phase 0 | Phase 0.5 |
|-------|---------|-----------|
| rustc | 1.94.0 (4a4ef493e 2026-03-02) | 1.96.0 (ac68faa20 2026-05-25) |
| cargo | 1.94.0 (85eff7c80 2026-01-15) | 1.96.0 (30a34c682 2026-05-25) |

**Change:** `rustup update stable` bumped both tools. No API or ABI changes
expected; native cdylib artifacts are ABI-stable and Python-version-independent.
Native backends do NOT need a rebuild for this Rust version change alone.

---

## Dependency Changes

### 1. `cudf-polars-cu12` — BLOCKING PIN CHANGE

| Field | Old | New |
|-------|-----|-----|
| Constraint | `>=26.2,<26.3` | `>=26.6` |
| Resolved | 26.2.1 | 26.6.0 |
| Blocker reason | 26.2.x wheels only cover `cp310`–`cp313`; no `cp314` wheel published |
| Resolution | 26.6.0 is first release with `cp314` ABI wheels |

**Related packages also bumped by resolver:**
- `pylibcudf-cu12`: 26.2.1 → 26.6.0
- `libcudf-cu12`: 26.2.1 → 26.6.0
- `libkvikio-cu12`: 26.2.0 → 26.6.0
- `librmm-cu12`: 26.2.0 → 26.6.0
- `rmm-cu12`: 26.2.0 → 26.6.0
- `nvidia-libnvcomp-cu12`: 5.1.0.21 → 5.2.0.13
- New in 26.6: `cuda-core==1.0.1`, `cupy-cuda12x==14.1.1`,
  `librapidsmpf-cu12==26.6.0`, `libucx-cu12==1.19.0`,
  `libucxx-cu12==0.50.0`, `rapidsmpf-cu12==26.6.0`, `ucxx-cu12==0.50.0`

**Classification:** Required stack-lift change (not a regression).
The `gpu` extra is not installed in the base dev environment; this change
only affects machines that `uv sync --extra gpu`. CUDA toolkit is not in
the uv venv on this machine; GPU code paths exercise at Phase 2+.

**Revisit trigger:** When RAPIDS releases 26.8 or later, verify whether
`cudf-polars-cu12>=26.6` still resolves to a working release. If the 26.6
API is removed, update the lower bound and re-baseline. Track at:
https://github.com/rapidsai/cudf/releases

### 2. `polars` — CONSTRAINT WIDENED

| Field | Old | New |
|-------|-----|-----|
| Constraint | `>=1.35.2,<1.36` | `>=1.35.2` |
| Resolved | 1.35.2 | 1.35.2 (unchanged) |

**Change:** Removed the artificial `<1.36` tight upper bound. The resolver
still picks 1.35.2 (latest stable at time of Phase 0.5) — no version change.
The bound was over-constraining; removing it allows natural upgrades when
later phases bump polars for new API features.

**Classification:** Constraint cleanup, no version change, no numerical impact.

### 3. `numpy` — FLOOR BUMPED

| Field | Old | New |
|-------|-----|-----|
| Constraint | `>=1.26.0,<3` | `>=2.0,<3` |
| Resolved | 2.4.6 | 2.4.6 (unchanged) |

**Change:** Floor bumped from 1.26 to 2.0 to reflect that numpy 2.x is the
stable target (1.x is EOL). Resolved version is identical to Phase 0.

**Classification:** Constraint cleanup, no version change, no numerical impact.

---

## Packages with NO Change Under 3.14

All of the following resolved to the same version under Python 3.14 as under
Python 3.11, confirming 3.14 compatibility:

| Package | Version | 3.14 compat |
|---------|---------|-------------|
| numpy | 2.4.6 | ✅ same |
| polars | 1.35.2 | ✅ same |
| pandas | 2.3.3 | ✅ same |
| psycopg | 3.3.4 | ✅ same |
| psycopg-binary | 3.3.4 | ✅ same |
| ta-lib | 0.6.8 | ✅ same |
| python-dotenv | 1.2.2 | ✅ same |
| PyYAML | 6.0.3 | ✅ same |
| pytest | 9.0.3 | ✅ same |
| ruff | 0.15.16 | ✅ same |

**5 originally-blocking deps — resolution status:**

| Dep | Constraint | Resolved | 3.14 status |
|-----|-----------|----------|-------------|
| ta-lib | `>=0.6.8` | 0.6.8 | ✅ cp314 wheels available |
| psycopg | `>=3.3.3` | 3.3.4 | ✅ cp314 wheels available |
| cudf-polars-cu12 | `>=26.6` | 26.6.0 | ✅ resolved (pin bumped — see §1 above) |
| optuna | `>=4.7.0` | *(in `optimize` extra, not in base dev env)* | ✅ pure Python + C extensions, resolves under 3.14 |
| numba | `>=0.64.0` | *(in `optimize` extra, not in base dev env)* | ✅ resolves under 3.14 |

---

## Summary

- **1 package required a pin change** (`cudf-polars-cu12`: 26.2→26.6 for cp314 wheels)
- **0 packages are outright unavailable on 3.14** (no hard blockers remaining)
- **All other deps resolve identically** on 3.14 vs 3.11
- **Revisit trigger:** cudf-polars-cu12 API changes at RAPIDS 26.8+ release
