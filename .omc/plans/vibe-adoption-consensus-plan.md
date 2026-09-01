# Vibe-Trading → LuminaQuant Adoption — Consensus Execution Plan (crypto-perp, framework-faithful)

> Authoritative requirements source: `/home/hoky/Quants-agent/.omc/specs/deep-interview-vibe-trading-adoption.md`
> (di-vibe-adoption-20260701, Ambiguity 11.5%, PASSED). Components 1–6, AC-11…AC-65.
> ralplan mode: **DELIBERATE (--consensus)** — high-risk, real-money live crypto-perp engine.

## Status
- **PENDING APPROVAL** (user-gated; no execution, no branch, no commits until approved).
- Last automated verdicts: **architect = NEEDS_WORK**, **critic = ITERATE** (deliberate/real-money review).
- **Iterations reflected:** revision 2 (critic blocker + 2 major + 2 minor) and revision 3 (architect + critic blocker + major + 2 minor) are folded in; this document additionally folds the **final architect NEEDS_WORK synthesis + critic ITERATE** as a **revision-4 applied fix set** (FIX-B1 eigen-free N_eff, FIX-M1 pinned DSR dispersion, FIX-M2 third bench arm, FIX-m1 flat-payload value-type guard, FIX-m2 branch re-baseline procedure). Wave topology and lane count are UNCHANGED (5 waves, 14 lanes).
- Because the last critic verdict was ITERATE, the fixes below are authored into the plan but the plan itself remains **pending user approval**; team execution starts only on explicit approval.
- Unresolved blockers after revision-4 fixes: **none** — all blocker/major/minor items from the final architect+critic passes are addressed additively (see ADR §Follow-ups and Open Risks for the residual modeling choices that need human inspection, not code blockers).

---

## 0. Immutable Guardrails (non-negotiable; verified at every barrier)

These override any lane-level convenience. A lane that violates one FAILS its barrier.

1. **Golden byte-identical.** `tests/test_golden_roundtrip_pnl.py` and `tests/integration/test_walk_forward_golden.py` MUST stay byte-identical with **all new gates OFF**. Never edit golden fixtures except by an explicit, recorded user decision (see Delivery §branch re-baseline).
2. **Config-gated OFF by default; additive-subblock no-leak.** Every new numeric behavior lands behind a default-OFF config key. New fields appear ONLY as **new top-level JSON sub-objects or new files** — NEVER mutated into an existing metric dict. Specifically, nothing may insert a nested container into the flat `dict[str,float]` returned by `research_metrics.resolve_compute_metric_payload` (`:453`) / `compute_metric_summary` (`:525`); doing so changes the value-type and can flip golden bytes (FIX-m1).
3. **Determinism, no scipy/sklearn.** One fixed-seed `np.random.default_rng(seed)` threaded everywhere; sorted/ordered iteration (no dict/set-order reliance). No scipy, no sklearn (BLAS/version non-reproducibility). SLSQP → pure-numpy projected-gradient; norm CDF/PPF → in-repo `research_metrics._norm_cdf` (`:23`) / `_norm_ppf` (`:28`); block-bootstrap → reuse `research_metrics.spa_like_pvalue` (`:318`). **N_eff is computed eigen-free** (trace/Frobenius closed form) — `np.linalg.eigh` is reserved for C4 max-div where eigenvectors are genuinely needed, so the centerpiece gate does not import BLAS last-bit drift (FIX-B1). Never copy Vibe's i.i.d. `rng.choice`.
4. **Hotspot-only Rust.** Default = Polars-vectorized formulaic IR + numpy canonical-order reductions. Escalate to a `native/lumina_compute` kernel ONLY when a PRE-COMMITTED benchmark proves a hotspot AND a realizable constrained-kernel speedup exists (bench arm 3, below). Every kernel carries a pure-Python parity oracle at **rtol=1e-8** matched to the numpy canonical-order reference.
5. **Scope = Binance USDⓈ-M crypto-perp only.** Options / forex / TradFi land ONLY as config-gated-OFF seams / reference / research-only. No live multi-asset engine. Backtest↔live parity is mandatory (RiskManager gate + go-live stages).
6. **Do-not-port (reference logic only).** `FALLBACK_CHAINS` auto-resolve, `_sanitize_data_map` silent row-drop, per-bar `on_bar` ABC dispatch, runtime LLM transpile to Pine/TDX/MQL, sklearn KMeans rule-mining.
7. **Delivery.** Branch `feat/vibe-adoption`, atomic commit per item, golden + full suite + C0 guard green before push, `gh pr create`. **No merge to main, no `LUMINA_ENABLE_LIVE_REAL` flip** — user's decision.

---

## 1. RALPLAN-DR Summary

### 1.1 Principles
1. **Golden byte-identical is the load-bearing invariant.** Every change additive + default-OFF; each wave boundary runs the FULL `uv run pytest` suite as a precondition PLUS both golden tests PLUS the committed `tests/test_metric_dict_keyset_guard.py`. C0's honest scope = "single-seam dynamic capture (`json.dump/dumps` + `dataclasses.asdict`) + per-class capture for enumerated `to_dict`/`model_dump` emitters + a static net for unexecuted paths (idiom-bounded) + a canary proving the hooks fire" — NOT a repo-wide no-leak proof (that stays with the two golden tests + full suite) and NOT a nonexistent blanket `object.to_dict()` hook.
2. **Determinism over convenience, one mechanism proven BEHAVIORALLY.** One fixed-seed rng; in-repo erf/`_norm_ppf` (`:23`/`:28`); `spa_like_pvalue` reused in its verified single-strategy role; pure-numpy projected-gradient; **eigen-free N_eff**; and all final float reductions done in numpy over a canonical `(symbol,timestamp)` order — Polars is elementwise-transform-only, proven by (a) bit-identity to the pure-numpy reference, (b) identity across ≥2 `POLARS_MAX_THREADS`, (c) row-shuffle-invariance of the input lazy frame. The static "no-reduction-in-Polars" lint is only a fast secondary check. No scipy/sklearn; never Vibe's i.i.d. `rng.choice`.
3. **Hotspot-only Rust with an honest, ship-consistent gate.** Batch factor-IC escalates to a native kernel only after a PRE-COMMITTED quantitative benchmark (N, T×S, wall-clock) — measured against a kernel spec that already concedes ordered/pairwise reduction — is exceeded, cost is attributed to the numpy-sole-reduction relocation, and the expected constrained-kernel speedup is stated. **The escalate/skip verdict is measured against a fully numpy-vectorized batched-reduction arm** (arm 3), not a per-candidate Python-loop strawman (FIX-M2). Always a pure-Python parity oracle at rtol=1e-8.
4. **The alpha discovery loop is the centerpiece; its TRUST is protected STATISTICALLY, not just by wiring.** Multiple-testing correction = `deflated_sharpe_ratio` as the PRIMARY correlation-aware gate (`research_metrics.py:192`), deflating by BOTH `num_trials=N_eff` AND an **empirically-pinned `variance_across_trials`** computed from the candidate-Sharpe cross-section (FIX-M1) — not left to the conservative single-series stand-in `_estimate_sr_variance_across_trials` (`:180`). `N_eff` is a PINNED participation-ratio effective-independent-trials count computed **eigen-free** as `N_eff = (trace C)² / ‖C‖_F²` (raw_N logged only as an upper bound, never fed as the independent count). Per-candidate `spa_like_pvalue` (`:318`, VERIFIED single-strategy: docstring "single-strategy case" at `:327`, no family-max, no joint resample) is a single-strategy SANITY check with NO family claim; `approx_pbo` (`:245`) is the overfit check. Proven by an executable B3 gate: an EXACT-scalar N_eff KAT, a dispersion-term KAT, and a MIDDLE correlated-near-duplicate CALIBRATION KAT varying BOTH DSR inputs, plus reject-at-N / accept-at-1 endpoints. No alpha is trusted without cost-realism A/B (wave-4, de-serialized from delivery) + DSR(N_eff, dispersion)/spa/PBO + RiskManager go-live parity.
5. **Respect entity boundaries, function semantics, and multi-asset scope.** Factor provenance → NEW `AlphaCandidateRecord`/`ResearchLedger` (distinct file), never the frozen+slots trade-outcome `CandidateOutcomeRecord`. `research_metrics` used in verified roles; EXTENDed only for a labelled optional non-load-bearing joint-SPA advisory. Options/forex/TradFi = config-gated-OFF reference/research-only. Live engine stays single-asset Binance USDⓈ-M crypto-perp, strict backtest↔live parity, no auto-fallback drops.

### 1.2 Decision Drivers
- **Real-money live crypto-perp.** Any silent numeric drift, thread-dependent nondeterminism, backtest↔live divergence, or MIS-CALIBRATED (not just mis-wired) survivorship gate — including crediting a single-strategy `spa` with family-level robustness it lacks, or routing the primary correction through BLAS-backed `eigh` — is catastrophic. Golden discipline, the single-seam+per-class keyset guard, the numpy-sole reduction proven by shuffle-invariance, the eigen-free DSR(N_eff, dispersion) primary, and RiskManager parity dominate sequencing.
- **User priority = the alpha find/create/improve loop.** Waves land its deterministic substrate (operators, purity guards, factor-IC) before the engine; a wave-2 determinism spike pulls reproducible generation forward and the wave-3 B3 gate pulls the correlation-aware trust signal forward, without reordering waves.
- **Determinism + additive-no-leak constraints** force known-answer fixtures, parity oracles, a single-seam+per-class+static-net committed keyset guard, a PINNED eigen-free effective-trials haircut with an EXACT-scalar KAT, an empirically-pinned dispersion term, and a NEW research-ledger entity instead of library calls and schema mutation.

### 1.3 Options Considered
- **Option A (CHOSEN): feeder-first phased waves** with early determinism/objectivity fixes, behavioral falsifiers, a DSR(num_trials=N_eff, variance_across_trials=empirical) correlation-aware primary trust gate, and de-serialized cost-realism.
  - *Pros:* max wave-1 parallelism at zero numeric-path risk; centerpiece lands on a verified deterministic substrate; per-wave FULL-suite+golden+keyset barriers catch drift/leak/parity/regression earliest; the wave-2 spike + pre-committed ordered-reduction bench (with the numpy-vectorized arm) + numpy-sole reduction proven by shuffle-invariance pull the biggest unknowns forward; the AC-23 gate uses each `research_metrics` function in its VERIFIED role; cost-realism split into wave-4 de-serializes trust from delivery; atomic commits stay small.
  - *Cons:* the loop is fully usable only at wave-3, fully trusted at wave-4; more barriers = more full-suite runs + cross-lane C0 registration coordination; N_eff and the dispersion term are pinned-but-still-modeling choices, cost-realism params need human review.
- **Option B: centerpiece-first vertical slice.** *Pros:* fastest thin end-to-end loop. *Cons:* forces the highest-risk deterministic components before purity/keyset guards and parity oracles exist → golden/parity/leak/under-deflation regression risk; harder to keep commits atomic; violates feeders-before-engine; risks shipping a mis-characterized gate before the calibration KATs exist. **Rejected.**
- **Option C: Rust-first batch factor-IC.** *Pros:* peak throughput early, front-loads maturin/native risk. *Cons:* violates hotspot-only (no bench evidence; numpy-sole-reduction — not Polars — may be the actual cost AND caps achievable speedup); fp-summation parity risk vs the numpy reference; native-build friction on the critical path; premature. **Rejected.**

---

## 2. Premortem (top 3 load-bearing failure scenarios)

**P1 — Silent field leak flips golden / mutates a frozen record.** A new factor/operator, or the Sharpe-CI / execution-attribution / cost-realism emitter, leaks a field into an existing metric dict or the flat `resolve_compute_metric_payload` (`:453`) `dict[str,float]` (or C2c mutates the frozen `CandidateOutcomeRecord`), flipping golden bytes and drifting the live path — via a per-class `to_dict`/`model_dump` idiom no single blanket hook can catch.
  - *Likelihood:* medium. *Mitigation:* additive default-OFF sub-blocks only; provenance → NEW `AlphaCandidateRecord`/`ResearchLedger` file, never `CandidateOutcomeRecord`; the committed `tests/test_metric_dict_keyset_guard.py` (C0) runtime-captures single-seam idioms (`json.dump/dumps` + `asdict`) AND patches the enumerated per-class `to_dict`/`model_dump` producers surfaced by static discovery, diffing every captured key-set vs snapshot, with a static AST/grep net for unexecuted paths and a canary proving the hooks fire; it FAILS on any un-snapshotted producer or changed key-set with all gates OFF, and **additionally asserts the flat payload's value-type (all values remain float)** so a nested sub-object cannot be smuggled in (FIX-m1). FULL suite + both golden + this guard run at every barrier; `lq config validate` in CI. No reliance on a nonexistent blanket `object.to_dict()` hook.

**P2 — Survivorship gate MIS-CALIBRATES on correlated near-duplicates / rests on nothing.** The search emits massively correlated near-duplicate candidates: feeding raw post-dedup N over-deflates and kills genuine edges; dedup-collapsing N under-deflates and promotes phantom alphas into the live path — OR a single-strategy function is credited with family robustness it lacks — OR the deflation's dispersion half is left unpinned so the calibration KAT passes by construction while `variance_across_trials` is miscalibrated on real families.
  - *Likelihood:* medium. *Mitigation:* `deflated_sharpe_ratio` (`:192`) is PRIMARY, deflating by BOTH `num_trials=N_eff` AND an **empirically-pinned `variance_across_trials`** from the candidate-Sharpe cross-section (seeded, canonical order) (FIX-M1). `N_eff` is PINNED and **eigen-free**: `N_eff = (trace C)²/‖C‖_F² = S²/Σ_ij C_ij²` over the candidate return-CORRELATION matrix (unit diagonal ⇒ trace = S exactly; automatic bounds [1,S]; clamp to [1, raw_N]; no eigendecomposition, no eigenvalue clipping, no near-zero-trace guard) (FIX-B1); raw_N logged only as an upper bound. Per-candidate `spa_like_pvalue` (`:318`) is single-strategy SANITY only; `approx_pbo` (`:245`) overfit check. Executable B3 gate asserts the EXACT N_eff scalar (k-duplicates+m-independents KAT: `N_eff=(k+m)²/(k²+m)`, e.g. k=5,m=3→64/28), a dedicated dispersion-term KAT, N_eff≤raw_N, plus reject-known-null-at-realistic-breadth / accept-edge-at-N=1 endpoints AND a MIDDLE correlated-near-duplicate family varying BOTH DSR inputs (no over-reject of a true edge, no under-reject of a null = calibration, not monotonicity), all byte-reproducible across ≥2 thread counts. Optional joint-block-bootstrap max-SPA is advisory only, never load-bearing.

**P3 — Batch factor-IC disagrees with the reference (fp/thread reorder) or Rust is escalated on a strawman.** A Polars parallel agg or an escalated Rust kernel diverges beyond rtol=1e-8 due to fp-summation order across thread counts (→ phantom-alpha promotion), a hidden reduction slips past a grep-only lint, or a kernel is escalated for raw cost the ordered-reduction constraint cannot actually speed up.
  - *Likelihood:* medium. *Mitigation:* all final float reductions in numpy over a canonical `(symbol,timestamp)` order, Polars elementwise-transform-only, proven BEHAVIORALLY by (a) bit-identity to the pure-numpy reference, (b) identity across ≥2 `POLARS_MAX_THREADS`, (c) row-shuffle-invariance (a reduction inside Polars breaks shuffle-invariance); the "no-reduction" lint is only a fast secondary check; the numpy canonical-order reduction is the SOLE reference any Rust oracle must match at rtol=1e-8; the pre-committed bench threshold is measured against a kernel spec conceding ordered/pairwise reduction, with cost attributed to the reduction relocation and the expected constrained-kernel speedup stated — **and the escalate/skip verdict is measured against a third bench arm (a fully numpy-vectorized batched reduction over the `(T·S, N)` matrix in canonical order)**, so the cheapest deterministic Python path is exhausted before any native kernel is justified (FIX-M2). Else skip per hotspot-only policy.

---

## 3. Extended Test Plan

### 3.1 Unit
- `ts_mean`/`ts_mean_series` vs a hand-computed fixture, matching verified `ts_sum`/`ts_rank` NaN/min-window semantics.
- ERC known-answer (2 uncorrelated equal-vol assets = 50/50) + max-div/MV/HRP KAT incl. a degenerate/near-singular covariance case AND an HRP `cluster_by_correlation` equal-distance tie-break-determinism case (fixed linkage order + identical weights across runs) + run-twice determinism.
- Options-erf BS/greeks vs analytic reference fixtures (reusing VERIFIED `_norm_cdf:23`/`_norm_ppf:28`).
- Execution-attribution FIFO pairing + delta-PnL bucket math on synthetic fills.
- Sharpe-CI seed-fixed reproducibility (`spa` in its single-strategy CI role).
- Factor-IC vs numpy canonical-order pure-Python reference at rtol=1e-8 PLUS row-shuffle-invariance (behavioral no-reduction-in-Polars proof), static lint secondary.
- `effective_trials` PINNED **eigen-free** participation-ratio N_eff — an EXACT-scalar KAT on a hand-constructed correlation matrix (k perfectly-correlated duplicates + m independents ⇒ `N_eff=(k+m)²/(k²+m)`, e.g. k=5,m=3→64/28) plus N_eff≤raw_N and byte-identity across ≥2 thread counts (now genuinely BLAS-independent because the formula is pure sums-of-squares).
- **Dispersion-term KAT (FIX-M1):** `variance_across_trials` computed from a constructed candidate-Sharpe cross-section equals a hand-computed value; seeded, canonical-order, reproducible across ≥2 thread counts.
- AC-31 shift-invariance per-alpha on a seeded synthetic panel (>95% NaN windows skipped).
- C0 dynamic guard: runtime-captures single-seam producer key-sets (`json.dump/dumps` + `asdict`) and per-class enumerated `to_dict` emitters (incl. `CandidateOutcomeRecord.to_dict()`), asserts byte-unchanged, fails on any un-snapshotted producer, canary asserts the hooks fire, AND asserts the flat `resolve_compute_metric_payload` payload's value-type stays `float` (FIX-m1).

### 3.2 Integration
- Alpha search gen→eval→survivorship→promote over a fixed panel writes a deterministic NEW research ledger (run twice across ≥2 `POLARS_MAX_THREADS`, byte-identical).
- Survivorship gate uses `deflated_sharpe_ratio(num_trials=N_eff, variance_across_trials=empirical)` (`:192`) PRIMARY, per-candidate `spa_like_pvalue` (`:318`) SANITY, `approx_pbo` (`:245`) overfit — AC-23 KAT proves REJECT-known-null-at-realistic-breadth / ACCEPT-edge-at-N=1 AND a MIDDLE correlated-near-duplicate family proving the deflation neither over-rejects a true edge nor under-rejects a null (calibration), **varying BOTH `N_eff` and the dispersion input**, with the EXACT eigen-free N_eff fed to DSR and N_eff/raw_N/dispersion/DSR-verdict/PBO/spa-sanity reproducible across ≥2 thread counts.
- `portfolio.allocation_method` registry round-trips (single `@register` site in `portfolio_optimizers.py`, bootstrapped via `portfolio/__init__.py`) with `tuning/param_registry.py` schema; default `equal_weight` keeps the inverse-vol RP sleeve byte-identical.
- Cost-realism A/B (C6a) wires into the promotion gate; a RiskManager go-live parity test asserts no backtest↔live divergence.
- `lq alpha rank/card/promote` over a fixture; qlib158 factors pass the documented IC/IR eligibility bar before candidate-eligibility; TradFi snapshot→replay with `LUMINA_ENABLE_TRADFI_EXTERNAL_FETCH` OFF passing `assert_source_registry_is_free_unauthenticated`; MCP read-only smoke asserting no order/place/cancel tools; wave-2 determinism spike run-twice byte-identical.

### 3.3 E2E
- Full `feat/vibe-adoption` suite (`uv run pytest`) green AT EVERY BARRIER B1–B5; both golden tests byte-identical with ALL new gates OFF; `tests/test_metric_dict_keyset_guard.py` green (single-seam capture + per-class net + static net + canary + flat-payload value-type); spot-check that toggling library/optimizer/search gates ON does not change default outputs.
- `scripts/bench_factor_ic.py` asserts measured bench vs the pre-committed ordered-reduction hotspot threshold, attributes cost to the reduction relocation, states the expected constrained-kernel speedup, AND compares against the numpy-vectorized batched-reduction arm (FIX-M2) before an escalate/skip verdict; `lq config validate` passes; `scripts/build_native_backends.py` green and, if a kernel was added, its rtol=1e-8 parity oracle runs in CI; a promoted alpha carries a cost-realism A/B and passes the RiskManager order-risk gate + go-live stages before "trusted".

### 3.4 Observability
- Per-factor crypto-perp IC/IR readout advisory artifacts checked against the documented eligibility bar (gutted factors flagged, not silently carried); NEW research-ledger provenance (formula, seed, IC, IR, turnover, DSR, SPA_p single-strategy-sanity, PBO, cost_ab, raw_N upper-bound, N_eff eigen-free effective-trials, variance_across_trials) in a distinct file; factor-card md+json; dashboard IC-heatmap + candidate-queue read-only views on `DashboardBridgeContractV2`; `bench_factor_ic.py` report documenting the pre-committed threshold, the reduction-relocation cost attribution, the numpy-vectorized-arm comparison, the expected constrained-kernel speedup, and the Rust escalate/skip decision; execution-attribution delta-PnL bucket report (new sub-object); CI-surfaced golden-diff + the single-seam+per-class executable keyset drift assertions (C0); the B3 correlation-aware calibration assertions (DSR(N_eff, dispersion) primary verdict + EXACT eigen-free N_eff + N_eff≤raw_N + middle-KAT calibration) surfaced as explicit CI checks.

---

## 4. Wave-by-Wave Team-Lane Decomposition

> Common verification prelude for every lane: `uv run pytest <lane tests>` green AND (if the lane touches any emitter) `uv run pytest tests/test_metric_dict_keyset_guard.py`. All lanes are additive / default-OFF / test-only unless a config gate is named.

### WAVE 1 — independent default-OFF / test-only primitives (parallel, zero numeric-path risk)

#### Lane `keyset-guard-lane` — C0-keyset-guard
- **Goal:** Metric-dict/JSON key-set drift EARLY-WARNING: DYNAMIC runtime-capture of single-seam idioms (primary) + per-class patch net for enumerated emitters + static discovery net + canary. Honest scope, NOT a repo-wide no-leak proof.
- **work_items:**
  - [ ] `tests/conftest.py` runtime hook monkeypatches ONLY genuine single-seam serialization points — `json.dump` / `json.dumps` + `dataclasses.asdict` (+ `pandas.DataFrame.to_json` IF referenced in `src/lumina_quant`) — to RECORD every key-set emitted through those seams during a full-suite run, diffing each captured `(producer_call_site → key-set)` vs `tests/fixtures/metric_dict_keyset_snapshot.json`. NO blanket `object.to_dict()` hook (none exists).
  - [ ] STATIC + PER-CLASS PATCH NET: AST/grep walk over `src/lumina_quant` ENUMERATES persisted-JSON / metric-dict producers (40+ `json.dump` sites + 20+ `to_dict`/`asdict` emitters — candidate_outcome_ledger, research/edge_calibration, research/run_card, alpha_zoo/factor_card, cli/optimize, cli/config, workflows/autonomous_portfolio_research_loop, strategy_factory/selection, eval/*). Patch the enumerated per-class `to_dict`/`model_dump` producers so their key-sets are captured when exercised; the static net FAILS LOUD on any producer neither dynamically captured nor snapshotted.
  - [ ] CANARY: a seeded test producer emits a known key-set through EACH covered idiom (`json.dump`, `json.dumps`, `asdict`, a representative enumerated `to_dict` class), asserting the hook ACTUALLY FIRES and is diffed. Covered-idiom list PINNED EXPLICITLY in the test module (blind spots auditable).
  - [ ] With ALL new gates OFF, assert every captured/snapshotted key-set is byte-unchanged; fail loud on any added/removed key. `CandidateOutcomeRecord.to_dict()` JSONL row (frozen+slots, `asdict`→json `sort_keys`, consumed by `tests/test_edge_calibration.py`) explicitly asserted unchanged and enumerated as one of the patched per-class emitters.
  - [ ] **FIX-m1:** additionally assert the flat `dict[str,float]` from `resolve_compute_metric_payload` (`:453`) / `compute_metric_summary` (`:525`) keeps ALL values of type `float` (value-type guard), so no nested sub-object can be smuggled into the flat payload.
  - [ ] Expose `register_additive_subblock(name)` so emitter lanes (C5 exec-attr, C5b Sharpe-CI/options, C2c research ledger, C6a cost_ab) each add their NEW top-level sub-object key EXPLICITLY and assert their emitter never mutates an existing metric dict.
  - [ ] HONEST claim scope documented in the test module: idiom-bounded early-warning; the load-bearing no-leak guarantee remains the two golden tests + full suite.
- **files_touched:** `tests/conftest.py`, `tests/support/keyset_runtime_capture.py`, `tests/support/metric_producer_discovery.py`, `tests/support/keyset_canary.py`, `tests/fixtures/metric_dict_keyset_snapshot.json`, `tests/test_metric_dict_keyset_guard.py`
- **rust/python rationale:** python (test-only, read-only, zero numeric-path touch; runtime monkeypatch confined to `tests/conftest.py`; no hotspot; **Rust N/A**).
- **config_gate:** none (test-only; asserts the default-OFF key-set + value-type state; makes no repo-wide-completeness claim).
- **depends_on:** none. **Mapped AC:** AC-13, AC-32, AC-54, AC-61.
- **Verification:** `uv run pytest tests/test_metric_dict_keyset_guard.py -q`

#### Lane `guards-lane` — C3-guards
- **Goal:** Lookahead & purity guards (test-only, adopt_now).
- **work_items:**
  - [ ] `tests/property/test_alpha101_lookahead_purity.py` parametrizing `indicators/alpha101/registry.list_alpha_ids()` (101) + `ALPHA_FUNCTION_SPECS`.
  - [ ] Recompute each factor on a shifted synthetic panel via `alpha101/compiler.evaluate_compiled_formula` (full-series path, NOT `AlphaFunctionSpec.callable` last-scalar); assert shift-invariance / no future leakage.
  - [ ] Seed `np.random.default_rng(42)`; skip >95% NaN degenerate windows; xfail/allowlist genuine leaks to land CI-green (purity debt tracked as a burn-down allowlist).
  - [ ] AC-33 (optional, separate): extend `scripts/check_architecture.py` with an os/sys/eval/subprocess/socket/urllib import BLOCKLIST for factor modules (blocklist, not Vibe whitelist; polars/numpy/lumina internals pass).
- **files_touched:** `tests/property/test_alpha101_lookahead_purity.py`, `scripts/check_architecture.py`, `tests/fixtures/`
- **rust/python rationale:** python (test-only, read-only, zero numeric-path touch; no hotspot; **Rust N/A**).
- **config_gate:** none (test-only).
- **depends_on:** none. **Mapped AC:** AC-31, AC-32, AC-33.
- **Verification:** `uv run pytest tests/property/test_alpha101_lookahead_purity.py -q && uv run python scripts/check_architecture.py`

#### Lane `operators-lane` — C1a-operators
- **Goal:** Missing formulaic operators `ts_mean` + `ts_mean_series`.
- **work_items:**
  - [ ] Add `ts_mean(values, window)` and `ts_mean_series(values, window, *, index)` to `src/lumina_quant/indicators/formulaic_operators.py` mirroring verified `ts_sum`/`ts_sum_series` + `ts_rank_series` NaN/min-window semantics.
  - [ ] Wire into the operator dispatch resolved by `alpha101/compiler` (importlib of `formulaic_operators`) so the formulaic IR can reference them.
  - [ ] Unit-test vs a hand-computed fixture (`tests/test_formulaic_operators.py`), matching `ts_sum` edge semantics.
  - [ ] Purely additive: inert unless a gated factor references it; never imported on the backtest/optimization hot path.
- **files_touched:** `src/lumina_quant/indicators/formulaic_operators.py`, `tests/test_formulaic_operators.py`
- **rust/python rationale:** python (scalar+series rolling op via numpy/pandas Series; not a proven hotspot; **Rust N/A**).
- **config_gate:** none (pure additive operator; inert unless referenced by a gated factor).
- **depends_on:** none. **Mapped AC:** AC-11.
- **Verification:** `uv run pytest tests/test_formulaic_operators.py -q`

#### Lane `portfolio-lane` — C4-optimizers
- **Goal:** Pure-numpy ERC / max-div / MV / HRP portfolio optimizers, single registration site.
- **work_items:**
  - [ ] New `src/lumina_quant/portfolio/portfolio_optimizers.py` with `@register('portfolio','ERC')` pure-numpy equal-risk-contribution over full LW-shrunk cov; REUSE (read-only) `optimizer_core.ledoit_wolf_shrunk_covariance` (`:230`) / `project_simplex_with_upper_bounds` (`:512`) / `apply_caps` (`:679`).
  - [ ] `@register('portfolio','MaxDiversification')` + `@register('portfolio','MeanVariance')` pure-numpy projected-gradient (NO scipy SLSQP), fixed iteration count + tolerance, seeded `default_rng`; `@register('portfolio','HRP')` reusing the currently-unused `optimizer_core.cluster_by_correlation` (`:470`). Only C4 (max-div) uses `np.linalg.eigh` where eigenvectors are genuinely needed.
  - [ ] Single unambiguous registration site: all four self-register via `@register` in `portfolio_optimizers.py`, imported EXACTLY ONCE at `portfolio/__init__.py` bootstrap; registry infra `core/plugin_registry.py` NOT modified (only EqualWeight registered today).
  - [ ] Add config key `portfolio.allocation_method` default `'equal_weight'` + `tuning/param_registry.py` schema entries.
  - [ ] KAT: ERC of 2 uncorrelated equal-vol assets = 50/50; run-twice determinism; degenerate/near-singular covariance case exercising projected-gradient convergence; PLUS an HRP tie-break determinism KAT (tied/equal-distance correlation matrix → fixed reproducible `cluster_by_correlation` linkage order + identical HRP weights across runs, a nondeterminism source DISTINCT from projected-gradient tolerance). Near-singular handling here stays internally consistent (thread-count stable).
- **files_touched:** `src/lumina_quant/portfolio/portfolio_optimizers.py`, `src/lumina_quant/portfolio/__init__.py`, `src/lumina_quant/tuning/param_registry.py`, `config.yaml`, `tests/test_portfolio_optimizers.py`
- **rust/python rationale:** python (small N×N pure-numpy projected-gradient; not a proven hotspot; **Rust N/A**).
- **config_gate:** `portfolio.allocation_method=equal_weight` (default keeps inverse-vol RP sleeve byte-identical).
- **depends_on:** none. **Mapped AC:** AC-41, AC-42, AC-43, AC-44.
- **Verification:** `uv run pytest tests/test_portfolio_optimizers.py -q && uv run pytest tests/test_golden_roundtrip_pnl.py tests/integration/test_walk_forward_golden.py -q`

#### Lane `products-attr-lane` — C5-execattr-scaffold
- **Goal:** Execution-attribution scaffold (additive, gated OFF, new-subblock only).
- **work_items:**
  - [ ] New `src/lumina_quant/research/execution_attribution.py`: FIFO round-trip pairing, pure-math bias severity, delta-PnL attribution buckets (noise/early/late/overtrading/missed=residual).
  - [ ] Reconcile with the LuminaQuant fee/funding model; REJECT Vibe reporter/codegen/j2/CJK parser/sklearn rule-mining.
  - [ ] Add config key `research.execution_attribution_enabled: false`; never imported on hot path.
  - [ ] Emit ONLY into a NEW top-level JSON sub-object (never mutate an existing metric dict; never the flat `resolve_compute_metric_payload` payload — FIX-m1); register the sub-object key via `register_additive_subblock` in the C0 snapshot.
  - [ ] Unit tests on synthetic fills (FIFO pairing + delta-PnL bucket math).
- **files_touched:** `src/lumina_quant/research/execution_attribution.py`, `config.yaml`, `tests/test_execution_attribution.py`
- **rust/python rationale:** python (offline post-trade analytics; not hot path; **Rust N/A**).
- **config_gate:** `research.execution_attribution_enabled=false`.
- **depends_on:** C0-keyset-guard. **Mapped AC:** AC-52.
- **Verification:** `uv run pytest tests/test_execution_attribution.py tests/test_metric_dict_keyset_guard.py -q`

### WAVE 2 — factor library + batch factor-IC (parallel after B1)

#### Lane `factorlib-lane` — C1b-factorlib
- **Goal:** Qlib158 factor families in the formulaic IR (crypto-perp) with a documented IC/IR eligibility bar.
- **work_items:**
  - [ ] Express Qlib158-family factor defs in the existing formulaic IR / `src/lumina_quant/indicators/formulaic_definitions.py` (per-symbol OHLCV, NO cross-sectional rank).
  - [ ] Exclude & cite duplicates already in `alpha_zoo/crypto_fx_factors.py` (kmid/klen/kup/klow ~4).
  - [ ] Register behind `strategies.qlib158_formula.enabled=false` (or candidate-library opt-in) via `strategy_factory/candidate_library._add_alpha101_formula_candidates` parallel path + `tuning/param_registry` schema.
  - [ ] Each factor gets a lookahead/purity test (via C3) + a crypto-perp IC/IR readout advisory artifact.
  - [ ] Define a DOCUMENTED minimum crypto-perp IC/IR advisory bar (explicit `|rank-IC|` + IC-IR thresholds in `config`/`docs`) and make "inspect the advisory readout against that bar" an EXPLICIT gated checklist item before any qlib158 factor becomes candidate-library-eligible; factors whose per-symbol adaptation gutted the (originally cross-sectional) signal are FLAGGED — never silently carried.
- **files_touched:** `src/lumina_quant/indicators/formulaic_definitions.py`, `src/lumina_quant/strategy_factory/candidate_library.py`, `config.yaml`, `src/lumina_quant/tuning/param_registry.py`, `docs/qlib158_crypto_ic_bar.md`, `tests/test_qlib158_factors.py`
- **rust/python rationale:** python/Polars (formulaic IR vectorized; no hotspot; **Rust N/A**).
- **config_gate:** `strategies.qlib158_formula.enabled=false`.
- **depends_on:** C1a-operators, C3-guards. **Mapped AC:** AC-12, AC-13, AC-14.
- **Verification:** `uv run pytest tests/test_qlib158_factors.py tests/property/test_alpha101_lookahead_purity.py -q && uv run pytest tests/test_golden_roundtrip_pnl.py -q`

#### Lane `eval-ic-lane` — C2a-batchic
- **Goal:** Batch factor-IC evaluation: Polars elementwise-only + numpy canonical-order reduction as the SOLE reduction; behavioral shuffle-invariance falsifier; Rust hotspot-only with a realizable-gain gate measured against a numpy-vectorized arm.
- **work_items:**
  - [ ] New `src/lumina_quant/research/factor_ic.py`: batch rank-IC / IC-IR / turnover / decay per candidate over a fixed panel. Polars is ELEMENTWISE-TRANSFORM-ONLY (shift, per-symbol ranking); EVERY final float reduction (cross-sectional mean/cov → IC, IC-IR, decay) done in numpy over a canonically-sorted `(symbol,timestamp)` index — the SOLE reduction and the single reduction order any Rust oracle must match.
  - [ ] DETERMINISM MECHANISM: the numpy canonical-order reduction is the SOLE guarantee — DROP reliance on runtime `POLARS_MAX_THREADS=1` scoping (Polars fixes its thread pool at first import → per-entrypoint scoping is a no-op or a process-global bleed onto the hot path). Retain the ≥2 `POLARS_MAX_THREADS` run-twice byte-identity CI check as a FALSIFIER only.
  - [ ] BEHAVIORAL FALSIFIER: a test runs the factor-IC path and asserts the result is (i) bit-identical to the pure-numpy canonical-order reference AND (ii) INVARIANT under a Polars row-shuffle of the input lazy frame (a reduction inside Polars breaks shuffle-invariance). The static "no-reduction-in-Polars" lint is retained ONLY as a fast secondary check.
  - [ ] Establish the pure-Python/numpy reference + rtol=1e-8 golden fixture (`tests/fixtures/factor_ic_golden.json`); CI asserts run-twice byte-identity, identity across ≥2 `POLARS_MAX_THREADS`, AND row-shuffle-invariance.
  - [ ] Commit the QUANTITATIVE hotspot threshold INTO `scripts/bench_factor_ic.py` BEFORE wave 2: explicit candidate count N, panel dims T×S, wall-clock/throughput budget. The threshold is measured against a kernel SPEC that ALREADY concedes deterministic ordered/pairwise reduction matching the numpy canonical reference (NO unrestricted SIMD/parallel float reduction). The bench report MUST (a) attribute measured cost to the numpy-sole-reduction RELOCATION (banning `group_by().agg()` can itself manufacture the hotspot), (b) state the EXPECTED post-kernel speedup UNDER the ordered/pairwise-reduction constraint, AND (c) **FIX-M2:** include a THIRD bench arm — a fully numpy-vectorized batched cross-sectional reduction over a `(T·S, N_candidates)` matrix in canonical order (einsum/matmul) — and measure the escalate/skip verdict AND the expected constrained-kernel speedup against THIS arm, not a per-candidate Python-loop strawman. If the constrained kernel cannot beat the numpy-vectorized arm's throughput budget, SKIP.
  - [ ] Escalate to a `native/lumina_compute` `batch_factor_ic` kernel (`#[pyfunction]` in `lib.rs` + maturin + fixed deterministic ORDERED/PAIRWISE reduction) ONLY if the bench exceeds the committed threshold AGAINST the numpy-vectorized arm AND the expected constrained speedup clears the budget; every kernel carries a pure-Python parity oracle at rtol=1e-8 (AC-64) matching the numpy canonical-order reference; module never imported on the hot path.
  - [ ] Wave-2 determinism SPIKE: `tests/test_alpha_search_determinism_spike.py` — a minimal deterministic gen→eval→ledger stub over a tiny fixed panel (seeded `default_rng`, ordered enumeration, threads the search-breadth artifacts, writes the NEW research ledger) with a run-twice byte-identical assertion, proving centerpiece reproducibility at B2.
- **files_touched:** `src/lumina_quant/research/factor_ic.py`, `tests/fixtures/factor_ic_golden.json`, `scripts/bench_factor_ic.py`, `native/lumina_compute/src/lib.rs`, `scripts/build_native_backends.py`, `tests/test_factor_ic.py`, `tests/test_factor_ic_shuffle_invariance.py`, `tests/test_alpha_search_determinism_spike.py`
- **rust/python rationale:** python-first (Polars elementwise transforms + numpy canonical-order reduction as SOLE reduction, proven by bit-identity + shuffle-invariance); Rust `batch_factor_ic` is the prime hotspot candidate but added ONLY after the pre-committed bench threshold (measured against an ordered/pairwise kernel spec, cost attributed to the reduction relocation, expected constrained speedup clearing the budget, **compared against the numpy-vectorized batched-reduction arm — FIX-M2**) is exceeded, always with a pure-Python parity oracle rtol=1e-8 (AC-64).
- **config_gate:** `research.alpha_search.enabled=false` (module off the hot path).
- **depends_on:** C1a-operators. **Mapped AC:** AC-22, AC-64.
- **Verification:** `uv run pytest tests/test_factor_ic.py tests/test_factor_ic_shuffle_invariance.py tests/test_alpha_search_determinism_spike.py -q && uv run python scripts/bench_factor_ic.py --assert-threshold && uv run python scripts/build_native_backends.py`

### WAVE 3 — centerpiece discovery + survivorship gate (serial-ish)

#### Lane `search-engine-lane` — C2b-searchengine
- **Goal:** Alpha discovery/search generate→evaluate→select loop (centerpiece) with raw-N + candidate-return-matrix bookkeeping.
- **work_items:**
  - [ ] Extend `src/lumina_quant/workflows/alpha_research_pipeline.py` + `strategy_factory/candidate_library.py` into a generate→evaluate→select loop.
  - [ ] Operator-tree/formula generation over the formulaic operator set: seeded deterministic enumeration or seeded `np.random.default_rng` sampling (NOT Vibe i.i.d. `rng.choice`); sorted/ordered iteration everywhere.
  - [ ] Evaluate candidates via C2a batch factor-IC (numpy canonical-order reduction); deterministic selection loop.
  - [ ] Track and expose TWO deterministic artifacts, NOT a single raw trial count: (i) `raw_N` = candidates enumerated post-dedup/canonicalization AND INCLUDING those the survivorship gate later rejects (plus any gated LLM-proposer drafts) — an UPPER BOUND, reproducible, LOGGED but NEVER fed as the independent-trials count; and (ii) the candidate return/IC matrix over the fixed panel that C2c reduces to `N_eff` via the pinned eigen-free participation-ratio haircut. Both threaded to the C2c gate so breadth couples to deflation via a CORRELATION-AWARE effective count.
  - [ ] Whole loop config-gated; running it does not alter any default backtest/live numeric output (AC-26); reuses the wave-2 determinism spike harness as the run-twice regression.
- **files_touched:** `src/lumina_quant/workflows/alpha_research_pipeline.py`, `src/lumina_quant/strategy_factory/candidate_library.py`, `src/lumina_quant/research/factor_ic.py`, `config.yaml`, `tests/test_alpha_search_engine.py`
- **rust/python rationale:** python (orchestration/enumeration; no hotspot; **Rust N/A**).
- **config_gate:** `research.alpha_search.enabled=false`.
- **depends_on:** C2a-batchic, C1b-factorlib. **Mapped AC:** AC-21, AC-26.
- **Verification:** `uv run pytest tests/test_alpha_search_engine.py tests/test_alpha_search_determinism_spike.py -q`

#### Lane `survivorship-lane` — C2c-survivorship-ledger
- **Goal:** Survivorship gate — `deflated_sharpe_ratio(num_trials=N_eff, variance_across_trials=empirical)` PRIMARY (correlation-aware) + pinned eigen-free N_eff + per-candidate spa sanity + PBO + NEW `AlphaCandidateRecord`/`ResearchLedger` + gated LLM proposer.
- **work_items:**
  - [ ] Add a NEW `AlphaCandidateRecord` dataclass + `ResearchLedger` writer in `src/lumina_quant/research/candidate_outcome_ledger.py` (spec ontology 133/137) carrying formula, seed, IC, IR, turnover, DSR, SPA_p, PBO, cost_ab, raw_N, N_eff, variance_across_trials; written to a DISTINCT file (`research_ledger.jsonl`). Do NOT add any field to the frozen+slots `CandidateOutcomeRecord` (per-round-trip TRADE record, `asdict`→json `sort_keys` JSONL, consumed by `tests/test_edge_calibration.py`).
  - [ ] Regression: assert existing `CandidateOutcomeRecord.to_dict()` key-set byte-unchanged (folded into C0 as one enumerated per-class emitter); the new research ledger registers as a NEW file / new top-level sub-object only.
  - [ ] PRIMARY correlation-aware gate = `deflated_sharpe_ratio(num_trials=N_eff, variance_across_trials=empirical)` (`:192`). N_eff is the legitimate correlation correction. `spa_like_pvalue` (`:318`, VERIFIED single-strategy: docstring "single-strategy case" at `:327`) is DEMOTED to a per-candidate single-strategy SANITY p-value with NO family-level claim. `approx_pbo` (`:245`) reused as overfit check. `raw_N` retained + LOGGED as an UPPER BOUND, NEVER fed as the independent count.
  - [ ] **FIX-B1 — N_eff PINNED, EIGEN-FREE:** compute N_eff in new `src/lumina_quant/research/effective_trials.py` by the closed form `N_eff = (trace C)² / ‖C‖_F² = S² / Σ_ij C_ij²` (pure sums-of-squares over the canonical candidate order) over the candidate return-CORRELATION matrix C. A correlation matrix has unit diagonal so `trace C = S` exactly and bounds are automatic ([1,S]). **NO `np.linalg.eigh`, NO eigenvalue clipping, NO near-zero-trace guard** — keep only the `[1, raw_N]` clamp. Documented rationale (participation ratio = number of effectively-contributing independent directions). This removes the BLAS/version last-bit nondeterminism the no-scipy rule targets; `eigh` is reserved for C4 max-div only.
  - [ ] **FIX-M1 — dispersion PINNED:** compute `variance_across_trials` from the empirical candidate-Sharpe cross-section (seeded, canonical order) and pass it explicitly to `deflated_sharpe_ratio` alongside `num_trials=N_eff` — do NOT leave it to the conservative single-series stand-in `_estimate_sr_variance_across_trials` (`:180`). Add a dedicated dispersion-term KAT.
  - [ ] EXACT-SCALAR N_eff KAT (replaces the prior inequality-only assertion): a hand-constructed correlation matrix of k perfectly-correlated duplicates (rank-1 block) + m independent assets has closed-form `N_eff = (k+m)²/(k²+m)`; assert the EXACT scalar (k=5,m=3 → 64/28 = 2.2857…) via the trace/Frobenius identity (`trace=k+m`, `‖C‖_F²=k²+m`), and assert byte-identity across ≥2 thread counts (now genuinely BLAS-independent).
  - [ ] CORRECTED anchors: `deflated_sharpe_ratio(returns, *, num_trials=1, variance_across_trials=None)` at `:192`; `expected_max_sharpe(*, variance_across_trials, num_trials)` at `:92`; `_estimate_sr_variance_across_trials` at `:180`; `spa_like_pvalue` at `:318` (single-strategy docstring `:327`); `approx_pbo` at `:245`; `resolve_compute_metric_payload` at `:453`; `compute_metric_summary` at `:525` (survivorship/compute-metric entry points that thread num_trials).
  - [ ] EXECUTABLE B3 gate (CALIBRATION not just monotonicity): (a) plumbing — assert the EXACT eigen-free N_eff scalar is what DSR receives via `num_trials=N_eff`, that the empirical `variance_across_trials` is what DSR receives, that N_eff≤raw_N, that raw_N is logged as an upper bound and never fed as the independent count, that DSR(N_eff, dispersion) is the PRIMARY verdict, and per-candidate spa is a single-strategy sanity check only; (b) ENDPOINT KATs — REJECT a constructed known-null (zero-edge) family at realistic breadth, ACCEPT a lone genuine edge at N=1; (c) MIDDLE/CALIBRATION KAT — a constructed CORRELATED near-duplicate family (one genuine edge + many near-duplicates of a null) **varying BOTH `N_eff` and the dispersion input**, asserting the deflation NEITHER over-rejects the true edge NOR under-rejects the null family (calibrated bite, not just monotonicity); (d) N_eff, raw_N, variance_across_trials, the DSR verdict, PBO and the per-candidate spa sanity p-value byte-reproducible across ≥2 thread counts.
  - [ ] Optional LLM proposer behind `research.llm_proposer.enabled=false`: drafts candidate formulas, but only those passing lookahead+purity+DSR(N_eff,dispersion)/spa/PBO are registered; registration path deterministic; proposer outputs are frozen/seeded artifacts feeding the deterministic gate, NEVER a live dependency; proposed drafts count toward raw_N and enter the correlation matrix feeding N_eff.
  - [ ] OPTIONAL non-load-bearing additive (gated advisory only): a NEW pure-numpy JOINT circular-block-bootstrap max-statistic SPA (White's Reality Check proper — SAME resample block indices across ALL candidates simultaneously, realized-MAX studentized statistic, seeded/deterministic) MAY be EXTENDED into `research_metrics.py` behind its own default-OFF gate as a FAMILY-level advisory cross-check; if added it carries run-twice byte-identity across ≥2 thread counts, candidate-column-order invariance, input row-shuffle invariance, and an rtol=1e-8 parity fixture. NOT the load-bearing primary; NOT required to land.
  - [ ] Never copy Vibe i.i.d. `rng.choice`; seed everything; ledger writes go through the numpy canonical-order reductions so the ledger is byte-identical run-to-run across thread counts.
- **files_touched:** `src/lumina_quant/research/candidate_outcome_ledger.py`, `src/lumina_quant/research/effective_trials.py`, `src/lumina_quant/workflows/alpha_research_pipeline.py`, `src/lumina_quant/strategy_factory/research_metrics.py`, `src/lumina_quant/research/llm_proposer.py`, `config.yaml`, `tests/test_survivorship_gate.py`, `tests/test_survivorship_trial_count.py`, `tests/test_effective_trials_haircut.py`, `tests/test_research_ledger.py`
- **rust/python rationale:** python (DSR(N_eff, dispersion) primary + pinned eigen-free participation-ratio haircut + per-candidate spa sanity + approx_pbo + ledger IO; reuses `research_metrics`, EXTENDs it only for the optional joint-SPA advisory; no hotspot; **Rust N/A**).
- **config_gate:** `research.alpha_search.enabled=false` + `research.llm_proposer.enabled=false`.
- **depends_on:** C2b-searchengine, C2a-batchic, C0-keyset-guard. **Mapped AC:** AC-23, AC-24, AC-25.
- **Verification:** `uv run pytest tests/test_survivorship_gate.py tests/test_survivorship_trial_count.py tests/test_effective_trials_haircut.py tests/test_research_ledger.py tests/test_metric_dict_keyset_guard.py -q`

### WAVE 4 — product surfaces + de-serialized cost-realism (parallel)

#### Lane `cli-lane` — C5a-cli
- **Goal:** `lq alpha` CLI (rank / card / promote).
- **work_items:**
  - [ ] New `src/lumina_quant/cli/alpha.py` + register in the `cli/main.py` dynamic dispatch map (`autonomous_research.py` precedent).
  - [ ] `lq alpha rank` (IC/IR/turnover/decay), `lq alpha card` (factor card md+json), `lq alpha promote` (candidate/research-ledger workflow).
  - [ ] Reuse C2a evaluation + C2c NEW `ResearchLedger`; CLI opt-in surface only (no default numeric-path).
- **files_touched:** `src/lumina_quant/cli/alpha.py`, `src/lumina_quant/cli/main.py`, `tests/test_cli_alpha.py`
- **rust/python rationale:** python (CLI surface; no hotspot; **Rust N/A**).
- **config_gate:** CLI-only invocation; compute gated behind `research.alpha_search.enabled=false`.
- **depends_on:** C2c-survivorship-ledger, C2a-batchic. **Mapped AC:** AC-51.
- **Verification:** `uv run pytest tests/test_cli_alpha.py -q`

#### Lane `seams-lane` — C5b-multiasset-seams
- **Goal:** Gated multi-asset seams: options erf-math, forex seam, Sharpe-CI, TradFi fetchers.
- **work_items:**
  - [ ] Options BS/greeks pure-erf math library for analytics/validation ONLY (reuse `math.erf` + VERIFIED `research_metrics._norm_cdf` `:23` / `_norm_ppf` `:28`), own golden fixtures, NO execution path; + forex parameter seam only.
  - [ ] Sharpe-CI advisory sub-block (block-bootstrap via `spa_like_pvalue` `:318` in its single-strategy CI role, seed fixed), `emit_bootstrap_sharpe_ci=0.0` default; NEVER fed into hurdle/robust score.
  - [ ] Sharpe-CI + options-erf analytics emit ONLY into NEW top-level JSON sub-objects (never the flat `resolve_compute_metric_payload` `dict[str,float]` payload — FIX-m1); register each via `register_additive_subblock` in C0; additive default-OFF emitters keep golden byte-identical.
  - [ ] TradFi research fetchers (yahoo/stooq/sec_edgar) research-only, snapshot→replay, env `LUMINA_ENABLE_TRADFI_EXTERNAL_FETCH` default False, must pass `assert_source_registry_is_free_unauthenticated`; source-pinned + fail-loud (do-not-port `FALLBACK_CHAINS`/`_sanitize_data_map` — reference logic only).
- **files_touched:** `src/lumina_quant/research/options_math.py`, `src/lumina_quant/research/forex_seam.py`, `src/lumina_quant/research/sharpe_ci.py`, `src/lumina_quant/research/external_source_registry.py`, `config.yaml`, `tests/test_options_math.py`, `tests/test_sharpe_ci.py`, `tests/test_tradfi_snapshot_replay.py`
- **rust/python rationale:** python (analytics/validation, research-only; off hot path; **Rust N/A**).
- **config_gate:** `emit_bootstrap_sharpe_ci=0.0` + `LUMINA_ENABLE_TRADFI_EXTERNAL_FETCH=false` (options analytics gated OFF).
- **depends_on:** C0-keyset-guard (may start as early as post-B1). **Mapped AC:** AC-53, AC-54, AC-55.
- **Verification:** `uv run pytest tests/test_options_math.py tests/test_sharpe_ci.py tests/test_tradfi_snapshot_replay.py tests/test_metric_dict_keyset_guard.py -q`

#### Lane `surfaces-lane` — C5c-mcp-dashboard
- **Goal:** MCP read-only bridge + dashboard IC-heatmap/candidate-queue.
- **work_items:**
  - [ ] MCP read-only stdio bridge (`lq mcp` console script) as `[project.optional-dependencies] mcp` extra, lazy import; ONLY read-only tools (backtest/rank/report); order/place/cancel NEVER exposed; read-only smoke test asserts no order tools present.
  - [ ] Dashboard: Next.js factor IC-heatmap + candidate-queue views wired to the existing `dashboard/bridge.DashboardBridgeContractV2` read-only surface (additive `DashboardRouteDescriptor` routes/services); MCP + Next.js kept strictly optional/lazy so core pytest + golden stay green without them.
- **files_touched:** `src/lumina_quant/cli/mcp.py`, `pyproject.toml`, `src/lumina_quant/dashboard/bridge.py`, `apps/dashboard/`, `tests/test_mcp_readonly_smoke.py`
- **rust/python rationale:** python + typescript (read-only surfaces; no numeric hot path; **Rust N/A**).
- **config_gate:** optional-dep `mcp` extra (lazy import; absent by default).
- **depends_on:** C5a-cli, C2c-survivorship-ledger. **Mapped AC:** AC-56, AC-57.
- **Verification:** `uv run pytest tests/test_mcp_readonly_smoke.py -q`

#### Lane `cost-realism-lane` — C6a-costrealism
- **Goal:** Cost-realism A/B + RiskManager go-live parity feeding the promotion gate (trust signal de-serialized from delivery).
- **work_items:**
  - [ ] New `src/lumina_quant/research/cost_realism.py` `CostRealismProfile`: cost-realism A/B for EVERY promoted alpha (flat vs `sqrt_impact` slippage + funding coverage), parameters reconciled against the LIVE fee/funding model; an alpha is "trusted" only if it passes cost-realism A/B AND the correlation-aware survivorship gate from C2c (DSR(N_eff, dispersion) PRIMARY + per-candidate spa sanity + approx_pbo).
  - [ ] `cost_ab` written ONLY into a NEW top-level sub-object of the research-ledger/promotion record (never the flat payload — FIX-m1); registered via `register_additive_subblock` in C0.
  - [ ] New live-eligible factors routed through the existing RiskManager order-risk gate + go-live stages (no backtest↔live divergence, AC-63); an explicit parity test asserts the go-live path matches backtest assumptions.
  - [ ] SEQUENCING: this lane is SPLIT OUT of the old terminal C6 into wave 4 so the trust signal feeds the C2c promotion gate directly and is COMPLETE before the wave-5 delivery gate.
  - [ ] Miscalibration guard: cost-realism `sqrt_impact` + funding-coverage parameters get a KAT reconciling them against the live fee/funding model, plus a documented human-review checkpoint before any alpha is marked trusted.
- **files_touched:** `src/lumina_quant/research/cost_realism.py`, `src/lumina_quant/workflows/alpha_research_pipeline.py`, `src/lumina_quant/research/candidate_outcome_ledger.py`, `src/lumina_quant/live/`, `config.yaml`, `tests/test_cost_realism_ab.py`, `tests/test_risk_gate_parity.py`
- **rust/python rationale:** python (offline promotion-gate analytics + live-gate parity; not hot path; **Rust N/A**).
- **config_gate:** cost-realism part of the promotion gate under `research.alpha_search.enabled=false` (default OFF).
- **depends_on:** C2c-survivorship-ledger, C2a-batchic, C0-keyset-guard, C5-execattr-scaffold. **Mapped AC:** AC-62, AC-63.
- **Verification:** `uv run pytest tests/test_cost_realism_ab.py tests/test_risk_gate_parity.py tests/test_metric_dict_keyset_guard.py -q`

### WAVE 5 — delivery & final verification (single lane)

#### Lane `delivery-lane` — C6-delivery
- **Goal:** Delivery & final verification (golden + full-suite + config-validate + native-build + PR). PURE delivery — cost-realism (AC-62) and go-live parity (AC-63) already landed in wave-4 C6a.
- **work_items:**
  - [ ] `scripts/build_native_backends.py` green; any new kernel has a pure-Python parity oracle at rtol=1e-8 matching the numpy canonical-order reference (AC-64).
  - [ ] Full suite green (`uv run pytest`); `tests/test_golden_roundtrip_pnl.py` + `tests/integration/test_walk_forward_golden.py` byte-identical with ALL gates OFF; C0 guard green (single-seam runtime-capture + per-class net + static net + canary + flat-payload value-type); `lq config validate` passes; spot-check that toggling library/optimizer/search gates ON does not change default outputs.
  - [ ] Atomic commits per item on `feat/vibe-adoption`; golden+full-suite+keyset-guard green; `gh pr create` (NO main merge, NO `LUMINA_ENABLE_LIVE_REAL` flip).
- **files_touched:** `scripts/build_native_backends.py`, `.github/`, `tests/test_golden_roundtrip_pnl.py`, `tests/integration/test_walk_forward_golden.py`
- **rust/python rationale:** python + Rust build/parity verification (build_native_backends green + rtol=1e-8 oracle for any kernel).
- **config_gate:** delivery/verification only (all new gates default OFF; asserts golden byte-identity in that state).
- **depends_on:** ALL prior lanes (C0, C1a, C3, C4, C5, C1b, C2a, C2b, C2c, C5a, C5b, C5c, C6a). **Mapped AC:** AC-61, AC-64, AC-65.
- **Verification:** `uv run pytest -q && uv run python scripts/build_native_backends.py && uv run lq config validate`

---

## 5. Wave Barriers (hard gate at each wave boundary)

Every barrier B1–B5 runs, AS A PRECONDITION to crossing: the FULL `uv run pytest` suite + both golden tests (`tests/test_golden_roundtrip_pnl.py` + `tests/integration/test_walk_forward_golden.py`, byte-identical) + the committed `tests/test_metric_dict_keyset_guard.py` (C0). FULL-SUITE-AT-EVERY-BARRIER catches non-golden regressions (plugin registry, `tuning/param_registry` schema, `cli/main.py` dispatch) per-wave, not deferred to B5. Atomic commit per item at each barrier.

- **B1 (after Wave 1):** full suite + two golden byte-identical + C0 guard green (single-seam capture + per-class net + static net + canary + flat-payload value-type). Command: `uv run pytest -q && uv run pytest tests/test_golden_roundtrip_pnl.py tests/integration/test_walk_forward_golden.py tests/test_metric_dict_keyset_guard.py -q`
- **B2 (after Wave 2):** B1 set + `scripts/bench_factor_ic.py` bench-vs-threshold assertion (cost attributed to reduction relocation + expected constrained-kernel speedup stated + **numpy-vectorized-arm comparison — FIX-M2**) + factor-IC bit-identity + shuffle-invariance + (if kernel added) native parity oracle rtol=1e-8 + `scripts/build_native_backends.py` green + the wave-2 determinism spike run-twice byte-identical.
- **B3 (after Wave 3, BLOCKER GATE):** B1 set + C0 guard (`CandidateOutcomeRecord` key-set unchanged + new ledger is a new file) + the AC-23 CALIBRATION gate — DSR(num_trials=N_eff, variance_across_trials=empirical) primary verdict, the EXACT eigen-free participation-ratio N_eff scalar fed to DSR (k-duplicates+m-independents KAT: `N_eff=(k+m)²/(k²+m)`), the pinned dispersion-term KAT, N_eff≤raw_N with raw_N never fed as the independent count, per-candidate spa sanity-only, and THREE KATs (reject-known-null-at-realistic-breadth / accept-edge-at-N=1 endpoints + a MIDDLE correlated-near-duplicate family varying BOTH DSR inputs proving calibration not just monotonicity; N_eff/raw_N/dispersion/DSR-verdict/PBO/spa-sanity byte-reproducible across ≥2 thread counts) + provenance completeness.
- **B4 (after Wave 4):** B1 set + C0 guard (Sharpe-CI/options/exec-attr/cost_ab confirmed as NEW top-level sub-objects, default-OFF, flat-payload value-type preserved) + read-only assertions (MCP exposes no order/place/cancel tools) + cost-realism / RiskManager parity green.
- **B5 (after Wave 5, RELEASE GATE):** golden byte-identical + full suite green + C0 guard green + `lq config validate` + native build green + `gh pr create` on `feat/vibe-adoption` (no main merge, no live-real).

---

## 6. ADR

### ADR-1 — Eigen-free N_eff for the primary correlation-aware survivorship correction
- **Decision:** Compute the effective-independent-trials count as the participation ratio via the eigen-free closed form `N_eff = (trace C)² / ‖C‖_F² = S² / Σ_ij C_ij²` over the candidate return-correlation matrix, and feed it as `num_trials` to `deflated_sharpe_ratio`. No `np.linalg.eigh` in `effective_trials.py`.
- **Drivers:** Guardrail 3 (no scipy/sklearn, BLAS/version non-reproducibility) applied to the centerpiece real-money gate; the plan's own EXACT-scalar N_eff + byte-reproducibility claims cannot rest on LAPACK/BLAS last-bit drift across rebuilds of a long-lived branch.
- **Alternatives considered:** (a) `np.linalg.eigh` participation ratio with tiny-negative-eigenvalue clipping + near-zero-trace guard — reintroduces BLAS nondeterminism, more complex; (b) entropy of eigenvalues / eigenvalue-count-above-threshold — additional modeling choice, still eigh-backed; (c) raw dedup count — under/over-deflation, rejected as the blocker this revision fixes.
- **Why chosen:** the participation ratio is a symmetric function of eigenvalues with an eigen-free identity (correlation matrix unit diagonal ⇒ trace = S exactly, `Σλ² = trace(C²) = ‖C‖_F²`); it reproduces the k-duplicates+m-independents KAT EXACTLY (`(k+m)²/(k²+m)`), yields automatic bounds [1,S], removes clipping/guard complexity, is genuinely bit-stable across BLAS builds, and unifies with the C2a numpy-canonical-order reduction discipline.
- **Consequences:** `eigh` is reserved for C4 max-div (eigenvectors genuinely needed); the "shared eigh path C4↔C2c" coupling is REMOVED from open risks / premortem; N_eff determinism now derives from sums-of-squares over canonical order, identical in kind to the C2a reduction guarantee.
- **Follow-ups:** the participation-ratio MAPPING (vs entropy) remains a pinned modeling choice needing human inspection before an alpha is trusted; documented in `effective_trials.py` and Open Risks.

### ADR-2 — DSR deflation pins BOTH parameters (num_trials=N_eff AND empirical variance_across_trials)
- **Decision:** Pass `deflated_sharpe_ratio` an EMPIRICAL `variance_across_trials` computed from the candidate-Sharpe cross-section (seeded, canonical order) alongside `num_trials=N_eff`; add a dispersion-term KAT and extend the B3 middle-calibration KAT to vary BOTH inputs.
- **Drivers:** DR principle #4 (trust protected STATISTICALLY, not just wired); `deflated_sharpe_ratio` (`:192`) deflates by BOTH `num_trials` AND `variance_across_trials` (via `expected_max_sharpe` `:92`); with `None` it falls back to the conservative single-series stand-in `_estimate_sr_variance_across_trials` (`:180`), leaving half the deflation magnitude unpinned.
- **Alternatives considered:** (a) leave `variance_across_trials=None` (stand-in) — conservative (biases toward over-rejection) but the calibration KAT would pass by construction while the dispersion is miscalibrated on real families; (b) a fully bespoke joint-SPA max-statistic as the primary — more net-new statistical machinery, higher risk on a real-money branch, contradicts the AC-23 REUSE intent.
- **Why chosen:** the empirical cross-candidate Sharpe dispersion is available in a real search; pinning it fully anchors the two-parameter DSR so "the deflation bites correctly" is a real calibration claim, not a half-claim, while staying within AC-23's reuse-of-`research_metrics` intent.
- **Consequences:** the B3 middle KAT varies both inputs; provenance records `variance_across_trials`; determinism proofs cover it.
- **Follow-ups:** the joint-block-bootstrap max-SPA (White's Reality Check proper) MAY be added later as an advisory family-level cross-check, non-load-bearing, with its own determinism proofs.

### ADR-3 — Batch factor-IC: numpy-vectorized batched-reduction arm as the escalation baseline
- **Decision:** `scripts/bench_factor_ic.py` carries three arms — per-candidate Python loop, a fully numpy-vectorized batched cross-sectional reduction over the `(T·S, N)` matrix in canonical order, and (conditionally) a Rust kernel — and the escalate/skip verdict + expected-speedup are measured against the numpy-vectorized arm.
- **Drivers:** Guardrail 4 (hotspot-only Rust); banning `group_by().agg()` forces reductions through numpy and can itself manufacture the measured hotspot; comparing Rust against a per-candidate-loop strawman can wrongly justify a native kernel.
- **Alternatives considered:** (a) two-way Polars-vs-Rust framing with an implicit per-candidate loop baseline — strawman, rejected; (b) Rust-first (Option C) — violates hotspot-only, rejected.
- **Why chosen:** the vectorized batched reduction is the cheapest deterministic Python path within the no-scipy/determinism envelope and likely eliminates the loop hotspot entirely; measuring against it makes the realizable-gain gate honest.
- **Consequences:** Rust escalation almost certainly fails the realizable-gain gate unless a genuine hotspot survives the vectorized arm; the numpy canonical-order reduction remains the SOLE reference any Rust oracle must match at rtol=1e-8.
- **Follow-ups:** if the vectorized arm still exceeds the pre-committed budget, escalate to the ordered/pairwise-reduction Rust kernel with its rtol=1e-8 parity oracle.

### ADR-4 — Additive sub-blocks never touch the flat metric payload; keyset guard covers value-type
- **Decision:** Sharpe-CI, execution-attribution, and cost_ab emit into a SEPARATE artifact / new nested container, NEVER the flat `dict[str,float]` returned by `resolve_compute_metric_payload` (`:453`) / `compute_metric_summary` (`:525`); the C0 guard covers that payload's key-set AND value-type.
- **Drivers:** Guardrail 2 (additive-subblock no-leak); inserting a nested container into the flat payload changes the value type and can flip golden bytes.
- **Alternatives considered:** (a) nest sub-objects into the flat payload — breaks value-type, rejected; (b) key-set-only guard — misses the value-type flip, rejected.
- **Why chosen:** separate artifacts + a value-type assertion make the no-leak discipline enforceable and fail-loud.
- **Consequences:** each emitter lane registers a NEW top-level sub-object via `register_additive_subblock`; C0 asserts flat-payload values stay float.
- **Follow-ups:** none beyond per-barrier enforcement.

### ADR-5 — spa demoted to single-strategy sanity; DSR(N_eff) is the family gate
- **Decision:** `spa_like_pvalue` (`:318`, verified single-strategy, docstring "single-strategy case" at `:327`) is a per-candidate SANITY check with NO family-level correlation-robustness claim; the family-level multiple-testing correction is `deflated_sharpe_ratio(num_trials=N_eff, …)`; `approx_pbo` (`:245`) is the overfit check.
- **Drivers:** source verification showed the prior "spa = correlation-robust realized-MAX family gate" framing was a MISREAD (1-D single series, no candidate-family axis, no max-over-candidates statistic, no joint resample).
- **Alternatives considered:** (a) keep spa as the family gate — statistically unfounded, rejected; (b) build joint-SPA-as-primary — higher risk, deferred to optional advisory.
- **Why chosen:** closest to AC-23's REUSE intent, low-risk, uses each `research_metrics` function in its verified role.
- **Consequences:** the load-bearing trust signal is DSR(N_eff, dispersion); spa is sanity-only; the optional joint-SPA is advisory.
- **Follow-ups:** optional joint-SPA advisory as in ADR-2 follow-up.

### ADR-6 — Long-lived branch golden/keyset re-baseline is user-gated on main-merge-in
- **Decision:** golden fixtures AND the C0 keyset snapshot are re-baselined ONLY by an explicit, recorded user decision, triggered on a main-merge-in that legitimately changes them; a stale-golden barrier failure is thereby distinguishable from a genuine regression.
- **Drivers:** `feat/vibe-adoption` is long-lived; main may legitimately drift golden/keyset state; silent re-baselining would erode barrier trust.
- **Alternatives considered:** (a) never re-baseline — stale-golden false positives block progress; (b) auto-re-baseline on merge — silent drift, catastrophic for backtest↔live parity. Both rejected.
- **Why chosen:** an explicit user-gated procedure preserves parity while allowing legitimate main-side updates.
- **Consequences:** the delivery procedure records each re-baseline; barriers stay trustworthy.
- **Follow-ups:** document the re-baseline record location (e.g. a note in the PR + a commit message tag) at delivery time.

---

## 7. Delivery Procedure

1. **Branch:** create `feat/vibe-adoption` off the current tip (NOT main; no main merge, no `LUMINA_ENABLE_LIVE_REAL` flip — user's decision).
2. **Atomic commits (one per lane/item, in dependency order):**
   - `C0` keyset guard (single-seam capture + per-class net + static net + canary + flat-payload value-type)
   - `C3` lookahead/purity guards (+ optional AC-33 architecture blocklist)
   - `C1a` `ts_mean` / `ts_mean_series` operators
   - `C4` ERC / max-div / MV / HRP optimizers (+ `portfolio.allocation_method` default equal_weight)
   - `C5` execution-attribution scaffold (gated OFF)
   - `C1b` Qlib158 formulaic factor library (+ IC/IR eligibility bar doc)
   - `C2a` batch factor-IC (Polars elementwise + numpy canonical-order reduction; bench 3-arm; determinism spike) (+ Rust kernel commit ONLY if the bench justifies it)
   - `C2b` alpha discovery search loop (raw_N upper-bound + candidate return matrix)
   - `C2c` survivorship gate (DSR(N_eff, dispersion) primary + eigen-free effective_trials + spa sanity + PBO) + new `AlphaCandidateRecord`/`ResearchLedger` + gated LLM proposer
   - `C5a` `lq alpha` CLI
   - `C5b` options-erf / forex / Sharpe-CI / TradFi gated seams
   - `C5c` MCP read-only bridge + dashboard views
   - `C6a` cost-realism A/B + RiskManager go-live parity feeding the promotion gate
   - `C6` delivery/verification
   - Commit-message footer: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`
3. **Green gate before push:** at every barrier and before push — `uv run pytest` full suite green + both golden tests byte-identical with all gates OFF + `tests/test_metric_dict_keyset_guard.py` green + `scripts/build_native_backends.py` green (+ kernel rtol=1e-8 oracle if added) + `lq config validate` passes + toggling-gates-ON spot-check unchanged.
4. **Re-baseline discipline (ADR-6):** golden + C0 snapshot re-baselined ONLY by explicit recorded user decision on a legitimate main-merge-in; never silently.
5. **PR:** `gh pr create` targeting the repo default review flow (NOT auto-merge to main). PR body ends with: `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.
6. **Explicitly out of scope for this workflow (user's decision):** merge to main, `LUMINA_ENABLE_LIVE_REAL` flip, running the actual backtest numbers on production data.

---

## 8. Open Risks

1. **Hotspot verdict.** Whether batch factor-IC benchmarks as a genuine hotspot is decided at B2 against a PRE-COMMITTED threshold measured against the numpy-vectorized batched-reduction arm; the numpy-sole-reduction choice (banning `group_by().agg()`) can itself manufacture the hotspot AND caps the achievable Rust speedup (ordered/pairwise reduction to pass rtol=1e-8), so the bench must attribute cost to the reduction relocation AND state the expected constrained-kernel speedup vs the vectorized arm before an escalate/skip verdict; if the constrained kernel cannot clear the budget it is correctly skipped.
2. **Determinism substrate.** Determinism rests SOLELY on the numpy canonical-order reduction over a sorted `(symbol,timestamp)` index (Polars elementwise-only), proven behaviorally by bit-identity + ≥2-thread identity + row-shuffle-invariance; residual risk is a Polars op that both reduces AND is shuffle-invariant (not a realistic fp-reorder source).
3. **N_eff mapping choice.** N_eff is now eigen-free (BLAS-independent, ADR-1), but the participation-ratio MAPPING (vs entropy / eigenvalue-count-above-threshold) is a pinned modeling choice needing human inspection before any alpha is trusted; mitigated by the EXACT-scalar KAT + the MIDDLE calibration KAT + run-twice reproducibility.
4. **DSR dispersion term.** `variance_across_trials` is now empirically pinned (ADR-2), but the empirical cross-candidate Sharpe dispersion estimator is itself a modeling choice; the dispersion-term KAT + the two-input middle-calibration KAT anchor it, but real candidate families should be inspected before trust.
5. **C0 scope honesty.** C0 is idiom-bounded (single-seam dynamic capture + per-class capture for enumerated emitters + static net + canary + flat-payload value-type), NOT a repo-wide no-leak proof; the true guarantee stays with the two golden tests + full suite. Residual risk = a producer neither exercised nor enumerated — mitigated by the canary + pinned covered-idiom list + fail-loud on newly-discovered producers.
6. **Qlib158 economic hollowness.** Several Qlib158 formulas assume cross-sectional rank; per-symbol crypto-perp adaptation may weaken IC — gated behind a DOCUMENTED minimum IC/IR eligibility bar with flagged (not silently carried) gutted factors, but the bar's threshold values are a modeling choice needing inspection.
7. **Optimizer conditioning.** LW-shrinkage cov + pure-numpy projected-gradient for max-div/MV on ill-conditioned crypto covariance may need iteration/tolerance tuning to stay deterministic; the KAT set includes a degenerate/near-singular covariance case AND an HRP `cluster_by_correlation` tie-break-determinism case, but coverage of all singular/tied regimes is not guaranteed. (Note: the former C4↔C2c shared-eigh coupling is REMOVED — N_eff no longer uses eigh; only C4 max-div uses eigenvectors.)
8. **Cost-realism calibration.** cost-realism `sqrt_impact` + funding-coverage parameters are modeling choices reconciled against the live fee/funding model; miscalibration could over/under-reject alphas — needs the C6a reconciliation KAT and human review before trust; landed in wave 4 so it is reviewable before delivery.
9. **Non-Python toolchain.** MCP optional-dep + Next.js dashboard add a non-Python surface; CI must keep them strictly optional/lazy so the core pytest suite and golden path stay green without them.
10. **Long-lived branch drift.** `feat/vibe-adoption` is long-lived and main may drift; golden fixtures AND the C0 snapshot must only ever be re-baselined by explicit recorded user decision on main-merge-in (ADR-6), never silently, to preserve backtest↔live parity.
11. **LLM proposer nondeterminism.** Even gated OFF and deterministic-registration-only, the LLM proposer introduces a non-reproducible generation source; its outputs must be frozen/seeded artifacts feeding the deterministic gate, must count toward raw_N and enter the correlation matrix feeding N_eff, and never be a live dependency.
12. **C0 registration coordination.** `register_additive_subblock` is a cross-lane obligation (C5/C5b/C2c/C6a); a lane that forgets to register its new top-level sub-object will correctly FAIL the guard (fail-loud via dynamic capture or static net), but this must be enforced at each barrier.

---

*End of plan. Team executes only on explicit user approval; last critic verdict was ITERATE with the fixes above folded in as revision-4.*
