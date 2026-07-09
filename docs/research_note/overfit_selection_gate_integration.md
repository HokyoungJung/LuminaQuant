# Overfit Selection Gate -- Data-PC Integration Guide

> **Handoff scope.** The canonical selection / stamping / DSR-SPA-PBO aggregation
> pipeline lives on the **data PC** (`LuminaQuant/` layout, `private-main`
> branch), NOT this repo's monthly-refit walk-forward engine. This repo ships the
> reusable, unit-proven gate helper (`apply_selection_reject_and_dedup` +
> `passes_dsr_spa_hard_gate`) and a WORKED REFERENCE wiring in
> `scripts/research/run_alpha_zoo_69_asset_monthly_refit_walkforward.py`. This
> document is the implementation-ready spec for porting that gate into the
> data-PC pipeline.

## 한줄 요약 (Korean summary)

2026-07-04 walk-forward 런은 train/validation 성적만으로 22개 후보를 골라
`research_selected`로 찍었는데, 이들의 평균 validation Sharpe는 +2.62였지만
locked-OOS에서 -4.29로 붕괴했고(그중 14개는 동일 9-심볼 4h 바스켓의 클론),
파이프라인이 스스로 기록한 DSR/SPA/PBO 게이트를 실제 selection에 **강제하지
않았다**. 해결책: 후보 admit 및 최종 merge 단계 양쪽에서
`apply_selection_reject_and_dedup`를 통과시켜 (1) DSR/SPA/PBO hard-reject과
(2) basket/lineage/family 중복 제거를 강제한다. 이 저장소의 엔진은 기준
파이프라인이 아니라 **참고 예시**이며, 모든 신규 동작은 config-gated + 기본
OFF(바이트 동일)다. 아래는 데이터 PC 파이프라인이 그대로 배선하기 위한 정밀
스펙이다.

---

## 1. Problem recap

The 2026-07-04 full walk-forward run selected 22 candidates on **train +
validation only**:

- Mean validation Sharpe of the selected cohort: **+2.62**.
- Their **locked-OOS** mean Sharpe: **-4.29** (a full collapse).
- **14 of the 22** are the SAME 9-symbol 4h crypto basket wearing different
  `strategy_class` hats (identical-symbol clone cluster).
- The pipeline **recorded** per-candidate DSR / SPA / PBO in each fold's
  `selected[].validation` block but **never enforced** those numbers as a
  rejection gate -- selection ranked on a soft score that let weak-DSR clones in.

Recovering each selected candidate's validation-block DSR / SPA / PBO from the
recorded artifact and applying the strict floors rejects **22 / 22** (proven
data-free in `tests/test_overfit_selection_reject_gate.py`). The single
locked-OOS "survivor" (`candidate_id` starting `1f1fd241c12f0bc2`, validation
Sharpe 5.409) is ALSO rejected at its own validation DSR of 0.234 -- so the gate
rejects on uniform a-priori merit, not by cherry-picking the collapsers.

## 2. The reusable API (exact signatures)

Import path:

```python
from lumina_quant.strategy_factory import (
    apply_selection_reject_and_dedup,
    passes_dsr_spa_hard_gate,
)
```

### 2.1 `apply_selection_reject_and_dedup`

```python
def apply_selection_reject_and_dedup(
    rows: Iterable[dict[str, Any]],
    *,
    mode: str = "oos",
    robust_score_params: dict[str, Any] | None = None,
    max_per_symbol_basket: int | None = None,
    max_per_lineage: int | None = None,
    max_per_family_basket: int | None = None,
    enabled: bool = False,
) -> list[dict[str, Any]]:
```

- `enabled=False` (default) is a **STRICT identity no-op**: returns `list(rows)`
  with the same objects, in the same order (nothing dropped / reordered /
  mutated). This is the byte-identical OFF-path.
- `enabled=True`, reject-only (all `max_per_*` are `None`): applies
  `passes_dsr_spa_hard_gate` per row, drops failures, keeps survivors in **input
  order** (no reorder).
- `enabled=True`, at least one `max_per_*` cap active: rows are processed
  **best-`hurdle_score`-first** (descending, with the original input index as a
  deterministic tie-break) BEFORE applying the caps, so the surviving basket /
  lineage / family representative is the **best-scoring** clone -- not merely the
  earliest one. Each cap is applied only when its argument is not `None`.

Use `mode="val"` on the data PC so the gate reads the VALIDATION block (see
section 3). The three caps key off `symbols` only (no DSR plumbing needed), so
basket dedup works even before the DSR/SPA/PBO metrics land.

### 2.2 `passes_dsr_spa_hard_gate`

```python
def passes_dsr_spa_hard_gate(
    candidate: dict[str, Any],
    *,
    mode: str = "oos",
    robust_score_params: dict[str, Any] | None = None,
) -> bool:
```

Returns `True` unless the (opt-in) DSR / SPA / PBO hard gate rejects the
candidate. The gate is **OFF** unless `robust_score_params` carries
`enforce_selection_reject_gate` (or the legacy `strict_selection_gate`) truthy;
with it OFF this is a strict no-op returning `True`. The floors
`dsr_gate_floor` / `spa_gate_ceiling` / `pbo_gate_ceiling` are read from
`robust_score_params`.

## 3. Required per-candidate VALIDATION-block metric keys

For the gate to work the data-PC pipeline MUST emit these keys into each
candidate's **validation** metric block (the block `mode="val"` reads --
`validation`, then `val`, then `locked_oos_report_only` as a last-resort
fallback):

| key | meaning | fail-direction if missing |
| --- | --- | --- |
| `deflated_sharpe` | Deflated Sharpe Ratio, deflated against the WHOLE-SEARCH trial count | missing -> `0.0` -> fail-**CLOSED** |
| `spa_pvalue` | SPA reality-check p-value | missing -> `1.0` -> fail-**CLOSED** |
| `pbo` (alias `approx_pbo`) | probability of backtest overfitting | missing -> `0.0` -> fail-**OPEN** |

**Deflate against the whole search, not `num_trials=1`.** The `deflated_sharpe`
value MUST be computed against the full candidate-search trial count. This is an
achievable, honest bar: a genuine edge reaches `DSR ~= 0.978` at
`num_trials ~= 1400` (proven in
`test_strict_dsr_floor_is_achievable_for_a_genuinely_strong_candidate`), whereas
`num_trials=1` makes the 0.90 floor near-trivial and would let overfit clones
through. Record the `num_trials` basis on the block (e.g. a `num_trials` field)
so downstream review can audit that the deflation was honest (review [MED]).

Compute the DSR with this repo's canonical estimator:

```python
from lumina_quant.strategy_factory.research_metrics import deflated_sharpe_ratio

dsr = deflated_sharpe_ratio(returns, num_trials=whole_search_trial_count)
```

### Diversity keys

The dedup caps key off either explicit fields on the row or derived keys:

- `symbol_basket` (else derived by `candidate_symbol_basket_key(row)` from
  `symbols`).
- `lineage` (else `candidate_lineage_key(row)` from
  `family` + `strategy_class` + `timeframe` + `symbols`).
- `family_basket` (else `candidate_family_basket_key(row)` from
  `family` + `symbols`).

So each candidate row needs at minimum `symbols`, plus `family` /
`strategy_class` / `strategy_timeframe` (or `timeframe`) for the lineage and
family caps to discriminate. Supplying the pre-computed
`candidate_symbol_basket_key` / `candidate_lineage_key` /
`candidate_family_basket_key` fields on the row is equivalent and avoids
recomputation.

## 4. Strict thresholds (from `configs/profiles/research.yaml`)

```yaml
research:
  enforce_selection_reject_gate: true   # master ON switch (reject axis)
  dsr_gate_floor: 0.90                   # reject deflated_sharpe < 0.90
  spa_gate_ceiling: 0.05                 # reject spa_pvalue > 0.05
  pbo_gate_ceiling: 0.50                 # reject pbo > 0.50
  max_cross_trial_pbo: 0.50              # cross-trial CSCV/PBO run-tail ceiling
  max_per_symbol_basket: 2               # basket-dedup caps
  max_per_lineage: 1
  max_per_family_basket: 1
```

`robust_score_params` fed to the gate:

```python
robust_score_params = {
    "enforce_selection_reject_gate": True,
    "dsr_gate_floor": 0.90,
    "spa_gate_ceiling": 0.05,
    "pbo_gate_ceiling": 0.50,
}
```

Every field defaults to a strict no-op (`enforce_selection_reject_gate=False`,
`dsr_gate_floor=0.0`, the ceilings `1.0`, caps `None`), so a default config load
is byte-identical; the honest-research profiles
(`configs/profiles/research.yaml`, `configs/profiles/backtest_cost_realistic.yaml`)
turn them ON.

## 5. Fail-direction semantics (deliberate, review [LOW])

- **DSR missing -> 0.0 -> fail-CLOSED.** A candidate with no recorded DSR is
  rejected (0.0 < any positive floor).
- **SPA missing -> 1.0 -> fail-CLOSED.** A candidate with no recorded SPA
  p-value is rejected (1.0 > any ceiling < 1.0).
- **PBO missing -> 0.0 -> fail-OPEN.** A candidate with no recorded PBO is NOT
  rejected on that axis (0.0 <= any ceiling).

This asymmetry is intentional: DSR and SPA are the primary multiple-testing
bars and must not be silently skipped, while PBO is a secondary fold-instability
estimate. **Instruct the pipeline to always record all three** so the fail-OPEN
PBO path is never the reason a real overfit slips through.

## 6. Call sites -- enforce at BOTH boundaries

The gate must run at TWO places, or a candidate rejected per-fold re-enters at
merge:

1. **Per-fold admit.** After evaluating a fold's candidates, filter the admitted
   set through `apply_selection_reject_and_dedup(..., mode="val", enabled=True)`.
2. **Final merge / consolidation stamp.** When candidates selected in `>= 2`
   folds are merged into the run's `research_selected` set, re-run the gate on
   the consolidated rows. A candidate selected in multiple folds must be rejected
   in **ALL** folds, otherwise it re-enters at the merge stamp. Do NOT trust the
   per-fold pass alone.

Worked reference in this repo (byte-identical / OFF by default):
`scripts/research/run_alpha_zoo_69_asset_monthly_refit_walkforward.py`
-> `_configure_selection_gate` (resolves floors + caps: explicit CLI cap >
`--dedupe-baskets` default > config cap > None-OFF) and
`_apply_selection_gate_rows` (the admit-boundary hook that calls
`apply_selection_reject_and_dedup`).

## 7. Cross-trial multiplicity (run-tail reject)

Per-candidate DSR does not catch family-wide overfitting. Assemble the candidate
fold-return matrix (shape `(n_candidates, n_periods)`) and reject the run tail
when the family-wise CSCV/PBO exceeds the ceiling:

```python
from lumina_quant.strategy_factory.research_metrics import cross_trial_pbo_rejects_run

reject_run_tail = cross_trial_pbo_rejects_run(
    candidate_fold_return_matrix,
    max_cross_trial_pbo=0.50,   # from research.max_cross_trial_pbo
    enabled=True,
)
```

`enabled=False` OR `max_cross_trial_pbo >= 1.0` is a strict no-op (returns
`False`). Proven in `test_cross_trial_pbo_rejects_overfit_family_when_enabled`
and the surrounding cases.

## 8. Data-free proof pointers

- `tests/test_overfit_selection_reject_gate.py` -- recovers the recorded 22
  candidates' validation DSR/SPA/PBO from the artifact and proves **22 / 22**
  rejected under the strict floors (non-circular: uniform a-priori floors reject
  even the OOS survivor); proves the 14-member clone cluster collapses to `<= 2`
  with the **best-scoring** representative surviving; proves all-flags-OFF is a
  byte-identical passthrough; proves DSR-floor achievability at
  `num_trials ~= 1400`; exercises the cross-trial CSCV/PBO reject consumer and
  the fail-closed strict-research env guard.
- `tests/test_overfit_selection_engine_gate.py` -- pins the reference wiring:
  OFF-path identity no-op, basket-dedup collapse, strict-profile config arms both
  the reject floors AND the 2/1/1 dedup caps, CLI overrides config.

## 9. Worked example (per-fold admit + merge stamp)

```python
from lumina_quant.strategy_factory import apply_selection_reject_and_dedup

STRICT = {
    "enforce_selection_reject_gate": True,
    "dsr_gate_floor": 0.90,
    "spa_gate_ceiling": 0.05,
    "pbo_gate_ceiling": 0.50,
}

def admit_fold(candidate_rows):
    # candidate_rows carry a 'validation' block with deflated_sharpe (whole-search
    # deflated), spa_pvalue, pbo, plus symbols/family/strategy_class/timeframe.
    return apply_selection_reject_and_dedup(
        candidate_rows,
        mode="val",
        robust_score_params=STRICT,
        max_per_symbol_basket=2,
        max_per_lineage=1,
        max_per_family_basket=1,
        enabled=True,
    )

def stamp_merge(consolidated_rows):
    # Re-run the SAME gate at the merge/consolidation boundary so a candidate that
    # passed in one fold but should be rejected in all does not re-enter.
    return apply_selection_reject_and_dedup(
        consolidated_rows,
        mode="val",
        robust_score_params=STRICT,
        max_per_symbol_basket=2,
        max_per_lineage=1,
        max_per_family_basket=1,
        enabled=True,
    )
```

With `enabled=False` (or `robust_score_params=None` and all caps `None`) both
functions are strict identity no-ops, so the gate can be shipped OFF and turned
on only under the honest-research profiles.

## 10. E1 worked example -- feeding the gate from the alpha-zoo WF runner

> **Added for the v2 alpha-pool-expansion batch (2026-07-09).** The E1 code-lane
> (`1eeb1f7`) made this repo's reference runner *emit* the two per-candidate
> metrics the gate consumes; whole-search DSR stays a data-PC responsibility (it
> is not computable inside the runner's per-window metric step). This section is
> the end-to-end wiring: emission -> validation block -> `passes_dsr_spa_hard_gate`
> -> `apply_selection_reject_and_dedup`.

### 10.1 What E1 emits (and what it deliberately does not)

`scripts/research/run_alpha_zoo_69_asset_monthly_refit_walkforward.py` gained a
`_candidate_overfit_stats` helper (near `:840`) called from the per-window metric
path. When (and only when) `emit_candidate_overfit_stats` is armed it stamps two
keys into the candidate's `validation` block:

| emitted key | source | selection role |
| :-- | :-- | :-- |
| `spa_pvalue` | `research_metrics.spa_like_pvalue` (`:674`) -- single-strategy seeded bootstrap | read by `passes_dsr_spa_hard_gate`; rejected if `> spa_gate_ceiling` |
| `approx_pbo` | `research_metrics.approx_pbo` (`:309`) -- fold-instability estimate | accepted as the `pbo` alias per `selection.py:374`; rejected if `> pbo_gate_ceiling` |

Arming is config-gated and **default OFF -> byte-identical output**. Flip it only
under a strict `RuntimeConfig` (`--config` / `LQ_CONFIG_PATH` ->
`configs/profiles/research.yaml`, which sets `research.emit_candidate_overfit_stats:
true`). With the flag OFF, `_candidate_overfit_stats` no-ops and the runner's
artifacts are unchanged; proven in `tests/test_overfit_selection_reject_gate.py`
(default-OFF byte-identity + strict-config emission populates the block +
determinism).

**What E1 does NOT emit: whole-search DSR.** The runner's per-window metric step
(`_period_metrics`, `:803`) only sees ONE candidate's return slice. The
`deflated_sharpe` the gate consumes MUST be deflated against the **full
candidate-search trial count** (`num_trials = candidate_count`), a search-global
quantity that structurally does not exist inside `_period_metrics`. So E1 ships
the two per-candidate metrics in code and leaves DSR stamping to the data-PC (next
subsection). This is the honest split: the runner is a WORKED REFERENCE, not the
canonical selection pipeline.

### 10.2 Activating whole-search DSR on the data-PC

The data-PC computes `deflated_sharpe` at the **aggregation layer**, where the
candidate count is known, exactly as this repo's own aggregation precedent does:

- `research_runner.py:6649` calls `deflated_sharpe_ratio(..., num_trials=candidate_count)`
  -- the DSR is deflated against the whole search, not a single trial. (The same
  `num_trials=candidate_count` threading recurs at `:6727`, `:6738`, `:6891`,
  `:6903`, `:6917`, ... -- the pattern is: resolve `candidate_count` once for the
  run, forward it into every `deflated_sharpe_ratio` call.)
- `selection.py:347-353` (the `passes_dsr_spa_hard_gate` docstring) makes the
  contract explicit: *"A per-candidate `num_trials=1` DSR would understate
  deflation and must not be used to feed this gate."* A genuine edge still reaches
  `DSR ~= 0.978` at `num_trials ~= 1400` (achievability proof in
  `test_strict_dsr_floor_is_achievable_for_a_genuinely_strong_candidate`), so the
  0.90 floor is honest, not trivial.

**Ownership.** Per-fold, the data-PC stamps each candidate's `validation` block
with (a) the E1-emitted `spa_pvalue` + `approx_pbo`, and (b) a
`deflated_sharpe` computed with `num_trials = candidate_count` for that search,
plus a `num_trials` field recording the basis so a reviewer can audit that the
deflation was honest (review [MED], section 3). Missing `deflated_sharpe` -> 0.0 ->
fail-CLOSED (section 5), so a fold that forgets to stamp DSR rejects rather than
leaks.

### 10.3 How the emitted metrics reach the gate

Once the `validation` block carries `deflated_sharpe` (whole-search),
`spa_pvalue`, and `approx_pbo`, the gate is purely mechanical:

1. `passes_dsr_spa_hard_gate(candidate, mode="val", robust_score_params=STRICT)`
   reads the three keys (section 2.2 / section 4 thresholds), rejecting on
   `deflated_sharpe < 0.90`, `spa_pvalue > 0.05`, or `pbo > 0.50` (with `pbo`
   resolved from `approx_pbo` when the explicit `pbo` field is absent --
   `selection.py:374`).
2. `apply_selection_reject_and_dedup(rows, mode="val", enabled=True, ...)` calls
   `passes_dsr_spa_hard_gate` per row and then applies the basket / lineage /
   family caps. Run it at BOTH boundaries (section 6): per-fold admit AND the
   final merge stamp -- a candidate must be rejected in ALL folds, or it re-enters
   at merge.

So the data-PC path is: **E1 emits `spa_pvalue` + `approx_pbo` (arm via
`research.yaml`)** -> **data-PC stamps whole-search `deflated_sharpe`
(`num_trials=candidate_count`)** -> **`apply_selection_reject_and_dedup` with strict
`robust_score_params` at admit + merge** -> overfit clones (the recorded 22 /
14-clone cohort) reject 22 / 22, a genuine edge survives. The v2
alpha-pool-expansion hand-off
([`alpha_pool_expansion_v2_handoff.md`](alpha_pool_expansion_v2_handoff.md),
section 3.2) requires this exact routing for all nine lanes.
