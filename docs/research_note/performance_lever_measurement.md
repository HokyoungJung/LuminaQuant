# Performance-Lever Measurement Foundation -- Honest Portfolio / Cost Contract

> **Scope.** This note documents the MEASUREMENT foundation (not any performance
> lever) shipped on branch `fix/overfit-selection-gates-20260708`. It exists so
> that later performance levers are measured truthfully -- on the REAL nonlinear
> cost/funding engine, deflated by every config-grid cell tried -- instead of
> producing another overfit / cost-shuffle illusion. Every new behavior is
> config-gated and DEFAULT OFF: the shipped `config.yaml` / `RuntimeConfig()` load
> is byte-identical; the corrections turn ON only under
> `configs/profiles/backtest_cost_realistic.yaml` and/or
> `configs/profiles/research.yaml`.

## 한줄 요약 (Korean summary)

성능 레버를 정직하게 재려면 먼저 측정 배관을 고쳐야 한다. 이 저장소에는 **서로
다른 두 개의 비용 엔진**이 있다. (a) 벡터화 리서치 스코어러는 `net = returns -
turnover * cost_rate`로 **선형 비용(펀딩 없음)**만 물리고 DSR/PBO/SPA 게이트를
먹인다. (b) 실제 백테스트 엔진은 **비선형 sqrt-impact + 스프레드 + 수수료 +
펀딩**을 물린다(감사의 -11..-18 net 수치의 출처). (a)에서만 잰 레버는
반증불가능한 cost-shuffle다. 또한 후보별 정직 게이트는 시도한 **할당 설정 격자
셀 수**로 포트폴리오 Sharpe를 deflate하지 않아, 그리드를 튜닝해 최적 셀만
출하하면 게이트를 너무 쉽게 통과한다. 이 작업은 (P1) 리서치 스코어러의
config-gated 현실 비용 바닥, (P2) 포트폴리오 계층 정직 게이트 + **교차-런 시행
회계사**(시도한 모든 그리드 셀로 deflate), (P3) 생존자 다양성 진단(데이터 PC
핸드오프)으로 측정을 고친다. 레버 자체는 아직 출하하지 않는다. 아래
**6단계 측정 계약**을 통과하지 못한 레버는 실거래로 승격 금지.

---

## 1. The two-cost-engines problem (why this work exists)

There are TWO disjoint cost engines, and a lever measured only on the first is
unfalsifiable:

* **(a) VECTORIZED research scorer** -- `strategy_factory/research_runner.py`.
  `_load_candidate_signal_payload` computes `net = returns_raw - turnover *
  cost_rate` (the scored stream), and `_candidate_oos_cost_stress_metrics`
  applies the x2 / x3 stress the same linear way. `cost_rate` comes from the
  per-class map in `_candidate_cost_rate`. This is a **LINEAR** per-turnover cost
  with **NO funding**, and it is what feeds the per-candidate DSR / PBO / SPA
  gate. Any "improvement" that only shuffles this linear scalar is a cost-shuffle,
  not an edge.
* **(b) REAL backtest engine** -- `backtesting/portfolio_backtest.py`,
  `backtesting/execution_sim.py`, `backtesting/cost_models.py`. This charges
  **nonlinear sqrt-impact + spread + fees + funding** (the source of the audit's
  -11..-18 net numbers). This is the only engine whose numbers can promote a
  strategy to real money.

Second defect: the per-candidate honest gate does **not** deflate the
portfolio/allocation Sharpe by the number of ALLOCATION configs tried. So tuning a
config grid (bands, min-hold, allocator, regime thresholds, order policy, funding
window) and shipping the best cell clears the gate too easily -- the classic
config-grid overfit.

This foundation fixes the MEASUREMENT. It ships NO performance lever.

---

## 2. What shipped (P1 / P2 / P3)

### P1 -- Config-gated realistic cost floor (research scorer)

* `ResearchConfig.cost_rate_multiplier: float = 1.0` and
  `ResearchConfig.cost_rate_bps_override: float | None = None`
  (`configuration/schema.py`).
* `research_runner._candidate_cost_rate(candidate, *, scoring_config=None)` now
  FIRST resolves the exact legacy per-class map value, then: if
  `cost_rate_bps_override is not None` returns `override / 10_000`; elif
  `cost_rate_multiplier != 1.0` returns `map_value * multiplier`; else returns the
  exact map value. With the shipped defaults (`override is None` AND
  `multiplier == 1.0`) it is **byte-identical** (`x * 1.0 == x`; the override
  branch short-circuits on `None`). The flag is read via the existing
  `_research_flag(scoring_config, name, default)` seam, threaded through
  `_load_candidate_signal_payload` -> `_evaluate_candidate`, and emitted into
  `score_config['research']` by `research_run_support.research_config_to_overrides`.
* Profile calibration (`backtest_cost_realistic.yaml`):
  `research.cost_rate_multiplier: 2.2`. Rationale: the scorer's base map is ~5
  bps/turnover; this profile's one-way execution cost is
  `taker_fee_rate (0.0004) + spread_rate (0.0002) + slippage_rate (0.0005) =
  0.0011` (11 bps), so `11 / 5 = 2.2x`. It aligns a **SCALAR** only -- it does NOT
  reproduce the nonlinear sqrt-impact + funding shape of engine (b) -- and it is
  monotonically `>=` the legacy cost for every class, so it can only make a lever
  look WORSE, never manufacture a gain. `cost_rate_bps_override` stays `null` so
  the multiplier path preserves the per-class relative structure. (Purpose is
  measurement integrity, not a new backtest number: engine (a) is still just a
  cheap pre-filter; promotion is decided on engine (b).)

### P2 -- Portfolio-layer honest gate + cross-run trial accountant (the linchpin)

* `ResearchConfig.portfolio_honest_gate: bool = False` (ON in both strict
  profiles).
* `portfolio_followup_rules.portfolio_honest_gate_report(net_returns, *,
  num_trials=1, dsr_gate_floor=0.0, spa_gate_ceiling=1.0, pbo_gate_ceiling=1.0,
  stream="net")` runs DSR (`research_metrics.deflated_sharpe_ratio`, deflated by
  `num_trials`), SPA (`spa_like_pvalue`), and PBO (`approx_pbo`, the single-stream
  fold-instability estimator the per-candidate hard gate also uses) on the
  weighted-portfolio NET return stream. Reuses the SAME floors already in the
  profiles (`dsr_gate_floor` / `spa_gate_ceiling` / `pbo_gate_ceiling`).
* **Cross-run trial accountant:** `count_config_grid_trials(axes)` returns the
  PRODUCT of the declared config-grid axis sizes (e.g. `{"band_min_hold": 6,
  "allocator": 3, "regime_threshold": 4, "order_policy": 2, "funding_window": 2}`
  -> 288 cells). The portfolio DSR must be deflated by THIS count -- the total
  cells searched across runs -- NOT `candidate_count`. The data-PC harness
  DECLARES how many cells it searched and passes it as `num_trials`; default
  `num_trials=1`.
* **Uplift mode:** `portfolio_uplift_gate_report(on_rows, off_rows, ...)` gates the
  ON-minus-OFF uplift stream (aligned by date) so a lever is judged on the
  marginal edge it adds, deflated by the same `num_trials`.
* `evaluate_weighted_portfolio(rows, *, honest_gate=False, num_trials=1, ...)`:
  with the gate OFF (the shipped default) the output dict is **byte-identical** --
  no new key. Only when `honest_gate=True` is a `"portfolio_honest_gate"` key
  added.
* **Honesty proof (data-free, `tests/test_portfolio_honest_gate.py`):** a synthetic
  "best-of-many" stream (the MAX in-sample Sharpe over N noise trials) is REJECTED
  once deflated by `num_trials=N` (DSR 0.28 < 0.90 floor), while a genuine
  single-config stream (`num_trials=1`) with real edge PASSES (DSR ~1.0). The
  accountant is load-bearing: the SAME best-of-many stream would PASS at
  `num_trials=1` (DSR 0.96). This proves the accountant stops config-grid
  overfitting.

### P3 -- Survivor-diversity diagnostic (data-PC handoff)

* `scripts/research/survivor_diversity_report.py` runs on the data PC where real
  funding-charged returns exist. Given the survivor sleeves (net-Sharpe > 0 under
  funding-correct realistic cost) it reports: survivor count; the pairwise return
  correlation matrix; the number of genuinely low-correlation clusters
  (`|corr| < --low-corr-threshold`, default 0.3, via
  `portfolio.optimizer_core.cluster_by_correlation`); the CRASH-period correlation
  (correlation conditioned on benchmark deep-drawdown bars); and a VERDICT:
  diversification upside **BOUNDED** (`< 3` low-corr clusters OR high crash
  correlation) vs **REAL** (`>= 3` low-corr clusters AND low crash correlation).
* Accepts survivor returns via CSV / parquet / JSON path so it is runnable on the
  data PC; `--apply-cost-drag` nets GROSS inputs with the realistic regime via
  `research.cost_realism.apply_cost_drag` first. Synthetic-return unit test
  (`tests/test_survivor_diversity_report.py`): 2 highly-correlated streams ->
  BOUNDED; 4 uncorrelated streams -> REAL.
* **Purpose:** P3 SIZES the diversification upside BEFORE any allocation lever is
  built. If the upside is BOUNDED, the HRP/ERC/vol levers below are not worth the
  complexity; only a REAL verdict justifies pursuing them.

---

## 3. The SIX-STEP measurement contract (every lever must pass)

No performance lever promotes to real money until it passes ALL SIX, in order:

1. **OFF byte-identical.** The lever ships config-gated, DEFAULT OFF. Prove the
   golden / default outputs are byte-identical with the flag off (grep tests for
   `byte_identical` / `shipped_config`; OFF-path is a strict identity no-op /
   exact IEEE arithmetic).
2. **Profile ON.** Turn the lever ON only under
   `backtest_cost_realistic.yaml` / `research.yaml`; confirm it activates
   end-to-end (config -> `research_config_to_overrides` -> `score_config['research']`
   / backtest wiring).
3. **Cost grid on the REAL engine.** Re-measure on engine (b) across a
   `10 / 15 / 20 bps + funding` cost grid (NOT the linear scorer). Funding must be
   charged on the UTC settlement boundary (`funding_on_utc_boundary: true`), and
   `require_funding_coverage: true` so a leveraged run fails loudly instead of
   silently charging 0 funding.
4. **Fresh-forward lockbox.** Evaluate on a never-touched forward lockbox window
   (`use_lockbox_split: true`, `purge_embargo_bars >= 1`); rank/select on
   validation, report the lockbox as the sole OOS.
5. **Deflated portfolio gate.** Run `portfolio_honest_gate_report` on the
   weighted-portfolio NET stream (and `portfolio_uplift_gate_report` on the
   ON-minus-OFF uplift) with `num_trials = count_config_grid_trials(grid)` counting
   EVERY grid cell searched -- bands x min-hold, allocator set, regime axis x
   threshold, order policy, funding window. The lever must clear
   `dsr_gate_floor=0.90 / spa_gate_ceiling=0.05 / pbo_gate_ceiling=0.50` AFTER
   deflation.
6. **Net-positive proof.** Report OFF-vs-ON for: turnover, cost, funding,
   gross-Sharpe, net-Sharpe, Calmar, MDD. The lever is accepted only if net-Sharpe
   and Calmar improve and MDD does not worsen, on the REAL engine, after the
   deflated gate.

If a gate cannot be made honest AND green, STOP and report -- never weaken the
gate to pass.

---

## 4. Deferred levers (how to A/B them once P3 sizes the upside)

None of these ship here. Pursue them only after P3 returns a REAL verdict, and A/B
each through the six-step contract above (measure the UPLIFT stream, deflated by
the full grid).

* **L-A -- HRP / ERC offline allocation manifest.** Build via
  `portfolio.quality_gated_allocation` (`allocate_quality_gated`,
  `build_allocation_manifest`) over the NET (post-cost) covariance and drive it
  with `strategies/artifact_portfolio_mode.py` (`ArtifactPortfolioModeStrategy`).
  A/B: `{legacy, erc, hrp}` x cap set -> each cell is a `num_trials` count.
* **L-B -- Down-only vol overlay.** Wrap the portfolio with
  `strategies/vol_managed_risk_overlay.py` (`VolManagedRiskOverlayStrategy`) as a
  DOWN-ONLY overlay (de-lever in high vol, never lever up). **Check it does not
  double-count the `ArtifactPortfolioMode` gross caps** -- the overlay and the
  artifact mode must not each independently clamp the same gross exposure.
* **L-C -- No-trade band + min-hold, RE-TARGETED into the REAL engine.** Wire the
  existing but currently DEAD `portfolio/cost_aware_constructor.py`
  `no_trade_band_bps` (and a min-hold) into `backtesting/portfolio_backtest.py` /
  the `portfolio.strategy_quality` (`StrategyQualityOverlay`) seam -- i.e. into
  engine (b). **The earlier "insert at `strategy_signal_dispatch`" idea is WRONG:**
  that seam feeds the LINEAR research scorer (a), where a no-trade band merely
  shuffles linear cost and cannot show the nonlinear-impact / funding savings that
  are the entire point. Measure L-C ONLY on the real engine.
* **L-D -- Funding-entry guard (pre-registered).** A FIXED, pre-registered rule:
  skip entry when the remaining intended hold is shorter than one funding interval
  (so a sub-interval round trip that would pay a full funding charge is never
  opened). Pre-register the rule BEFORE measuring; do not tune its threshold on the
  test window.
* **L-E -- Maker / LMT execution.** Measurement scaffolding ONLY until real fill /
  quote data exists. A maker/limit assumption cannot be honestly measured without
  real queue-position / fill data, so do not promote any maker-based edge from the
  simulator alone.

---

## 5. Files

* `src/lumina_quant/configuration/schema.py` -- `ResearchConfig.cost_rate_multiplier`,
  `cost_rate_bps_override`, `portfolio_honest_gate`.
* `src/lumina_quant/strategy_factory/research_runner.py` -- `_candidate_cost_rate`
  cost floor + `scoring_config` threading.
* `src/lumina_quant/strategy_factory/research_run_support.py` -- emits the cost
  fields into `score_config['research']`.
* `src/lumina_quant/portfolio_followup_rules.py` -- `count_config_grid_trials`,
  `portfolio_honest_gate_report`, `portfolio_uplift_gate_report`, and the
  config-gated `honest_gate` path in `evaluate_weighted_portfolio`.
* `scripts/research/survivor_diversity_report.py` -- P3 diagnostic.
* Profiles: `configs/profiles/backtest_cost_realistic.yaml` (cost floor +
  portfolio gate), `configs/profiles/research.yaml` (portfolio gate).
* Tests: `tests/test_portfolio_honest_gate.py`,
  `tests/test_survivor_diversity_report.py`,
  `tests/test_research_profile_activation.py` (override-shape).
