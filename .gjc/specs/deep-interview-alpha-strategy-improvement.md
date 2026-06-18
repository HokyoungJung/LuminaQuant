# Deep Interview Spec: LuminaQuant Alpha Strategy Improvement

## Metadata
- Interview ID: 5ac753f6-d6bd-48d3-9074-87f6031d2a4f
- Rounds: 19 + Round 0 topology confirmation
- Final Ambiguity Score: 4.50%
- Type: brownfield
- Generated: 2026-06-18T12:40:00Z
- Threshold: 0.05
- Threshold Source: default
- Initial Context Summarized: no
- Status: PASSED
- Auto-Researched Rounds: []
- Auto-Answered Rounds: []
- Architect Failures: 0
- Lateral Reviews: 3 milestone panels
- Lateral Panel Failures: 0
- Refined Rounds: [1]
- Closure Overrides: none
- Restated Goal: LuminaQuant에서 기존 registered/runnable 전략과 신규 crypto+TradFi 알파를 넓게 smoke하고, clean 규율과 strict promotion gates를 통과한 후보만 full WF·fresh-forward·paper/shadow-ready 구현으로 승격해, MDD≤30%와 no-liquidation 조건 안에서 return/MDD·Calmar·return을 극대화하는 실제 구현 가능한 Strategy class와 artifact portfolio manifest를 만든다.

## Clarity Breakdown
| Dimension | Score | Weight | Weighted |
|-----------|-------|--------|----------|
| Goal Clarity | 0.95 | 0.35 | 0.3325 |
| Constraint Clarity | 0.97 | 0.25 | 0.2425 |
| Success Criteria | 0.95 | 0.25 | 0.2375 |
| Context Clarity | 0.95 | 0.15 | 0.1425 |
| **Total Clarity** | | | **0.9550** |
| **Ambiguity** | | | **0.0450** |

## Topology
| Component | Status | Description | Coverage / Deferral Note |
|-----------|--------|-------------|--------------------------|
| Existing Alpha/Strategy Reassessment | active | 기존 registered/runnable strategy와 과거 우수 후보를 넓게 smoke하고, audit·correlation·promotion 기준에 따라 selective full WF로 승격한다. | Broad smoke + selective full; smoke는 완화, full WF는 strict audit pass만 허용; JSON+Markdown에 candidate metrics, audit flags, correlation matrix, survivor list, full-WF promotion list, rejection reasons 포함. |
| New Alpha Discovery | active | 기존 데이터 안에서 crypto 및 데이터가 충분한 TradFi/equity/commodity/FX-linked alpha family/source/parameter 후보를 최대한 넓게 찾는다. | 1~2시간 broad smoke 후 survivors만 full WF/long run; crypto는 strict data sufficiency, weak-data TradFi는 shadow/research-only. |
| Strategy/Portfolio Improvement | active | 살아남은 leaf alpha를 구현 가능한 Strategy class로 만들고, 최종 조합은 artifact portfolio manifest로 risk/correlation/return 기준에 맞게 구성한다. | Train/validation-only optimizer; return/MDD 또는 Calmar objective; gross cap, MDD≤30%, correlation penalty, manifest replay reproducibility, fail-closed tests 필수. |
| Validation/Promotion Gates | active | clean/paper/shadow/real-ready-pre-stage의 승격·탈락 기준을 명확히 하여 overfit과 실전 불가능 후보를 차단한다. | Strict matrix: tier benchmark + MDD≤30% + no-liquidation + data sufficiency + cost/slippage + telemetry + fresh-forward 2 folds; real-money는 별도 승인. |

## Established Facts
- Round 1: 1차 목표는 return 극대화지만, MDD는 가급적 30% 이하이고 청산에 가까운 리스크·미래참조·post-hoc rule injection은 금지한다. Disputed: false.
- Round 2: 신규 알파 탐색은 crypto와 기존 데이터가 충분한 equity/commodity/FX-linked TradFi까지 포함한다. Disputed: false.
- Round 3: 성과 개선 인정 기준은 two-tier다. Shadow는 risk_trim 64.42% comp 또는 return/MDD 3.49를 넘고, clean/paper는 clean baseline 34.39% comp를 넘어야 한다. Disputed: false.
- Round 4: crypto 신규 알파는 strict data sufficiency를 요구하고, short/incomplete TradFi는 이론적으로 타당해도 shadow/research-only로 둔다. Disputed: false.
- Round 5: search budget은 broad smoke 1~2시간 후 survivors만 full WF/long run으로 확장한다. Disputed: false.
- Round 6: 최종 산출 목표는 real-ready 전단계이며, 실제 투입은 별도 승인이다. Disputed: false.
- Round 7: smoke survivor와 full WF promotion은 strict validation/theory/cost/risk/data/liquidity gate를 통과해야 한다. Disputed: false.
- Round 8: surviving new leaf alpha는 Strategy class/registry surface로 구현하고, 조합·리스크 관리는 artifact portfolio manifest로 구현한다. Disputed: false.
- Round 19: 이번 iteration은 신규 외부 vendor/on-chain/news 데이터 수집, real-money execution, locked-OOS 기반 튜닝, sub-minute live execution을 제외한다. Disputed: false.

## Trigger Metadata
| Round | Trigger | Status | Affected Component / Dimension | Ambiguity Direction | Evidence |
|-------|---------|--------|--------------------------------|---------------------|----------|
| 4 | D scope expansion | active_resolved_into_next_target | New Alpha Discovery / Success Criteria | 32.70% -> 33.95% up | User expanded discovery mandate to as many alpha family/source/parameter candidates as feasible. |
| all other scored rounds | none | none | n/a | ambiguity decreased or stayed controlled | Answers clarified scope, gates, implementation surface, or non-goals. |

## Lateral Review Panel
- Initial → progress: researcher/contrarian/simplifier panel noted that current top models are highly correlated, shadow variants cannot be promoted without fresh-forward evidence, and new alpha boundaries must be explicit.
- Progress → refined: panel emphasized hard gate criteria, separate weak-data TradFi shadow handling, and tried-universe logging to prevent cherry-picking.
- Refined → ready: panel passed crystallization. Remaining implementation-stage work is to align benchmark constants and choose exact optimizer/telemetry defaults in ralplan/implementation.

## Goal
Build a disciplined alpha-improvement pipeline for LuminaQuant that broadly reassesses existing runnable strategies, aggressively but cleanly searches new crypto and existing-data TradFi alpha families, promotes only strict-gate survivors into full walk-forward and fresh-forward shadow/paper readiness, and emits implementable Strategy classes plus artifact portfolio manifests that improve return under MDD≤30%, no-liquidation, cost/slippage, and anti-overfit constraints.

## Constraints
- No locked-OOS selection, threshold fitting, tie-break, pruning, or portfolio tuning.
- No future-reference, post-hoc rule injection, or non-theoretical rules.
- Crypto candidates require sufficient train + prior validation coverage for clean/paper promotion.
- Short or incomplete TradFi candidates may be researched aggressively but remain shadow/research-only until data sufficiency improves.
- Full WF promotion requires two-tier benchmark pass, MDD≤30%, no-liquidation, data sufficiency, cost/slippage feasibility, and audit pass.
- Final portfolio sizing must use train/validation-only information and apply gross caps plus correlation penalty.
- Live preflight must include execution realism and research hygiene telemetry.
- Actual real-money deployment is out of scope and requires separate approval.

## Non-Goals
- New external vendor data integration.
- New on-chain or news data collection.
- Real-money execution or exchange order routing changes.
- Locked-OOS-based tuning or post-OOS meta-selection.
- Sub-minute live execution.
- Promoting a candidate solely because it looks good in a single cherry-picked result.

## Acceptance Criteria
- [ ] A broad smoke lane evaluates as many feasible existing strategies and new alpha family/source/parameter candidates as time/memory allow within the staged budget.
- [ ] Smoke output records candidate-level metrics, audit flags, tried-universe coverage, rejection reasons, survivor list, and full-WF promotion list in JSON and Markdown.
- [ ] New alpha discovery covers crypto price/volume, cross-asset/residual/dispersion, funding/OI/taker-flow/BBO, existing-winner overlays, and TradFi/equity/commodity/FX-linked buckets where existing data permits.
- [ ] Full WF candidates are selected only from train/validation smoke evidence, never from locked-OOS evidence.
- [ ] A promoted shadow candidate beats risk_trim 64.42% comp or return/MDD 3.49 while satisfying MDD≤30%, no-liquidation, data, cost, and telemetry gates.
- [ ] A promoted clean/paper candidate beats clean baseline 34.39% comp while satisfying MDD≤30%, no-liquidation, data, cost, and telemetry gates.
- [ ] Surviving leaf alphas are implemented as Strategy classes/registry entries only when they pass theoretical and audit checks.
- [ ] Final portfolio manifests replay deterministically, fail closed on missing artifacts/components, apply gross cap and correlation penalty, and do not exceed risk limits.
- [ ] Fresh-forward promotion requires at least two new monthly folds passing benchmark, risk, cost, and telemetry checks.
- [ ] If no candidate passes strict gates, the result is no-promotion plus best shadow watchlist, with existing baseline retained.

## Deferrals
- Exact optimizer objective weights, grid breadth, telemetry numeric thresholds, and runtime split belong to ralplan/implementation planning.
- Real-money enablement is deferred until separate approval after paper/shadow/fresh-forward evidence.
- New data vendor/on-chain/news integrations are deferred outside this iteration.
- Convergence pacing deferral: no min-round floor, score-drop cap, or artificial dampening was used; bidirectional scoring controlled pacing.

## Assumptions Exposed & Resolved
| Assumption | Challenge | Resolution |
|------------|-----------|------------|
| Higher return alone is enough. | Very high return can hide liquidation-like risk or unacceptable MDD. | Return-first is allowed only under MDD≤30%, no-liquidation, and theoretical validity constraints. |
| New alphas should only be crypto. | Existing data may include useful TradFi-linked predictors. | Include crypto and existing-data TradFi/equity/commodity/FX-linked candidates, but weak TradFi data remains shadow-only. |
| A few promising candidates are enough. | User explicitly requested maximum breadth. | Use staged broad smoke over many families, then full WF only for survivors. |
| Shadow winners can become real strategies quickly. | Current top shadow models need fresh-forward and telemetry evidence. | Output target is real-ready pre-stage only; real-money requires separate approval. |
| Portfolio can be optimized on final results. | Locked-OOS tuning would contaminate evaluation. | Optimize only on train/validation, report locked-OOS without using it for selection. |

## Technical Context
- Strategy classes and registry live under `src/lumina_quant/strategies/`, including `artifact_portfolio_mode.py` and `registry.py` surfaces for implementable strategies and artifact-backed portfolio modes.
- Research lanes live under `scripts/research/`, especially `run_alpha_zoo_clean_new_alpha_discovery.py` for clean discovery/shortlist smoke and `run_alpha_zoo_69_asset_monthly_refit_walkforward.py` for monthly refit walk-forward evaluation.
- Current top-model reports live under `var/reports/current_top_models/` and indicate that the best recent raw/shadow candidates outperform clean baselines but are not real-ready.
- Prior planning under `.gjc/plans/ralplan/2026-06-07-0457-b89b/pending-approval.md` established no locked-OOS selection, no nested/hybrid contamination, candidate freeze before OOS, post-OOS/lagged-shadow quarantine, and no-promotion acceptability.
- Recent new-alpha smoke tests passed technically but did not produce adoption-ready candidates; the next effort must broaden search and strengthen gates rather than promote weak results.

## Ontology (Key Entities)
| Entity | Type | Fields | Relationships |
|--------|------|--------|---------------|
| Alpha Candidate | core domain | family, source, parameters, data sufficiency, theory, smoke metrics, WF metrics, audit flags | Candidate may become Strategy class only after gates pass. |
| Existing Strategy | core domain | registry name, runnable status, smoke metrics, correlation, audit status | Existing strategies enter broad smoke and selective full WF. |
| Strategy Class | implementation surface | signal logic, parameters, registry entry, fail-closed behavior | Strategy class implements surviving leaf alpha. |
| Artifact Portfolio Manifest | implementation surface | component weights, gross cap, correlation penalty, replay metadata, ready_for_real flag | Manifest combines strategies and risk management without locked-OOS tuning. |
| Promotion Gate | governance | benchmark, MDD, no-liquidation, data, cost, telemetry, fresh-forward | Promotion gate blocks clean/paper/shadow escalation. |
| Fresh-Forward Fold | validation evidence | monthly fold, benchmark pass, risk pass, cost pass, telemetry pass | At least two new folds required before real-ready consideration. |
| Live Preflight Gate | operational governance | cost mean/p95, spread/slippage, fill/reject/reconcile, turnover, liquidity, margin buffer, data freshness, as-of coverage | Blocks real-money execution until separately approved. |

## Ontology Convergence
| Round | Entity Count | New | Changed | Stable | Stability Ratio |
|-------|-------------|-----|---------|--------|-----------------|
| 1 | 5 | 5 | - | - | - |
| 6 | 6 | 1 | 0 | 5 | 83% |
| 11 | 7 | 1 | 0 | 6 | 86% |
| 14 | 7 | 0 | 0 | 7 | 100% |
| 19 | 7 | 0 | 0 | 7 | 100% |

## Interview Transcript
<details>
<summary>Full Q&A (19 rounds)</summary>

### Round 0
**Q:** Topology confirmation: Existing Alpha/Strategy Reassessment, New Alpha Discovery, Strategy/Portfolio Improvement, Validation/Promotion Gates로 읽어도 되는가?
**A:** 맞다 — 이 4개 컴포넌트로 진행.
**Ambiguity:** not scored.

### Round 1
**Q:** 이번 개선 작업의 1차 성공 기준은 무엇으로 고정할까요?
**A:** Return-first, MDD preferably ≤30%, no liquidation-like risk, no future-reference/post-hoc rules, theoretically defensible.
**Ambiguity:** 43.75%.

### Round 2
**Q:** 신규 알파 탐색의 허용 범위는 어디까지입니까?
**A:** Crypto + TradFi까지 허용; 기존 데이터가 충분한 경우 equity/commodity/FX-linked도 포함.
**Ambiguity:** 36.45%.

### Round 3
**Q:** 성과 개선으로 인정할 benchmark 기준은 무엇입니까?
**A:** Shadow는 risk_trim 64.42% comp 또는 return/MDD 3.49 초과, clean/paper는 clean baseline 34.39% comp 초과.
**Ambiguity:** 32.70%.

### Round 4
**Q:** Crypto + TradFi 신규 후보의 데이터 충분성 기준은 무엇입니까?
**A:** Crypto strict; TradFi aggressive shadow/research if theoretically valid; search as many feasible candidates as possible.
**Ambiguity:** 33.95%.

### Round 5
**Q:** 최대 탐색을 이번 실행의 budget으로 어떻게 고정합니까?
**A:** 1~2h broad smoke, survivors only to full WF/long run.
**Ambiguity:** 31.45%.

### Round 6
**Q:** 최종 산출물/승격 목표는 어디까지입니까?
**A:** Paper/shadow implementation plus live preflight checklist and telemetry gate; actual deployment needs separate approval.
**Ambiguity:** 29.70%.

### Round 7
**Q:** 승격/탈락 gate matrix는 무엇입니까?
**A:** Strict matrix: smoke validation/theory/cost feasibility; full WF tier benchmark + MDD≤30% + no liquidation + data sufficiency + cost/slippage; weak-data TradFi shadow-only.
**Ambiguity:** 27.70%.

### Round 8
**Q:** 최종 구현 surface는 무엇입니까?
**A:** Hybrid: new leaf alpha as Strategy class; final composition/risk through artifact portfolio manifest.
**Ambiguity:** 25.95%.

### Round 9
**Q:** 기존 알파/전략 재평가 범위는 무엇입니까?
**A:** Broad smoke + selective full over runnable/registered strategies.
**Ambiguity:** 23.45%.

### Round 10
**Q:** 신규 알파 탐색 runner lane은 무엇입니까?
**A:** Extend/wrap `run_alpha_zoo_clean_new_alpha_discovery.py` for smoke/shortlist; survivors into monthly refit walk-forward.
**Ambiguity:** 22.25%.

### Round 11
**Q:** 최종 portfolio sizing/조합 방식은 무엇입니까?
**A:** Train/validation-only optimizer maximizing return/MDD or Calmar with gross cap, MDD≤30%, correlation penalty.
**Ambiguity:** 21.75%.

### Round 12
**Q:** 1차 smoke에 포함할 alpha family bucket은 무엇입니까?
**A:** All major buckets: crypto price/volume, cross-asset/residual/dispersion, funding/OI/taker-flow/BBO, existing-winner overlays, TradFi/equity/commodity/FX-linked.
**Ambiguity:** 18.50%.

### Round 13
**Q:** 기존 전략 broad smoke skip/demotion 기준은 무엇입니까?
**A:** Two-stage audit: lenient broad smoke, strict audit only for full WF promotion.
**Ambiguity:** 15.75%.

### Round 14
**Q:** live preflight와 paper/shadow telemetry gate 필수 항목은 무엇입니까?
**A:** Both execution realism and research hygiene.
**Ambiguity:** 15.45%.

### Round 15
**Q:** 기존 전략 재평가의 최종 산출물 형태는 무엇입니까?
**A:** JSON + Markdown with candidate-level metrics, audit flags, correlation matrix, survivor list, promotion list, rejection reasons.
**Ambiguity:** 15.00%.

### Round 16
**Q:** 최종 strategy/portfolio policy acceptance criteria는 무엇입니까?
**A:** Strict: two-tier benchmark, MDD≤30%, no liquidation, cost/slippage pass, correlation penalty, manifest replay, fail-closed tests.
**Ambiguity:** 12.50%.

### Round 17
**Q:** 아무 후보도 strict gate를 통과하지 못하면 어떻게 처리합니까?
**A:** No-promotion + best shadow watchlist; baseline remains.
**Ambiguity:** 10.50%.

### Round 18
**Q:** fresh-forward/shadow 관찰 기간은 무엇입니까?
**A:** Minimum two new monthly folds passing benchmark/risk/cost/telemetry.
**Ambiguity:** 8.50%.

### Round 19
**Q:** 이번 iteration의 non-goals는 무엇입니까?
**A:** No new external vendor/on-chain/news collection, no real-money execution, no locked-OOS tuning, no sub-minute live execution.
**Ambiguity:** 4.50%.

</details>
