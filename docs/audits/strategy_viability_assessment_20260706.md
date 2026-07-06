# Strategy Viability Assessment — 2026-07-06

A three-axis, adversarially-verified assessment of whether LuminaQuant's strategies
have a viable path to real net profitability. Each axis was a fan-out with per-finding
verification:

1. **Internal performance** — what the actual backtest/walk-forward/G005 artifacts show (5 lanes).
2. **External reality** — what the crypto-perp systematic literature + market data say is achievable (4 web lanes + fact-check).
3. **Theoretical soundness** — whether the signal/simulation/validation/portfolio math is correct (5 code-review lanes + verify).

The three converge tightly, which is itself the most important result: they tell one
consistent story from three independent directions.

---

## BOTTOM LINE

**No robust, cost-realistic, out-of-sample edge is demonstrated today — but this is
over-selection, not a broken idea, and a narrow path exists.** The honest signal that
survives is real but tiny (cross-asset lead-lag momentum, ~Sharpe 0.24, which flips
negative at a 20 bps round-trip). Every 100–200 % headline is a methodology artifact
(flat costs, zero funding, pre-fidelity-fix, overfit, outlier-month-concentrated). The
research *engine* is sound; the validation *wiring* is not, so the promotion pipeline
has been over-admitting noise and calling it "winners." External reality caps a
disciplined crypto-perp operation at ~0.3–0.8 net Sharpe — an order of magnitude below
the headlines but not zero. The way forward is to fix four validation-wiring defects,
run the honest measurements that were designed but never executed, and concentrate on
the two alpha families with real external support, on a small liquid book, shadow-only,
0 % real allocation until a funding-charged deflated walk-forward clears a real bar.

---

## AXIS 1 — INTERNAL PERFORMANCE (what the artifacts actually show)

Every lane, independently, found no defensible edge.

- **The curated "winner" portfolio is a corpse.** Fit Sharpe 0.79 / +5.47 % collapses to
  **OOS Sharpe −0.42 / −15.22 %** (`portfolio_superiority_curated/portfolio_optimization_latest.json`).
  Its top candidate: OOS deflated-Sharpe **0.0039**, PBO **0.50**, SPA p **1.0**, hurdle
  fail on all three splits. All five shortlist names carry sentinel penalty scores
  (−500 k to −1.5 M) — they were *retained by curation, not by passing the gate*.
- **The most rigorous search found nothing.** G005 (2026-07-03, post-audit, DSR + PBO +
  walk-forward, ~964 candidates fully evaluated from a 24,794-effective-trial budget):
  **max deflated Sharpe 0.026, PBO = 1.0 on 100 % of 1d candidates.** Raw Sharpes up to
  6.64 are pure selection artifact. Zero cleared even a lenient DSR > 0.5. *This is the
  pipeline working correctly when the correction is actually binding.*
- **The one honest cost grid ever run** (a discovered leaf, clean): realistic PF 1.166,
  **+1.71 %/yr @10 bps → +0.46 % @15 bps → −0.78 % @20 bps → −3.21 % @30 bps.** The edge
  flips negative at a 20 bps round-trip.
- **The only repeatedly-positive clean signal:** cross-asset lead-lag momentum, +2.51 %
  OOS comp, **Sharpe 0.24**, dies at 20–30 bps (`docs/research_note/research_note.md`).
- **The flashy 159 %/214 %/197 % headlines** are grid-search meta-selectors (not registered
  strategies), all `ready_for_real=false`, never deflated, never funding-charged, with
  gains concentrated in ~2 outlier months (one June fold alone +63 %) and profit-factors
  of 23–30 (the repo's own documented cost-optimism tell vs realistic PF ~1.2).
- **The 114-strategy registry is breadth-by-parametrization**, not 114 edges: 36 curated
  classes, 78 glob-discovered sleeves, one class = 14 % of all candidates; 68/114 sit in
  "live_default" tier by omission, not validation. `alpha_scoreboard.py` has never been
  run to persist a single audited ranking.

**Every performance artifact except G005 predates the 2026-07-03/06 fidelity fixes and
uses the flat/zero-funding cost profile — so even the positive numbers are upward-biased.
The one genuinely post-fix search is precisely the one that found nothing.**

### What has already been tried (the research trail is disciplined, not naïve)
Calendar-teacher + FRED filter (dead) → Optuna integer-leverage 69-asset hybrids (caught
a leakage bug, +296 % discarded) → profit_moonshot 1,050 candidates + 1.3 M-combo
sub-searches (best OOS Sharpe 0.10) → monthly-refit clean walk-forward (core engine) →
pre-registered clean alphas (the honest +2.5 %/0.24) → 30+ return-rider sleeves (code +
synthetic tests only, never backtested) → meta-ensemble spine (theory + synthetic tests,
0 % allocation) → G005 full-pool (incomplete: 1h shard unfinished, G006 cost-stress +
G007 decision never ran). **The team's own discipline repeatedly caught and killed its
own false positives** — the negative knowledge is high quality.

**Unexhausted, never executed:** (a) the cost-realistic remeasurement (funding + sqrt-impact,
post-fix) — the guide exists, no artifact does; (b) multi-month fresh-forward on a frozen
rule (only ~1 week exists); (c) backtesting the spine + rider sleeves; (d) finishing G005
+ G006/G007; (e) historical funding/OI/taker/BBO/depth backfill (feature families are
coverage-gated at ~5 %).

---

## AXIS 2 — EXTERNAL REALITY (is this domain even winnable?)

Web evidence, adversarially fact-checked, says: **yes but narrow, low-turnover, cost-governed.**

- **High-turnover crypto-perp alpha is a costs mirage (HARD).** A peer-reviewed 2026 study
  (Frontiers in Blockchain) shows 5-min signals with genuinely positive *gross* Sharpe
  (0.43–0.96) flip to **net Sharpe −10.7 to −18.4 on Binance USDT-M** at 124–203× daily
  turnover — and that excludes funding, so it's an upper bound for retail. *Any edge that
  lives at short horizons is net-negative.*
- **Realistic disciplined live net Sharpe is ~0.3–0.8, not 2–4 (HARD).** The SG CTA Index
  (net-of-fee, investable managed-futures benchmark) has run ~0.6 live since 2000. That's
  what billions in AUM + cheap execution + decades of research actually deliver.
- **Only two alpha families have hard, peer-reviewed edge**, and both are compressing:
  **funding-rate carry** (BIS: historically >10 %/yr, but ~19 % in 2025 and only ~40 % of
  top basis spreads net-positive after costs) and **trend/momentum** (real gross alpha but
  flips +69 % IS → −2 % OOS and dies at ~125 bps). For both, the binding constraint is net
  execution, not signal discovery.
- **Capacity ceiling is the alt long-tail**, ~low-single-digit-million USD per rebalance if
  weight sits in mid/low-caps; BTC/ETH absorb $1 M+ at sub-bp cost. The shipped ~100-symbol
  universe is too wide for real size.
- **DSR + N_eff + walk-forward is above-average but insufficient**: a rigorous shop adds PBO
  via CPCV (not a single WF path), an honest trial-count budget (undercounting N re-inflates
  DSR), and a t≈3 (not t≈2) discovery hurdle assuming most candidates are false.

**External reality corroborates internal:** the honest lead-lag signal (0.24) sits right at
the bottom of the plausible band, and the "no robust edge" result is exactly what the base
rates predict for a wide, cost-optimistic search.

---

## AXIS 3 — THEORETICAL SOUNDNESS (is the method correct?)

**Verdict: mostly-sound at the engine, flawed at the promotion gates.** The estimators are
individually correct; the way they are *wired into accept/reject decisions* is not.

### Sound (this calibrates trust in the numbers)
- **Point-in-time discipline is clean** — every `compute/ops.py` primitive uses only trailing
  data; cross-sectional IC is strictly per-timestamp; forward-return labels are causal;
  `rolling_zscore` excludes the scored bar; walk-forward lags signals one bar and tiles
  test windows without overlap. *The thing most systems get wrong is correct here.*
- **The simulator is conservative, not optimistic** — next-bar fill only, correct funding
  sign, always-adverse slippage with a 2× volatile-bar multiplier, liquidation triggers
  marginally early, forced-liquidation fills at the adverse intrabar extreme. (Two
  "optimistic" claims — mark-vs-last and liquidation slippage — were checked and refuted as
  actually conservative.)
- **The statistical primitives are individually correct** — `expected_max_sharpe` matches
  Bailey/López de Prado, the DSR non-normality denominator is the correct Mertens/Lo form,
  Ledoit-Wolf shrinkage + ERC + simplex projection are textbook, the offline ERC/HRP
  allocator is train/val-fitted and OOS-fail-closed.
- **The live/default book rests on recognized risk premia** — TS/regime momentum, order-flow
  exhaustion reversal, crowding-fade are economically coherent and causally computed. *This
  is a different, less-contaminated population than the mining funnel.*

### Flawed (why the "winners" are contaminated) — ranked
1. **[CRITICAL] The binding pass/reject gate has no multiple-testing correction.** The gate
   is raw `oos.sharpe ≥ 0.35 AND oos.pbo ≤ 0.45`. The deflated Sharpe *is* computed with the
   correct multiplicity count but enters only as a soft score weight (1.2–1.4×), never a pass
   condition; the DSR/SPA hard gate defaults OFF (`selection.py:38`). Under the null a pure-noise
   candidate clears the SR ≥ 0.35 gate with P ≈ 0.40 on a 0.5-yr OOS. *The one correct
   multiplicity defense the codebase owns is deliberately unplugged from the decision.*
2. **[CRITICAL] Selection on the same OOS window that is reported as OOS (no lockbox).** Splits
   are train/val/oos only, with ranking and the binding hurdle both on `oos.*`. Ranking N
   candidates by OOS and reporting the survivor's OOS makes the headline ≈ E[max of N] ≈
   +3.3 OOS-SE of pure selection inflation at N≈200 — contradicting the framework's own
   `locked_oos_report_only` discipline that the alpha_zoo path already honors.
3. **[MAJOR] N_eff double-discounts, making the one binding DSR gate (alpha_search) too lenient.**
   The cross-trial variance already equals ~(1−ρ)V, and then N_eff = (trace C)²/ΣC² collapses
   again — applying the correlation discount twice, so the DSR benchmark is 2.6–10× too small
   at ρ=0.5. Correlated overfit candidates clear it.
4. **[MAJOR] DSR/Sharpe assume iid; no Lo(2002)/HAC autocorrelation correction.** For positively
   autocorrelated perp P&L the true SE is 1.2–1.7× larger at ρ=0.2–0.5, so raw n overstates the
   DSR z behind the gate.
5. **[MAJOR] Rank-IC t-stat treats overlapping horizon-4 ICs as iid** → FP(|t|>1.96) = 0.28 at
   h=4, 0.40 at h=8 (MC-reproduced) vs nominal 0.05, so the factor-discovery gate over-admits at
   ~6–9× the intended rate.
6. **[MAJOR] Portfolio "superiority" argmaxes across K challengers on locked OOS, no correction**,
   and its promotion score mixes scale-dependent total_return with scale-free Sharpe → **rewards
   leverage, not risk-adjusted edge** (promotes the most-levered candidate that got lucky on OOS).
7. **[MAJOR] Funding charged on an entry-anchored 8h clock, not Binance 00/08/16 UTC snapshots**
   → any sub-8h round trip pays *zero* funding; ~1–5 %/yr return inflation for high-turnover
   long-biased books. (Orthogonal to the already-fixed funding-rate work.)

Minor: `approx_pbo` is a fold-instability heuristic, not CSCV PBO, yet is a binding gate; no
purge/embargo between splits (negligible for 1-bar labels, material for multi-bar); several
research sleeves are economically mislabeled (funding-"carry" is really positioning-momentum;
single-asset Alpha101 replaces cross-sectional rank with a time-series self-rank, destroying the
alphas' economic content) — narrative inflation, not estimator bugs.

### Net effect on "is there an edge?"
The reported edges of anything that came *through the research/promotion funnel* are **likely
overstated, and a meaningful fraction of promoted "winners" are false discoveries** — a three-layer
cascade (factor discovery → strategy promotion → portfolio superiority) where each layer
independently over-admits noise and the primary multiplicity defense (DSR) is unplugged from every
binding decision. **But this is over-selection, not fabrication:** look-ahead discipline is clean,
the simulator is conservative, and the live book rests on real risk premia — so a *smaller real
edge probably survives*, concentrated in the economically-grounded live sleeves, not the
top-of-leaderboard mined candidates. **Trust the execution-level numbers (modulo funding timing);
do not trust the pipeline's "beat OOS"/"superior" claims without re-running the gates corrected.**

---

## THE PATH FORWARD ("방도")

There is a path. It is not "pick these winners" (the curated portfolio is a −15 % OOS corpse).
It is, in priority order:

**Fix the validation wiring first (these decide whether any future verdict is trustworthy — all
are wiring/parameter changes over correctly-implemented primitives, not rebuilds):**
1. Make the deflated Sharpe a **hard reject** in the strategy gate (flip `selection.py:38
   dsr_spa_hard_gate` on; it's already computed with the right trial count).
2. Fix the **N_eff double-discount** so the one binding DSR gate is actually stringent.
3. Add a **fourth never-touched lockbox split scored exactly once**; rank/gate on validation,
   report lockbox as the sole OOS (extend the existing `locked_oos_report_only` discipline to the
   strategy_factory and portfolio-superiority paths).
4. **HAC/block-correct every iid inference** (rank-IC SE and the n behind DSR/Sharpe).
5. Make the portfolio objective **exposure-normalized** so leverage can't buy "superiority."
6. Charge **funding on crossed 00/08/16 UTC boundaries**.

**Then run the honest measurements that were designed but never executed:**
7. Run `backtest_cost_realistic.yaml` (funding-charged + sqrt-impact, post-fix) on the least-bad
   survivors (cross-asset lead-lag momentum, the G005 4h soft-passes). *This single datapoint does
   not exist anywhere today.*
8. Finish G005 → G006 (cost-stress) → G007 (decision).
9. Persist one `alpha_scoreboard.py` gated leaderboard over a cost-realistic WF; collapse the
   parametrization fan-out to families that survive DSR **and** OOS; kill the tail.

**Concentrate where external evidence says edge exists:** low-turnover **trend/momentum** and
**funding-rate carry**, on a **small liquid book** (BTC/ETH/majors — where sqrt-impact, feed-death,
and delisting risk are bounded), not the 100-symbol alt universe.

**Discipline / expectations:**
- Never report a raw Sharpe without its DSR and PBO on the same line.
- Require 1–2 months of frozen-rule fresh-forward (best today is ~1 week).
- **Realistic target if a leaf eventually clears the corrected gates: low-single-digit to low-teens
  annualized *net* Sharpe ~0.3–0.8** — an order of magnitude below the headlines, and not
  demonstrated today.
- **Real-money is separately No-Go** until the live-safety fixes (readiness audit, PR #39) are
  turned on and validated; the most that is ever defensible on the best honest candidate is
  **shadow/paper, 0 % real allocation**, until a funding-charged deflated walk-forward Sharpe
  meaningfully above 0 survives a 20 bps round-trip plus multi-month fresh-forward.

**One-sentence answer:** the searches have produced high-quality *negative* knowledge rather than an
edge; the method's engine is sound but its selection gates lie by omission; fix the four wiring
defects, run the four never-executed honest measurements, and if a real (small) edge exists it will
show up in low-turnover momentum/carry on a liquid book — otherwise the honest conclusion is that
there is no edge here worth real capital.
