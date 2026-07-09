# Alpha Pool Expansion v2 -- Data-PC Handoff Spec

> **Handoff scope.** This document is the data-PC measurement contract for the
> v2 alpha-hunt batch shipped on branch `feat/alpha-pool-expansion-v2`. This PC
> built code + tests + CI only -- **no backtests, no data collection, build-not-
> measure**. Every return/OOS/cost number is measured on the data-PC (the
> `LuminaQuant/` layout, `private-main` branch) under the two-tier gate below.
> Until that gate runs, the verdict is **do_not_promote / research_only /
> 0% real** for every lane. Source of truth for the design:
> `.omc/plans/alpha-pool-expansion-consensus.md` (untracked session artifact);
> adversarial inputs: `.omc/research/alpha-graveyard-ledger-20260709.md`,
> `.omc/research/external-literature-verdicts-20260709.md`. The falsification
> content those files carry is reproduced inline here so this hand-off is
> self-contained and survives in-repo.

## Honest executive banner (load-bearing -- required disclosure)

**A1-anchored expansion -- this cycle is a falsification protocol, NOT a promise
of edge.** Expected value **concentrates in A1**, the only leaf with a strong
external crypto-specific OOS prior (Jia, Simkins, Yan, Zhang & Zhao 2025,
*J. Banking & Finance*; long-short ~= +130 bps/week **GROSS**, cost-unverified).
**L1 / L2 / A2 / N4 are lower-prior orthogonal FALSIFICATION probes** whose value
this cycle is *measuring an axis* (each carries an explicit EXPECTED NULL), NOT
manufacturing edge: L1 rides a *decaying* factor; L2's diversification return
*shrinks* under BTC-correlation; A2 likely dies like its own ancestor (graveyard
#6); N4 is cost-fragile. Do NOT collapse to A1-only -- the gated `research_only`
probes are cheap and their falsification-yield is real. Every candidate -- CORE
included -- clears a **two-key admission gate**: (key 1) an author-time
deterministic divergence BUILD gate (already resolved in-lane, keep/drop), and
(key 2) the data-PC incremental orthogonal `factor_ic` ADMISSION gate below.
"CORE" means *high-prior*, NOT gate-exempt.

## What shipped (cite these commits)

Nine lanes, all `research_only`, none in the live map. `.venv/bin/python` (3.14).

| Lane | Commit | What landed |
| :-- | :-- | :-- |
| A1 (flagship) | `0f99e48` | `CrossSectionalNearHighAnchoringStrategy` sleeve + B2 hard build gate + breadth-fallback test |
| L1 | `bcae576` | `LowTurnoverTrendPersistenceStrategy` sleeve + MTTE orthogonality test + min-hold rescue test |
| L2 | `6e08e6f` | `RebalancingPremiumHarvestStrategy` sleeve + forecast-free property test |
| X1 | `49ed97d` | `vol_estimator` param on existing `VolManagedRiskOverlayStrategy` (config-gated OFF, byte-identical) |
| E1 | `1eeb1f7` | per-candidate `spa_pvalue` + `approx_pbo` emission in the WF runner (config-gated OFF) |
| MR2 | `f2cb076` | `RegimeAdaptiveDisagreementEnsembleStrategy` standalone variant (regime-adaptive gate) |
| MR1 | `42752f0` | turnover-penalty + hand-rolled Ledoit-Wolf shrinkage on the M2 allocator (offline, OFF) |
| N4 | `a99c568` | `StationarityGatedResidualReversionStrategy` sleeve (ADF isolation) |
| A2 | `7408f0e` | `SlowCrossSectionalLeadLagStrategy` sleeve (3-incumbent divergence + RPT design gate) |
| W3 integration | `b8edf44` | 6 sleeves registered `research_only` atomically; manifest snapshot re-pinned (sha `fa49ae95fa502b6a9f053a701d9ff623e28a111f7475d3a0f6e5d7973ea5dfaf`); hardcoded-params baseline regenerated; tier-guard green; none in `live_default` |

MR1 (offline allocator extension), X1 (edits an already-registered class), and E1
(engine emission path) carry **no new `@register`** -- they are config-gated OFF,
flag-OFF byte-identical (golden rtol 1e-8), and re-verified at W3.

## 1. Two-tier gate

### 1.1 Pool admission (relaxed -- verbatim from spec)

A candidate is admitted to the research pool iff **ALL THREE** hold:

1. **net-of-cost 20 bps edge > 0**, AND
2. **DSR > 0 after the `N_eff` penalty**, AND
3. **incremental orthogonal `factor_ic` > 0 vs the existing sleeves.**

Beating the default set is **NOT** required for admission. This is the relaxed
bar; it exists so orthogonal probes can enter the pool on their own axis without
first having to out-Sharpe the incumbents.

### 1.2 Live promotion (strict, unchanged -- (a)-(f) verbatim)

Promotion to any non-zero live allocation requires **ALL** of:

- **(a)** 20 bps net DSR-adjusted Sharpe / IR of the NEW combined book > current
  default set on **IDENTICAL** walk-forward windows and cost model.
- **(b)** No degradation at 30 bps.
- **(c)** DSR > 0 after `N_eff` via `evaluate_survivorship_gate` +
  `effective_number_of_trials`.
- **(d)** Per-leaf incremental orthogonal `factor_ic` > 0.
- **(e)** Coverage-gate pass on latest-OOS.
- **(f)** 0% real until explicit human sign-off.

## 2. NO-SURVIVORSHIP MANDATE (B6b -- binding)

**The data-PC report MUST report ALL authored candidates INCLUDING rejects.**
For every one of the lanes below, the report records: (i) the candidate's
**prior-of-death** (mechanism-class base rate), (ii) the **single falsifying
measurement** that was actually taken, and (iii) the **outcome vs its EXPECTED
NULL**. A candidate silently dropped from the report is a **survivorship-bias
defect** and fails the hand-off. "It didn't work so I left it out" is exactly the
failure this mandate exists to prevent -- a null result is a reported result.

## 3. Operational handoff

### 3.1 Candidate family + `strategy_class` (exact, all lanes)

Route measurement by these exact strings. `allow_multi_asset` is **mandatory** for
every cross-sectional sleeve (A1 / A2 / L2 / N4) so the diversified-shortlist
selector does not silently drop them -- and it is achieved by the honest
`family="cross_sectional"`, **NOT** by attaching a fake carry tag.

| Lane | `strategy_class` | `family` | `allow_multi_asset` | candidate builder |
| :-- | :-- | :-- | :-- | :-- |
| A1 | `CrossSectionalNearHighAnchoringStrategy` | `cross_sectional` | **True (mandatory)** | `_build_near_high_anchoring_candidates` |
| L1 | `LowTurnoverTrendPersistenceStrategy` | `time_series_momentum` | **False** (per-symbol single-asset; NO fake tag) | `_build_low_turnover_trend_persistence_candidates` |
| L2 | `RebalancingPremiumHarvestStrategy` | `cross_sectional` | **True (mandatory)** | `_build_rebalancing_premium_harvest_candidates` |
| A2 | `SlowCrossSectionalLeadLagStrategy` | `cross_sectional` | **True (mandatory)** | `_build_slow_leadlag_xs_candidates` |
| N4 | `StationarityGatedResidualReversionStrategy` | `cross_sectional` | **True (mandatory)** | `_build_stationarity_gated_residual_reversion_candidates` |
| MR2 | `RegimeAdaptiveDisagreementEnsembleStrategy` | `trend` | N/A (per-symbol single-asset ensemble) | `_build_regime_adaptive_disagreement_candidates` |

L1 is per-symbol single-asset (`time_series_momentum`) and **must NOT** be given
`allow_multi_asset` -- that would be a fake-family tag. MR1 (offline allocator),
X1 (`vol_estimator` param on `VolManagedRiskOverlayStrategy`), and E1 (WF-runner
emission) have no candidate rows / no `strategy_class`.

### 3.2 Clean walk-forward invocation

- **Selection uses train + validation ONLY.** Locked-OOS is report-only, monthly WF.
- **`no_nested_oos_mining=true`.** No fold's OOS may inform any other fold's
  selection, tuning, tie-break, or weighting.
- Route **per-fold admit AND final merge** through
  `selection.apply_selection_reject_and_dedup` with **strict**
  `robust_score_params` (`enforce_selection_reject_gate=True`,
  `dsr_gate_floor=0.90`, `spa_gate_ceiling=0.05`, `pbo_gate_ceiling=0.50`) -- see
  the E1 worked example in
  [`overfit_selection_gate_integration.md`](overfit_selection_gate_integration.md)
  (data-PC owns whole-search `num_trials` / DSR).

### 3.3 Cost grid -- `backtest_cost_realistic.yaml` MANDATORY for admission

- Run the **10 / 15 / 20 / 30 bps** round-trip cost grid. Admission is decided at
  **net 20 bps**; promotion (b) checks **no degradation at 30 bps**.
- Admission MUST use `configs/profiles/backtest_cost_realistic.yaml`
  (sqrt-impact + charged funding). The research scorer's linear / no-funding cost
  is **optimistic** and is NOT sufficient for admission. Method:
  [`../COST_REALISM_REMEASUREMENT.md`](../COST_REALISM_REMEASUREMENT.md).
- **RPT-per-split >= 10 bps is mandatory for L1 and A2** -- this is the exact gate
  turnover-death entry #14 (debounced momentum hysteresis, RPT 9.36-9.48 bps)
  failed. Any split below 10 bps RPT rejects the leaf.

### 3.4 Per-leaf measurement notes

- **A1 horizon `factor_ic` sweep: 10 / 20 / 30 / 52 wk vs `NearHighMomentum`.**
  This is the cheapest experiment isolating the published effect. Strict 52 wk
  drops young alts toward a mega-cap-only cross-section where anchoring is
  weakest; the sweep resolves anointing-vs-breadth empirically. The sleeve uses
  `min(52wk, max_available)` with a per-symbol min-history gate, so young alts are
  ADMITTED via the fallback (not dropped) -- proven by the author-time
  breadth-fallback test.
- **MR1 -- M2 manifest-provenance checklist.** The turnover-penalized /
  Ledoit-Wolf-shrunk allocation must round-trip through the ACTUAL
  `ArtifactPortfolioModeStrategy` consumer WITHOUT fail-closing, with:
  all `real_money` keys false (top-level + child), oos clean, optimizer /
  correlation provenance (source + selection_inputs, ready=True), source-artifact
  id / path / sha / freshness, and per-child `no_current_fold_oos` +
  train/validation provenance. Flags-OFF the manifest is byte-identical to today.
- **`emit_candidate_overfit_stats=ON`** via `configs/profiles/research.yaml`
  (`research.emit_candidate_overfit_stats: true`) so the WF runner stamps
  `spa_pvalue` + `approx_pbo` into each candidate's `validation` block. Default
  OFF is byte-identical; the honest-research profile arms it. Details in the E1
  worked example.

### 3.5 Goal-completion recommendation

- **G005 completion** -- the 1h shard is unfinished (`g005_*_1h_chunk_01..11`).
- **G006 cost-stress** -- never ran under sqrt-impact + funding at the 10/15/20/30
  grid.
- **G007 decision** -- last verdict `do_not_promote / blocked_fail_closed`.

Run G005 -> G006 -> G007 to close the loop before any promotion decision on this
batch.

## 4. Per-candidate falsification table (NO-SURVIVORSHIP -- report every row)

Each row is a contract: measure the **single falsifying measurement**, then report
the outcome **vs the EXPECTED NULL**. The EXPECTED NULL for the leaves is
`reject` -- honestly, we expect most of these to die. Reporting the death is the
deliverable.

### Leaves

**A1 -- `CrossSectionalNearHighAnchoringStrategy` (flagship)**
- **Prior-of-death:** anchoring / XS-momentum variants historically collapse
  val -> OOS unless externally pre-registered. Nearest cautionary graveyard entry:
  #10 Alpha101 single-asset (single-asset destroys XS content) -- dodged (XS
  long-short, never single-asset).
- **Single falsifying measurement:** net-20 bps XS long-short Sharpe <= 0 OR
  `factor_ic` <= 0 vs `NearHighMomentum` on the 10/20/30/52 wk horizon sweep.
- **EXPECTED NULL:** the published +130 bps/wk GROSS effect does not survive
  weekly-turnover cost on the tradeable-majors cross-section -> reject. (This is
  the flagship; if it survives, it is the one leaf worth promoting.)

**L1 -- `LowTurnoverTrendPersistenceStrategy`**
- **Prior-of-death:** HTF momentum sweeps promoted 0/many and died at RPT < 10 bps
  (graveyard #13 HTF sweeps, #14 debounced hysteresis). L1 IS the rescued version
  (min-hold -> 36 analog lifts RPT clear of 10 bps, per the repo's proven rescue).
- **Single falsifying measurement:** RPT < 10 bps on ANY split OR `factor_ic` <= 0
  vs `MultiTimeframeTrendEnsemble`.
- **EXPECTED NULL:** a decaying factor at low turnover clears cost but does not
  beat the incumbent trend ensemble -> reject.

**L2 -- `RebalancingPremiumHarvestStrategy`**
- **Prior-of-death:** mechanical harvests underperform in trending single-asset
  regimes. No dead relative (cleanest); #15 all-paper unscaled is a different
  failure (correlated-forecast over-search) -- N/A to a forecast-free rule.
- **Single falsifying measurement:** net-20 bps excess growth <= buy-and-hold
  basket.
- **EXPECTED NULL:** BTC-dominated correlation collapses the diversification
  return below cost -> reject. (Growth-rate / variance benefit, not high Sharpe --
  a modest expectation stated ex-ante.)

**A2 -- `SlowCrossSectionalLeadLagStrategy` (conditional, first-to-cut)**
- **Prior-of-death:** the repo's own lead-lag died at 20-30 bps (graveyard #6
  cross_asset_lead_lag_momentum, clean OOS Sharpe 0.24) and leadership-unwind found
  0 all-gate survivors (#2). This is the highest-prior-of-death leaf.
- **Single falsifying measurement:** does not clear 20 bps where #6 died OR
  `factor_ic` <= 0 vs ALL THREE incumbents (`CrossCryptoSlowDiffusion`,
  `SemisLeadLagRotation`, `IntermarketLeadLagContinuation`).
- **EXPECTED NULL (high):** A2 dies like its ancestor -> reject. Do NOT resurrect
  #6.

**N4 -- `StationarityGatedResidualReversionStrategy` (conditional secondary)**
- **Prior-of-death:** pairspread low-turnover rescue hard_reject 4/4 (#9);
  reversal on liquid majors is wrong-signed (verdict 9 -- largest/liquid coins show
  daily MOMENTUM, not reversal). Dodge: beta-HEDGED idiosyncratic residual (not
  price-level reversal) + ADF gate; does NOT re-attempt the dead pairspread rescue.
- **Single falsifying measurement:** net-20 bps <= 0 OR `factor_ic` <= 0 vs
  `EquityBenchmarkResidualReversal`.
- **EXPECTED NULL:** residual reversion churns away its edge at cost -> reject.

### Meta / overlay / engine lanes

**MR1 -- turnover-aware M2 allocator (offline, config-gated OFF)**
- **Prior-of-death:** the M2 spine is a *promising-unexplored* entry (ledger B1) --
  implemented, synthetic tests only, NEVER backtested, 0% allocation. Attacks
  cost-death at the allocation layer.
- **Single falsifying measurement:** the turnover-penalized (`net_sharpe_20bps -
  lambda*turnover`) + hand-rolled Ledoit-Wolf allocation's OOS net-20 bps
  DSR-adjusted book <= the flags-OFF (byte-identical) ERC/HRP manifest on identical
  windows.
- **EXPECTED NULL:** encoding the RPT graveyard lesson into allocation does not
  improve the OOS-weighted book -> keep flags OFF.

**MR2 -- `RegimeAdaptiveDisagreementEnsembleStrategy` (variance-reduction thesis)**
- **Prior-of-death:** M1 is a *promising-unexplored* entry (ledger B1). Ensemble
  over-claim is the trap: replication-ratio literature (arXiv 2501.03938 /
  2512.12735; McLean-Pontiff 2016) warns the "lifts net Sharpe above the best
  single sleeve" claim is only conditionally supported. The thesis is therefore
  **variance-reduction / drawdown-smoothing (Calmar / MDD), NOT Sharpe-lift.**
- **Single falsifying measurement:** MDD / Calmar of the combined book WITH MR2 >=
  base M1 fixed-gate (i.e. no drawdown improvement). (Its author-time divergence
  test already passed in-lane; a divergence-fail would have DROPPED it.)
- **EXPECTED NULL:** the regime-adaptive gate yields no MDD/Calmar improvement over
  base M1 -> do not promote the variant.

**X1 -- `vol_estimator` param on `VolManagedRiskOverlayStrategy` (sizing, OFF)**
- **Prior-of-death:** range-vol-as-directional-alpha was killed (verdict 3: OHLCV
  cannot proxy VRP) and repurposed to its literature-endorsed SIZING role; the
  train/val-only risk-throttle is a *promising-unexplored* entry (ledger B9,
  never built).
- **Single falsifying measurement:** the overlay under `parkinson` / `garman_klass`
  / `yang_zhang` shows no OOS drawdown / risk-adjusted improvement over the
  `close_to_close` default (which is byte-identical to today).
- **EXPECTED NULL:** range-vol sizing does not beat close-to-close sizing -> keep
  the default estimator. Any calibration is train/val-only (no OOS peeking).

**E1 -- `spa_pvalue` + `approx_pbo` emission (engine path, OFF)**
- **Prior-of-death:** this is not a return candidate -- it attacks failure-cause
  (iv), post-OOS-informed / undeflated headlines (graveyard #12 lagged-leaf-router,
  PF ~30 cost-optimism tell). Config-gated OFF, flag-OFF byte-identical.
- **Falsifying (inverse) check:** if the emitted `spa_pvalue` / `approx_pbo` plus
  whole-search-deflated DSR do NOT reject the known overfit cohort (the recorded
  22-candidate cluster, 14 of which are one 9-symbol basket wearing different
  `strategy_class` hats), the emission is mis-wired. The gate already rejects
  **22 / 22** data-free (`tests/test_overfit_selection_reject_gate.py`).
- **EXPECTED behavior:** the whole-search-deflated gate rejects overfit clones and
  admits only a genuine edge (DSR ~= 0.978 achievable at num_trials ~= 1400) -- see
  the E1 worked example.

## 5. Dropped by the adversarial filter (recorded -- do NOT re-mine)

Not survivorship: these were killed by EVIDENCE, and the reason is on record so a
future cycle does not re-mine a dead axis.

- **Range-VRP directional leaf** -- verdict 3: OHLCV cannot proxy VRP -> repurposed
  to sizing overlay X1.
- **Amihud illiquidity-premium leaf (level AND change-form)** -- verdict 5: the
  premium lives in the illiquid tail where sqrt-impact / delisting kill it;
  un-harvestable net at size. The change-form (illiquidity innovation) was ALSO
  dropped -- it still tilts the illiquid tail, so it inherits the same cost trap.
- **Calendar / seasonality leaf** -- verdict 2 + graveyard #1: data-mining,
  sign-flips.
- **Breadth / dispersion standalone leaf** -- verdict 7: equities-only ->
  conditioning / gate only.
- **Naive reversal on majors** -- verdict 9: wrong-signed -> only N4's beta-hedged
  residual form survives, conditionally.
- **Crowded-leadership-unwind v2 / microstructure-squeeze** -- variant of dead #2 /
  coverage ~5%. No literature-killed padding.

## 6. Follow-ups (data-PC owned)

- A2 / N4 promotion only on measured `factor_ic` + 20 bps survival; A1 horizon
  sweep result feeds the final lookback.
- Production `selection.py` allowlist for XS sleeves is EARNED on the data-PC, not
  pre-granted here.
- Port the E1 worked example into the data-PC canonical selection / merge stamp.
- Funding-charged deflated WF before any sign-off. Default real-money allocation
  remains **0%**.

See also:
[`overfit_selection_gate_integration.md`](overfit_selection_gate_integration.md) *
[`../COST_REALISM_REMEASUREMENT.md`](../COST_REALISM_REMEASUREMENT.md) *
[`research_note.md`](research_note.md) *
[`../CONFIG_SPEC.md`](../CONFIG_SPEC.md) *
[`../METRICS.md`](../METRICS.md)
