# Equity-Flow Analysis of Promising Candidates — 2026-07-10

Analyst: worker-eq (equity-flow lane). Scope: study the **cumulative-return FLOW**
(fold-by-fold equity path), not just endpoint metrics, of the promising
`research_only` + historical-incumbent candidates, then design flow-driven
complements. Read-only; no source edits.

## 0. Data provenance & method

Primary artifact (richest, per-fold reconstructable):
`var/reports/strategy_performance_improvement_20260707/full_universe_walkforward/full_universe_walkforward_summary_latest.json`
(2026-07-08 run; 110 symbols; 11 monthly folds `2025-09`→`2026-07`; 2M validation /
1M locked-OOS; selection uses train+validation only, locked-OOS is report-only).

**Per-fold equity flow IS reconstructable.** Each of the 1733 `fold_candidate_rows`
carries `locked_oos.total_return` (+ `mdd`, `sharpe`, `validation.total_return`,
`final_weights`) keyed by `fold_id`. 149 candidates have all 11 non-null folds. I
rebuilt each candidate's compounded equity curve `Π(1+r_fold)` and characterized
where money is made/lost, drawdown depth/duration, and val→OOS divergence shape.

Supporting artifacts cross-checked:
- `.../full_universe_walkforward_summary_latest.md` (best-candidate monthly detail table — matches my reconstruction exactly).
- `var/reports/best_historical_strategy_improvement/best_historical_strategy_improvement_latest.md` (historical-85 vs expanded-110 fold returns for the lead-lag champion; the risk-trimmed `fallback_mdd20_cap2` variant).
- `var/reports/portfolio_superiority_dense_pairs/{promotable_candidate,dense_pairs_benchmark_comparison}_latest.md` (state_vwap pair carry sleeve, cash-heavy, Sharpe ~3.5 / MaxDD ~0.2%).
- `docs/research_note/research_note.md` (lead-lag survivor provenance; mdd-cap rationale line 365; universe survivorship caveat line 505).

Regime tag per fold = the fold's own best-OOS winner (`fold_summaries[*].best_oos_candidate/return/mdd`), used as a proxy for the market state (no BTC benchmark series is embedded in the artifact).

---

## 1. Regime map (what each fold "was")

| Fold | OOS window | Fold winner (regime signature) | Winner ret / MDD | Regime read |
|---|---|---|---:|---|
| 2025-09 | Sep | mdd30_risk_scaled | +3.7% / 15.7% | mild up, choppy |
| 2025-10 | Oct | asset_tf_leverage | +5.3% / 11.3% | mild up |
| **2025-11** | Nov | regime_opportunity_leaf_switch | **+33.5% / 27.7%** | **big rip + high vol (universal alpha fold)** |
| 2025-12 | Dec | *cash wins* | 0.0% / 0.0% | down/chop — flat is best |
| 2026-01 | Jan | tradfi_vol_managed | +30.9% / 17.6% | big up |
| 2026-02 | Feb | profile_optuna | +20.8% / 18.7% | big up |
| 2026-03 | Mar | cross_candidate_hybrid | +1.7% / 2.0% | flat/chop |
| 2026-04 | Apr | *cash wins* | 0.0% / 0.0% | down/chop — flat is best |
| 2026-05 | May | cross_candidate_hybrid | +8.6% / 5.6% | mild up |
| **2026-06** | Jun | profile_optuna aggressive | **+19.4% / 26.8%** | big up + high vol |
| **2026-07** | Jul 1–4 (4-day PARTIAL) | profile_optuna aggressive | +1.0% / 3.8% | **whipsaw partial; deployed momentum crowd all lose** |

Two folds define everything: **2025-11 (the universal win)** and **2026-07 (the
universal loss)**. 2025-12 and 2026-04 reward *being in cash*.

---

## 2. Top candidates — endpoint ranking (clean-eligible + historical incumbents)

From `clean_promotion_rankings` (131 clean-eligible) and the demoted/aggregate
lead-lag incumbents. The 4 threshold variants (t0.85/0.90/0.95/1.00) of each
dynamic_conviction row are byte-identical (threshold never binds) — collapsed here.

| # | Candidate (family) | Status | Comp OOS | Sharpe | MaxOOS-MDD | Pos folds | Deployed folds |
|---|---|---|---:|---:|---:|---:|---:|
| A | `dynamic_conviction_switch:…calmar80_gate_val_mdd20_scaled` | clean | **+7.99%** | 0.38 | 23.6% | 3/11 | 4/11 |
| A′ | `…calmar80_gate_val_mdd15_scaled` | clean | +6.97% | 0.37 | 19.3% | 3/11 | 4/11 |
| B | `codex_lagged_leaf_router_grid:…lagged_plus_val025_exact_unscaled` | research_only* | **+63.4%** | 1.50 | 27.7% | 4/11 | 8/11 |
| B′ | `…lagged_plus_val025_fallback_mdd20_cap2` | research_only* | +51.2% | 1.52 | 25.0% | 4/11 | 8/11 |
| C | `lagged_shadow_leaf_router:core_warmup4_avg2_val05_mdd12` | research_only* | +46.3% | 1.31 | 27.7% | 3/11 | 6/11 |
| D | `strict_efficiency:static_guarded` | clean | +0.55% | 0.12 | 11.8% | 3/11 | 11/11 |
| E | `tradfi_intraday_session_v1:open_impulse_close_top10_mdd15` | clean | +0.48% | 0.79 | 0.36% | 2/11 | 3/11 |
| F | `production_guarded_40_state_vwap_pair_25_cash_35` (dense_pairs) | promotable challenger | +0.46% (OOS split) | 3.46 | 0.22% | — | cash-heavy |

\* research_only = flagged `post_oos_research_variant, requires_fresh_forward_shadow`
(not clean-promotable; `ready_for_real=false`). These are the historical incumbents
the mission calls out — the highest raw returns in the whole board.

---

## 3. Fold-by-fold equity FLOW (the diagnosis)

### Candidate A — dynamic_conviction_switch (best CLEAN, +7.99%)

```
fold     OOS%     VAL%    cum-equity   note
2025-09   +0.00    +0.00   1.000       cash
2025-10   +0.00    +0.00   1.000       cash
2025-11  +28.71   +70.70   1.287  WIN  deploys strict_efficiency @ 2.5x gross
2025-12   +0.00    +0.00   1.287       cash
2026-01   +0.00    +0.00   1.287       cash  (misses Jan +30% regime)
2026-02   +0.26   +12.51   1.290       deploy, ~flat
2026-03   +0.00    +0.00   1.290       cash
2026-04   +0.00    +0.00   1.290       cash
2026-05   +0.47    +2.86   1.297       deploy, ~flat
2026-06   +0.00    +0.00   1.297       cash  (misses Jun +19% regime)
2026-07  -16.71  +180.50   1.080  LOSS deploys profile_optuna @ 1.5x into 4-day partial
```

**Flow pathology — SINGLE-FOLD DEPENDENCE + a pro-cyclical conviction cliff.**
- The *entire* +7.99% is **one fold** (2025-11 +28.71%). Everything else is cash
  (7/11) or flat (Feb +0.26%, May +0.47%), and the final fold gives back -16.71%.
- Money is made in exactly one high-vol rip and nearly half of it is handed back in
  one partial fold. Sharpe 0.38 / "clean #1" is a **1-observation artifact**.
- **val→OOS is a directional CLIFF, worst where conviction is highest.** The Jul
  deploy had the *highest* validation (+180.5%) and produced the *worst* OOS
  (-16.71%) — a sign flip. Nov: val +70.7% → OOS +28.7% (positive but 2.5× decay).
  The gate sizes gross ∝ validation conviction, so it levers *most* into the months
  whose validation is most overfit. Its "conviction" signal is a blowoff-top signal.
- Drawdown: eq-MDD 16.7% is entirely the last fold and **unrecovered** (dd_len=1 but
  terminal). No recovery observed.
- It "avoids" the down-chop folds (Dec/Mar/Apr cash) — good — but also **misses the
  two genuine alpha folds LL harvests** (Jan +16%, Jun +12%). Cash discipline here is
  really *non-participation*, not risk management.

### Candidate B / B′ / C — lead-lag routers (the historical champions, +46–63%)

```
             LL_exact(B)  LL_mdd12(C)   strict_eff(D)   tradfi(E)
2025-09         +0.00       +0.00         -8.56          +0.00
2025-10         +0.00       +0.00         -0.72          +0.00
2025-11        +33.48      +33.48        +12.20          +0.00   <- universal WIN
2025-12         +0.00       +0.00         -1.01          +0.00
2026-01        +16.03      +10.55         +0.79          +0.00
2026-02         +8.44       +3.28         +8.44          +0.00
2026-03         -0.05       -2.51         -3.25          +0.00
2026-04         -4.46       +0.00         -3.58          +0.00
2026-05         -0.62       -0.62         -0.98          +0.39
2026-06        +12.11       +0.00         -0.31          +0.38
2026-07         -8.59       -0.91         -0.91          -0.30   <- universal LOSS
cum            1.634       1.463         1.006          1.005
eq-MDD          8.6%        4.0%          8.8%           0.3%
```

**Flow pathology — genuinely distributed alpha, but a common tail + a leverage
knob that trades tail for upside.**
- B (exact_unscaled) is the **highest-quality flow on the board**: 4 win folds
  spread across the year (Nov +33%, Jan +16%, Feb +8%, Jun +12%), losses small
  (-4.5%, -0.6%, -8.6%), eq-MDD only 8.6%. This is what a real edge looks like —
  money made in *multiple* folds, not one.
- **Same 2026-07 tail as everyone else**: val +173% → OOS -8.59%. The July
  overfit cliff is *universal*, not candidate-specific.
- **The MDD cap is the single most powerful lever.** C (`…_mdd12`) is the same
  signal with a tighter cap: it converts 2026-07 from **-8.59% → -0.91%** (tail
  cut) but also clips upside (Jan +16→+10.5, Jun +12→0). `best_historical` confirms
  the sweet spot is **`fallback_mdd20_cap2`**: comp 79.4%→64.4% but MaxOOS-MDD
  27.7%→18.5%, return/MaxMDD **2.87→3.49** on the expanded-110 universe. A *static*
  cap gives up too much June/Jan upside; a *state-aware* cap is the opportunity.
- **Universe fragility (survivorship-flavored optimism).** Same candidate, same
  params, different universe: 2026-06 = **+64.8% on historical-85** vs **+12.6% on
  expanded-110**; 2026-05 = +12.5% (85) vs -0.6% (110). The headline 197%/79% comp
  leans on a June fold that only the 85-symbol snapshot delivers → treat the
  historical figure as optimistic (research_note line 505 flags the megacap/AI
  survivorship tilt).

### Candidate D — strict_efficiency:static_guarded (+0.55%, 11/11 deployed)

**Flow pathology — LONG SHALLOW BLEED, no regime timing.** The only *always-on*
sleeve. It bleeds -8.56% in the very first fold (Sep), then grinds through ~8 small
losses (Oct -0.7, Dec -1.0, Mar -3.3, Apr -3.6, May -1.0, Jun -0.3, Jul -0.9),
netting its two wins (Nov +12.2, Feb +8.44) back to ~flat. Longest fold-drawdown =
5 folds. It is *good raw alpha with bad standalone risk-timing* — note that A
(dynamic_conviction) literally deploys strict_efficiency @2.5x in Nov and skips it
in the bleed months, which is exactly the missing wrapper.

### Candidate E / F — low-beta diversifiers (tradfi_intraday, state_vwap pair)

**Flow pathology — none pathological; just too small.** E deploys only the last 3
folds, MaxMDD 0.36%, Sharpe 0.79, +0.48%. F (dense_pairs) is 35–53% cash, val→OOS a
*smooth decay* (+5.3%→+0.46%, no cliff), Sharpe 3.46, MaxMDD 0.22%. Both are genuine
**uncorrelated capital-preservation carry** — their returns do not depend on the
2025-11 momentum spike. They are the only true diversifiers in the set.

---

## 4. Cross-candidate structure — do bad folds coincide?

**YES — the momentum families are almost entirely UN-diversified from each other.**

- **2025-11 is everyone's win** (A +28.7, B +33.5, C +33.5, D +12.2). Same high-vol
  rip, same cross-sectional-momentum beta. Stacking A+B+C+D does **not** add breadth
  here — it concentrates one common alpha.
- **2026-07 is everyone's loss** (A -16.7, B -8.6, C -0.9, D -0.9, E -0.3). Common
  tail; magnitude scales with leverage / cap-looseness. It is a **4-day partial
  fold** (`2026-07-01 → 07-04 06:30`) whose frozen params were fit to a full-size
  validation showing +100–180% — a structural overfit cliff, not a tradeable signal.
- The down-chop folds (2025-12, 2026-04) reward cash; A sits out and "wins" by 0%,
  while D holds and bleeds. That is the one place A's cash-gate genuinely helps.

**Where complementarity is REAL:** only across the **momentum ↔ low-beta boundary.**
E (tradfi_intraday) and F (state_vwap pair) carry positive/flat in the exact folds
(2026-05/06/07) where the momentum crowd rolls over. Within momentum (A/B/C/D) there
is no complementary pair worth stacking — one momentum sleeve is enough; a second is
redundant tail. **The portfolio lesson: pair ONE momentum sleeve with the low-beta
carry sleeves, not with a second momentum sleeve.**

---

## 5. Improvement briefs (flow-driven, mechanism-first, no curve-fitting)

Every proposal states the regime mechanism (WHY), not "it would have helped." Tagged
**[here]** = wireable in-repo (config/slice/overlay) vs **[data-PC]** = needs the
data-bearing machine to backtest.

### Brief C1 — AvgCorrelationCrashGuardOverlay around lead-lag `fallback_mdd20_cap2`  **[here]**  ★ top EV
- **Pathology:** LL (the actual champion) has distributed alpha but one common tail
  (2026-07, and the high-vol side of 2025-11). The tail is a **high-cross-correlation
  event** — every sleeve moves together.
- **Mechanism WHY:** average pairwise correlation spikes precisely in the
  crash/blowoff folds (2025-11 high-vol, 2026-07 whipsaw); the distributed alpha
  folds (Jan/Feb/Jun leaf wins) are *lower-correlation, idiosyncratic*.
  `AvgCorrelationCrashGuardOverlayStrategy` (registered `research_only`,
  `src/lumina_quant/strategies/correlation_crash_guard_overlay.py:145`) de-risks when
  average correlation crosses a band — it trims the common tail **without touching**
  the idiosyncratic wins. This is surgical: it attacks the ONE universal loss fold.
- **Wiring:** overlay-wrap the already-built `…_fallback_mdd20_cap2` variant (make it
  the default over `exact_unscaled` — best_historical already shows return/MaxMDD
  2.87→3.49). Overlay params (correlation window/threshold) are pre-registered, then
  the effect is measured on the data-PC.

### Brief F1 — Pre-registered 2-sleeve momentum+low-beta composite (M2 allocation cell)  **[here]**
- **Pathology:** momentum families share 2025-11 win and 2026-07 loss → stacking
  them concentrates the common tail; the only real fold-complementarity is across the
  momentum/low-beta boundary.
- **Mechanism WHY:** E (`tradfi_intraday_session_v1`) and F (dense_pairs
  `state_vwap` pair) carry flat/positive in 2026-05/06/07 with MaxMDD 0.2–0.4% and
  returns uncorrelated with the momentum spike. A composite = 1 momentum sleeve
  (LL_mdd20_cap2) + the two low-beta carry sleeves buffers the July momentum tail
  structurally, not by fitting.
- **Wiring:** a pre-registered composite cell for the M2 offline allocator / family
  tilt (no new signals; reuses registered sleeves). Weight search is data-PC; the
  cell and its members are declared here.

### Brief B2 — MomentumCrashDynamicScalingOverlay around dynamic_conviction_switch  **[here]**
- **Pathology:** A levers *most* (1.5–2.5x) into blowoff months; its -16.71% July is
  a levered entry right after a big June run-up.
- **Mechanism WHY:** `MomentumCrashDynamicScalingOverlayStrategy` (registered
  `research_only`, `src/lumina_quant/strategies/momentum_crash_scaling_overlay.py:147`)
  cuts gross after extended run-ups / vol spikes — exactly the 2026-07 setup (post
  June +19%). It targets A's specific failure mode (pro-cyclical leverage), where a
  static cap would not, because A's problem is *timing of leverage*, not average size.

### Brief A3 — Conviction-gate val-return saturation + partial-fold down-weighting (selection-gate treatment)  **[here for wiring / data-PC for thresholds]**
- **Pathology:** A's deploy-size map is monotone-increasing in validation return, but
  validation return is a *contrarian/overfit* signal at the extremes (val +180% →
  OOS -16.7%). Also the terminal partial fold (4 days) dominates `latest_oos` /
  `max_oos_mdd` rankings.
- **Mechanism WHY:** this is a **val→OOS cliff → flag for selection-gate treatment,
  NOT parameter surgery** (per mission guidance). Two monotone re-shapings: (a) a
  validation-return *saturation ceiling* — beyond ~40%/mo, treat extra validation as
  an overfit flag that *reduces* deploy scale rather than increasing it; (b)
  `bar_count`-aware down-weighting of partial final folds so a 4-day OOS cannot set a
  candidate's headline MDD/latest-OOS. Both are selection-layer, not backtest-number
  surgery. Thresholds tuned on data-PC.

### Brief D4 — Cash/regime gate (or corr-crash overlay) around strict_efficiency  **[here]**
- **Pathology:** always-on long shallow bleed; two wins eaten by eight small losses;
  worst fold is the very first (Sep -8.56%).
- **Mechanism WHY:** strict_efficiency is good raw alpha but has no regime filter. A
  (dynamic_conviction) already demonstrates the fix — it deploys strict_efficiency
  *only* in the up-vol folds (Nov/Feb) and sits cash through the down-chop (Dec/Mar/
  Apr), avoiding exactly strict_efficiency's bleed. Wrap strict_efficiency in the
  same cash/corr-crash gate so the base sleeve stops paying rent in chop. Cheapest
  alternative: tighter exit / min-hold retune cell (data-PC).

### Brief D5 — Universe-survivorship re-run of the lead-lag champion  **[data-PC]**
- **Pathology:** 197%/79% headline leans on a 2026-06 fold that is +64.8% on the
  85-symbol snapshot but only +12.6% on the 110-symbol universe.
- **Mechanism WHY:** the pre-registered 85-universe is a 2026-06-13 megacap/AI
  snapshot (research_note line 505) → optimistic for momentum. Re-run LL on a
  rolling/point-in-time universe with survivorship controls before trusting the
  historical high-water mark. Pure data-PC experiment.

---

## 6. Top-5 implementable improvements, ranked by expected value

1. **C1 — AvgCorrelationCrashGuardOverlay ⊗ lead-lag `fallback_mdd20_cap2`.**
   Highest EV: LL is the real champion (distributed +46–63% flow), and its one
   weakness is a common high-correlation tail (2026-07). The overlay attacks that
   single fold while preserving Jan/Feb/Jun idiosyncratic wins; both the variant and
   the overlay already exist and are registered. `[here]`
2. **F1 — 2-sleeve momentum+low-beta composite (M2 cell).** Structural
   diversification is the only real fold-complementarity in the data (momentum↔carry
   boundary). Reuses existing sleeves; buffers the July tail without curve-fitting.
   `[here]`
3. **B2 — MomentumCrashDynamicScalingOverlay ⊗ dynamic_conviction_switch.** Directly
   cuts A's -16.71% terminal blowup, which is a levered entry after a run-up — the
   overlay's exact target. Converts A from a 1-fold lottery into a risk-timed sleeve.
   `[here]`
4. **A3 — conviction-gate val saturation + partial-fold down-weighting.** Fixes the
   pro-cyclical cliff at the source and de-noises the ranking; selection-gate work,
   no backtest-number surgery. `[here]` wiring, `[data-PC]` thresholds.
5. **D4 — cash/regime gate around strict_efficiency.** Converts the always-on
   bleeder into a gated base sleeve, mirroring the pattern A already proves works.
   `[here]`

(`D5` universe-survivorship re-run is the top **data-PC** experiment; it governs how
much to trust the 197% historical headline.)

## 7. Honest limitations
- Per-fold series exist and were used; there is **no embedded BTC/market benchmark
  series**, so regimes are tagged via each fold's own OOS winner (a proxy). Attributing
  the 2026-07 loss to a specific BTC move is inference, not measured here.
- All numbers are locked-OOS *diagnostic* (report-only; not a selection input) and
  research/paper — `ready_for_real=false` throughout. None of these briefs implies
  promotion; they are pre-registration + wiring proposals to be measured on the
  data-bearing PC.
- The dynamic_conviction "clean #1 / +7.99%" rests on a single fold (2025-11) and a
  4-day partial fold (2026-07); its Sharpe 0.38 should not be read as a stable edge.
