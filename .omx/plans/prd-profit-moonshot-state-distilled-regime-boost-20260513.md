# PRD — Profit Moonshot State-Distilled Regime Boost Portfolio — 2026-05-13

## Objective

Deliver a reproducible research-only `StateDistilledRegimeBoostPortfolio` lane that overlays the existing non-calendar state-distilled candidates with tunable regime classification, side bias, volatility-targeted leverage, a conditional booster sleeve capped at 25x, and a neutral high-dispersion pair overlay.

This PRD intentionally does **not** create a live strategy class first. The output is a research artifact and deployability decision. Any live/deploy class must be a later opt-in follow-up after the strict gate passes.

## User-facing target result

- Preserve the existing state-distilled candidate alphas instead of replacing them.
- Add a thin, explainable profit-amplifying overlay:
  - Core A: external-risk state-distilled 4x reference candidate.
  - Core B: pure state-distilled leadership/unwind 4x reference candidate.
  - Booster C: conditional high-leverage sleeve derived from Core B, configurable up to 25x based on long-term asset volatility and margin/liquidation safety.
  - Overlay D: neutral high-dispersion long/short pair sleeve.
- Tune all thresholds, weights, maps, allocation rules, and leverage gates via explicit config/grid parameters. No hidden hardcoded pseudocode thresholds may determine selection.
- Keep total session memory under 8 GiB and record peak RSS.
- Update research notes, handoff, `.omx/notepad.md`, plan/result artifacts, and report artifacts.
- Commit with Lore protocol, push to `private/main`, and verify GitHub Actions `ci` and `private-ci` green.

## Hard constraints

1. No calendar/month/day/hour entry or selection rule.
2. Calendar/current-base tuple remains `hypothesis_reference_only`; it is not a selection or promotion target.
3. Selection/tuning uses train+validation only.
4. Locked-OOS opens only after the candidate/config is frozen, and is gate/report-only.
5. Strategy/factor validity fails closed for calendar fields, OOS selection leakage, missing provenance, liquidation in strict lane, or non-positive margin buffer in strict lane.
6. Strict deploy lane and diagnostic high-leverage/nonfatal lane must be reported separately.
7. Strict promotion is prohibited if liquidation count >0 or min margin buffer <=0 in any split.
8. Booster leverage may be parameterized up to 25x, but never exceeds 25x and must be reduced/disabled by high long-term volatility, high short-term vol ratio, stress, or weak margin safety.
9. OOS MDD must remain <=25% for any promoted strict result.
10. Use risk metrics such as Sharpe/Sortino/smart Sortino/Calmar for risk quality; strict promotion requires positive validation/OOS return and positive locked-OOS Sharpe/Sortino/smart Sortino/Calmar, while return/MDD is diagnostic-only per latest user instruction unless an existing global live gate independently requires it.
11. Memory peak RSS must be <8 GiB.

## Inputs / evidence anchors

- Existing state-distilled candidates:
  - `fresh_state_distilled_ext_both_lb168_fast72_z075_ret180_h168_tp600_fl0_xr125`
  - `fresh_state_distilled_both_lb168_fast72_z075_ret180_h168_ls590_ss100_tp600`
- Existing artifacts:
  - `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/liquidation_aware_state_distilled_external_risk_filter_20260512/liquidation_aware_current_base_latest.json`
  - `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/liquidation_aware_state_distilled_20260511/liquidation_aware_current_base_latest.json`
  - `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/external_market_state_20260512/external_market_state_lagged.csv`
- Existing reusable code:
  - `scripts/research/replay_profit_moonshot_fresh_start.py`
  - `scripts/research/run_profit_moonshot_liquidation_aware_validation.py`
  - `scripts/research/replay_crypto_fx_alpha_zoo_state.py`

## Required deliverables

1. New or updated research runner for `StateDistilledRegimeBoostPortfolio`.
2. Config/dataclass model that serializes every tunable parameter and selected parameter value.
3. Real current-tail data replay and selection ledger.
4. Train/validation-only selection provenance and freeze artifact.
5. Locked-OOS gate/report artifact opened after freeze.
6. Strict zero-liquidation lane result and diagnostic high-leverage/nonfatal lane result.
7. Unit/regression tests for no calendar rule, no OOS selection leakage, leverage cap/vol downshift, strict liquidation fail-closed, and memory/provenance payloads.
8. Research notes and handoff updates.
9. Verification evidence and Lore commit/push/CI links.

## Out of scope

- Building a live trading strategy class or live deployment toggle.
- Using the invalid calendar/current-base result as a promotion target.
- Expanding into a broad new alpha family search beyond this thin overlay.
- Treating diagnostic high-leverage liquidations as strict deploy success.

## Acceptance checklist

- [ ] All overlay parameters are explicit config/grid fields and are written to artifacts.
- [ ] `calendar_primary=false` and no month/day/hour fields/rules are introduced.
- [ ] Selection ledger contains no locked-OOS metrics used for scoring/ranking.
- [ ] Frozen config timestamp precedes locked-OOS gate/report in artifacts.
- [ ] Booster effective leverage is capped at <=25x and varies with asset long-term volatility.
- [ ] Strict lane uses zero-liquidation and positive-buffer checks across train/validation/OOS.
- [ ] Diagnostic high-leverage lane is labeled nonfatal/diagnostic and cannot promote live success.
- [ ] Memory peak RSS is recorded and <8 GiB.
- [ ] Research docs and handoff list artifact paths, metrics, provenance, and remaining risks.
- [ ] Required tests/static checks pass locally; commit pushed; `ci` and `private-ci` are green.

## Critic iteration hardening — required before execution

The first Critic pass returned `ITERATE`; these hardening requirements are now part of the execution contract.

### Immutable freeze provenance

The frozen candidate artifact must include:

- selected core candidate id(s) and selected booster/pair sleeve identifiers
- full selected overlay config and every selected tunable value
- train/validation selection ledger path and SHA-256 hash
- input artifact manifest with SHA-256 hashes for all JSON/CSV/parquet inputs used by the runner
- code commit SHA, plus `dirty_tree=true/false` and dirty file list if applicable
- configured grid size, evaluated count, skipped/pruned count, search-space hash, and selection-score field list
- `frozen_at` timestamp
- freeze artifact path; its SHA-256 hash is recorded in a separate immutable sidecar/manifest, not inside the freeze payload

The locked-OOS gate artifact must record `locked_oos_opened_at`, reference the exact freeze artifact path/hash from the sidecar/manifest, and must not contain selected parameter mutations after `frozen_at`.

### Bounded-grid / validation-overfit guard

The default research grid cap is `64` evaluated trials. The hard maximum is `256` unless a future PRD explicitly raises it. The runner must deterministically enforce:

```text
evaluated_count <= min(config.grid_limit, 256, product_space_size)
```

Artifacts must emit configured grid size, evaluated count, skipped/pruned count, search-space hash, and selection-score field list. Selection/pruning/ranking must not read locked-OOS metrics.

### Neutral-pair leakage guard

Neutral-pair universe, pair choice, hedge ratio, dispersion trigger, and overlay weights must be fit from lagged/as-of train/validation data only, then frozen before locked-OOS. OOS-only-good pairs must not be selectable. Pair features must carry an as-of timestamp/provenance statement.

### OOS MDD gate

A strict promoted result must have locked-OOS max drawdown `<=25%` in addition to zero liquidations and positive min margin buffer across train, validation, and locked-OOS. This gate is separate from the user’s latest instruction that return/MDD ratio should be diagnostic-only.
