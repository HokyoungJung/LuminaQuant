# Alpha Hunt Consensus Plan — Meta Spine + One Leaf (APPROVED)

- Status: **APPROVED for execution** (user-authorized team run, 2026-07-03)
- Consensus record: Planner v1 → Architect SOUND-WITH-CHANGES (7 changes) → Critic ITERATE (3 blockers) → Planner v2 → Architect round-2 SOUND-WITH-CHANGES (4 mechanical residuals) → Critic round-2 **APPROVE** (+1 mechanical residual #5; hardening H1/H2 non-gating). Residuals 1-5 are folded into the lane specs below.
- Base: main @ 059cafe+ (rebase before every wave; other workers push to main concurrently).
- Execution mode: /team + /ultragoal. Waves W1/W2 parallel authoring -> W3 serial integration (single owner) -> W4 independent verification + handoff.

## Goal

Build alphas that can beat the CURRENT live default set — via meta/combination, not another lone leaf. No backtests and no data downloads on this PC (the data-bearing PC runs the clean walk-forward). Everything is live-applicable (completed-bar OHLCV, >=30m cadence, never-raise, state (de)serialization), theory-anchored, deterministic pure Python, golden-safe, real allocation 0%.

## Principles (consensus)

1. Meta over single-leaf: every single-leaf line failed locked-OOS (indicator/Kalman/ML -8.77%; deep-research leaves +1.71% -> -0.78% at 20bps). The spine is combination (M1) + allocation (M2).
2. <=1 new leaf now: orthogonality vs the ~77-sleeve live book is UNMEASURABLE on this PC. Any 2nd/3rd leaf (incl. deferred N4) is gated on measured orthogonal incremental factor_ic at the data-PC.
3. OHLCV-only core: funding/OI/taker coverage is 0% latest-OOS — hard blocker. No core component depends on it.
4. Golden-safe & candidate-only: new numerics research_only/offline-only; existing strategy numerics byte-identical; deterministic (same input -> bit-identical output). All verification uses `.venv/bin/python` (3.14) — system python3 is 3.12 and false-fails PEP 758.
5. Build-not-measure: deterministic unit + wiring verification here; return/OOS validation on the data-PC.

## CRITICAL LIVE-SAFETY MECHANISM (machine-enforced)

`_discover_plugin_strategies()` (registry.py:330) globs ALL strategies/*.py; an un-hinted `@register` class becomes `live_default` the moment the file lands. Therefore:
- W1/W2 authoring lanes ship strategy modules **WITHOUT the `@register` decorator** (verified: an unregistered module in strategies/ is truly inert — base Strategy has no metaclass/auto-registration; no meta-test requires registration).
- W1/W2 lane tests import the class DIRECTLY (`from lumina_quant.strategies.<module> import <Class>`) and must NOT assert registry/tier presence.
- W3 adds `@register` + the `"...Strategy": "research_only"` `_STRATEGY_TIER_HINTS` entry ATOMICALLY in the same commit, plus the CI guard test:
  - `test_no_unhinted_registered_strategy`: every registered strategy class must satisfy `name in _STRATEGY_TIER_HINTS or name in _LEGACY_UNHINTED_LIVE_DEFAULT or name in _STRATEGY_MAP` — membership assertion, NOT tier-value (tier value cannot distinguish explicit vs defaulted live_default).
  - `_LEGACY_UNHINTED_LIVE_DEFAULT` = programmatically generated frozen snapshot of today's ~68 pre-existing un-hinted names (generate at freeze time, do not hand-type), documented append-only-for-legacy.
  - Second assertion (Critic H1): this batch's new class names must appear SPECIFICALLY in `_STRATEGY_TIER_HINTS` as `research_only`.

## Scope — final portfolio (2 meta + 1 leaf + 1 conditional)

### Lane M1 — `DisagreementGatedEnsembleStrategy` (strategies/disagreement_ensemble_alpha_sleeves.py) — meta; first consumer of `ensemble_weights`
- Theory: forecast combination (Bates & Granger 1969; Timmermann 2006); ensemble diversity (Krogh & Vedelsky 1995); disagreement as real-time reliability proxy.
- Signal (OHLCV-only; funding component CUT): per symbol, 4 internal directional scores — (i) EMA-slope TSMOM sign, (ii) rolling-z reversion, (iii) Donchian channel position, (iv) Kaufman trend-efficiency (signed). Weights = `inverse_error_weights` over trailing sign-error per component; `disagreement_coefficient` (CV) gate -> flat/EXIT when CV > gate; else composite = sum(w_i * score_i); LONG/SHORT on +/-band; vol-targeted sizing. `direction_hit_rate` maintains the trailing error window.
- Distinct-from: `DiversifiedMultiFactorEnsembleStrategy` is a cross-sectional factor book; M1 is a per-symbol time-series signal combiner whose novelty is the disagreement GATE (trades only on consensus).
- Params via `get_param_schema` (HyperParam), shared rider-style knobs where applicable; never-raise; `get_state`/`set_state`.
- Bespoke tests (LCG-seeded): flat when components disagree (CV > gate); weights sum to 1; consensus long/short produce expected side; state roundtrip; determinism (run twice bit-identical); never-raise on degenerate input. NO @register in this lane.

### Lane M2 — Quality-gated sleeve allocator, OFFLINE manifest-generator CLI (portfolio/quality_gated_allocation.py + scripts/research/build_quality_gated_allocation.py) — meta; NO live surface
- Theory: ERC risk parity (Maillard-Roncalli-Teiletche 2010); HRP (Lopez de Prado 2016).
- Impl: pure function `allocate(streams, turnover, cost_regime)` + CLI wrapper. STATIC deterministic quality score per sleeve (NOT runtime StrategyQualityOverlay state):
  - net returns = `cost_realism.apply_cost_drag(gross_returns, turnover=<per-child turnover>, regime=<CostRegime built for 20bps>)` — NOTE: apply_cost_drag takes a CostRegime object + per-child turnover float, NOT a scalar bps (Architect residual 2).
  - sharpe/calmar = `optimizer_core.metrics(net_returns)["sharpe"|"calmar"]` (metrics is a FUNCTION returning a dict; periods_per_year=365 default) (Critic verification).
  - hit-rate = self-computed `mean(net_returns > 0)` — NOT provided by metrics() (Architect residual 2).
  - Gate: drop sleeves with net_sharpe_20bps <= 0; survivors -> `optimizers_extra.ERCPortfolio`/`HRPPortfolio` weights with `upper` cap; emit manifest artifact for the EXISTING `ArtifactPortfolioModeStrategy` consumer.
- Manifest contract (Critic residual #5 — the consumer's real fail-closed contract is DEEPER than 5 top-level fields): must satisfy artifact_portfolio_mode.py including — real_money keys false (top-level AND per-child), oos_contaminated/forbidden-OOS false, non-empty `optimizer_provenance` with source + selection_inputs, non-empty `correlation_input_provenance` with ready=True, per-source-artifact integrity (id/path, sha, freshness), per-child `no_current_fold_oos_provenance=True` and `train_validation_optimizer_provenance` (or lagged_completed_shadow) True.
- Bespoke tests (two-sided, MANDATORY):
  - Happy path: round-trip the emitted manifest through the ACTUAL `ArtifactPortfolioModeStrategy` consumer entry and assert it does NOT fail-close (cash_weight != 1.0; expected deterministic weights). Mock-only field checks are INSUFFICIENT.
  - Fail-closed path: parametrize over the consumer's ENUMERATED reason codes (manifest_missing, manifest_real_money_enabled, forbidden-oos, optimizer/correlation_provenance_invalid, source_artifact_sha/freshness_missing, child_current_fold_oos_provenance_missing, child_train_validation_optimizer_provenance_missing) — each induced omission must fail-close to cash.
  - Byte-golden on the emitted manifest for fixed synthetic inputs (determinism).
- No @register needed (not a strategy); lives in portfolio/ + scripts/ (new files only).

### Lane N1 — `CrossSectionalFlowShareRotationStrategy` (strategies/flow_share_rotation_alpha_sleeves.py) — the ONE leaf; first consumer of `flow_share`
- Theory: high-volume return premium (Gervais-Kaniel-Mingelgrin 2001); attention (Barber-Odean 2008); Amihud (2002).
- Signal: per bar per symbol quote-volume share via `cross_sectional_share`; `dense_rank_desc`; trailing z of share; `cdf_extremeness(z)` bounded gauge. LONG rising-share + positive short-horizon return confirmation; SHORT collapsing share OR blow-off (extreme share + negative return); cross-sectional z-scores; inverse-vol sizing; `share_weighted_return` composite diagnostics. Self-skip below min_symbols. Family=`cross_sectional`.
- Admission (honest route — Architect r1 #2, Critic r1 #2): N1 is cross-sectional+momentum but NOT carry -> the {cross_sectional,carry,momentum} tag-superset route does NOT apply; NO fake tags. Route = `allow_multi_asset=True` at the data-PC handoff. Production selection.py allowlist edit is an EARNED follow-up only.
- ACCEPTANCE TEST (Architect residual 1): assert the candidate appears in `select_diversified_shortlist(candidates, allow_multi_asset=True)` output (selection.py:515; or via `build_shortlist_payload(..., allow_multi_asset=True)` pipeline.py:105). `build_default_shortlist` DOES NOT EXIST.
- Distinct-from: no sleeve uses cross-sectional turnover SHARE (existing volume indicators are per-symbol level); doubly live-safe (research_only + non-carry XS excluded from default shortlist).
- Bespoke tests: shorts on share-collapse; longs on rising share + return; min_symbols self-skip; never-raise; state roundtrip; determinism. NO @register in this lane.

### Lane I2 (CONDITIONAL) — `RegimeRouterConfirmedRotationStrategy` (strategies/regime_router_confirmed_alpha_sleeves.py)
- Theory: regime-switching (Ang & Bekaert 2002); vol/cycle confirmation reduces chop whipsaw.
- Signal: breadth + BTC benchmark (as parent `BullBearRegimeRotationStrategy`) PLUS GARCH sigma regime (indicators/garch.py) and/or spectral cycle-phase confirmation (indicators/spectral_cycle.py); 3-state hysteresis (bull-long / bear-short / chop-flat); flips to bear-short only when vol + breadth + benchmark concur.
- KEEP ONLY IF its deterministic non-redundancy test passes, else DROP THE LANE: unit test constructs a chop input where `BullBearRegimeRotationStrategy` (and/or `AdaptiveRegimeMomentumStrategy`) flips bear-short but I2 stays FLAT absent vol/cycle confirmation — divergent action on identical input.
- NO @register in this lane.

### Dropped/Deferred (do not implement)
- DROPPED: N2 spectral phase rotation, N3 GARCH vol-managed momentum (funding-adjacent validation risk + saturation), I1 overlay-compat variant, O1 factor-IC wiring, O2 funding-basis carry.
- DEFERRED: N4 stationarity-gated residual reversion — behind measured orthogonal incremental factor_ic vs `DispersionConditionedReversionStrategy` at the data-PC. No "documented failure repair" claim (unverified).

## Waves (team execution)

- W1 (parallel authoring, new-file-only): Lane M1, Lane N1. Module + tests + atomic commit each. No shared-file edits; no @register. Barrier: full suite + ruff green (suite baseline 2946 passed / 21 skipped + our additions).
- W2 (parallel authoring, new-file-only): Lane M2, Lane I2 (commit only if non-redundancy test passes). Same rules.
- W3 (SERIAL integration, single owner): add @register + research_only hints (+ optuna overrides if applicable) atomically; thin `_build_*_candidates` builders + `build()` call lines in candidate_library.py (SLICE constants live in the lane modules); re-pin tests/test_candidate_manifest_snapshot.py (SHA/counts/tags — recompute, don't hand-derive); regenerate .github/hardcoded_params_baseline.json via `uv run python scripts/audit_hardcoded_params.py --write-baseline`; add the CI guard test. ONE commit. Full barrier.
- W4 (SERIAL verify + handoff, independent verifier — no self-approval): full suite + ruff check + ruff format --check + manifest + baseline + guard green; write the data-PC handoff spec (below) to docs/research_note/ or reports/; research_note entry.

Push discipline: rebase on origin/main before each wave's push (other workers push concurrently); authoring pushes first, W3 rebases last, W4 after.

## Data-PC Handoff — operationalized promotion decision rule (verbatim)

(a) At 20bps reference cost, net-of-cost DSR-adjusted Sharpe/IR of the NEW combined book > current default set — evaluated on IDENTICAL walk-forward windows and cost model as the default-set benchmark (H2).
(b) No degradation below the default set at 30bps.
(c) DSR > 0 after the N_eff penalty via `evaluate_survivorship_gate` (survivorship.py) + `effective_number_of_trials` (effective_trials.py).
(d) Per-leaf marginal test: incremental orthogonal factor_ic > 0 vs existing sleeves (gates N1 now; N4/2nd leaf later).
(e) Coverage-gate pass on latest-OOS.
(f) 0% real allocation until sign-off.
Plus: candidate family + strategy_class names; clean-WF invocation (train/validation-only selection, locked-OOS report-only monthly WF, no_nested_oos_mining=true); 10/15/20/30bps cost grid; `allow_multi_asset=True` note for N1; M2 manifest-provenance checklist.

## Risks

- live_default discovery window: mitigated by design (no @register until W3) + CI guard hard gate.
- Funding coverage 0% latest-OOS: hard blocker honored — OHLCV-only core.
- candidate_library merge risk: single W3 owner; thin builders only.
- Snapshot/baseline churn: re-pin exactly once in W3 (audit precedent 4b2b8e0).
- Concurrent main pushes: new-file-only authoring -> trivial rebases.
- Determinism drift: fixed-order reductions; pure-Python indicator modules; no scipy/sklearn/statsmodels.
- Follow-ups (non-gating): H1 stronger guard / flip un-hinted default tier to research_only with explicit legacy live hints; N1 production allowlist if earned; N4 if factor_ic earns it.

## ADR

- Decision: ship meta spine (M1 disagreement-gated ensemble + M2 offline quality-gated allocator CLI) + one honest XS leaf (N1 flow-share rotation) + conditional I2, all research_only/offline, atomic registration + CI guard, operationalized data-PC promotion rule. No backtests; 0% real allocation.
- Drivers: single-leaf locked-OOS failures; unmeasurable orthogonality here; funding coverage 0%; live-discovery window.
- Alternatives rejected: broad 8-alpha portfolio (premature breadth); meta-only (leaves flow_share unused — kept as fallback if N1 fails); retuning the +197% shadow champion (OOS-informed, clean_promotion_eligible=false); chasing 100%+ annualized clean.
- Consequences: +3~4 research_only/offline families; both orphaned indicator modules gain consumers; snapshot+baseline+guard re-pins; nothing live.
