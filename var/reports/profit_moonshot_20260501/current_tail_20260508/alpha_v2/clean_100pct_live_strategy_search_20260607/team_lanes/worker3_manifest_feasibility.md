# Worker 3 — Immutable Manifest + Runner/Backtest Feasibility

Generated: `2026-06-07T07:02:00Z`  
Team task: `5`  
Goal reference: `.omx/ultragoal G002-contamination-and-eligibility-audit`  
Repo commit: `f9da35e126234f05f5fc223a99b1f1b5fa4c5845`

## Decision

**Runner/backtest feasibility: PASS with a required manifest gate before eligibility.**

Existing Optuna/hybrid/backtest surfaces are sufficient for a report-only static feasibility decision, but a new or reused candidate should not be classified as `eligible_control` until an immutable preflight manifest is emitted and validated fail-closed.

- Current lane core-code change: **not made**; task scope is the lane artifact pair under `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/clean_100pct_live_strategy_search_20260607/team_lanes/`.
- Implementation needed before eligibility/promotion: **yes** — add a small immutable manifest writer/validator or wrapper around the selected runner.
- Locked OOS role: **post-freeze gate/report only**.
- 100pct annualized-return threshold: **excluded** from objective, pruning, search constraints, promotion gates, and tie-breaks for this audit.
- Leader clarification applied: mailbox `587f766f-d0a5-45a3-ab3d-e192c2ccb6e0`; root-level `team_lanes/` copies are superseded and removed.

## Evidence Anchors

- `scripts/research/run_alpha_zoo_integer_leverage_optuna_hybrid_decision.py:14-16,1379-1405` — train/validation-only selection and locked-OOS false flags already exist.
- `scripts/research/run_profit_moonshot_hybrid_v35_v36_fixed_inputs.py:13-15,1326-1330` — fixed input universe with locked-OOS excluded from objective/pruning/selection.
- `scripts/research/run_alpha_zoo_69_asset_clean_oos_gate.py:2-7` — clean locked-OOS replay is post-freeze and fails on overlap.
- `src/lumina_quant/optimization/search_policy.py:17-22` — shared locked-OOS search flags default false.
- `src/lumina_quant/alpha_zoo/optuna_hybrid_config.py:280-300` — live config rejects Optuna artifacts if locked-OOS policy flags are not false.
- `tests/test_alpha_zoo_10bps_full_retune_artifact_assertions.py:327-388` — tests reject locked-OOS selection/objective/discovery contamination.

## Immutable Manifest Field Groups

The manifest must freeze these groups before any optimization starts:

1. **Identity** — version, kind, manifest id, run id, task/goal references, classification target.
2. **Hypothesis freeze** — thesis, strategy family, allowed/forbidden decision use, prior artifacts and roles.
3. **Universe freeze** — ordered symbols, inclusion/exclusion rules, source manifests, data cutoff, timestamp-index hash.
4. **Split freeze** — train/validation/locked-OOS timestamps, split generator hash, no-overlap checks, locked-OOS role.
5. **Code/source freeze** — git commit, dirty status, source/config/dependency hashes, external-reference hash or unavailable reason.
6. **Search contract** — optimizer, search space hash, bounds, seeds, objective formula, fit/selection/pruning inputs, forbidden fields.
7. **Cost/metrics/labels** — primary and sensitivity costs, turnover/RPT definitions, liquidation model, metrics by split, labels, paper-only flags.
8. **Command/output ledger** — ordered commands, env overrides, input/output paths, trial/memory logs, output hashes.
9. **Validation contract** — static scans, artifact assertions, targeted tests, compile/type/lint evidence, known gaps.

See `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/clean_100pct_live_strategy_search_20260607/team_lanes/worker3_manifest_feasibility.json` for the exact machine-readable field list.

## Hard Search Constraints

- Optimization input splits: `train`, `validation` only.
- Locked OOS may be used only for `post_freeze_gate`, `post_freeze_report`, or `shadow_monitoring`.
- Forbidden locked-OOS uses: discovery, objective, pruning, warmup/initial-state learning, parameter fitting before selection, selection, tie-break, search-space expansion, and promotion-threshold design.
- Required false flags: `uses_locked_oos_for_discovery/objective/pruning/parameter_fitting/selection`.
- 100pct annualized-return threshold: report-only context at most; never a selector, Optuna objective, promotion gate, pruning input, or tie-break.

## Runner/Backtest Feasibility

Feasible path:

1. Emit immutable manifest before runner starts Optuna/search.
2. Canonicalize and hash the manifest JSON.
3. Run Optuna/hybrid over train+validation only.
4. Attach locked-OOS metrics only after params/candidate freeze.
5. Validate emitted result against the manifest and fail closed on locked-OOS or 100pct threshold contamination.
6. Write command ledger, trial ledger, memory guard, result JSON/CSV/MD, and output sha256s.

No heavy architecture rewrite is required. The minimum follow-up implementation is a manifest writer/validator plus regression tests.

## Source Hashes

| Path | sha256 |
| --- | --- |
| `scripts/research/run_alpha_zoo_69_asset_clean_oos_gate.py` | `b2c51ae3b7af6f4e6871b3645297728049b4b39590b069f56bd33e706bbd1da7` |
| `scripts/research/run_alpha_zoo_69_asset_monthly_refit_walkforward.py` | `73c6211004d93082d96c1504e599fb959197fb4129f841c855f3d3022a11936b` |
| `scripts/research/run_alpha_zoo_integer_leverage_optuna_hybrid_decision.py` | `e87fc87e97638a5f610ad1c6fe17ad72b5bc57851e81dcfde7f0e1897670b7bb` |
| `scripts/research/run_profit_moonshot_hybrid_v35_v36_fixed_inputs.py` | `1e617e0372a0bfbd3dc67c4637514af553000fd5e3482143255e67103c5c2234` |
| `src/lumina_quant/alpha_zoo/optuna_hybrid_config.py` | `cb73a2f56290758fa4d4ce4502884bafa996c444c31b45ab380bdc80da76f08b` |
| `src/lumina_quant/cli/backtest.py` | `97965f519df5dc48f2fa05a9060fd5ab0610df9434f8d109e405992fe199d81a` |
| `src/lumina_quant/optimization/search_policy.py` | `8875f3348ef29b91ab8f7ef0c98f9db2e3072c9fe36d0cf09a2382a327ce7d80` |
| `tests/test_alpha_zoo_10bps_full_retune_artifact_assertions.py` | `aec0cc181d662106ac1e854fde808e65aaa279b8229c5c4c8e5cdacec43d8d19` |
| `tests/test_optimization_search_policy.py` | `1c61bd29ddb7ca5e07352a0e047d076177fbea218259d0fc6d22337059a6857e` |


## Delegation Compliance

Subagent skip reason: Available native subagent tool surface did not expose required `agent_type`/model fields; using it would violate the OMX instruction to set `agent_type` and the task request for `gpt-5.4-mini`. Local scoped probe was safer and sufficient for this artifact-only lane.

## JSON Payload Hash

`bcd9d6d9b0a0c916f37c3be843583fa6e0b425e66f2c2bfcafcacce98577dfc9`
