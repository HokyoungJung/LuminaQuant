# Plan — Alpha Zoo top-candidate hybrid v3.5/v3.6 cost validation

Generated: 2026-05-18 KST
Mode: direct planning handoff for a new execution session
Do not execute real money; research/paper artifacts only.

## Target outcome

Build a latest-data, validation-to-March research run that takes the current Alpha Zoo leaderboards' top candidates, builds Hybrid v3.5 and Hybrid v3.6 portfolios from those candidate return streams, and validates **all individual seed candidates plus the two hybrids** under **round-trip slippage/fee = 5 bps and 10 bps**. The report must show train, validation, and locked-OOS metrics for every model, while keeping locked-OOS strictly out of training/selection.

## Evidence anchors from current repo

- High-leverage Alpha Zoo runner already defines the latest-data train/validation/locked-OOS split, MDD budget, base OOS reference, and split names in `scripts/research/run_alpha_zoo_validation_march_high_leverage.py:45-60`.
- Alpha Zoo train+validation objective uses validation-heavy metrics and explicitly reads `train`/`validation` metrics in `_tv_score` at `scripts/research/run_alpha_zoo_validation_march_high_leverage.py:349-386`.
- Candidate grid construction loops over strategy specs, leverage, and allocation and stores split metrics/audits at `scripts/research/run_alpha_zoo_validation_march_high_leverage.py:417-545`.
- Current candidate CSV records train/validation return plus locked-OOS metrics and live gate flags at `scripts/research/run_alpha_zoo_validation_march_high_leverage.py:597-642`.
- Latest-data report writing records grid summary/output paths and locked-OOS contamination evidence at `scripts/research/run_alpha_zoo_validation_march_high_leverage.py:727-803`.
- Existing fixed-input Hybrid v3.5/v3.6 runner documents method semantics: v3.5 is warmup-fixed default + rolling error/return weights + high-vol boost; v3.6 refreshes the default candidate online from rolling scores at `scripts/research/run_profit_moonshot_hybrid_v35_v36_fixed_inputs.py:2-15`.
- Hybrid implementation learns parameters only on train+validation masks at `scripts/research/run_profit_moonshot_hybrid_v35_v36_fixed_inputs.py:455-465` and writes Optuna metadata with `selection_inputs=["train", "validation"]` and `uses_locked_oos_for_selection=false` at `scripts/research/run_profit_moonshot_hybrid_v35_v36_fixed_inputs.py:545-553`.
- Existing fixed-input stream assembly is hard-coded to `A0 + P0 + E0 + S1 + S2 + S3 + S4` at `scripts/research/run_profit_moonshot_hybrid_v35_v36_fixed_inputs.py:568-663`; do **not** mutate that contract silently. Add a new runner or explicit alternate input mode for this seed-union experiment.
- Current Alpha stream reconstruction maps trades into timestamped return streams at `scripts/research/run_profit_moonshot_hybrid_v35_v36_fixed_inputs.py:666-723`; generalize this pattern for arbitrary Alpha Zoo candidate specs/leverage/allocation/cost scenarios.
- Integrated margin/live-promotion safety tests already cover locked-OOS isolation and deployability at `tests/test_profit_moonshot_hybrid_v35_v36_fixed_inputs.py:20-150`.
- Existing cost model for the live Alpha Zoo aligned lane subtracts `round_trip_slippage_fee_bps / 10000` inside isolated trade returns and reports 1/3/5/10/20 bps rows at `scripts/research/run_live_notional_risk_aligned_alpha_zoo.py:113-215`.

## Current seed selection snapshot

Source artifact: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/live_notional_risk_aligned_alpha_zoo_20260518/live_notional_risk_aligned_alpha_zoo_latest.json` and candidate CSV in the same directory.

Latest grid actually used in that artifact:

- leverage: `1x..20x`
- allocation: `0.03, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20`
- rows: `1600`
- live-promotion rows: `113`
- current filter rows (`train > val/OOS`, val >= 10%, OOS >= current base 6.4281%, OOS MDD <= 25%, positive OOS Sharpe/Sortino/smart/Calmar): `50`

Top-3 buckets to seed from; dedupe after union:

| Bucket | Top 3 seed rows |
| --- | --- |
| live OOS return | `fast_residual 7x/0.15`; `fast_residual 6x/0.175`; `fast_residual 5x/0.20` |
| live OOS Sharpe | `fast_residual 6x/0.05`; `fast_residual 7x/0.075`; `fast_residual 3x/0.10` |
| live OOS Sortino | `fast_residual 6x/0.05`; `fast_residual 4x/0.075`; `fast_residual 5x/0.075` |
| live OOS smart Sortino | `fast_residual 5x/0.05`; `fast_residual 2x/0.125`; `fast_residual 6x/0.05` |
| live OOS Calmar | `fast_residual 6x/0.175`; `fast_residual 7x/0.15`; `fast_residual 5x/0.20` |
| live full compound | `quality_single_pair 7x/0.20`; `quality_single_pair 7x/0.175`; `quality_single_pair 6x/0.20` |
| filtered balanced score | `quality_single_pair 7x/0.20`; `high_confidence_single_pair 7x/0.20`; `high_confidence_single_pair 7x/0.175` |
| filtered validation return | `quality_single_pair 10x/0.15`; `high_confidence_long_only 7x/0.20`; `quality_single_pair 7x/0.20` |
| filtered OOS return | `high_confidence_single_pair 7x/0.20`; `high_confidence_single_pair 7x/0.175`; `high_confidence_single_pair 6x/0.20` |
| filtered OOS Calmar | `high_confidence_single_pair 7x/0.20`; `high_confidence_single_pair 7x/0.175`; `high_confidence_single_pair 6x/0.20` |

Deduped seed universe currently has 18 rows:

1. `alpha_zoo_fast_residual 2x/0.125`
2. `alpha_zoo_fast_residual 3x/0.10`
3. `alpha_zoo_fast_residual 4x/0.075`
4. `alpha_zoo_fast_residual 5x/0.05`
5. `alpha_zoo_fast_residual 5x/0.075`
6. `alpha_zoo_fast_residual 5x/0.20`
7. `alpha_zoo_fast_residual 6x/0.05`
8. `alpha_zoo_fast_residual 6x/0.175`
9. `alpha_zoo_fast_residual 7x/0.075`
10. `alpha_zoo_fast_residual 7x/0.15`
11. `alpha_zoo_high_confidence_long_only 7x/0.20`
12. `alpha_zoo_high_confidence_single_pair 6x/0.20`
13. `alpha_zoo_high_confidence_single_pair 7x/0.175`
14. `alpha_zoo_high_confidence_single_pair 7x/0.20`
15. `alpha_zoo_quality_single_pair 6x/0.20`
16. `alpha_zoo_quality_single_pair 7x/0.175`
17. `alpha_zoo_quality_single_pair 7x/0.20`
18. `alpha_zoo_quality_single_pair 10x/0.15`

The execution session should recompute these buckets from the latest artifact rather than trusting this snapshot if the candidate CSV changes.

## Design decision

Create a **new research runner** rather than changing the existing fixed-input hybrid contract:

- Recommended path: `scripts/research/run_alpha_zoo_top_seed_hybrid_v35_v36_cost_validation.py`
- Output directory: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_top_seed_hybrid_cost_validation_20260518/`
- Keep existing `run_profit_moonshot_hybrid_v35_v36_fixed_inputs.py` behavior intact for `A0 + P0 + E0 + S1 + S2 + S3 + S4`.

## Implementation steps

1. **Load current Alpha Zoo artifact and candidate CSV**
   - Inputs:
     - `.../live_notional_risk_aligned_alpha_zoo_20260518/live_notional_risk_aligned_alpha_zoo_latest.json`
     - `.../live_notional_risk_aligned_alpha_zoo_20260518/alpha_zoo_validation_march_high_leverage_candidates_latest.csv`
   - Validate grid summary equals the current artifact values unless user intentionally overrides.
   - Recompute top-3 buckets and seed union dynamically; write `seed_selection_latest.json` and `seed_selection_latest.csv`.

2. **Generalize Alpha Zoo candidate stream reconstruction**
   - Reuse `run_alpha_zoo_validation_march_high_leverage` split/data/calibration setup.
   - For each selected seed row, resolve the strategy spec by `candidate_name`, run signals, attach trade path extrema, and convert trades to an hourly return stream by exit timestamp.
   - Apply isolated liquidation accounting: if liquidation happens, trade return is `-allocation_fraction`; otherwise use `alpha._portfolio_trade_return(..., leverage, allocation_fraction, round_trip_slippage_bps=bps)`.
   - Generate separate stream matrices for `0bps`, `5bps`, and `10bps`. The required report focus is `5bps` and `10bps`; keep `0bps` as reference/headline only.

3. **Build Hybrid v3.5/v3.6 on the seed universe**
   - Reuse the HybridParams / `_run_optuna` / `_annotate_live_policy` mechanics from the fixed-input hybrid runner.
   - For each cost scenario (`5bps`, `10bps`), run v3.5 and v3.6 with train+validation-only objective on cost-adjusted train+validation returns; locked-OOS remains report-only after freeze.
   - Persist learned params, top trials, final weights, allocation history, split metrics, and margin replay for both hybrid versions.
   - Use a deterministic seed, default `42`, and a configurable `--n-trials` (start with 80; increase only if runtime permits).

4. **Evaluate all individual and hybrid models under both cost scenarios**
   - Required model rows:
     - all deduped seed candidates individually,
     - `hybrid_v3_5_seed_union`,
     - `hybrid_v3_6_seed_union`,
     - current reference `fast_residual 7x/0.15`,
     - strict zero-liquidation reference `fast_residual 6x/0.10` if reconstructed cleanly,
     - previous fixed-input hybrid v3.5/v3.6 as historical/reference rows if compatible with the latest split; label them clearly as reference if not rerun on the same seed universe.
   - For every model and every cost scenario, report train/validation/locked-OOS:
     - total return,
     - MDD,
     - Sharpe,
     - Sortino,
     - smart Sortino,
     - Calmar / return-MDD,
     - trade/event count,
     - liquidation count,
     - account-wipeout count,
     - min margin buffer,
     - live/deployable gate result.

5. **Produce artifact bundle**
   - `alpha_zoo_top_seed_hybrid_cost_validation_latest.json`
   - timestamped JSON copy
   - `alpha_zoo_top_seed_hybrid_cost_validation_latest.md`
   - `seed_selection_latest.csv`
   - `model_cost_metrics_latest.csv`
   - `hybrid_weights_latest.csv`
   - `local_verification_alpha_zoo_top_seed_hybrid_cost_validation_*.log`
   - Include contamination audit fields: `uses_locked_oos_for_objective=false`, `uses_locked_oos_for_pruning=false`, `uses_locked_oos_for_selection=false`.

6. **Tests**
   - Add/extend tests without relying on long real-data runs:
     - seed bucket top-3 dedupe behavior from a miniature CSV;
     - cost-adjusted Alpha stream subtracts 5bps/10bps as round-trip bps, not percent;
     - locked-OOS poisoning cannot change seed selection or hybrid Optuna objective;
     - v3.5/v3.6 runner preserves train+validation-only selection metadata;
     - hybrid output includes both 5bps and 10bps scenario rows for all models;
     - liquidation/account wipeout events are included in split MDD/gates.

7. **Documentation and handoff**
   - Update `docs/research_note/research_note.md` with run result summary and artifact paths.
   - Update `.omx/notepad.md` with final artifact paths, selected/failed models, and verification evidence.
   - If the execution session refreshes market data or adds a new source family, update/regenerate `docs/research_note/research_history.md` and matching `var/reports/.../research_history/` artifacts; otherwise explicitly state why not regenerated.

## Acceptance criteria

- `seed_selection_latest.*` records top-3 buckets and deduped seed universe, and recomputation matches the latest candidate CSV.
- Hybrid v3.5/v3.6 are built from the deduped Alpha Zoo seed streams, not from the legacy `A0 + P0 + E0 + S1 + S2 + S3 + S4` fixed-input universe unless explicitly labeled as reference.
- For both `round_trip_slippage_fee_bps=5` and `10`, every individual seed and both hybrids have train/validation/locked-OOS rows with return, MDD, Sharpe, Sortino, smart Sortino, Calmar, liquidation count, account-wipeout count, and min margin buffer.
- Locked-OOS is not used for objective, pruning, parameter fitting, or model selection; it appears only in gate/report fields after candidate/hybrid freeze.
- No real-money execution or real order placement occurs.
- OOS MDD hard cap remains `25%`; account wipeout count must be `0` for anything marked deployable.
- The final report explicitly calls out whether each row survives 5bps and 10bps and whether the result is cost-fragile.

## Verification commands for the execution session

Minimum targeted checks after implementation:

```bash
uv run --extra dev pytest tests/test_profit_moonshot_hybrid_v35_v36_fixed_inputs.py tests/test_common_split_alpha_zoo_hybrid_v35_v36.py -q
uv run --extra dev pytest tests/test_profit_moonshot_liquidation_aware_validation.py tests/test_profit_moonshot_live_final_selection.py tests/test_crypto_fx_alpha_zoo_state_strategy.py -q
uv run --extra dev pytest -q
uv run --extra dev ruff check .
uv run --extra dev python -m compileall -q src scripts tests
git diff --check
git diff --cached --check
```

If runtime is too high, first run the targeted suites and the new test file, then run full pytest before final commit.

## Risks and mitigations

- **Survivorship/selection leakage**: Top buckets include OOS metric leaders. Mitigation: label this as a post-hoc research basket, not a deployable training selection; if selecting a live hybrid, freeze by train+validation-only objective and keep locked-OOS gate/report-only.
- **Cost model mismatch**: 5bps means 0.05% round-trip, 10bps means 0.10% round-trip. Mitigation: unit-test bps conversion and label round-trip vs per-side costs clearly.
- **Duplicate notional rows**: e.g., `7x/0.15` and `6x/0.175` are both 105% notional/equity. Mitigation: preserve both if they enter via distinct top buckets; report notional/equity and margin/equity separately.
- **Hybrid overfits train+validation**: Mitigation: fixed seed, limited Optuna space, locked-OOS report-only, and transparent top trials/final weights.
- **Runtime/memory**: Rebuilding streams for 18 seeds across multiple bps scenarios may be heavier than current fixed-input run. Mitigation: cache per-candidate trades/signals once and fan out cost scenarios; keep RSS under 8 GiB.

## Recommended new-session prompt

```text
$ralplan $team $ralph 이어서 진행해. 작업 디렉터리는 /home/hoky/Quants-agent/LuminaQuant 이야.

먼저 최신 상태를 맞춰:
- git fetch private
- git checkout private-main
- git reset --hard private/main
- git status -sb

이번 목표는 Alpha Zoo 최신 live-aligned 후보들로 seed-union hybrid v3.5/v3.6 cost validation을 새 artifact 디렉터리에 만드는 것이다. 반드시 먼저 읽어:
- .omx/plans/plan-alpha-zoo-hybrid-v35-v36-cost-validation-20260518.md
- docs/research_note/research_note.md
- var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/live_notional_risk_aligned_alpha_zoo_20260518/live_notional_risk_aligned_alpha_zoo_latest.json
- var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/live_notional_risk_aligned_alpha_zoo_20260518/alpha_zoo_validation_march_high_leverage_candidates_latest.csv
- scripts/research/run_alpha_zoo_validation_march_high_leverage.py
- scripts/research/run_profit_moonshot_hybrid_v35_v36_fixed_inputs.py
- scripts/research/run_live_notional_risk_aligned_alpha_zoo.py

핵심 요구:
1. 현재 artifact의 live/gate-pass leaderboard에서 top-3 buckets를 재계산해 deduped seed universe를 만들어라: OOS return, OOS Sharpe, OOS Sortino, OOS smart Sortino, OOS Calmar, full compound, 그리고 필터 통과 후보의 balanced/validation-return/OOS-return/OOS-Calmar top3.
2. 이 seed universe의 개별 Alpha Zoo candidate streams를 latest split 기준으로 재구성하고, isolated liquidation 손실을 account equity/MDD에 반영해라.
3. 이 seed streams로 Hybrid v3.5와 Hybrid v3.6을 새로 만들고, 기존 fixed-input A0+P0+E0+S1+S2+S3+S4 결과와 혼동하지 않게 artifact와 이름을 분리해라.
4. round-trip slippage/fee 5bps(0.05%)와 10bps(0.10%) 각각에서 모든 개별 seed + hybrid_v3_5 + hybrid_v3_6 + 현재 fast_residual 7x/0.15 reference + strict zero 6x/0.10 reference 성적을 train/validation/locked-OOS split별로 전부 찍어라.
5. locked-OOS는 selection/objective/pruning/parameter fitting에 절대 쓰지 말고 gate/report-only로만 써라. 이 오염 audit를 JSON에 명시해라.
6. 전체 계좌 wipeout은 절대 금지. isolated liquidation이 있으면 account equity와 MDD에 포함해라.
7. real money 실행은 하지 마라. paper/research artifact만 만든다.
8. 결과 artifact는 새 디렉터리에 저장해라:
   var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_top_seed_hybrid_cost_validation_20260518/
9. 연구노트와 .omx/notepad.md를 업데이트하고 Lore commit으로 private/main에 push해라. 가능하면 GitHub Actions ci/private-ci green까지 확인해라.

필수 검증:
uv run --extra dev pytest tests/test_profit_moonshot_hybrid_v35_v36_fixed_inputs.py tests/test_common_split_alpha_zoo_hybrid_v35_v36.py -q
uv run --extra dev pytest tests/test_profit_moonshot_liquidation_aware_validation.py tests/test_profit_moonshot_live_final_selection.py tests/test_crypto_fx_alpha_zoo_state_strategy.py -q
uv run --extra dev pytest -q
uv run --extra dev ruff check .
uv run --extra dev python -m compileall -q src scripts tests
git diff --check
git diff --cached --check

최종 보고에는 commit hash, artifact paths, seed universe count/list, hybrid v3.5/v3.6 configuration, 5bps/10bps train/val/OOS tables, liquidation-inclusive MDD, account wipeout count, locked-OOS contamination audit, verification evidence, CI links를 포함해.
```
