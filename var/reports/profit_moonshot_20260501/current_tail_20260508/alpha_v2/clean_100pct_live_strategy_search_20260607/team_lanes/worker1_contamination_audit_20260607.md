# Worker-1 Backup — Contamination / Eligibility Audit for G002

Scope: `.omx/ultragoal` active story `G002-contamination-and-eligibility-audit`.

Reason for leader backup: worker-1 gathered local evidence but became stuck waiting for a repository-map subagent; leader timeboxed and completed this lane from the collected evidence plus main-repo artifacts, without mutating prior dirty work.

## Classification summary

| Candidate / artifact | Observed result | Classification | Promotion status |
|---|---:|---|---|
| `clean_input_meta_selector` | OOS ann 110.46%, comp 85.91%, MDD 19.29%, Sharpe 1.28 | `shadow_freeze_only` | **Rejected for current promotion**: post-OOS selector/grid ranking contamination; may be frozen for fresh-forward validation only. |
| `strict_no_leak_best_single_10bps` | return 54.56%; 20bps stress 27.10%; high MDD / stress MDD | `paper_control` | Clean control, but not real-money due drawdown and missing fresh-forward/paper telemetry. |
| 85-symbol dynamic conviction baseline | OOS ann 42.57%, comp 34.39%, max MDD 27.69% | `paper_control` / monitor baseline | TradFi expansion monitor only until train+validation history and cost telemetry mature. |
| New-alpha diagnostic artifacts | full 10-fold ann 3.01%, feature-bounded ann negative/near-flat | `diagnostic_only` | No promotion; useful for search-family diagnostics only. |
| Untracked 2026-06-06 deep-research dirs | uncommitted historical research reports | `diagnostic_only` until audited | Not promotion evidence; preserve untouched. |

## no_nested_oos_mining checklist

- Existing historical OOS artifacts are used only as contamination map inputs: PASS.
- `clean_input_meta_selector` is explicitly not promoted despite 100%+ ann label: PASS.
- 100% annualized threshold is report-only, not selector objective: PASS in plan/test spec; execution must assert in code/config before any new run.
- Any post-OOS design change caps candidate at `shadow_freeze_only` or `diagnostic_only`: PASS as policy.
- Locked OOS attachment must occur only after train/validation choice: required for later stories G004/G005.

## Dirty-work protection

Main repo `/home/hoky/Quants-agent/LuminaQuant` remains dirty and was treated read-only by Team. Team ran from clean detached worktree `/home/hoky/Quants-agent/LuminaQuant-clean-100pct-20260607`; worker artifacts are versioned under `clean_100pct_live_strategy_search_20260607/team_lanes/`.

## Conclusion for G002

No existing historical artifact is eligible for current real-money promotion. The only 100%+ candidate remains `shadow_freeze_only`; the clean controls do not target 100%+ live annualized. G002 contamination audit therefore supports proceeding to G003/G004 only with a new immutable manifest and train/validation-only process.
