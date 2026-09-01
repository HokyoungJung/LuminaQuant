# Verification summary — 30m+ Alpha Zoo feedback discovery

- Generated artifact: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_30m_plus_alpha_feedback_discovery_20260523/alpha_zoo_30m_plus_alpha_feedback_discovery_latest.json`
- Candidate count: `18450`
- Paper/testnet-only candidates: `4`
- Ready for real: `false`
- Real-money execution: `false`
- Discovery runner max RSS: `1,908,808 KB` (`1864.07 MiB`), 8GB cap pass `true`
- Discovery elapsed: `3:11.64`, exit `0`
- Full pytest: `1414 passed in 70.71s (0:01:10)`, max RSS `2,771,784 KB`, elapsed `1:10.00`
- Targeted/quality checks: `ruff`, targeted Alpha Zoo tests (`20 passed`), `compileall`, hardcoded-parameter audit (`new=0`), `git diff --check`, `git diff --cached --check` all passed.
- Team feedback loop: worker-1 detected an initial >8GB loader shape and validated chunked aggregation; leader runner uses per-file native 1s→30m aggregation and final main-worktree run stayed below 8GB.
