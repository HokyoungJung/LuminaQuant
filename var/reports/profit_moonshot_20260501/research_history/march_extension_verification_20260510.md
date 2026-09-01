# March research-history extension verification

- scope: `2026-03-01..2026-05-10`
- git commits: `278`
- artifact inventory: `2384`
- inventory/ledger: `2667` / `2666`
- chronology entries: `15`

## Checks
- generator: exit `0`, elapsed `0:01.31`, max RSS `192972 kB`
- targeted_tests: exit `0`, elapsed `0:01.17`, max RSS `172316 kB`
  - stdout: `.....................                                                    [100%]
21 passed in 0.09s`
- full_pytest: exit `0`, elapsed `4:00.73`, max RSS `2816720 kB`
  - stdout tail: `........................................................................ [ 86%] | ........................................................................ [ 91%] | ........................................................................ [ 97%] | ..............................                                           [100%] | 1254 passed in 268.86s (0:04:28)`
- ruff: exit `0`, elapsed `0:00.04`, max RSS `34816 kB`
  - stdout: `All checks passed!`
- compileall: exit `0`, elapsed `0:00.16`, max RSS `34560 kB`
- git_diff_check: exit `0`, elapsed `0:00.15`, max RSS `31668 kB`
