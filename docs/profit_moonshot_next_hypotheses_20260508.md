# Profit moonshot next hypotheses — 2026-05-08

## Non-negotiable constraints
- Keep `<8 GiB` memory guard and current RSS evidence fields.
- Keep locked-OOS as **report-only / gate-only**. Selection objectives may use train/validation only.
- Do not promote OOS-ranked diagnostics or family-diversity candidates unless they also pass train/validation and existing OOS gates.
- No new dependencies. Reuse existing replay, portfolio tuning, Optuna, and validator paths.

## Evidence from latest all-family expansion
- Latest promoted portfolio pass: train `3.5993%`, validation `2.6755%`, locked-OOS `1.2181%`, OOS MDD `0.1662%`, OOS Sharpe `6.7264`.
- Prior promoted portfolio OOS return was `0.8789%`; current pass improved by `+0.3392%p`.
- Replay: `6,805` specs, `300` pass; portfolio tuning: `58,224` specs, `6,129` pass; Optuna: `128` trials, `8` pass.
- All passing replay families were `calendar_rotation`. New standalone non-calendar families had zero promoted successes.
- High-return diagnostic portfolio exists but is not promotable: train `20.4013%`, validation `19.4308%`, locked-OOS `3.9710%`, but OOS MDD `0.8977%` fails `oos_mdd_beats_shadow`.
- Diagnostic best OOS portfolio also fails MDD: locked-OOS `6.3563%`, OOS MDD `1.2530%`.

## Working diagnosis
The next edge probably is **not** another standalone all-hours non-calendar alpha. The data says calendar/TRX sleeves have the durable train/validation sign, while residual/flow/funding/trend standalone families are negative or too low-amplitude. Therefore the next hypotheses should use new features as risk controls, gates, or calendar-conditioned modifiers instead of independent sleeves.

## Execution status — 2026-05-09
- H6/H1/H2 were processed first in commit `333f8c7931e75ea3b828f7f7edc714ab66857c10`; CI was green and no improved candidate was promoted.
- H3/H4/H5 were then processed as `h3_h4_h5_calendar_conditioned`:
  - H3 added calendar-conditioned residual/funding/market/flow veto variants.
  - H4 added day-of-month + entry-hour calendar sub-window variants.
  - H5 added an explicit TRX/ETH calendar spread state machine.
- Bounded replay evaluated `80` H3/H4/H5 specs: `0` replay survivors, `0` success candidates, peak RSS `250.707 MiB`.
- Portfolio follow-up evaluated `18` train/validation-positive sleeves and `20,145` portfolio specs: `0` improved candidates, peak RSS `389.789 MiB`.
- High locked-OOS return diagnostics remain quarantined as `diagnostic_not_promoted`; best diagnostic OOS was `3.7780%` with MDD `0.5981%`, failing MDD / return-risk gates.
- Conclusion: all listed hypotheses H1-H6 have now been tried under the locked-OOS/report-only and `<8 GiB` constraints; current champion is preserved.

## Ranked hypotheses

### H1 — Drawdown-capped calendar sleeve allocator
**Hypothesis:** The high-return additive calendar portfolios fail because correlated sleeves stack the same drawdown. A train/validation-only allocator that clusters overlapping calendar sleeves and caps each cluster's drawdown contribution can keep most of the return while passing OOS MDD.

**Why now:** Equal-weight passes at `1.2181%` OOS, while additive diagnostics show much higher train/validation/OOS return but fail MDD. This suggests allocation/overlap control, not alpha discovery, is the bottleneck.

**Implementation sketch:**
1. Add a portfolio candidate mode such as `cluster_capped_validation_weight`.
2. Cluster sleeves by calendar long/short symbols, hold window, take-profit, and return correlation on train/validation equity curves.
3. Apply per-cluster cap and per-sleeve MDD budget from train/validation only.
4. Emit diagnostic rows for capped-vs-uncapped comparison.

**Acceptance target:** train `>=5%`, validation `>=4%`, locked-OOS return report `>1.2181%`, OOS MDD still passing current gate. The OOS threshold is a report/gate check, not a selection criterion.

### H2 — Calendar parameter-neighborhood stability objective
**Hypothesis:** The robust calendar edge is a parameter neighborhood around TRX long, `thr≈0.018`, `hold≈168`, `take_profit≈0.060`, not a single point. Scoring parameter neighborhoods by train/validation median and dispersion should improve return without selecting fragile spikes.

**Why now:** Optuna and expanded grid found strong high-validation rows around the same TRX take-profit region, but some high-return variants failed MDD. A neighborhood score can prefer stable adjacent parameter blocks.

**Implementation sketch:**
1. Add a replay/portfolio summary that groups calendar candidates by symbol pair + threshold/hold/take-profit neighborhood.
2. Score with train/validation median return, lower-quartile return, validation MDD, and trip sufficiency.
3. Feed top neighborhood representatives into portfolio tuning before broad combinatorics.

**Acceptance target:** fewer candidate sleeves than broad top-N, equal or better train/validation return than current champion, OOS pass preserved, replay/portfolio memory stays below current peak order.

### H3 — Calendar-conditioned non-calendar veto filters
**Hypothesis:** Residual/flow/funding/trend features are not profitable as standalone entries, but can reduce drawdown when used as veto filters for calendar sleeves.

**Why now:** Standalone non-calendar families have zero train+validation pass, but some families occasionally show small positive OOS or validation signs. Their best use is likely “do not enter when adverse,” not “enter independently.”

**Implementation sketch:**
1. Add calendar variants with optional veto filters:
   - residual z-score too adverse for selected long/short leg,
   - funding sign conflicts with calendar long/short direction,
   - market trend/fade extreme near entry,
   - flow imbalance exhaustion near entry.
2. Keep candidate names explicit, e.g. `calendar_trx_tp600_resid_veto`.
3. Score by train/validation return retention plus MDD reduction.

**Acceptance target:** retain at least `75%` of base calendar train/validation return while reducing train/validation MDD by `>=20%`; OOS gate pass and OOS return above current single-sleeve baseline.

### H4 — Monthly/day-window decomposition of the calendar edge
**Hypothesis:** The calendar edge is concentrated in specific day-of-month and hour buckets inside March-May / January-Feb. Restricting entries to robust sub-windows can raise return per drawdown and reduce failed additive MDD.

**Why now:** Calendar is the only passing family, so sub-window structure is more plausible than unrelated alpha families.

**Implementation sketch:**
1. Add calendar candidate grids for day-of-month buckets and entry-hour blocks.
2. Use train/validation-only minimum sample/trip gates to avoid starving trades.
3. Combine with H1 cluster caps.

**Acceptance target:** validation return/MDD improves versus current calendar champion; OOS trips remain non-starved and OOS pass preserved.

### H5 — TRX/ETH spread-style calendar sleeve
**Hypothesis:** The durable signal is a relative TRX-vs-ETH/weakest seasonal spread. Explicit hedge-ratio or dollar-neutral spread construction may keep return while lowering drawdown compared with independently scaled long/short legs.

**Why now:** Best sleeves repeatedly use TRX long with ETH/weakest or paired calendar shorts. Treating this as a spread rather than two independently scaled exposures may address MDD failures.

**Implementation sketch:**
1. Add a spread-calendar family where train/validation estimates hedge ratio or fixed notional ratio.
2. Keep target exposure capped and report beta/exposure diagnostics if available.
3. Compare against existing `calendar_rotation` using same split/gates.

**Acceptance target:** positive train and validation, OOS return competitive with current champion, OOS MDD lower than current additive diagnostics and passing gate.

### H6 — High-return diagnostic quarantine
**Hypothesis:** The high-return rows are useful as research signals but dangerous as promoted candidates. A quarantine report will prevent accidental promotion while preserving evidence for allocator research.

**Implementation sketch:**
1. Add a report section listing high-return failed-MDD portfolios.
2. Require explicit `diagnostic_not_promoted` label.
3. Add a regression test that selected-by-validation rows failing OOS MDD are never marked success.

**Acceptance target:** no behavior change to strategy returns; clearer safety evidence and fewer future promotion mistakes.

## Execution order
1. Implement H6 first if touching reports/tests: it locks the risk boundary.
2. Implement H1 + H2 next: highest expected return lift without inventing unrelated alpha.
3. Implement H3 after H1/H2: use non-calendar features as vetoes, not standalone strategies.
4. Implement H4/H5 only if H1/H2/H3 do not exceed current champion under gates.

## Stop condition for next run
A new candidate can be called improved only if:
- train and validation are both positive and stronger than the current promoted portfolio on return/risk,
- locked-OOS is report-only but passes all current gates,
- OOS return is greater than `1.2181%`,
- peak RSS remains `<8 GiB`,
- full local tests/ruff/compile/diff-check pass before commit/push.
