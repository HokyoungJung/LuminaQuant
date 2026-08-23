# Alpha-research evaluation contract

This contract prevents execution coverage, research quality, portfolio selection, and infrastructure state from collapsing into one `pass` field.

## Evidence precedence

1. Wrapper-owned immutable acceptance or rejection whose exact bytes, invocation, service exit, and ancestry are bound.
2. Named evaluation artifact with window, universe, timeframe, costs, status, and content hash.
3. Source/registry fact bound to a commit or source manifest.
4. Secondary research-note summary with an explicit missing-primary flag.
5. Literature or a proposed hypothesis. This can shape a test but never populate repository performance.

A lower class cannot override a higher class. Missing metrics remain `null` or `not_available`; exclusions and unrun work are never recorded as zero.

## Execution statuses

| Status | Meaning | Performance claim |
|---|---|---|
| `completed` | The named evaluation completed its declared work. | Only the artifact's stated outcome/metrics. |
| `failed` | The evaluator or contract failed. | None unless a separately sealed valid unit exists. |
| `interrupted` | Work stopped before an eligible terminal artifact. | None; partial output is non-reusable. |
| `superseded` | A method was retired by a newer design. | None from unfinished output. |
| `non_admitted` | The run was intentionally outside official admission. | None. |
| `pending` | Work is scheduled but not complete. | None. |
| `not_run` | The gate correctly prevented execution. | None. |

## Evaluation outcomes

| Outcome | Meaning |
|---|---|
| `execution_pass` | Runner/interface completed. It is not quality survival. |
| `quality_survivor` | Candidate passed the named positive-quality gate. It is not suite or portfolio acceptance. |
| `suite_rejected` | Portfolio-level requirements failed even if a candidate survived. |
| `passed_gate` | The exact named gate passed; downstream authority remains separate. |
| `failed_gate` | A valid completed evaluation failed its preregistered gate. |
| `allowed_exclusion` | Data/universe/feature contract permitted exclusion; performance unavailable. |
| `resource_excluded` | Capacity policy excluded evaluation; performance unavailable. |
| `feature_excluded` | Causal required feature was unavailable; performance unavailable. |
| `invalid` | The result violated causality, contract, or evidence identity and cannot be interpreted. |
| `inconclusive` | Valid observations exist but do not support the requested decision. |

## Candidate, suite, and portfolio invariants

- Candidate pass does not imply quality survival.
- Candidate quality survival does not imply a surviving suite or portfolio.
- Suite rejection does not automatically reject every strategy concept.
- Locked OOS may run only after its preregistered admission gate. G003 selection-v11 correctly did not launch it after one survivor failed the six-survivor floor.
- Smoke results cannot select parameters, promote a candidate, or authorize allocation.
- Infrastructure failure, interruption, supersession, and non-admission cannot reject a strategy.

## Comparability

Metrics are comparable only when domain, point-in-time universe, start/end, cadence, execution model, costs/funding, fold construction, and evidence class match. Nested recent windows are sensitivity checks, not independent OOS. Cross-family combination claims require common aligned net-return panels and incremental-correlation or factor evidence.

## Controls

- Rebalancing requires an identical-asset, identical-initial-weight no-rebalance control.
- Every overlay requires an equal-gross unwrapped child.
- Every router requires a static equal-risk leaf blend.
- Event continuation/reversal arms must be preregistered before observing the target window.
- Formula/horizon/stop variants count toward the effective-trials denominator.
- Causal feature age and point-in-time membership are part of the contract, not optional diagnostics.

## Safety

Parity, source/canonical audit, checkpoint, and observability artifacts establish execution provenance, not alpha performance. Official acceptance remains wrapper-owned. `order_routing_enabled=false` throughout this research graph.
