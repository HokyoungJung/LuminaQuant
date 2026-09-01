# RALPLAN-DR — Web-Grounded Autonomous Research Expansion (2026-03-15)

## Principles
- Preserve the current best incumbent until a challenger clears the locked-OOS promotion rule.
- Prefer web-grounded ideas with low implementation risk and clear mapping onto existing LuminaQuant primitives.
- Keep one heavy lane at a time and stay inside the 8 GiB total-session cap.
- Every new idea must end in keep/discard/crash with artifact evidence.

## Decision Drivers
1. Improve locked-OOS total return without giving back the current incumbent's drawdown discipline.
2. Minimize engineering overhead by reusing current strategy/eval/optimizer surfaces.
3. Maintain reproducible, reversible artifacts that can be promoted or discarded cleanly.

## Viable Options
1. Web-grounded factor/residual momentum expansion around current cross-sectional + pair + trend sleeves.
2. New pair-state modeling lane (better spread estimation / dynamic hedge ratio / residual filters).
3. More radical allocator changes (dynamic/overlay/online learning).

## Chosen Direction
- Prioritize option 1 first, then option 2, while keeping option 3 as a secondary benchmark lane.
- Immediate candidate lanes:
  - residual / factor-neutral momentum overlay ideas
  - crash-aware momentum gating / momentum decomposition ideas
  - carry + momentum hybrid ideas
  - improved pair-state / Kalman-filter-inspired pair features

## ADR
- Decision: Continue with web-grounded research intake and bounded execution under team+ralph, anchored to the current 3-sleeve incumbent.
- Drivers: better locked-OOS performance, minimal new architecture, reproducible artifact flow.
- Alternatives considered: stop at current incumbent; pure allocator search; full architecture rewrite.
- Why chosen: highest expected improvement per unit of implementation risk.
- Consequences: requires ongoing disciplined keep/discard loops and careful memory budgeting.
- Follow-ups: tune/backtest shortlisted ideas, refresh decision artifacts, then Ralph verification/cleanup/private sync.

## Staffing Guidance
- available-agent-types roster: executor
- team headcount: 1 executor worker (memory-safe lane)
- leader lane: monitor status, web-ground idea intake, artifact triage
- verification lane: linked Ralph follow-up after team delivery
- launch hint: omx team ralph 1:executor "<task>"
