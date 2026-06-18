Approved plan: `.gjc/plans/ralplan/2026-06-07-0457-b89b/pending-approval.md`
Source spec: `.gjc/specs/deep-interview-alpha-strategy-improvement.md`
Global constraints: no locked-OOS tuning/selection/threshold/tie-break/correlation/sizing; weak-data TradFi is shadow/research-only; MDD<=30%; no liquidation/account wipeout; shadow benchmark >64.42% comp or return/MDD >3.49; clean/paper benchmark >34.39% comp; real-money excluded; no commit/push.

@goal: Implement alpha promotion schema and discovery reporting gates
Add reusable gate/report fields for candidate identity, family/source bucket, theory, data sufficiency, weak-data shadow-only status, locked-OOS usage flags, train/validation freeze hashes, benchmark tier, MDD/liquidation/cost/telemetry gate results, promotion status, rejection reasons, and tried-universe coverage. Extend the clean new-alpha discovery smoke output so JSON+Markdown reports expose these fields while preserving train/validation-only selection.

@goal: Add existing strategy reassessment smoke wrapper
Create or extend a research script that enumerates registered/runnable strategies, records runnable status, tier metadata, skip/audit flags, tried-universe coverage, current known evidence where available, full-WF promotion eligibility, survivor list, correlation/report placeholders, and rejection reasons. Smoke is lenient for coverage, but full-WF promotion requires strict gates.

@goal: Expand candidate family and survivor manifest workflow
Broaden discovery/family coverage using existing data only across crypto price/volume, cross-asset/residual/dispersion, funding/OI/taker-flow/BBO/depth/liquidation, existing-winner overlays, and TradFi-linked candidates where data permits. Implement survivor manifest/freeze contract so only train/validation-frozen candidates can be forwarded to full WF, with locked-OOS attached only after freeze.

@goal: Implement artifact portfolio manifest fail-closed mode
Add manifest-driven portfolio composition beside existing artifact portfolio aliases. It must validate source artifact sha/freshness, child readiness, no-current-fold-OOS provenance, train/validation-only optimizer provenance, gross cap, per-leaf netting, correlation input provenance, and fail closed to cash on missing/stale/unreconciled/OOS-contaminated/gross-cap-breaching children. Keep real-money disabled.

@goal: Run focused verification and produce execution evidence
Run focused tests for discovery invariants, existing strategy reassessment, survivor manifests, strategy/registry tiering, artifact portfolio manifest fail-closed behavior, and report schemas. Run a bounded smoke/probe only if safe within time and available data; otherwise record no-promotion/watchlist evidence without weakening gates.
