# PRD — Profit Moonshot Alpha Zoo Real-Data Calibration (2026-05-13)

## Objective
Promote the Alpha Zoo path from synthetic smoke to real current-tail research by wiring `CryptoFxAlphaZooStateStrategy` and its factor/outcome/calibration stack to real crypto/FX/FRED data, then evaluating it under train/validation-only selection and strict liquidation-aware promotion gates.

## Scope
1. Implement or connect a real current-tail data adapter for `scripts/research/run_crypto_fx_alpha_zoo_screen.py`.
2. Persist factor cards with source coverage, `calendar_primary=false`, and `uses_locked_oos_for_selection=false`.
3. Produce a real candidate outcome ledger using triple-barrier labels.
4. Calibrate lower-confidence edge buckets using train/validation only, fail-closing non-positive or tail-loss buckets.
5. Replay `CryptoFxAlphaZooStateStrategy` against narrow interpretable grids seeded by Alpha Zoo, state-distilled external-risk, and residual-pair hypotheses without hand-tuned calendar proxies.
6. Freeze candidates before opening locked-OOS, then run locked-OOS as report/gate only.
7. Run integer leverage 1x..6x with strict zero-liquidation promotion lane and separate diagnostic nonfatal 5x/6x lane.
8. Update notepad, plan, research note, handoff, and report artifacts; note whether global research history/source ledger changed.
9. Run required local verification, Lore commit, push to `private/main`, and confirm GitHub Actions `ci` and `private-ci` green.

## Non-goals
- No new hand-tuned calendar/month/day/hour entry rule.
- Do not use the current-base/calendar tuple as a selection or promotion target.
- Do not select or tune by locked-OOS.
- Do not conflate diagnostic nonfatal leverage with live promotion.

## Acceptance criteria
- Real-data screen artifacts identify data sources, coverage, row counts, split counts, and fail-closed strategy validity.
- Real-data adapter artifacts distinguish observed vs imputed fields per source/symbol and fail closed when required real OHLCV coverage is missing; smoke/default-filled coverage cannot be treated as valid economic evidence.
- Candidate ledger and calibration artifacts record train/validation-only provenance and locked-OOS exclusion.
- Edge calibration must physically filter records to `split in {"train", "validation"}` before estimating buckets and must report `locked_oos_calibration_record_count=0`.
- Strict validation reports all leverages 1x..6x, zero-liquidation strict lane, and diagnostic 5x/6x lane.
- Any promoted success has zero liquidations across train/validation/OOS, positive min buffers, OOS MDD <= 25%, OOS return above the invalid current-base reference, and positive Sharpe/Sortino/smart Sortino/Calmar. Return/MDD is diagnostic/report-only, not a hard promotion hurdle.
- If no deployable candidate exists, artifacts explicitly set live promotion/deployable success false and explain the gate that failed.
- Peak RSS is reported and < 8 GiB.
- Required tests/lint/compile/diff checks pass.
- Lore commit is pushed and GitHub Actions `ci`/`private-ci` are green.

## Available agent-type roster and staffing guidance
- `explore` / low reasoning: map data loaders, artifacts, and scripts.
- `executor` / medium reasoning: implement adapters, CLI/report changes, and tests.
- `test-engineer` / medium reasoning: verify targeted/focused/full tests and acceptance coverage.
- `architect` / high reasoning: review train/validation/OOS separation and promotion gates.
- `verifier` / high reasoning: final evidence audit.

## Execution lane
Use `$team` for bounded parallel discovery/implementation support when helpful, then `$ralph` as the single-owner persistence/verification loop. Team launch hint: `omx team 3:executor "Profit Moonshot Alpha Zoo real-data calibration implementation and evidence lanes"`. Ralph stop condition is this PRD's acceptance criteria plus the user's final reporting contract.

## ADR
- Decision: reuse the existing Alpha Zoo/triple-barrier/calibration/liquidation-aware stack and add real-data adapters/reporting instead of creating a new strategy family.
- Drivers: avoid calendar proxy leakage, preserve train/validation-only selection, make real-data economics reproducible, and keep memory under 8 GiB.
- Alternatives considered: (a) retune calendar/current-base tuple — rejected because invalid for live promotion; (b) build a new hand-tuned proxy — rejected as likely calendar overfit; (c) use smoke scaffold only — rejected because user requires real current-tail evidence.
- Consequences: strict gates may produce no live promotion; that is acceptable if evidence is complete and fail-closed.
