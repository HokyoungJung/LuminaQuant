# Alpha-research knowledge graph

This directory records alpha-research facts and hypotheses without converting either into allocation or routing authority.

## Current authority

- **Binding selection decision:** `strategy_evidence_index.json` → `eval:g003-selection-v11`. G003 selection-v11 executed 24 candidates, accepted 16 allowed exclusions, produced 20 active panels and one Turtle quality survivor; six were required. No allocation was emitted, locked OOS was not launched, and order routing is `false`.
- **Status rules:** `evaluation_contract.md`. An execution pass is not a quality survivor or promotion.
- **Human map:** `strategy_taxonomy.md`; **typed graph:** `strategy_relationships.json`; **claim/source registry:** `strategy_evidence_index.json`.

## Evidence boundaries

The structured files are the current canonical graph. `research_note.md` and `research_history.md` are immutable-style provenance/history, not current-status authority. Their historic conclusions are not silently promoted into this graph. `strategy_evidence_20260812.json`, when present, is literature/design-prior evidence only, never repository performance.

Operational recovery, parity, resume, and handoff documents are archives/provenance leaves. They can establish sealing, causality, interruption, completeness, or routing state; they cannot establish alpha profitability. In particular, interrupted infrastructure is not strategy rejection.

## Reading order

1. Read the contract.
2. Find a stable strategy/candidate/evaluation ID in the evidence index.
3. Follow graph edges; `hypothesis` edges are proposals, not results.
4. Read source paths and windows before comparing metrics.

The `143` versus `144` registry snapshots remain an explicit missing-evidence flag, not a reconciled count. All paths and hashes are preserved verbatim from observed receipts.