# G003 alpha-research integration — 2026-08-23

## Research objective

Integrate the G003 Alpha-Max candidate and its exact evaluation machinery into the existing LuminaQuant/Quants-agent framework as research-only infrastructure. The result must support strategy comparison, lineage, combination, and new-alpha discovery without enabling order routing or treating interrupted evidence as reusable.

## Frozen evidence retained

- Source/canonical deep clearance: `03b670f27a2b2847ff588c65565f444e69eeef7b57f87225391d4cd2a54141bb`.
- Strategy/named-suite clearance: `595faa303b8f8d40843cd59a49ca3aa5da251da680cfe4064245b3f969541547`.
- Exact-native candidate capsule: `7f5374685ef7a0a0acb3a1353ca87c4976ceba9f80e63ededf6e8a1a8644c222`.
- Native finalization: `a5e30170bf43d17ce8714c987644b2d853ba8f9f1ddb7e5480ae2b4808cdd1e6`.
- Invocation-sealed v10 candidate: `437911de1532db5a0d76dc3c1d6cfd9000bf75cbd047f06c324f6ccd9fae6aa7`.
- Current full verification clearance: `3ba94620329ccf4862f89734e149797237078f7a211e7f52638c36d67153b4b9`.

## Superseded oracle

The non-checkpointed one-second Python oracle was deliberately stopped before acceptance. Direct byte accounting mapped its progress to about 80.42 of 366 UTC days (21.97%) after 10.48 hours, with roughly 37 hours still expected. A WSL shutdown would have discarded all oracle progress.

The service and downstream path triggers are disabled. Its empty staging output is immutable and non-reusable. Supersession receipt SHA-256: `f1793acc435a626f2e42fdbad739a12decfa8f8c33f1a39c9fed57e86161dd24`.

## Integrated optimization boundary

- Rust owns only deterministic folding of authenticated canonical 1-second OHLCV into completed 4h/1d bars and retained working-bucket state.
- Python retains cross-sectional release ordering, strategy calls, event draining, final forming-bar handoff, finalization, capsule serialization, and hashes.
- Economic replay remains separate. Funding, costs, strict LMT touches, orders, fills, liquidation, equity, and drawdown semantics are unchanged.
- Parity restart state is sealed only at complete UTC-day boundaries before finalization.
- Prelock retains complete row/cost-cell checkpoints; historical evaluation gains the same complete 10-fold logical-cell boundary.
- Interrupted folds, partial output, unsealed checkpoints, and failed candidates remain non-reusable.

## Alpha-research knowledge model

Each strategy note must distinguish observed evidence from hypotheses and record:

- family and implementation lineage;
- current, historical, excluded, rejected, or research-only state;
- evaluated universe, timeframe, interval, costs, and point-in-time selection rules;
- observed metrics or exact rejection reason;
- closest related strategies and overlap drivers;
- complementary and conflicting exposures;
- reusable insights, improvement hypotheses, combination hypotheses, and unexplored variants;
- links to immutable run, selection, and clearance evidence.

The Obsidian graph uses typed links for lineage, similarity, derivation, complementarity, conflict, evaluation, rejection, and evidence. Operational receipts remain provenance leaves rather than central strategy nodes.

## Safety state

`order_routing_enabled=false`. G003 remains research-only until independently sealed evaluation and final current-state audit are complete.
