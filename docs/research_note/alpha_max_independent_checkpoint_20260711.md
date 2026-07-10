# Alpha-Max Independent Research Checkpoint — 2026-07-11

## 1. Checkpoint purpose

This is a portable **implementation checkpoint**, not a completed alpha-performance claim. It preserves every normative plan, goal, audit artifact, current implementation slice, known gap, and verification result needed to continue on a data-bearing PC.

- Repository: `hoky1227/Quants-agent`
- Branch: `feat/alpha-max-20260710`
- Frozen baseline: `252910e54e280cc593365484cbc99d6ca87893f9`
- Original isolated worktree: `/home/hoky/Quants-agent-alpha-max-20260710`
- Shared worktree intentionally untouched: `/home/hoky/Quants-agent`
- Approved design: Ralplan Revision `5.14`
- Experiment id: `alpha_max_portfolio_20260710`
- Runtime model floor: GPT-5.5 or newer. Output produced after a worker fell back to `gpt-5.4-mini` was excluded.

On another PC, fetch this branch and treat the committed `.omx/plans`, `.omx/ultragoal`, and this document as the continuation authority.

## 2. User decisions and non-negotiable objective

The interview closed with these decisions:

1. **Return-first** after fixed eligibility gates.
2. **Portfolio-first**, with separately measurable component sleeves and leave-one-out ablations.
3. Normal MDD band is `<= 30%`.
4. A row in `(30%, 35%]` may survive only if it has both strictly higher CAGR and strictly higher Calmar than the deterministic return-first best matched normal-band row.
5. MDD `> 35%` is a hard rejection.
6. Selection ranking is cumulative return, CAGR, Calmar, net Sharpe, lower MDD, then lexicographic row id.
7. Local data collection and local performance backtesting are out of scope. The data PC performs the real replay.
8. No arbitrary/OOS-mined rule, unsupported return claim, real-money promotion, or modification of the other active session is allowed.

The goal is therefore not “invent a high backtest number.” It is to create the strongest causally executable, cost-aware, falsifiable portfolio experiment the current repository can test without contaminating historical selection.

## 3. Proposed alpha portfolio

The frozen candidate book combines three low-frequency crypto-perpetual mechanisms over ten liquid symbols:

- Universe: `ADAUSDT`, `AVAXUSDT`, `BNBUSDT`, `BTCUSDT`, `DOGEUSDT`, `ETHUSDT`, `SOLUSDT`, `TONUSDT`, `TRXUSDT`, `XRPUSDT`.
- **Daily trend persistence**: inherited time-series momentum/ADX/efficiency/volatility-persistence mechanics, with exact frozen parameters in the current-node registry.
- **Daily cross-sectional near-52-week-high anchoring**: inherited distance-to-high ranking with an atomic admitted-symbol daily barrier, causal completed-bucket handling, weekly rebalance, and symmetric long/short support.
- **4h funding-harvest carry**: inherited funding carry with causal as-of funding points, trend-fight avoidance, volatility sizing, controlled adds, and explicit funding settlement.

Portfolio variants are:

- each component at 1x;
- full equal weight, Ledoit-Wolf/ERC equal risk, and shrunk-HRP;
- leave-one-component-out variants for all three allocators;
- selected full/LOO equal-risk and shrunk-HRP risk-scaled siblings using `clip(0.25, 2.25, 0.27 / max(validation_1x_mdd, 1e-12))`;
- incumbent and diagnostic rows retained for frozen comparison/reporting but not silently materialized as executable new rows.

The allocation input is exact-inner-calendar daily arithmetic net-equity returns at the 20 bps cell with at least 252 observations. Selection eligibility is evaluated at 30 bps.

## 4. Theoretical basis and falsifiers

Primary mechanisms are grounded in the following literature. These sources justify hypotheses and falsifiers, **not** an expected CAGR or Sharpe:

- Time-series momentum: <https://www.sciencedirect.com/science/article/pii/S0304405X11002613>
- 52-week-high anchoring: <https://onlinelibrary.wiley.com/doi/full/10.1111/j.1540-6261.2004.00695.x>
- Common crypto risk factors: <https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.13119>
- Crypto carry: <https://pubsonline.informs.org/doi/abs/10.1287/mnsc.2024.05069>
- Volatility-managed portfolios: <https://onlinelibrary.wiley.com/doi/abs/10.1111/jofi.12513>
- Multifactor volatility-management counterevidence: <https://onlinelibrary.wiley.com/doi/10.1111/jofi.13395>
- Equal-weight benchmark difficulty: <https://academic.oup.com/rfs/article-lookup/doi/10.1093/rfs/hhm075>
- Deflated Sharpe ratio: <https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2460551>
- Superior Predictive Ability test: <https://www.tandfonline.com/doi/abs/10.1198/073500105000000063>
- Probability of backtest overfitting: <https://www.risk.net/journal-of-computational-finance/2471206/the-probability-of-backtest-overfitting>
- Crypto-momentum counterevidence: <https://link.springer.com/article/10.1007/s11408-025-00474-9>
- Volatility-scaling counterevidence: <https://www.sciencedirect.com/science/article/pii/S1386418116301379>

Critical falsifiers are cost death, funding/timestamp leakage, validation-to-historical collapse, insufficient admitted coverage, concentration, allocator instability, PBO/SPA/DSR failure, liquidation/ruin, and scaled rows that improve only leverage rather than alpha quality.

## 5. Normative artifacts and hashes

These files are committed even though `.omx/` is normally ignored. Do not regenerate or edit them casually.

The portable source/test/artifact checksum manifest is `docs/research_note/alpha_max_checkpoint_sha256_20260711.txt`; verify it with `sha256sum -c docs/research_note/alpha_max_checkpoint_sha256_20260711.txt` before continuing.

| Artifact | SHA-256 |
|---|---|
| `.omx/plans/ralplan-alpha-max-independent-20260710.md` | `3b4601b489e906452f8b25e4e116e973954307caa0f4ada7a98a55f3d033ddf6` |
| `.omx/plans/prd-alpha-max-independent-20260710.md` | `bbb0f07dc019571081163baf1f672f8ba60bc38d663427e7fadc80bc2221e889` |
| `.omx/plans/test-spec-alpha-max-independent-20260710.md` | `99ee9b9760cf14041afb63f275d0371773b09835e09f8719bb7ccad61e3f8d2f` |
| `.omx/plans/alpha-max-current-trial-nodes-v1.json` | `cfe3a04620c52cc235d6f1cda1cac617ba30cd7327c753fc2f620d8250d51a4e` |
| `.omx/plans/alpha-max-incumbent-resolution-v1.json` | `5133bc40116399fe7af32e75a1ecc52a4f385dc8a0b5d3a4a9585e2437615ed8` |
| `.omx/plans/ralplan-consensus-alpha-max-independent-20260710.json` | `3d1f3447d676801b6c73e4805f57d23550ef29f1489604bc95be79ab87951746` |
| `.omx/plans/architect-review-alpha-max-independent-20260710-revision5.14.md` | `e21c4800546eae8b5edde336a97c9a32f0dfcbdd75d8c0305dabdd08dd1d3549` |
| `.omx/plans/critic-review-alpha-max-independent-20260710-revision5.14.md` | `e8dd5da4a3299e783d595a8b753484f29b165c86ffc07915d19df44296a46759` |

Frozen experiment config hashes:

- runtime contract: `b3859443c842cf8b04d04ed32923e6c6a8207af18e26f68a717ba623b4edfef9`
- payload: `b53c2274624fe4bc017ead59975efc805d166038f841773337bb48d55ee9692d`
- canonical document: `85ab64360d77265441d2eeaaa7a41a4df12589667bccdfec75b62572bfcf5e62`
- file: `34f1ea894b0af984d4f76348f52fbca09fab45b9e3d5d963f257ec9d128ee356`
- prior/current DSR family: 1,466 + 21 = 1,487 trials
- current key-set SHA-256: `3a4791cf353abcb82f9717ce89ee16b9d73d84f431d5b058135046c2ba8e332b`
- prior actual-LF SHA-256: `3b078011040f89e8d788b2cef9214c58f687221104381e26a688a7f8cdbddd78`

## 6. Durable goal status

The original Ultragoal files are preserved at `.omx/ultragoal/brief.md`, `.omx/ultragoal/goals.json`, and `.omx/ultragoal/ledger.jsonl`.

Important: the stored goal status still shows G001 `in_progress` because the Codex goal snapshot became `usageLimited` and OMX correctly refused to falsify an active/complete checkpoint. The real implementation status is:

| Goal | Checkpoint status | Evidence |
|---|---|---|
| G001 — isolated baseline and integrity plumbing | implementation complete | one-descriptor receipts, generic receipt propagation, strict constructor seams, default-neutral portfolio seams, focused and full regressions |
| G002 — sleeves and portfolio artifacts | implementation complete for planned local contracts | three research-only sleeves, native clocks/capsules, causal feature/funding seams, admission, allocators, manifest materializer, frozen config |
| G003 — runner/statistics/external replay | **partial** | execution attribution, pure metrics/statistics/trial ledger, and strict runtime preflight exist; actual replay, selection, terminal state, CLI, and data-PC bundle remain |
| G004 — integrate/review/verify/push | checkpoint-only | this portable checkpoint is tested and pushed; hosted CI has the full-history checkout required by the immutable prior-trial Git-blob contract; final completion review and data-backed performance verification remain |

The exact original objectives and audit ledger must remain intact. Do not mark the aggregate goal complete on the continuation PC until G003/G004 and the mandatory independent review gate are finished.

## 7. Implemented surfaces

### Already committed before this checkpoint

- `ArtifactReadReceipt` with one-descriptor hashing/parsing and fail-closed path identity checks.
- Generic manifest/source receipt propagation with tuple identity preservation.
- Strict `Backtest` constructor kwargs path without changing the legacy retry path.
- Optional portfolio attribution/funding seams with legacy `None` behavior preserved.
- Immutable causal raw/feature point accessors.
- Three research-only native-timeframe sleeves and registry entries.
- Portfolio wrapper decision cadence, native-bucket finalization, and indicator-only capsule forwarding.

### Added in this checkpoint

- `configs/research/alpha_max_portfolio_20260710.json`: exhaustive immutable Revision 5.14 experiment contract.
- `alpha_max_evidence.py`:
  - ordered funding roots and boundary resolver;
  - atomic funding ledger/settlement and rollback semantics;
  - train-only admission artifacts;
  - equal-weight, Ledoit-Wolf/ERC, and shrunk-HRP allocation;
  - exact 17 executable-row manifest materializer;
  - common RNG and strict UTC 4h arithmetic return streams;
  - canonical metrics plus full-event MDD/drawdown duration/type-7 VaR/worst-5% ES;
  - report-only turnover/RPT/capacity diagnostics;
  - immutable Git-blob-backed 1,487-trial ledger;
  - pre-gate Sharpe variance and canonical DSR/SPA/PBO calls.
- Execution pricing/application attribution with trace hashes, positive-fill bijection, no-fill taxonomy, liquidation separation, state persistence, and fail-loud sinks.
- Production portfolio bridge for one-call funding batch settlement and attribution.
- `alpha_max_engine_runner.py` pure foundation:
  - one-descriptor config receipt and exact hash/schema gates;
  - ambient `LQ_*` rejection;
  - immutable uppercase allowlist and deterministic read audit;
  - exact common seed schedule;
  - exact four cost-cell configs;
  - explicit phase-owned constructor plans.
- `tests/__init__.py` prevents a dependency-installed `tests` package from shadowing repository tests.

## 8. Intentionally incomplete work

Do not mistake the checkpoint for a runnable final experiment. The following are still required:

The independent verifier's exact file/line continuation audit is committed at `.omx/plans/verifier-continuation-blueprint-alpha-max-20260711.md`. It found six blockers that must be closed before orchestration: carry must consume canonical `feature_lookup`; ordered funding lookup must satisfy the engine `db_path` capability gate; indicator-only warmup needs a non-economic runner boundary; full-event evidence needs a bounded streaming design compatible with 8 GiB; pure selection/reconciliation evidence is incomplete; and final-refit manifests require a fresh permitted-prefix capsule replay rather than relabeling a validation capsule.

1. Bind the exact config/manifest receipt and immutable `PortfolioModeDefinition` immediately before the first replay.
2. Assemble train-only admission and prove the identical admitted tuple/object/value at config, strategy, data handler, manifest, funding resolver, and portfolio boundaries.
3. Implement the two-engine phase protocol:
   - warmup engine is indicator-only and economically discarded;
   - finalize completed native buckets at the exact watermark;
   - export deterministic indicator capsules;
   - construct a fresh economic engine and restore capsules before scoring;
   - prohibit orders, fills, funding, cash, and metrics from crossing the warmup boundary.
4. Implement raw-first native event ordering, 1s decision cadence, phase-owned raw/funding roots, and exact resolver/sink identity assertions.
5. Run all 21 frozen rows where resolvable across exactly 10/15/20/30 bps, record complete matrix evidence, and fail closed on missing cells.
6. Build the nominal-30 bps gate input, exact metric calendar reconciliation, DSR/SPA/PBO evidence, coverage/hash/manifest/funding/ruin gates, and only then MDD gates.
7. Implement deterministic normal/soft-band ranking, sole prelock champion fixation, report-only historical leader, final-weight refit rules, and terminal outcome precedence.
8. Implement physically separate CLIs/processes for prelock selection and historical exposed evaluation. The historical process must be incapable of mutating the prelock champion.
9. Emit deterministic JSON/Markdown evidence bundles and a data-PC runbook. No performance number is valid until those replays finish.
10. Run final ai-slop cleanup, independent `code-reviewer`, independent `architect`, architecture-invariant audit, full CI, and hosted CI.

## 9. Recommended continuation order

Use small sequential write scopes; do not start with CLI wiring.

### Slice A — seal and prepare phase inputs

- Extend `src/lumina_quant/research/alpha_max_engine_runner.py` only.
- Add immutable prepared-phase records for config receipt, phase window, admitted tuple, ordered raw/funding roots, manifest receipt, and strategy definition.
- Assert descriptor/receipt hashes, absolute immutable paths, symbol identity, 1s cadence, attribution sink identity, resolver identity, and exact four cost cells before any engine construction.

### Slice B — two-engine replay primitive

- Add a single-phase replay primitive and hostile unit tests.
- Warmup must be indicator-only; export/finalize/restore capsules; then create a fresh economic engine.
- Prove raw-before-derived ordering and prove zero warmup economic leakage.

### Slice C — row and matrix orchestration

- Materialize only the 17 executable rows; keep incumbents/diagnostic rows explicitly unavailable or report-only as declared.
- Execute common random numbers for each split/cost cell.
- Record trace/application/funding bijections and complete cost-cell evidence.

### Slice D — selection and terminal state

- Add pure gate/selection functions to `alpha_max_evidence.py` with dedicated tests.
- Gate order: DSR, SPA, PBO, positive metrics, native/funding coverage, hash/manifest validity, zero ruin, MDD.
- Apply the exact normal/soft/hard MDD policy and lexicographic tie-breaker.
- Fix at most one prelock champion before any historical replay.

### Slice E — process separation and outputs

- Add two entry points: prelock selection and historical evaluation.
- Use separate input/output paths and process-level tests proving historical code cannot mutate selection artifacts.
- Add deterministic bundle/report schema and the exact data-PC commands.

### Slice F — final quality gate

- Focused hostile tests, full pytest, Ruff, compile, architecture/purity/hardcoded-parameter checks, docs verification, Rust/dashboard/8 GiB checks where CI requires them.
- Run independent code-reviewer and architect lanes.
- Only after a clean gate may Ultragoal G003/G004 and the aggregate goal be completed.

## 10. Verification at checkpoint

Fresh checkpoint verification:

- Alpha-max plus adjacent constructor/receipt/portfolio tests: `265 passed`.
- G003-A metrics/statistics slice: `34 passed` focused, `106 passed` adjacent, `181 passed` all alpha-max at agent handoff.
- G003-B1 runtime-contract slice: `42 passed` focused+adjacent at agent handoff.
- Repository-wide Ruff check: passed.
- Repository-wide Ruff format check: passed after formatting four branch-owned files; AST parity was identical for all four.
- Hardcoded-parameter audit: `1668` total, `0` new, `1668` baselined. The deterministic baseline was regenerated because the new research strategy and inserted wrapper seams changed the scanner's line/column signatures; no scanner exemption was added.
- Python compile and `git diff --check`: passed.
- Full repository pytest: `4349 passed, 36 skipped, 3 xfailed` in `124.06s`; maximum RSS `562,336 KiB`, zero swap.
- Hosted CI portability: the quality job now checks out full Git history (`fetch-depth: 0`). This is required because prior-trial accounting deliberately reads the frozen baseline blob `252910e54e280cc593365484cbc99d6ca87893f9:var/reports/ultragoal_full_pool_strategy/g004_frozen_candidate_manifest.json`. Runtime auto-fetch or a missing-history fallback was rejected because either would add network side effects or weaken fail-closed evidence.

An earlier stable full-suite run after execution-attribution integration and namespace hardening produced `4287 passed, 36 skipped, 3 xfailed` with 4,326 collected nodes. This is evidence for that earlier state; use the fresh final run for the checkpoint commit.

## 11. Resume commands on the data PC

```bash
git clone https://github.com/hoky1227/Quants-agent.git
cd Quants-agent
git fetch origin feat/alpha-max-20260710
git switch --track -c feat/alpha-max-20260710 origin/feat/alpha-max-20260710
git cat-file -e 252910e54e280cc593365484cbc99d6ca87893f9^{commit}
uv sync --frozen --extra dev

# Verify all normative hashes before editing.
sha256sum \
  .omx/plans/ralplan-alpha-max-independent-20260710.md \
  .omx/plans/prd-alpha-max-independent-20260710.md \
  .omx/plans/test-spec-alpha-max-independent-20260710.md \
  .omx/plans/alpha-max-current-trial-nodes-v1.json \
  .omx/plans/alpha-max-incumbent-resolution-v1.json

# Re-establish the checkpoint before implementation.
uv run --frozen --extra dev pytest -q \
  tests/unit/test_alpha_max_*.py \
  tests/test_alpha_max_windowed_data_accessors.py \
  tests/test_portfolio_optional_seams.py \
  tests/unit/test_artifact_read_receipt.py \
  tests/unit/test_artifact_portfolio_mode.py \
  tests/unit/test_backtest_constructor_kwargs.py
uv run --frozen --extra dev ruff check .
uv run --frozen --extra dev ruff format --check .
```

Do not start the data replay by inventing paths or allowing profile/environment fallback. Complete and test the physical prelock/historical CLIs first, then point their explicit raw/funding roots at the data-PC datasets.

## 12. Safety and interpretation

- This branch is research-only and has zero real-money approval.
- CI demonstrates deterministic implementation contracts, not alpha performance.
- Historical exposed evaluation is report-only and cannot select or repair the prelock champion.
- Scaled-row improvement is a risk transform, not a distinct alpha claim.
- If no row survives the frozen 30 bps gates, the correct terminal result is `no_demonstrated_alpha`.
- Never weaken DSR/SPA/PBO, coverage, funding, ruin, MDD, or process-separation gates merely to produce a winner.
