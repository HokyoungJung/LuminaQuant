# Strategy recovery cross-session handoff — G056 — 2026-07-19

## Resume identity

- Repository: `/home/hoky/Quants-agent/LuminaQuant`
- Branch: `recovery/strategy-plan-20260714`
- Implementation checkpoint: `5bdc3b378cc477a939428d4455462ba651dc024b` (`Checkpoint G056 acquisition and raw durability work`). It is incomplete.
- Handoff-document commit: not known yet. The commit containing these three files must descend from the implementation checkpoint.
- Authoritative Ultragoal session: `019f603a-0e73-7000-88a7-c94f42950c09`
- Durable files: `.gjc/_session-019f603a-0e73-7000-88a7-c94f42950c09/ultragoal/{brief.md,goals.json,ledger.jsonl}`
- Machine state: [`strategy_recovery_resume_state_20260714.json`](strategy_recovery_resume_state_20260714.json)
- Copy-ready bootstrap: [`strategy_recovery_new_session_prompt_20260714.md`](strategy_recovery_new_session_prompt_20260714.md)

Stable aggregate objective:

> Complete the durable ultragoal plan in `.gjc/ultragoal/goals.json`, including later accepted/appended stories, under the original brief constraints; use `.gjc/ultragoal/ledger.jsonl` as the audit trail.

The three documents above are the portable bootstrap. `.gjc` remains the canonical same-machine state; `ledger.jsonl` remains the audit trail.

## Durable status and seals

Durable G056 is `active` and incomplete. The inline aggregate goal is paused for cross-session transfer. No completion claim is authorized.

Goal counts: pending 0, active 1, complete 10, failed 0, blocked 2, review_blocked 14, superseded 29. G036 and G037 are blocked. Review-blocked predecessors are G035, G038–G047, G049, G052, and G053. Completed replacements include G050, G051, and G055.

Latest durable event: `human_blocked` `44a0ea42-0861-41b6-a74d-f18c04bfd507` at `2026-07-19T13:53:05.860Z`.

| Durable file | SHA-256 | Bytes | Lines |
|---|---|---:|---:|
| `brief.md` | `faf6f83679e7ce93a8950af4df350fc4a92557d8eaaa40ea17c9c8b918c04e57` | 4836 | 22 |
| `goals.json` | `ab9119c28c1ca8e7822b5dbe21cb33888063b749ed9b525130025391eb580c17` | 132956 | 1256 |
| `ledger.jsonl` | `3af47660386c60e8e104022a3f6ea66b239b7a97458e39cf0aca6fed79acdc32` | 283276 | 321 |

## Stop state

Every detached worker and monitor is terminal. No acquirer process or network acquisition is running. V4 never started. There is no eligible receipt. No phase, prelock, historical, candidate, forward, order, or capital operation ran. Failed v2 and v3 roots are immutable.

V3 source: `/home/hoky/quants-external-data/alpha-max-g056-full-source-v3`
V3 report: `/home/hoky/quants-recovery-runs/G056-full-acquisition-report-20260719-v3`
V3 controller: `/home/hoky/quants-recovery-runs/G056-full-acquisition-controller-20260719-v3`

V3 stopped at raw 134/415 through BTCUSDT 2023-04 and funding 3834/12347 through BNBUSDT 2026-06-30. Its exact failure is `archive_trade_order_invalid`; no eligible receipt exists. Preserve descriptor sequence-5 receipt `/home/hoky/quants-recovery-runs/G056-full-acquisition-controller-20260719-v3/descriptor-refresh-v3-state-005.json`, SHA `47b6e8b9006718e0dad043be85b9a86757aa2e0e39de05528a9107b192589321`.

The official blocker is BTCUSDT 2023-05: `https://data.binance.vision/data/futures/um/monthly/aggTrades/BTCUSDT/BTCUSDT-aggTrades-2023-05.zip`; SHA `301acec76a7644aa73180fd7f8d913ce4eecfa7e7bca5057f1782f96d91b9ef0`, 468405603 bytes, 35,641,068 rows, aggregate IDs `1715206948..1751963960`, 1,357,200 ID regressions and 1,357,200 timestamp regressions in archive order, timestamps `1682899200101..1685577599904`. Official bytes are valid but interleaved.

## Committed but unverified remediation

- acquirer: `0a8417e3d8fb36ba1c6863c328d819ac32211ffeb81c9bad2f0f77a46e897503`
- acquirer tests: `2ca7a73e2b330fdc194cc84087c2bf2db4cdf4c773363bf99af50e98c3a2134c`
- wrapper: `2c51d6b69b695cdbdcdf61fe35394e342673bb4329b355021b1a81a051122268`
- wrapper tests: `7b711ecc998d28628d4c5956f46002258d03da615ef071e8e4b7a0b73723aeb5`
- data_sync: `f092cd2a30928e985da2433df15c0b26e0616202cbca8991ae146357203ae59c`
- ohlcv_repo: `61d531b79af3182f1ac479af0bfe141c7a71b7211671e4bc5a2a5bb5e9c44536`
- collect tests: `d96111bd756e3a68b23f08666b1212f660811d07a0719bae5c36f9171497ab71`
- WAL tests: `35b257edb8e0fc0ff91960aadf128d97a40f4cf972913a0578e30ede33cbaae6`

Implemented remediation is derivation v4; a singular exact known-artifact gate; stdlib packed 32-byte `>qqdd` external merge by aggregate ID; 250k chunks; fan-in 64; duplicate and post-sort timestamp rejection; retained ordinary streaming branch; forwarded `validate_complete` receipt; exclusive verify lock; and adversarial tests.

Unresolved: Ruff first failed SIM115, then an ExitStack fix landed but has not been rerun; the wrapper pin is stale and must be repinned after source freeze; source uses `.order-*` session directories while architect contract 414 required owned singly-linked mode-0600 `.aggtrades-*` scratch; review same-opened-inode authentication-to-parse binding; no real-artifact two-run benchmark; no ordered real-archive byte regression; and no post-change cleaner, architect, or executor QA.

## Exact remaining order

1. Resume exact durable session and inline goal.
2. Close scratch and same-open authentication review; run Ruff and focused tests via `uv`.
3. Run a real allowed-v3-provenance artifact two-run deterministic/bounded benchmark and the already-ordered BTCUSDT 2024-01 byte regression; admitted output SHA `ac99f6439d544901db0c09d8d2e7ad7ffcd9d328a2de73e7930231a849e2b1e8`.
4. Freeze acquirer, repin wrapper/tests, and run focused, integrated, and full gates.
5. Run mandatory cleaner, then fresh architect CLEAR/CLEAR/CLEAR APPROVE and executor QA/red-team.
6. Run fresh v4 complete acquisition with periodic repository descriptors, `validate_complete`, and offline verify-eligible.
7. Run six authenticated phase roots and exact one-touch runbook fences 4–11 after wrapper.
8. Create strict G056 checkpoint, then explicitly supersede predecessors.
9. Run G036 C-00..C-05/C-06 candidate contracts and execution.
10. Run G037 fresh-forward.
11. Perform current-state final aggregate audit/receipt/goal complete.

After approval only, use fresh v4 roots: source `/home/hoky/quants-external-data/alpha-max-g056-full-source-v4`, report `/home/hoky/quants-recovery-runs/G056-full-acquisition-report-20260719-v4`, controller `/home/hoky/quants-recovery-runs/G056-full-acquisition-controller-20260719-v4`. Never mutate, reuse, or copy into v2 or v3; advance again after any code change following a started v4.

## Binding safety contract

The lexical strings `/home/hoky/Quants-agent/LuminaQuant-data/alpha_max_20260711_listing_aware_source` and `/home/hoky/Quants-agent/Quants-agent-alpha-max-data-pc` are quarantined. Never inspect, stat, hash, traverse, read, copy, or use them; they may appear only as `--forbidden-root` argv.

Use official-only data. No synthetic/substitute/prelisting/date shift/retune/locked-OOS-selection. Zero order and capital. Use `uv` Python and profile-first only. Rust/native is permitted only after a material exact-equivalence benchmark. Never reset, rebase, amend, or push.
