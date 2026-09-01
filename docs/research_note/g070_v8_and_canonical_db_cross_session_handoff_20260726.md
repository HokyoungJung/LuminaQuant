# G070 v8 acquisition and canonical DB cross-session handoff (2026-07-26)

## Authoritative resume update (2026-07-28)

This section supersedes the older stop-state, hashes, and continuation status below. The historical sections remain for audit context.

### Terminal G073 result (2026-07-28)

- The storage-bounded acquisition and atomic shared-root publication implementation is frozen at commit `3b0cc6be253e29ae3ca11fbb92a456bafa0eee8a`. Its final verification passed 158 compact-acquirer tests, 340 control/policy tests plus 41 subtests, 19 publication/funding/WAL tests, 171 broader storage tests, Ruff, format, and `git diff --check`.
- Compact capacity was sufficient only for a fail-closed attempt, not guaranteed completion: the final audit kept the 20 GiB `/mnt/c` reserve, one live ZIP, per-write admission, temporary cleanup, and double-derivation retirement. Audit v3 SHA-256 is `ed4d5d59e061d700321ae6b3b8d1de0a981cd253c0f5a337e38da6b059c6e971`.
- Fresh run `cd0f446dd0e7a47c2698538a528ec30142d536da87fd9b0c532e9479cabebd86` failed before authority startup because transient `systemd-run` escaped the generated `%d` credential specifier. Observer, telemetry, network acquisition, source, report, and canonical publication never started. Failure receipt SHA-256: `f11a99aa4512f39daccbc40f209dcb3038d865bb6033e362580f0d2aad00b3f5`; the roots are terminal and reuse-forbidden.
- One explicitly authorized replacement used statically rendered and `systemd-analyze`-verified units with literal `%d`, `UMask=0077`, exact cgroup/sandbox directives, and per-role credentials. Run `85be5b266630adc6105456f7c3bbfd8a72630f5e1a4f25b6f3492687c5189b71` nevertheless failed in systemd credential setup with status `243/CREDENTIALS` and `ENOENT`, before `ExecStart`.
- Replacement terminal receipt SHA-256 is `3d9de04c55efb3a413daa5446925113f243ac5fee6f85e43ec17ff4861244d22`. It proves: no authority socket, no signed terminal receipt, no network acquisition, source/report absent, telemetry empty, phase/one-touch/publication absent, canonical root unchanged, and no active or linked Alpha-Max v8 units after containment.
- The independent replacement authorization explicitly prohibited another retry or review loop. In addition, canonical publication remains blocked by independent HIGH findings on prepared-resume link-count assumptions, merge reserve accounting, long-lived reader generation pinning, read-side swap races, and WAL writers outside the generation lock.
- Final data status: **zero fresh v8 rows collected and zero canonical DB writes**. G073 is terminal-blocked. Continuing requires a new user-authorized story that first resolves and proves the systemd credential-launch contract and the publication HIGH findings; this session must not retry, reuse either failed run, or claim collection success.

### Durable state and bounded execution

- Durable session remains `019f603a-0e73-7000-88a7-c94f42950c09`.
- `G071` and `G072` are complete. `G073` is the only active/final story. Do not create more goals; use ledger annotations for runtime facts.
- V6 and v7 remain crash-terminal/ineligible and must stay byte-for-byte preserved and unread.
- No fresh v8 network acquisition has started and G073 has made zero writes to `data/market_parquet`.
- The only attempted fresh control generation was run `d999226083feb7431cfabcd4e4405e22abb0622cdd83b6190a6940fb1385e5e4`; it failed before launch because the request prerequisites had extra `st_uid`/`st_gid` fields. Its five roots were quarantined, `COMPLETE.json` is absent, and no network/service/source/report action occurred. The schema defect was fixed at commit `d6d79788221951d3ddf21f614cf863054d973497`.
- WSL containment is active: `init.scope` has `MemoryHigh=5G`, `MemoryMax=7G`, `MemorySwapMax=2G`; `gjc-memory-guard-4933-v2.service` is active. Heavy verification uses separate user-systemd cgroups with zero swap.
- Do not iterate review stories indefinitely. One bounded implementation/review pass plus one concrete repair pass is the operational ceiling; unresolved safety failures are recorded as failures rather than hidden behind new goals.

### Capacity and actual data state

- Durable audit: `/home/hoky/quants-recovery-runs/g065-oom-safety-20260726/g073-data-capacity-audit-v1.json`, SHA-256 `a44429445c1031fe527b7926a7790818ce8b8630685ab74a47b16120ba7169d0`.
- Canonical `data/market_parquet` logical bytes: `26,909,667,462`; topology audit: 48,136 regular files, 45,865 directories, zero symlinks/special files, all files `nlink=1`.
- Existing presence is not source eligibility: 182 of 415 required symbol-months exist; 233 are missing. Eight symbols have 18/43 months, BTC has 20/43, and TON has 18/28.
- Windows C free space has fluctuated around 35-38 GB; the hard emergency reserve is 20 GiB and `/mnt/c` is monitored. The Ubuntu VHDX is non-sparse, about 350 GB, so WSL logical free space is not the controlling capacity signal.
- Safe cache cleanup removed 28.7 GiB from the uv cache and cleaned npm cache without touching data, venvs, Git, or v6/v7. Those freed ext4 blocks are reusable inside the non-sparse VHDX even though Windows host free space did not immediately increase.
- Full archive retention is prohibited at current capacity. Acquisition now permits at most one live official ZIP, performs two independent derivations, durably records derivation/retirement/deletion evidence, deletes the ZIP body, and enforces the 20 GiB host reserve before writes.

### Integrated acquisition and shared-DB contract

- Acquisition schemas are now plan v4, source manifest v5, eligible receipt v4, and partition receipt v2.
- Manifest and eligible receipt bind the same canonical archive-evidence digest. Offline resume authenticates retained checksum/request/derivation/retirement/deletion evidence and never needs retired ZIP bodies.
- The permanent target is still the existing `data/market_parquet`; there is no permanent Alpha-Max database.
- `publish_alpha_max_eligible_source.py` accepts only a native-verified signed `SUCCEEDED` acquisition receipt and exact v5/v4 source, contract/listing, and partition evidence.
- Publication builds a hidden same-filesystem generation. Existing canonical files are hardlink-cloned without duplicating data; signed source partitions are independently copied/merged one month/day at a time. OHLCV conflicts are checked by canonical timestamp/value, funding preserves unrelated feature columns and rejects differing non-null values, and listing records are committed with the same generation.
- The visibility point is one Linux `renameat2(RENAME_EXCHANGE)` of the top-level `data/market_parquet` pathname to a trusted sibling generation symlink. Direct-path and repository consumers therefore resolve either the complete old root or complete new root, never a per-partition prefix. Post-exchange validation rolls back on failure; prepared candidates resume; an already-active exact generation is idempotent.
- Accepted 1-second data remains the common base. Existing UTC-aligned loaders derive 1m, 30m, and higher intervals. Nothing synthesizes 1-second data from coarse bars.
- Reflink/FICLONE was probed and is unsupported (`errno 95`). Directory↔symlink `RENAME_EXCHANGE` was live-proven. The hardlink generation is capacity-safe because old roots are removed only after exact activation readback, returning active files to `nlink=1`.
- A crash during an incomplete generation before `prepared.json` is fail-closed and requires bounded operator recovery of that transaction-owned candidate; it cannot expose partial canonical data.

### Current frozen role hashes before commit

- terminal policy JSON: `10065c762baa9281015715f5db1305dcc2175637ee909a5b91037a3f2451495c`
- acquirer: `12737ea26e55166d82518f70e9cff249c46b2a77c02a5d323650d8c77219e365`
- phase wrapper: `125a3e6d08debd6d2a7ce3d414e33807586265b3a4153349d2bc4a1f1e6aaa47`
- v8 control builder: `0a19901838833500173110e0db3035264bb4a310c99945bd86e80a06b7fc63cd`
- terminal policy module: `0c3c5da07ca446cab160bfcd053ce2d890e8cafb55e8be22b0812020f46e3523`
- canonical publisher: `b1ab3352843b6546c9b030834464229f6bd2791eed37bd55536c39ddd5763c71`
- market-data facade: `049ab7431d967c0ef7f817930226d445d06b42144dd9b598ae5f0ce146435fef`
- parquet repository: `783580b5501de27886ccd890a075c14f2fd3e648a599c6ebf5e7e4d177ec5313`

### Current verification

All commands ran in bounded user-systemd services with `UMask=0077` and `MemorySwapMax=0`:

- compact acquirer: 158 passed;
- terminal policy, external envelope, v8 controls/telemetry, and phase wrapper: 340 passed plus 41 subtests;
- canonical generation/funding/WAL stack: 19 passed, including live tmp-root root exchange, injected post-exchange rollback, prepared resume, and idempotent replay;
- broader parquet/raw lineage and collector regressions: 171 passed;
- Ruff check passed; all 14 changed Python files are formatted.

### Remaining G073 sequence

1. Commit this frozen implementation and note.
2. Generate a new canonical current-state approval against that clean HEAD and obtain one independent bounded exact-hash review.
3. Generate exactly one fresh no-launch v8 control/key/evidence/telemetry/output package under new IDs and roots; inspect `COMPLETE.json`, native readback, unit properties, capacity contract, and source/report absence.
4. Launch only authority → telemetry → observer/acquirer under the existing cgroup limits. Monitor the signed terminal receipt and `/mnt/c`; do not launch phase preparation or one-touch.
5. After signed `SUCCEEDED` and native source eligibility, run the atomic shared-root publisher. Keep staging/report immutable through publication and verify all canonical targets before old-root cleanup.
6. Run the mandatory cleaner, full verification rerun, Architect review, Executor red-team QA, and terminal Critic once. Checkpoint G073 and complete the aggregate goal only after the durable final receipt exists.

## Stop boundary

- Work was stopped at `2026-07-26T14:43:17Z` (`2026-07-26T23:43:17+09:00`) at the user's request.
- Repository: `/home/hoky/Quants-agent/LuminaQuant`
- Branch: `recovery/strategy-plan-20260714`
- Pre-handoff HEAD: `1629ae2ba383acd48e95810b966cf591ab6f7e77`
- Durable GJC session: `019f603a-0e73-7000-88a7-c94f42950c09`
- No acquisition, authority, observer, or telemetry unit is running. No fresh v8 control/key/evidence/source/report root has been generated. No current-state approval has been created.
- All task subagents and background jobs are terminal. The only intentionally persistent service is the OOM guard `gjc-memory-guard-1298-v3.service` (`active/running`, `MemoryMax=67108864`, `MemorySwapMax=16777216`).
- Final service verification at `2026-07-26T14:49:07Z` found and stopped two leftover read-only lifecycle-test telemetry collectors, `g065-persist-collector.service` and `g065-persist-collector-v2.service`. They were not acquisition units. After stopping them, the only running user services are D-Bus and the intentional OOM guard.
- The latest source edits from tasks 63-65 are deliberately stopped **before verification**. Do not treat the current v8 builder/test hashes as reviewed or launch-authorized.

## Non-negotiable evidence and safety boundary

V6 and v7 are crash-terminal/ineligible. Preserve them byte-for-byte. Never inspect, reuse, copy, recover, relaunch, or compare private controls, keys, journals, or data from either generation. Freshness is established by new cryptographic IDs and exclusive roots, not by reading old values.

Lexically reject these roots before any syscall:

- `/home/hoky/Quants-agent/LuminaQuant-data/alpha_max_20260711_listing_aware_source`
- `/home/hoky/Quants-agent/Quants-agent-alpha-max-data-pc`

Keep every heavy command in a user systemd cgroup with explicit `MemoryHigh`, `MemoryMax`, `MemorySwapMax`, `OOMPolicy=kill`, and `UMask=0077`. A live probe confirmed that `OOMPolicy=kill` sets `memory.oom.group=1`; `MemoryOOMGroup` is not a supported transient property.

The prior WSL shutdown was caused by host OOM: Bun reached about 14.8 GiB RSS plus 8 GiB swap under the configured 16 GiB/8 GiB WSL limit. The current guard and per-command cgroups exist to stop recurrence.

## Durable Ultragoal state

Canonical files (not Git-tracked; retain the same session ID):

- `.gjc/_session-019f603a-0e73-7000-88a7-c94f42950c09/ultragoal/brief.md`
  - SHA-256 `faf6f83679e7ce93a8950af4df350fc4a92557d8eaaa40ea17c9c8b918c04e57`
- `.gjc/_session-019f603a-0e73-7000-88a7-c94f42950c09/ultragoal/goals.json`
  - SHA-256 `894ccfa56a1cefb265b2a869d836fe802b7897b57fdf419f8dac589186e46abc`
- `.gjc/_session-019f603a-0e73-7000-88a7-c94f42950c09/ultragoal/ledger.jsonl`
  - SHA-256 `fcc49d420f091066034fe3a510797a6f7d7d48eb3499c496d6187192e94ee28c`

State at stop:

- Aggregate status: `blocked`
- `G070` **review_blocked** — exact SOL identity patch is complete, but fresh v8 control generation/acquisition is not.
- `G071` **pending/current** — execute only verified policy/key-creator bytes and close authenticated-import TOCTOU.
- `G072` **pending** — isolate per-unit keys and pin reviewed interpreter/package/telemetry identities.
- `G066` **blocked** — common DB publication waits for a signed accepted v8 source.
- `G065`/v7 remains terminal/ineligible and preserved.

Do not manually edit `goals.json` or `ledger.jsonl`. Resume through `gjc ultragoal complete-goals`, normal checkpoint/steer commands, and the unified `goal` tool.

## Completed scientific and product work

### Official SOL diagnosis

The official Binance SOLUSDT November 2023 archive is approved only at this exact identity:

- URL `https://data.binance.vision/data/futures/um/monthly/aggTrades/SOLUSDT/SOLUSDT-aggTrades-2023-11.zip`
- archive SHA-256 `188c3145ecaab1cf546318c293fb4fef0e320a6dc05b14eea013a46209ebbd73`
- bytes `535864305`
- checksum URL: archive URL plus `.CHECKSUM`
- checksum SHA-256 `d1a92cf7d5775d5edd1960d75091c06af72955c99fb806dca4ccf670af983f9d`

G067 v3 read-only diagnostic evidence:

- report SHA-256 `3bdce0a8697e16059169abcc05449463d2e5dc3bdc8ec09831fb150b26f2cb9e`
- terminal cgroup SHA-256 `435f2ed7980f2db85639cf577a3144372cf2eac407717b6ee951418b2250cb0b`
- supplemental evidence `/home/hoky/quants-recovery-runs/g065-oom-safety-20260726/g067-diagnostic-v3-terminal-supplement-v1.json`
  - SHA-256 `2c35205431c4fb19272766bffeee7f1c8d59a0b2534150ffb419644d172816e2`
- exact contiguous IDs `426116753..468992821` (`42,876,069` rows)
- archive-order regressions `14,427,753`
- after exact aggregate-ID ordering: duplicate IDs `0`, ID gaps `0`, timestamp regressions `0`
- diagnostic peak `537,202,688` bytes, swap peak `0`, all max/OOM counters `0`

G069 algorithm boundary report:

- `/home/hoky/quants-recovery-runs/g065-oom-safety-20260726/g069-algorithm-boundary-report-v2.json`
- SHA-256 `8a48d1faa063e6ab9c6bdc42a7f20be050226b86a522b915e61f278ea0e7ea4`
- baseline plus 15 adversarial cases passed; G069 is complete.

### Exact product patch

Only the exact SOL identity was added to the existing canonical-order allowlist. No dates, listings, derivation, strategy behavior, official-only rule, synthesis rule, orders, or capital behavior changed.

Known frozen product hashes before the stopped tooling work:

- `scripts/research/acquire_alpha_max_official_source.py` — `300a85442f03db193efff9d7b7725aee0533bb75c8939821ae6548d30328bdbd`
- `tests/test_acquire_alpha_max_official_source.py` — `7c07982315a41342885a82700444e30ea12fc4a0ffcfc348b34411dd3b5891c7`
- `scripts/research/run_alpha_max_phase_preparation_from_eligible_source.py` — `406209fe76bd6fb49acf44029b9017fd4c2d9de0d60308f3c561c31704e93809`
- `configs/research/alpha_max_terminal_authority_policy_v1.json` — `4772e597826515e9318a25456e9a48bce6dfe23a98e0aa58df8f6b85c32e0376`
- `tests/test_alpha_max_terminal_policy.py` — `7dee16f348626f6c3a5b7d81367d3a3fc6789a3de3ee9ef9bdff9d8e20e97fc9`

Observed product verification: focused SOL `11 passed`; combined acquirer/policy/external-envelope `369 passed`; Ruff passed; Cleaner 39 PASS; Architect 40 CLEAR/CLEAR/CLEAR APPROVE; Executor 41 passed. Product test report:

- `/home/hoky/quants-recovery-runs/g065-oom-safety-20260726/g070-api-package-test-report.json`
- SHA-256 `dfa041c60bf68271aa45f0dacd201c9dfdf7457e839a613bacce16e6d097c11e`

## Accepted checkout and exact runtime pins

Accepted checkout:

- `/home/hoky/Quants-agent/LuminaQuant-alpha-max-fresh-20260718`
- HEAD `391000b40717386765bfa39bd212d91c2e3be794`
- baseline `629d91e5d4aac26911af65a4a5e15ebdcbded30f`
- was porcelain-clean at the last check

Approved alignment receipt:

- `/home/hoky/quants-recovery-runs/20260714T105113Z/alpha-max-rev515-alignment-receipt-v5.json`
- SHA-256 `8687b52180502a11de9fbe317a19d00bb4492c464b3bf33d4eda2437683ca812`
- 530 bytes, mode 0444

Pinned interpreters and package freezes:

- current Python `/home/hoky/Quants-agent/LuminaQuant/.venv-g056v8-current/bin/python-g056v8-current`
  - binary SHA-256 `a1512f9a07029c4a9b02a1bb63bbd156d36b0dcb26f49cb7f5ee175f19b222da`
  - bytes `32299584`, mode 0555
  - canonical freeze SHA-256 `3b8e4d900ddfc1bf05d65ff4fcf1eb6a04709dcc684ad8f4a49fe3bd4bba9724`, 32 normalized packages
- accepted Python `/home/hoky/Quants-agent/LuminaQuant-alpha-max-fresh-20260718/.venv-g056v8-accepted/bin/python-g056v8-accepted`
  - same binary identity
  - canonical freeze SHA-256 `df09a5a1d4d1ab657d6a11d28eaf00cea06df4d9e28c0ef81ec5382257d6abf6`, 29 normalized packages

A live descriptor-execution probe proved `subprocess.run(..., executable=/proc/self/fd/<fd>, pass_fds=(fd,))` preserves the current venv prefix, `sys.executable`, and Polars `1.35.2`.

## Stopped v8 tooling state (must be reverified)

Current unverified hashes at the stop boundary:

- `scripts/research/create_alpha_max_v8_acquisition_controls.py`
  - SHA-256 `1393fdd0b1203454e119f9f0926d4ce4373835fb680c84312d5a6063e937b0f4`
- `scripts/research/monitor_alpha_max_v8_resources.py`
  - SHA-256 `17db931cd816e8ab46bb1db9d62eb9b207b147cbac99475196197684c205b383`
- `tests/test_create_alpha_max_v8_acquisition_controls.py`
  - SHA-256 `eec9346a33273f9cefc259ed8f8f0f378092859710f2e23aa8c7e5dab5ae2a28`
- `tests/test_monitor_alpha_max_v8_resources.py`
  - SHA-256 `c5913c097f14d5f5e3dfb198099c9fc322c32131c5c0997169176ba63ab481ac`

Last verified revision **before** tasks 63-65: `g070-v8-tooling-verify-9.service` reported `30 passed`, Ruff clean, `2.0M` peak, zero swap. Since then:

1. Task 63 added exact interpreter/freeze/telemetry pins.
2. Task 64 replaced a failed same-path bind design with live-proven `InaccessiblePaths=<key_root>` plus one unit-scoped `LoadCredential`, and `%d/<credential>` argv.
3. Task 65 changed package-freeze execution to the verified open interpreter inode through `/proc/self/fd`, with FD/path revalidation.

Those three changes have **not** been formatted, tested, cleaner-reviewed, architect-reviewed, or QA-reviewed. The old tooling test report at `/home/hoky/quants-recovery-runs/g065-oom-safety-20260726/g070-v8-tooling-test-report-v1.json` is stale and must not be used for completion or launch.

A live user-systemd credential probe passed: with `InaccessiblePaths=<key_root>` and `LoadCredential=authority.private:<source>`, `$CREDENTIALS_DIRECTORY/authority.private` was readable while the original selected file and sibling key were unreadable. The earlier same-path `BindReadOnlyPaths` probe failed and must not be reintroduced.

No `g070-current-state-approval.json` exists. Generate it only after the four tooling files and the whole overlay are frozen and independently approved.

## Required continuation plan

### A. Resume and close G071/G072

1. Set the exact session ID and inspect status:

   ```bash
   export GJC_SESSION_ID=019f603a-0e73-7000-88a7-c94f42950c09
   cd /home/hoky/Quants-agent/LuminaQuant
   gjc ultragoal status --json
   gjc ultragoal complete-goals
   ```

2. Reformat and run only the focused tooling suite under a bounded transient user service. Use at most 1 GiB RAM and 256 MiB swap for this suite. Then run Ruff and format checks.
3. Re-run live authenticated policy load, both exact package freezes, descriptor-pinned interpreter execution, exact user-systemd directive probes, and per-unit `LoadCredential` negative sibling-key tests.
4. Re-run the internal Ultragoal `ai-slop-cleaner` fragment on the four tooling files until zero blockers.
5. Run frozen-snapshot Architect and Executor QA lanes. They must explicitly test:
   - exact-byte policy/key-creator execution under pathname replacement;
   - FD-pinned interpreter freeze under pathname replacement;
   - exact executable path/hash/freeze substitution rejection before root mutation;
   - one credential only per unit; original/sibling keys inaccessible;
   - concrete systemd directive allowlist and actual user-systemd support;
   - no launch, no source/report creation, quarantine/failure preservation;
   - pending versus fatal telemetry evidence and native signed-receipt binding.
6. Checkpoint G071 and G072 only with strict quality-gate JSON after their blocker objectives are proven. Then resume/reopen G070 through `gjc ultragoal complete-goals`.

### B. Freeze and generate v8 controls (no launch yet)

1. Freeze the full current overlay and create `/home/hoky/quants-recovery-runs/g065-oom-safety-20260726/g070-current-state-approval.json` with:
   - actual full HEAD;
   - raw porcelain v1 `-z` digest/count;
   - accepted-commit-to-HEAD binary diff digest/count;
   - HEAD-to-worktree binary diff digest/count;
   - canonical tracked+untracked path/type/mode/size/content inventory digest/count.
2. Obtain an independent exact-hash review of the approval, all current files, interpreters/freezes, and telemetry pin.
3. Generate independent random 64-hex run and request IDs. Run the builder under a bounded cgroup using only exact sibling roots:
   - `g056v8-controls-<run>`
   - `g056v8-keys-<run>`
   - `g056v8-acquisition-evidence-<run>`
   - `g056v8-telemetry-<run>`
   - `g056v8-acquisition-output-<run>`
4. Require `COMPLETE.json`, no `FAILED.json`, exact native loader/command readback, four fresh distinct key IDs, source/report still absent, and no service/network action.
5. Independently review generated hashes and live-render every systemd directive. Do not launch on WATCH/BLOCK or stale review.

### C. Launch only the acquisition scope

Start in this order only after exact-hash authorization:

1. authority: High 256 MiB, Max 512 MiB, SwapMax 64 MiB;
2. telemetry: Max 128 MiB, SwapMax 32 MiB;
3. observer/acquirer: High 2 GiB, Max 3 GiB, SwapMax 512 MiB.

All use `OOMPolicy=kill`, `UMask=0077`, no privilege escalation, private tmp/devices, protected system/home, and exact read/write roots. Authority and telemetry have no IP network. Observer alone permits AF_UNIX/AF_INET/AF_INET6. Each service receives only its one credential via `LoadCredential`; the key root is inaccessible. Launch acquisition only—never phase preparation or one-touch.

Terminal acceptance requires both cgroups gone; both ExecStopPost captures canonical and `success/exited/0`; `memory.oom.group=1`; max/OOM counters zero; peaks strictly below limits; native signed receipt verifies exact request and `SUCCEEDED`; command 0 execute+validate-complete and command 1 verify-eligible both return 0; source/report are complete.

### D. Canonical shared DB integration (G066)

Do not write the shared DB until a signed accepted v8 source exists. The permanent target is the existing repository-wide root `data/market_parquet`, not a separate Alpha-Max database.

Implementation owner and constraints:

- actual repository owner: `src/lumina_quant/storage/parquet/ohlcv_repo.py` and existing `ParquetMarketDataRepository` / `load_data_dict_from_parquet` surfaces;
- keep the v8 source root immutable as staging/provenance only;
- add a thin accepted-source publisher, not a second storage/loader/resampler stack;
- one generic same-filesystem atomic bundle transaction/catalog generation for OHLCV plus feature/funding records;
- exact duplicate rows are no-ops; any differing OHLCV value at the canonical key aborts the whole bundle; differing non-null feature values abort unless a separately signed policy authorizes replacement;
- publish only a complete signed source; no partial symbol/month promotion;
- common loaders read the committed generation and fail closed; no fallback to staging;
- use accepted 1-second data as the common base. Derive 1m, 30m, and higher by the existing UTC-aligned downsampler (first open, max high, min low, last close, sum volume, established gap/partial-bucket policy);
- never synthesize 1-second data from 1m or any coarser timeframe;
- profile fixed snapshots before optimization; preserve exact schema/dtype/value/order/null semantics; use predicate/projection pushdown, month/day pruning, chunked streaming, and provenance-bound materialization only when measured;
- atomic visibility/rollback tests must prove readers see the complete old or complete new generation, never a mixture.

## New-session prompt

Use the prompt below verbatim after opening the new session:

```text
Resume durable Ultragoal session 019f603a-0e73-7000-88a7-c94f42950c09 in /home/hoky/Quants-agent/LuminaQuant on branch recovery/strategy-plan-20260714. First read docs/research_note/g070_v8_and_canonical_db_cross_session_handoff_20260726.md and verify the handoff commit/worktree plus the three durable files under .gjc/_session-019f603a-0e73-7000-88a7-c94f42950c09/ultragoal. Invoke /skill:ultragoal and resume through `GJC_SESSION_ID=019f603a-0e73-7000-88a7-c94f42950c09 gjc ultragoal status --json` and `gjc ultragoal complete-goals`—do not create a new plan or session. Work was intentionally stopped with G070 review_blocked and G071/G072 pending. No v8 controls/keys/units/source/report exist, and the current task-63/64/65 builder/test edits are unverified. Preserve v6/v7 byte-for-byte and never inspect/reuse their artifacts or the two forbidden roots. Keep every heavy command OOM-bounded with user-systemd memory/swap limits and OOMPolicy=kill. Verify, clean, architect-review, and executor-red-team the current exact-byte authenticated loader, pinned interpreter/freeze/telemetry identities, descriptor-pinned freeze execution, and per-unit LoadCredential key isolation; checkpoint G071 and G072 strictly; then reopen G070. Only after a frozen current-state approval and independent exact-hash launch review, generate wholly fresh v8 roots and launch acquisition only. After a signed accepted v8 source exists, unblock G066 and atomically publish into the existing shared data/market_parquet repository so all strategies use the same committed 1s base and existing 1m/30m+ downsampling; never create a permanent Alpha-Max DB or synthesize 1s from coarse bars. Continue autonomously to completion unless a genuine human-only blocker exists.
```
