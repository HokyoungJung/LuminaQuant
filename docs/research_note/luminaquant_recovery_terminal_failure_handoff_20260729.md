# LuminaQuant recovery terminal-failure handoff — 2026-07-29

## Outcome

The approved replacement recovery used one fresh run and stopped at its signed no-retry boundary.

- Branch: `recovery/strategy-plan-20260714`
- Validated recovery implementation commit: `68bcafd755b9fe00c774fb4d9637829c45e85b46`
- Fresh run ID: `fa7feb0cd6901880aa36c461871c7be201dbd8b9472877620a1d235b64b1aa5f`
- Fresh request ID: `5e5b4667a769b17d5aa91ff2a9e7ac100e827bcd3222da009bbe26a50f096b1e`
- Terminal state: `FAILED_NO_RETRY`
- Failure stage: acquirer root preflight, before output-root creation and archive acquisition
- Primary error: `acquire_alpha_max_official_source: unsafe_root_parent`
- Secondary terminal telemetry error: `resource evidence error: cannot open proc cgroup: Not a directory`
- Network archive acquisition: 0
- Source/report rows written: 0
- Canonical mutation: 0
- Publication: not started

The earlier G073 blocked story and all of its evidence remain unchanged.

## Completed implementation and validation

Commit `68bcafd7` fixes the static systemd credential path and hardens the atomic generation protocol:

- static, manifest-bound systemd units with literal `%d`, `LoadCredential`, `UMask=0077`, sandboxing, and cgroup limits;
- live non-network effective `FragmentPath`/`ExecStart`/credential/resource readback;
- shared-reader/exclusive-writer generation locking and whole-request generation pinning;
- fail-closed corrupt parquet/manifest handling;
- path-confined materialized manifest files and required generation metadata validation;
- immutable publication source pins and enforced serialized-output quotas;
- durable generation/transaction directory publication;
- reader-rejected bootstrap generations, one top-level `RENAME_EXCHANGE`, authenticated resume, and subset-idempotent predecessor cleanup;
- exact, path-qualified coordination-lock omission and predecessor-detachment authorization.

Observed checks:

- live systemd credential probe: PASS;
- related recovery/storage suite: 81 passed;
- raw/WAL/manifest suites: 272 passed;
- added publisher/manifest safety suite: 26 passed;
- Ruff format/check: PASS;
- `py_compile`: PASS;
- `git diff --check`: PASS;
- independent terminal critic: `PASS_FOR_ONE_FAIL_CLOSED_PILOT`, publication ineligible.

## Capacity and runtime state

- Exact reserve: `21,474,836,480` bytes.
- Admitted prelaunch Windows C: free space: `25,758,502,912` bytes.
- Capacity-monitor low-water mark: `19,547,324,416` bytes.
- Final Windows C: free space after cleanup: `25,754,058,752` bytes.
- Completion capacity was not guaranteed; canonical publication remained forbidden.
- Production memory peak: authority `29,913,088` bytes; observer `10,166,272` bytes.
- Production swap peak: `0` bytes.
- OOM events/kills: `0`.
- Preserved OOM guard: `MemoryHigh=5G`, `MemoryMax=7G`, `MemorySwapMax=2G`.

## Canonical state

The failed observer never received canonical write access and exited before creating source/report roots.

- Existing OHLCV rows remain the previously verified `467,877,392`.
- Observed non-null canonical funding rows: `21,011`.
- Acquired/integrated symbol-month list: empty.
- Canonical root identity before and after: device `2096`, inode `195868`, mode `16877`, nlink `7`.
- Canonical identity digest before and after: `769e65708b4c8f534fbfe1c1358bcb879e416345b15b1fda645a486244fd4240`.
- Atomic publication state: not attempted; the original direct canonical root remains active.

## Durable evidence

| Evidence | SHA-256 |
|---|---|
| `/home/hoky/quants-recovery-runs/luminaquant-recovery-631242a65e5d9732/current-state-approval-v1.json` | `aa2da9b547c9f1e6540719c6aed5df6d60f8429f4f47acb6d2d331c6b122d419` |
| `/home/hoky/quants-recovery-runs/luminaquant-recovery-631242a65e5d9732/systemd-probe-hardened-pass-2d260fd00ab837c8/terminal-probe.json` | `b005b59fde4e05298790c69b621dcb91bd27b773a65f849b6e38f03cfb5ca54f` |
| `/home/hoky/quants-recovery-runs/luminaquant-recovery-631242a65e5d9732/fresh-control-build-v1.json` | `85495de5807428fd76420566d170c702ce150770819938bfe1dd730b404871c1` |
| `/home/hoky/quants-recovery-runs/luminaquant-recovery-631242a65e5d9732/fresh-control-readback-v1.json` | `1bef56b5e979a1aadab7f15e7920b2b0dc2aa49d41b58a22d7a83fa0f7741819` |
| `/home/hoky/quants-recovery-runs/luminaquant-recovery-631242a65e5d9732/terminal-critic-v1.json` | `d72091e8732f873ebe7ffe6cc03e13a34bad1b2def23fa8ea88f4ae42b6e1bcb` |
| `/home/hoky/quants-recovery-runs/luminaquant-recovery-631242a65e5d9732/capacity-audit-v3.json` | `2425d6ca42f419a68dfa93bcac1f6862abaf910bf2c846587bb731edf87884b2` |
| `/home/hoky/quants-recovery-runs/luminaquant-recovery-631242a65e5d9732/production-prelaunch-v1.json` | `0c86ff11b5e440cb67a3ffb872854c8c8f44c153065ba89e2c4ca26568e29228` |
| `/home/hoky/quants-recovery-runs/g056v8-acquisition-evidence-fa7feb0cd6901880aa36c461871c7be201dbd8b9472877620a1d235b64b1aa5f/terminal-authority.receipt.json` | `ea0871405c45c8e7d7cbe9e09b05dcc9ccc12f0924eb9a8f8653b26a78a6e1db` |
| `/home/hoky/quants-recovery-runs/g056v8-acquisition-evidence-fa7feb0cd6901880aa36c461871c7be201dbd8b9472877620a1d235b64b1aa5f/terminal-observer.journal.jsonl` | `7c995609407972e03bf0fee6210f97927cd594c6f738c2245010601101ec7eee` |
| `/home/hoky/quants-recovery-runs/luminaquant-recovery-631242a65e5d9732/production-terminal-failure-v1.json` | `a4eef66196ea1dec1be2c043a216c86cdf7358a1311bce4d697582329c58f0a1` |
| `/home/hoky/quants-recovery-runs/luminaquant-recovery-631242a65e5d9732/failure-reconciliation-v1.json` | `08beefb8f6e6ad4c53d824eece1e999824a172952cb256a3073b265b98ddc33b` |
| `/home/hoky/quants-recovery-runs/luminaquant-recovery-631242a65e5d9732/cleanup-v1.json` | `114b619f3a4806e9aa7c62ad782c6d3c8a33d99fe3233d6094a8a03306c985f0` |
| `/home/hoky/quants-recovery-runs/luminaquant-recovery-631242a65e5d9732/windows-c-capacity.jsonl` | `f58ab706035b94a3ba1f8d0424430d5f298bee976845bc24d579aa138a4bce5c` |

## Cleanup and continuation boundary

Fresh static-unit links, failed states, authority socket, private credential root, temporary capacity monitor, and test scratch were removed. Control, public summaries, signed terminal evidence, telemetry, OOM guard, canonical data, and prior recovery evidence were preserved.

Do not resume or reuse this run. A further acquisition attempt requires explicit new authorization, a completely fresh run/request/credential/control/output topology, a fix for the `unsafe_root_parent` preflight failure and secondary `/proc/.../cgroup` telemetry defect, repeated synthetic sandbox verification, and a new capacity gate. Publication remains prohibited.

## Renewed authorization and root-cause repair

The user subsequently gave explicit authorization for one completely fresh run and completion of acquisition, publication, and existing-framework integration. The terminal `fa7feb0c…` run remains immutable and is not resumed or reused.

The primary failure was reproduced inside the production-equivalent user-manager mount namespace. Host-root-owned `/` and `/home` are reported there with overflow UID `65534`; the acquirer rejected that namespace mapping before opening the UID-1000 mode-0700 immediate output parent. The repair binds non-immediate ancestor acceptance to the UID observed on the already-opened namespace root while retaining no-follow traversal and the strict current-user/non-writable immediate-parent rule. The telemetry failure was independently reproduced as `/proc/self` being a symlink rejected by `O_NOFOLLOW`; terminal capture now opens `/proc/<current-pid>/cgroup`.

Post-repair evidence includes 176 passing acquirer/telemetry tests, a production-equivalent sandbox preflight PASS, a live terminal cgroup capture PASS, and a fresh live systemd credential/ExecStart probe PASS. This document remains the immutable handoff for the first failed run; subsequent run evidence is recorded separately.
