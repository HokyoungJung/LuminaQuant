# G060/G061 repository-native terminal-authority resume plan — 2026-07-22

## Status and authority

This document materializes the cross-session G060/G061 plan and typed contract inside the repository. A future session must not depend on `agent://` artifacts from the stopped session.

- Repository: `/home/hoky/Quants-agent/LuminaQuant`
- Branch: `recovery/strategy-plan-20260714`
- Implementation checkpoint: `475f3f2ebe37994f574dc970e1b3fa9563da8009`
- Durable Ultragoal session: `019f603a-0e73-7000-88a7-c94f42950c09`
- Current durable story: G061, active/incomplete; G060 remains `review_blocked`.
- Inline aggregate goal: paused solely for the user-owned session transition.
- Latest durable blocker classification: event `7523ed6e-93fb-41f8-a937-18897ac3de8f`, `human_blocked`.
- No implementation in this checkpoint is execution authority. No v5 root may be created and no target may run until G061 receives a fresh clean cleaner, architect, and executor QA/red-team gate and a strict durable checkpoint.
- Never run `create-goals` or `complete-goals`. Existing durable state and the recovery master plan/runbook are canonical.

G061 objective, verbatim:

> Replace the rejected delegated integration with one interoperable typed wire contract; bind request identities to the one-direction external manifest and real files; prevalidate roots and keys before the first O_EXCL mutation; enforce exact launch/start/exit/recovery state, semantic child-0 gates, canonical acquisition/A-02/sealed one-touch artifacts, immutable before-after snapshots, no-bytecode bounded cwd, secure key provenance, and no-launch recovery; rerun the full G060 cleaner/review/QA gate before any external root or target execution.

## Immutable safety constraints

Never inspect, stat, hash, traverse, read, copy, or use either quarantined root. They may occur only as lexical forbidden-root strings:

1. `/home/hoky/Quants-agent/LuminaQuant-data/alpha_max_20260711_listing_aware_source`
2. `/home/hoky/Quants-agent/Quants-agent-alpha-max-data-pc`

Also preserve:

- official-only data; no synthesis, substitution, prelisting, date shifts, retuning, or locked-OOS selection;
- zero exchange orders and zero capital;
- Python through `uv` only;
- profile-first optimization; native/Rust only after a material exact-equivalence benchmark;
- no reset, rebase, amend, or push;
- V4 as immutable failed/ineligible evidence only;
- no reading or execution of rejected external terminal drafts;
- no acquisition, wrapper, phase, prelock, historical, or one-touch target execution during G061 verification.

Frozen target identities:

- acquirer: `b440d79899a4ed60e18decfcd8bc2656d2de012189f03572a8be65f90cd24978`
- phase wrapper: `054163d23e8d2f1446b225e281472bcc563ac76f06aa47552cc5f3953b7c4dd9`

## Why G061 exists

Architect 67 blocked G060 for twelve source-level integration defects: incompatible authority/observer wire formats; impossible publication/preflight types; request-selected target identities; circular or unverified manifest authority; missing child-start and semantic continuation gates; shallow acquisition verification; fictitious A-02 semantics; invalid sealed schemas; fake before/after evidence; incompatible crash recovery; inherited cwd/bytecode plus missing target identity checks; and insecure key creation/reads.

G061 is the explicit review-blocker replacement. G060 must remain non-terminal until G061 is independently clean. This is safety and scientific-validity work, not performance tuning.

## Frozen policy inputs

Accepted Alpha repository execution HEAD:

`391000b40717386765bfa39bd212d91c2e3be794`

Baseline ancestor only:

`629d91e5d4aac26911af65a4a5e15ebdcbded30f`

Exact policy pins:

| Binding | SHA-256 |
|---|---|
| runbook | `249694fb1513354d61f67552f5c1b9175382f3c2bf9f271ee64dc0358d3c663f` |
| Alpha `uv.lock` | `59d9de230be950761736c24e04af3456e229cf4aa077536167fb7e650a71c339` |
| alignment receipt | `8687b52180502a11de9fbe317a19d00bb4492c464b3bf33d4eda2437683ca812` |
| portfolio | `2f267451c4df6b6b7471d972b7756327e41c82522ae2ef4b9198fbf6aa8b5e9c` |
| contract | `ae272f70f65797b4c8a87c29b7f8e64511617f8e0f2d4bd841b2d1addb7d1220` |
| availability | `214e5da198307d8d32b30f69fb6b1f09002e0b31888dc476ed16060f79de9719` |
| preparer | `ea26b902bcec4458340e4c345fa648a3db9104e1b337fd42460d9a9461a738ac` |
| prelock | `838d633ae34d44443dad4990a79f4d8caa95f7102ffe2a649ed341b1bed16ad0` |
| historical | `951290033c7efd9b59ba5418e38d96fbdcf3885211915b29010b79ae545f3fb0` |
| process boundary | `f95e8e0d356ca36063a415a7b37919e72d9d1f47af7d2c447e228546fddfb94c` |
| acquirer | `b440d79899a4ed60e18decfcd8bc2656d2de012189f03572a8be65f90cd24978` |
| phase wrapper | `054163d23e8d2f1446b225e281472bcc563ac76f06aa47552cc5f3953b7c4dd9` |

The checkpoint is the one-way trust anchor: `checkpoint.authority_manifest_sha256` equals the SHA-256 of the canonical external launch-envelope bytes. The envelope contains no self-hash. Current-repository HEAD is not hardcoded in policy; the envelope and external clean receipt pin the final current HEAD, which must descend from the historical implementation/handoff ancestry.

## Trust and identity boundary

The terminal pipeline has one fully trusted leader UID. Distinct authority and observer keys separate signed claims for audit, but they do not create operating-system isolation between processes running as that same UID. The pinned acquisition and phase-preparation child scripts are trusted derivation components; policy authenticates their fixed identities, complete artifact relationships, and published coverage, but this design makes no sandbox claim.

Any previously signed policy, authority, or observer identity and its receipts are historical evidence only. Any future control-plane byte change requires a fresh external v3 launch envelope and v1 checkpoint that pin the new identities while retaining the existing target and runbook pins.

## Typed closed schemas

All JSON is UTF-8, sorted-key, compact-separator, `allow_nan=False`, LF-terminated canonical JSON. Missing, extra, duplicate, non-canonical, unsafe path, non-finite, and wrong-type values fail closed.

Core identities:

```text
DirectoryIdentity {path,st_dev,st_ino,st_uid,st_gid,mode}
FileIdentity {path,sha256,byte_count,st_dev,st_ino,st_uid,st_gid,mode,nlink}
AbsentOutput {path,parent:DirectoryIdentity,leaf,must_be_absent:true}
Environment {HOME,LANG,LC_ALL,PATH,PYTHONHASHSEED,PYTHONNOUSERSITE,PYTHONDONTWRITEBYTECODE,TZ}
PublicationPaths {claim,journal,stdout,stderr,receipt}
PrerequisiteRecord {kind,path,sha256,byte_count,st_dev,st_ino,mode,nlink}
```

Environment values are exact: `HOME=<evidence-root>`, `LANG=C.UTF-8`, `LC_ALL=C.UTF-8`, `PATH=/usr/bin:/bin`, `PYTHONHASHSEED=0`, `PYTHONNOUSERSITE=1`, `PYTHONDONTWRITEBYTECODE=1`, `TZ=UTC`.

Launch envelope, schema `alpha_max_terminal_launch_envelope.v3`:

```text
{schema,policy_sha256,current_head,accepted_alpha_commit,baseline_ancestor,
 repositories,files,interpreters,authority_key,observer_keys,forbidden_roots,scope_order}
RepositoryBinding {role,root:DirectoryIdentity,head,clean_receipt:FileIdentity}
FileBinding {role,file:FileIdentity}
InterpreterBinding {role,file:FileIdentity,package_freeze:FileIdentity}
KeyBinding {key_id,public_key_b64,public_key_sha256}
ObserverKeyBinding {scope,key_id,public_key_b64,public_key_sha256}
```

Ordered repository roles are `current_repository`, `accepted_alpha_repository`. Ordered interpreter roles are `current_python`, `accepted_alpha_python`. Ordered file roles are:

```text
policy_json, policy_module, authority_script, observer_script, key_creator,
acquirer, phase_wrapper, runbook, alpha_uv_lock, alignment_receipt,
portfolio, contract_manifest, availability_evidence, preparer,
prelock_script, historical_script, process_boundary
```

Ordered scopes are `acquisition`, `phase_preparation`, `one_touch`. Forbidden roots must equal the two lexical constants above in that exact order. Protected roots/files, request records, prerequisites, control paths, and outputs must be disjoint and outside them. Existing path components are opened descriptor-relatively with `O_NOFOLLOW` on every component.

Checkpoint schema `alpha_max_terminal_checkpoint.v1`:

```text
{schema,accepted_alpha_commit,baseline_ancestor,runbook_sha256,uv_lock_sha256,
alignment_receipt_sha256,portfolio_sha256,contract_sha256,availability_sha256,
preparer_sha256,prelock_sha256,historical_sha256,process_boundary_sha256,
acquirer_sha256,phase_wrapper_sha256,authority_manifest_sha256}
```

Common request fields:

```text
{schema,request_id,scope,checkpoint_pin_sha256,interpreter,repository_root,
evidence_root,authority_socket,environment,forbidden_roots,publication,prerequisites}
```

Scope additions:

```text
acquisition: acquirer,contract_manifest,availability_evidence,source_root:AbsentOutput,report_root:AbsentOutput
phase_preparation: phase_wrapper,acquirer,source_root:DirectoryIdentity,source_report:DirectoryIdentity,
                   contract_manifest,availability_evidence,preparer,phase_output:AbsentOutput
one_touch: portfolio,contract_manifest,prelock_script,historical_script,phase_output:DirectoryIdentity,
           prelock_output:AbsentOutput,historical_output:AbsentOutput
```

Prerequisite orders:

- acquisition: `checkpoint_pin`, `alignment_receipt`
- phase: `checkpoint_pin`, `alignment_receipt`, `source_eligible_receipt`, `source_manifest`, `source_journal`
- one-touch: `checkpoint_pin`, `alignment_receipt`, `phase_handoff_receipt`, `preparation_manifest`

Publication leaves are fixed beneath the private evidence root: `prelaunch.claim.json`, `terminal-observer.journal.jsonl`, `terminal-authority.receipt.json`, plus `child-N.stdout.log`/`child-N.stderr.log` for 2/1/2 commands.

## Exact command derivation

No request may supply executable argv, environment overrides, or result paths. Commands derive only from the typed envelope/request.

Acquisition command 0:

```text
python acquirer --contract-manifest CONTRACT --availability-evidence AVAILABILITY
--output-root SOURCE --report-dir REPORT
--forbidden-root F0 --forbidden-root F1 --execute --validate-complete
```

Acquisition command 1 is the identical common argv ending only in `--verify-eligible`.

Phase preparation:

```text
python wrapper --acquirer ACQUIRER --source-root SOURCE --source-report REPORT
--forbidden-root F0 --forbidden-root F1 --contract-manifest CONTRACT
--availability-evidence AVAILABILITY --preparer PREPARER --output-root PHASE_OUTPUT
```

Prelock:

```text
python prelock --config PORTFOLIO --contract-manifest CONTRACT --exchange binance
--output-root PRELOCK_OUTPUT
--warmup-raw-root PHASE/warmup/raw --warmup-feature-root PHASE/warmup/feature
--train-raw-root PHASE/train/raw --train-feature-root PHASE/train/feature
--purge-raw-root PHASE/purge/raw --purge-feature-root PHASE/purge/feature
--validation-raw-root PHASE/validation/raw --validation-feature-root PHASE/validation/feature
--embargo-raw-root PHASE/embargo/raw --embargo-feature-root PHASE/embargo/feature
```

Historical:

```text
python historical --sealed-prelock-directory PRELOCK_OUTPUT
--embargo-feature-root PHASE/embargo/feature
--historical-evaluation-raw-root PHASE/historical_exposed_evaluation/raw
--historical-evaluation-feature-root PHASE/historical_exposed_evaluation/feature
--exchange binance --output-root HISTORICAL_OUTPUT
```

The command-bundle digest is canonical `{schema:"alpha_max_terminal_command_bundle.v1",scope,commands,environment}`.

## Wire protocol and state machine

Transport is `AF_UNIX/SOCK_SEQPACKET` with one 4-byte big-endian length plus canonical JSON packet, maximum 1,048,576 bytes. Authority verifies `SO_PEERCRED`, PID/UID/start ticks, source hash, key identity, nonce/request/auth/sequence replay, and Ed25519 signatures. Signing preimage is:

```text
b"luminaquant.alpha_max.terminal.v1/" + message_type + b"\0" + canonical_bytes(unsigned)
```

Signature fields are exact:

- challenge, authorization, command clearance, terminal receipt: `authority_signature_b64`
- observer proof and process event: `observer_signature_b64`

Challenge fields:

```text
schema,type,authority_key_id,scope,request_id,checkpoint_pin_sha256,envelope_sha256,
request_sha256,command_bundle_sha256,nonce_b64,issued_utc,authority_signature_b64
```

Observer proof fields:

```text
schema,type,authority_key_id,scope,request_id,checkpoint_pin_sha256,envelope_sha256,
request_sha256,command_bundle_sha256,nonce_b64,observer_key_id,observer_pid,observer_uid,
observer_start_ticks,observer_source_sha256,claim_sha256,observer_signature_b64
```

Authorization fields:

```text
schema,type,authority_key_id,authorization_id,scope,request_id,checkpoint_pin_sha256,
envelope_sha256,request_sha256,command_bundle_sha256,claim_sha256,observer_key_id,
observer_pid,observer_uid,observer_start_ticks,observer_source_sha256,
not_before_utc,expires_utc,authority_signature_b64
```

Command clearance fields:

```text
schema,type,authority_key_id,authorization_id,scope,request_id,completed_command_index,
next_command_index,validated_artifact_snapshot_sha256,issued_utc,authority_signature_b64
```

Claim v1 remains exactly:

```text
schema,request_id,scope,checkpoint_pin_sha256,evidence_root,
observer_pid,observer_uid,observer_start_ticks,created_utc
```

Process-event base:

```text
schema,type:"process_event",event,authorization_id,sequence,command_index,
argv_sha256,environment_sha256,prior_clearance,observed_utc,observer_signature_b64
```

Event additions:

- `launch_intent`: base only
- `child_started`: base plus `child_pid,child_start_ticks,stdin_identity,stdout,stderr`
- `child_exited`: child-start fields plus `return_code,stdout_sha256,stdout_byte_count,stderr_sha256,stderr_byte_count`
- `start_failed`: base plus `errno,error_name`

Durable order is `launch_intent` before any log creation or `Popen`; no relaunch after durable intent. Authority contains no launch primitive. Observer is the sole `subprocess.Popen` owner. Children use non-TTY direct O_EXCL logs, `stdin=DEVNULL`, `close_fds=True`, empty `pass_fds`, fixed environment, bounded repository cwd, and no inherited secret or `LQ_*` environment.

Terminal states:

```text
{kind:"SUCCEEDED"}
{kind:"FAILED",failed_command_index}
{kind:"START_FAILED",command_index,errno}
{kind:"START_UNKNOWN",command_index}
{kind:"OBSERVER_LOST",command_index,child_pid,child_start_ticks}
{kind:"UNAUTHENTICATED_TERMINAL",last_authenticated_sequence}
```

Receipt fields:

```text
schema,type,authority_key_id,authorization_id,scope,request_id,checkpoint_pin_sha256,
envelope_sha256,request_sha256,claim_sha256,observer_key_id,observer_pid,
observer_start_ticks,command_bundle_sha256,events_sha256,journal_sha256,prerequisites,
target_results,terminal_state,publication,created_utc,authority_signature_b64
```

No-launch recovery validates the stored authorization and authenticated journal prefix. A strict partial tail may be completed only from the O_EXCL pending record; non-prefix or malformed state fails closed. Recovery never launches or advances a child. Receipt publication requires the persisted journal to equal exactly `[authorization,*events]`.

## Semantic completion evidence

`TargetResult` is exact:

```text
{command_index,argv_sha256,environment_sha256,return_code,stdout,stderr,
 validated_artifacts,sealed_artifacts,completed_utc}
ValidatedArtifact {kind,path,sha256,byte_count,st_dev,st_ino,mode,nlink}
SealedArtifact {ValidatedArtifact fields,sealed_payload_sha256,canonical_inventory_sha256,readback_sha256}
```

Acquisition command 0 and offline command 1 both authenticate `source_eligible_receipt`, `source_manifest`, and `source_journal`; command 1 must prove byte/identity-stable evidence.

A-02 phase completion authenticates sibling `.<phase-name>.alpha_max_phase_preparation.handoff.json`, root `preparation_manifest.json`, and exactly six immutable manifest-declared trees: `warmup`, `train`, `purge`, `validation`, `embargo`, `historical_exposed_evaluation`. Every manifest output path/hash/byte count is verified; phase inputs are revalidated before launch and after each one-touch child.

One-touch command evidence snapshots:

- child 0: `(phase_output, prelock_output)`
- child 1: `(phase_output, prelock_output, historical_output)`

Phase and prelock snapshots must remain identical across the child boundary. Authority clearance binds the exact child-0 snapshot digest.

Prelock sealed inventory requires:

- fixed artifacts: `admission/train.json`, `admission/train_computation.json`, `admission/train_liquidity_buckets.json`, `allocation/train_fit.json`, `allocation/train_validation_refit.json`, `diagnostics/validation/trend_liquidity_falsifier.json`, `inputs/config.json`, `inputs/contract_manifest.json`, `inputs/prior_trial_inventory.json`, `run/prelock_result.json`, `selection/prelock.json`, `status/matrix.json`, `terminal/prelock.json`, `trial/ledger.json`;
- 17 validation-train-fit manifests, 17 final-refit manifests, 204 validation capsules, 17 final-refit capsules, 68 validation cells, and 816 validation rows;
- `SEALED.json` inventory entries exactly `{byte_count,relative_path,sha256}`;
- canonical outcome/selection/terminal/matrix/observability readback and a single consistent prelock champion.

Historical sealed inventory requires:

- fixed artifacts: `admission/train_liquidity_buckets.json`, `binding/prelock_seal.json`, historical diagnostic, `report/historical_result.json`, `selection/historical_ranking.json`, `status/matrix.json`, `terminal/historical.json`;
- 153 final-refit capsules, 68 historical cells, and 680 historical rows;
- exact prelock seal/snapshot binding and canonical outcome/ranking/terminal/matrix/observability readback.

All sealed files are regular, single-linked, `0444`; directories are `0555`; symlinks and extra/missing inventory entries fail closed. Observability is read-only and sealed-payload-bound; policy never launches or synthesizes an exporter.

## Implementation checkpoint contents

Commit `475f3f2ebe37994f574dc970e1b3fa9563da8009` contains:

- `configs/research/alpha_max_terminal_authority_policy_v1.json`
- `src/lumina_quant/alpha_max_terminal_policy.py`
- `scripts/research/create_alpha_max_terminal_keys.py`
- `scripts/research/run_alpha_max_terminal_authority.py`
- `scripts/research/run_alpha_max_terminal_observer.py`
- `tests/test_alpha_max_terminal_policy.py`
- `tests/test_alpha_max_terminal_external_envelope.py`
- `tests/test_run_alpha_max_terminal_authority.py`
- `tests/test_run_alpha_max_terminal_observer.py`
- `pyproject.toml` with exact `cryptography==49.0.0` terminal-authority and dev declarations.

The ignored local `uv.lock` was refreshed for frozen local verification but must not be staged or force-added.

Post-stop snapshot verification:

```text
ruff format/check: passed
py_compile: passed
focused terminal suite: 135 passed
external/target execution: none
```

The 135 tests include canonical framing, pin/role mismatch, exact argv, prelaunch identity drift, symlink ancestors, lexical quarantine rejection, key rollback/provenance, launch ordering, challenge/authorization/clearance failure, receipt states, crash recovery, phase snapshot binding, and prelock/historical semantic inventory/readback fixtures.

## Incomplete quality gate

G061 is not complete. The stopped session did not run a post-135-test mandatory cleaner rerun, integrated/full sanitized verification, formal architect review, or executor QA/red-team. Do not infer approval from the focused suite.

Earlier cleaner blockers were implemented and focused-tested: complete cleanup error evidence, exact journal/receipt equality, prerequisite binding, no-symlink paths, fixed forbidden roots, negative pin/argv/preflight coverage, receipt state matrix, launch/clearance failure coverage, secure key-root provenance, immutable phase snapshots, and semantic sealed fixtures. They still require the mandated fresh cleaner rerun on the final committed snapshot.

## Exact resume order

1. Verify branch `recovery/strategy-plan-20260714`, clean worktree, and HEAD ancestry from implementation checkpoint `475f3f2ebe37994f574dc970e1b3fa9563da8009` and the handoff-content commit named in the canonical resume state.
2. Verify durable brief/goals/ledger seals, latest event `7523ed6e-93fb-41f8-a937-18897ac3de8f`, all workers/monitors/processes stopped, and no target execution. Do not inspect the quarantined roots. Do not create or inspect v5 roots before G061 is clean.
3. Set `GJC_SESSION_ID=019f603a-0e73-7000-88a7-c94f42950c09` on every native Ultragoal command. Run `gjc ultragoal status --json`; call `goal get`, then `goal resume`. Never run `create-goals` or `complete-goals`.
4. Verify frozen target hashes and rerun the focused 135-test terminal suite.
5. Run the mandatory ai-slop-cleaner fragment over only the ten changed implementation/test/config files. Fix blocking findings only through an executor, rerun focused verification, and repeat until zero blocking findings.
6. Run integrated terminal/acquirer tests and sanitized full repository verification under a network namespace. Preserve the established deselection for the byte-identical shipped-YAML HEAD test if the committed handoff docs make that test intentionally inapplicable; record, never hide, any failure.
7. Freeze the post-cleaner change set. Run architect architecture/product/code review and executor QA/red-team on the identical frozen snapshot. The executor must produce a real API/package/CLI artifact; no bare inline claim is sufficient. No target script may execute.
8. Any review blocker becomes durable replacement work; do not checkpoint G061 until all lanes are clean.
9. Strictly checkpoint G061 with the structured quality gate. Then explicitly supersede/resolve G060 from the completed G061 replacement evidence. Do not call `complete-goals`.
10. Only after terminal authority is independently CLEAR may G056 be explicitly reactivated, fresh v5 roots be created/frozen, and official acquisition begin under the terminal authority.
11. Continue the recovery order: full acquisition; `validate_complete`; offline `--verify-eligible`; six authenticated phase roots; one-touch; G056; genuinely diverse G036 then G037; fresh final aggregate audit/receipt; inline goal completion only after durable aggregate proof.

If any official archive fails ordering, preserve the current v5 evidence and create an explicit evidence-backed replacement subgoal. Never broaden canonical sorting automatically.

## G064 prelaunch addendum

G062 completed the strict terminal-authority gate and superseded G061/G060. Before any target or network launch, read-only prelaunch review 189 found that the original 300-second authorization was incorrectly reused as the live-time authority for command 1 after a potentially hours-long acquisition command 0. G056 is review-blocked by G064 until the following unchanged-wire semantics are independently clean:

- command 0 requires the original authorization to be current immediately before its intent and `Popen`;
- every later command carries the exact most recently issued signed command clearance in `prior_clearance`;
- the observer revalidates that clearance's signature, authorization/request/index/snapshot bindings, and existing 60-second freshness immediately before intent and again immediately before `Popen`;
- the authority requires the later intent to carry the exact clearance object it issued and brackets both the observer's `observed_utc` and authority receive-before/receive-after seconds within the same 60-second window;
- missing, future, stale, replayed, mismatched, wrong-index, or differently signed clearance fails closed.

No wire field, CLI, target argv, environment, accepted pin, acquirer byte, or phase-wrapper byte changes for this repair.

Recovery remains deliberately authorization-only and no-launch. It rejects any intent/process-event history because a post-crash process event cannot reconstruct the original live receive bracket without weakening authorization. A host/process crash after durable intent therefore makes that source/report/evidence set terminal and ineligible: preserve it byte-for-byte, create an explicit replacement subgoal and fresh roots, and never relaunch or advance it through `recover`.

The fixed envelope field named `clean_receipt` may bind a truthful reviewed-overlay receipt when the no-commit constraint prevents a porcelain-empty current checkout. That receipt is valid only when it records the actual HEAD/branch/ancestry, every porcelain path and exact SHA-256, an empty index, `git diff --check` PASS, zero unexpected tracked or untracked paths, the final quality-gate/adversarial-report hashes, and the separately envelope-bound ignored interpreter/package files. It must state `git_porcelain_empty:false` and `decision:"PASS_REVIEWED_OVERLAY"` rather than claim Git cleanliness. The accepted Alpha repository must remain genuinely porcelain-empty at the pinned accepted commit. A fresh independent architect must approve the exact receipt and envelope before launch.

## Market-cap-index scope clarification

The exact repository `HokyoungJung/Market-Cap-Weighted-Indices` is not directly imported or implemented in this checkpoint. LuminaQuant already contains adjacent TopCap-universe and turnover/flow-share index concepts, but those are not equivalent to a point-in-time daily market-cap-weighted index. A future inclusion must be an explicit research candidate using point-in-time constituent and capitalization evidence; current CoinGecko/current-universe data must not be backfilled as historical membership. It must remain research/shadow-only until clean walk-forward and cost/funding gates pass. The terminal pipeline work above improves correctness and provenance; it must never be used to tune headline performance.

## 2026-07-26 cross-session continuation

The current G070/G071/G072 recovery state, v8 acquisition safeguards, and canonical shared-DB continuation plan are recorded in [`g070_v8_and_canonical_db_cross_session_handoff_20260726.md`](g070_v8_and_canonical_db_cross_session_handoff_20260726.md). That handoff supersedes this file as the operational resume entry point while preserving all earlier scientific and safety constraints.