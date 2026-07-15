# Alpha-Max Revision 5.15 Data-PC Runbook

## Purpose, authority, and stop condition

This is the no-discretion handoff for the PC that owns the complete market
dataset. It runs the frozen research-only Alpha-Max experiment; it does not
authorize paper, testnet, live, or real allocation.

The operational stop condition is:

1. the prelock command exits zero and publishes one immutable bundle with
   `SEALED.json` written last;
2. that bundle passes the independent inventory, SHA-256, size, link, mode, and
   readback audit below;
3. the physically separate one-touch historical command exits zero and publishes
   its own immutable report-only bundle;
4. the prelock tree is byte-identical before and after historical evaluation;
5. the historical bundle passes the same independent audit.

Local/hosted CI proves implementation integrity only. It is not evidence of
alpha, profitability, robustness, or deployability.

## Frozen source preflight and run record

Run from the repository root on the exact pushed branch. Do not edit source,
config, dates, symbols, thresholds, costs, seeds, or output artifacts on the data
PC. A clean clone includes the tracked, frozen `uv.lock`; never regenerate it.
Before source-manifest checks or `uv sync`, obtain the approved external alignment
receipt and its supplied SHA-256 from the handoff. The receipt's supplied SHA-256
is the trust anchor and avoids an impossible commit self-reference. The external
receipt is not a tracked final-manifest entry.

```bash
set -euo pipefail
umask 077
REPO="$(pwd -P)"
ALIGNMENT_RECEIPT="${ALIGNMENT_RECEIPT:?set to the absolute approved receipt path}"
ALIGNMENT_RECEIPT_SHA256="${ALIGNMENT_RECEIPT_SHA256:?set to the approved lowercase receipt SHA-256}"

for variable in \
  PYTHONPATH PYTHONHOME PYTHONSTARTUP PYTHONINSPECT PYTHONUSERBASE \
  PYTHONPLATLIBDIR GIT_DIR GIT_WORK_TREE GIT_COMMON_DIR GIT_INDEX_FILE \
  GIT_OBJECT_DIRECTORY GIT_ALTERNATE_OBJECT_DIRECTORIES GIT_CONFIG_GLOBAL \
  GIT_CONFIG_SYSTEM GIT_CONFIG_NOSYSTEM GIT_CONFIG_COUNT \
  GIT_CONFIG_PARAMETERS GIT_EXEC_PATH GIT_NAMESPACE GIT_NO_REPLACE_OBJECTS \
  GIT_REPLACE_REF_BASE GIT_SHALLOW_FILE GIT_ATTR_NOSYSTEM GIT_ATTR_SOURCE \
  VIRTUAL_ENV CONDA_PREFIX UV_PROJECT_ENVIRONMENT UV_PYTHON \
  UV_PYTHON_PREFERENCE UV_SYSTEM_PYTHON UV_NO_SYNC
do
  test -z "${!variable+x}" || {
    printf 'forbidden preflight environment variable: %s\n' "$variable" >&2
    exit 1
  }
done
for executable in \
  /usr/bin/date /usr/bin/git /usr/bin/mkdir /usr/bin/python3 \
  /usr/bin/sha256sum /usr/bin/tee /usr/bin/time
do
  test -x "$executable"
done

RUN_ID="$(/usr/bin/date -u +%Y%m%dT%H%M%SZ)"
RUNLOG="/absolute/path/to/alpha-max-run-record-$RUN_ID"
DATA="/absolute/path/to/alpha-max-phase-roots"
PRELOCK_OUT="/absolute/path/to/new/alpha-max-prelock-$RUN_ID"
HISTORICAL_OUT="/absolute/path/to/new/alpha-max-historical-$RUN_ID"
/usr/bin/mkdir -p "$RUNLOG"

# These Git commands inspect metadata only; no configurable worktree conversion runs.
export GIT_ATTR_NOSYSTEM=1
GIT_TRUST_ARGS=(
  --no-replace-objects
  -c core.commitGraph=false
  -c core.attributesFile=/dev/null
)
GIT_INDEX_PATH="$(
  /usr/bin/git "${GIT_TRUST_ARGS[@]}" rev-parse --git-path index
)"
GIT_GRAFTS_PATH="$(
  /usr/bin/git "${GIT_TRUST_ARGS[@]}" rev-parse --git-path info/grafts
)"
GIT_ATTRIBUTES_PATH="$(
  /usr/bin/git "${GIT_TRUST_ARGS[@]}" rev-parse --git-path info/attributes
)"
test -e .git
test ! -L .git
test -f "$GIT_INDEX_PATH"
test ! -L "$GIT_INDEX_PATH"
test ! -e "$GIT_GRAFTS_PATH"
test ! -L "$GIT_GRAFTS_PATH"
test ! -e "$GIT_ATTRIBUTES_PATH"
test ! -L "$GIT_ATTRIBUTES_PATH"
REPLACE_REFS="$(
  /usr/bin/git "${GIT_TRUST_ARGS[@]}" \
    for-each-ref --format='%(refname)' refs/replace/
)"
test -z "$REPLACE_REFS"
test "$(/usr/bin/python3 -I -S -c 'import sys; print(int(sys.flags.isolated and sys.flags.no_site))')" = "1"

case "$ALIGNMENT_RECEIPT" in /*) ;; *) exit 1 ;; esac
case "$ALIGNMENT_RECEIPT_SHA256" in
  [0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]\
[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]\
[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]\
[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]\
[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]\
[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]\
[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]\
[0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f][0-9a-f]) ;;
  *) exit 1 ;;
esac
test -f "$ALIGNMENT_RECEIPT"
test ! -L "$ALIGNMENT_RECEIPT"
read -r RECEIPT_ACTUAL_SHA256 _ < <(/usr/bin/sha256sum "$ALIGNMENT_RECEIPT")
test "$RECEIPT_ACTUAL_SHA256" = "$ALIGNMENT_RECEIPT_SHA256"

RECEIPT_OUTPUT="$(
  /usr/bin/python3 -I -S - \
    "$ALIGNMENT_RECEIPT" \
    "$ALIGNMENT_RECEIPT_SHA256" \
    "$RUNLOG/alignment-receipt-readback.json" \
    "$GIT_INDEX_PATH" \
    "$REPO" <<'PY'
from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import sys
from pathlib import Path

EXPECTED_LOCK_SHA256 = "59d9de230be950761736c24e04af3456e229cf4aa077536167fb7e650a71c339"
EXPECTED = {
    "artifact_kind": "alpha_max_rev515_alignment_receipt",
    "schema_version": 1,
    "repository": "LuminaQuant",
    "branch": "recovery/alpha-max-rev515-alignment-20260714",
    "baseline_commit": "629d91e5d4aac26911af65a4a5e15ebdcbded30f",
    "final_manifest_path": "docs/research_note/alpha_max_final_sha256_20260711.txt",
    "lock_sha256": EXPECTED_LOCK_SHA256,
}
KEYS = set(EXPECTED) | {"accepted_commit", "final_manifest_sha256"}


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


INDEX_MODE_TEXT = {
    0o100644: b"100644",
    0o100755: b"100755",
    0o120000: b"120000",
}
EXPECTED_GITATTRIBUTES = (
    b"# Keep repository text diffs stable across Linux/Windows tools.\n"
    b"* text=auto eol=lf\n"
    b"\n"
    b"# Windows launchers remain CRLF for native shell compatibility.\n"
    b"*.bat text eol=crlf\n"
    b"*.cmd text eol=crlf\n"
    b"*.ps1 text eol=crlf\n"
)
CRLF_SUFFIXES = (b".bat", b".cmd", b".ps1")


def git_object_id(kind, payload):
    header = kind + b" " + str(len(payload)).encode("ascii") + b"\0"
    return hashlib.sha1(header + payload).digest()


def new_tree_node():
    return {"dirs": {}, "files": {}}


def summarize_tree(node, name):
    child_summaries = {
        child_name: summarize_tree(child, child_name)
        for child_name, child in node["dirs"].items()
    }
    records = []
    for child_name, (mode, oid) in node["files"].items():
        records.append(
            (
                child_name + b"\0",
                INDEX_MODE_TEXT[mode] + b" " + child_name + b"\0" + oid,
            )
        )
    for child_name, child in child_summaries.items():
        records.append(
            (
                child_name + b"/",
                b"40000 " + child_name + b"\0" + child["oid"],
            )
        )
    tree_payload = b"".join(record for _, record in sorted(records))
    entry_count = len(node["files"]) + sum(
        child["entry_count"] for child in child_summaries.values()
    )
    return {
        "children": child_summaries,
        "entry_count": entry_count,
        "name": name,
        "oid": git_object_id(b"tree", tree_payload),
    }


def validate_cache_tree(payload, root):
    def parse_node(offset, expected):
        name_end = payload.find(b"\0", offset)
        if name_end < 0 or payload[offset:name_end] != expected["name"]:
            raise SystemExit("Git cache-tree pathname is invalid")
        header_end = payload.find(b"\n", name_end + 1)
        if header_end < 0:
            raise SystemExit("Git cache-tree header is unterminated")
        header = payload[name_end + 1 : header_end]
        match = re.fullmatch(rb"(0|[1-9][0-9]*) (0|[1-9][0-9]*)", header)
        if not match:
            raise SystemExit("Git cache-tree header is noncanonical")
        entry_count = int(match.group(1))
        subtree_count = int(match.group(2))
        if (
            entry_count != expected["entry_count"]
            or subtree_count != len(expected["children"])
        ):
            raise SystemExit("Git cache-tree counts mismatch")
        oid_start = header_end + 1
        oid_end = oid_start + 20
        if oid_end > len(payload) or payload[oid_start:oid_end] != expected["oid"]:
            raise SystemExit("Git cache-tree object ID mismatch")

        offset = oid_end
        seen = set()
        for _ in range(subtree_count):
            child_name_end = payload.find(b"\0", offset)
            if child_name_end < 0:
                raise SystemExit("Git cache-tree child is unterminated")
            child_name = payload[offset:child_name_end]
            if child_name in seen or child_name not in expected["children"]:
                raise SystemExit("Git cache-tree child set mismatch")
            seen.add(child_name)
            offset = parse_node(offset, expected["children"][child_name])
        if seen != set(expected["children"]):
            raise SystemExit("Git cache-tree children are incomplete")
        return offset

    if parse_node(0, root) != len(payload):
        raise SystemExit("Git cache-tree payload has trailing bytes")


def validate_full_index(path):
    raw = path.read_bytes()
    if len(raw) < 32 or raw[:4] != b"DIRC":
        raise SystemExit("Git index header is invalid")
    version = int.from_bytes(raw[4:8], "big")
    if version not in {2, 3}:
        raise SystemExit("Git index version must be 2 or 3")
    if hashlib.sha1(raw[:-20]).digest() != raw[-20:]:
        raise SystemExit("Git index checksum mismatch")

    entry_count = int.from_bytes(raw[8:12], "big")
    entries = []
    previous_path = None
    offset = 12
    payload_end = len(raw) - 20
    for _ in range(entry_count):
        entry_start = offset
        if offset + 62 > payload_end:
            raise SystemExit("Git index entry is truncated")
        mode = int.from_bytes(raw[offset + 24 : offset + 28], "big")
        oid = raw[offset + 40 : offset + 60]
        flags = int.from_bytes(raw[offset + 60 : offset + 62], "big")
        if mode not in INDEX_MODE_TEXT:
            raise SystemExit("Git index mode is forbidden")
        if oid == b"\0" * 20:
            raise SystemExit("Git index object ID is zero")
        if flags & 0x8000:
            raise SystemExit("Git assume-unchanged index entries are forbidden")
        if flags & 0x4000:
            raise SystemExit("Git extended index entries are forbidden")
        if flags & 0x3000:
            raise SystemExit("Git nonzero-stage index entries are forbidden")

        name_length = flags & 0x0FFF
        path_start = offset + 62
        path_end = raw.find(b"\0", path_start, payload_end)
        if path_end < 0:
            raise SystemExit("Git index pathname is unterminated")
        index_path = raw[path_start:path_end]
        if name_length < 0x0FFF and name_length != len(index_path):
            raise SystemExit("Git index pathname length mismatch")
        if name_length == 0x0FFF and len(index_path) < 0x0FFF:
            raise SystemExit("Git long pathname marker is noncanonical")
        parts = index_path.split(b"/")
        if (
            not index_path
            or any(not part or part in {b".", b".."} for part in parts)
            or any(part.lower() == b".git" for part in parts)
        ):
            raise SystemExit("Git index pathname is forbidden")
        if previous_path is not None and index_path <= previous_path:
            raise SystemExit("Git index paths are duplicate or unordered")
        previous_path = index_path

        entry_length = path_end + 1 - entry_start
        next_offset = entry_start + ((entry_length + 7) // 8) * 8
        if next_offset > payload_end:
            raise SystemExit("Git index entry padding is invalid")
        if any(raw[path_end:next_offset]):
            raise SystemExit("Git index entry padding is nonzero")
        offset = next_offset
        entries.append((index_path, mode, oid))
    attributes_entry = {
        index_path: (mode, oid) for index_path, mode, oid in entries
    }.get(b".gitattributes")
    if attributes_entry != (
        0o100644,
        git_object_id(b"blob", EXPECTED_GITATTRIBUTES),
    ):
        raise SystemExit("tracked Git attributes are not the frozen built-in policy")

    root = new_tree_node()
    for index_path, mode, oid in entries:
        parts = index_path.split(b"/")
        node = root
        for part in parts[:-1]:
            if part in node["files"]:
                raise SystemExit("Git index file/directory collision")
            node = node["dirs"].setdefault(part, new_tree_node())
        leaf = parts[-1]
        if leaf in node["files"] or leaf in node["dirs"]:
            raise SystemExit("Git index path collision")
        node["files"][leaf] = (mode, oid)
    summary = summarize_tree(root, b"")
    if summary["entry_count"] != entry_count:
        raise SystemExit("Git index entry count mismatch")

    extensions = {}
    while offset < payload_end:
        if offset + 8 > payload_end:
            raise SystemExit("Git index extension is truncated")
        signature = raw[offset : offset + 4]
        extension_length = int.from_bytes(raw[offset + 4 : offset + 8], "big")
        offset += 8
        if offset + extension_length > payload_end:
            raise SystemExit("Git index extension payload is truncated")
        if signature != b"TREE" or signature in extensions:
            raise SystemExit("Git index extension is forbidden")
        extensions[signature] = raw[offset : offset + extension_length]
        offset += extension_length
    if offset != payload_end:
        raise SystemExit("Git index boundary mismatch")
    if extensions:
        validate_cache_tree(extensions[b"TREE"], summary)
    return entries, summary["oid"].hex()


def filesystem_inventory(repo):
    files = set()
    directories = set()

    def visit(absolute, relative):
        try:
            children = sorted(os.scandir(absolute), key=lambda entry: entry.name)
        except OSError as error:
            raise SystemExit(f"worktree inventory failed: {error}") from error
        for child in children:
            name = child.name
            if not relative and name == b".git":
                continue
            child_relative = name if not relative else relative + b"/" + name
            if child.is_symlink():
                files.add(child_relative)
            elif child.is_dir(follow_symlinks=False):
                directories.add(child_relative)
                visit(child.path, child_relative)
            elif child.is_file(follow_symlinks=False):
                files.add(child_relative)
            else:
                raise SystemExit("special worktree entries are forbidden")

    visit(repo, b"")
    return files, directories


def stable_identity(metadata):
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def read_regular_file(path, before):
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino):
            raise SystemExit("worktree file changed before open")
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after_read = os.fstat(descriptor)
        if stable_identity(opened) != stable_identity(after_read):
            raise SystemExit("worktree file changed while reading")
    finally:
        os.close(descriptor)
    after = os.lstat(path)
    if stable_identity(before) != stable_identity(after):
        raise SystemExit("worktree file changed during validation")
    return b"".join(chunks), opened


def validate_worktree(repo_path, entries):
    repo = os.fsencode(os.path.realpath(repo_path))
    expected_files = {index_path for index_path, _, _ in entries}
    expected_directories = set()
    for index_path in expected_files:
        parts = index_path.split(b"/")
        expected_directories.update(
            b"/".join(parts[:position]) for position in range(1, len(parts))
        )

    observed_files, observed_directories = filesystem_inventory(repo)
    if observed_files != expected_files:
        raise SystemExit("worktree file inventory differs from the index")
    if observed_directories != expected_directories:
        raise SystemExit("worktree directory inventory differs from the index")

    for index_path, mode, expected_oid in entries:
        absolute = os.path.join(repo, *index_path.split(b"/"))
        before = os.lstat(absolute)
        if mode == 0o120000:
            if not stat.S_ISLNK(before.st_mode):
                raise SystemExit("worktree symlink mode mismatch")
            payload = os.readlink(absolute)
            after = os.lstat(absolute)
            if stable_identity(before) != stable_identity(after):
                raise SystemExit("worktree symlink changed during validation")
        else:
            if not stat.S_ISREG(before.st_mode):
                raise SystemExit("worktree regular-file mode mismatch")
            payload, opened = read_regular_file(absolute, before)
            executable = bool(opened.st_mode & 0o111)
            if executable != (mode == 0o100755):
                raise SystemExit("worktree executable mode mismatch")
        actual_oid = git_object_id(b"blob", payload)
        if actual_oid != expected_oid and index_path.endswith(CRLF_SUFFIXES):
            remainder = payload.replace(b"\r\n", b"")
            if b"\n" in remainder or b"\r" in remainder:
                raise SystemExit(
                    f"worktree CRLF conversion is noncanonical: "
                    f"{os.fsdecode(index_path)}"
                )
            actual_oid = git_object_id(b"blob", payload.replace(b"\r\n", b"\n"))
        if actual_oid != expected_oid:
            raise SystemExit(
                f"worktree blob differs from the index: {os.fsdecode(index_path)}"
            )

    final_files, final_directories = filesystem_inventory(repo)
    if final_files != expected_files or final_directories != expected_directories:
        raise SystemExit("worktree inventory changed during validation")
    return len(expected_files), len(expected_directories)


receipt_path = Path(sys.argv[1])
expected_digest = sys.argv[2]
readback_path = Path(sys.argv[3])
index_path = Path(sys.argv[4])
repo_path = sys.argv[5]
raw = receipt_path.read_bytes()
if hashlib.sha256(raw).hexdigest() != expected_digest:
    raise SystemExit("alignment receipt SHA-256 mismatch")
receipt = json.loads(raw, object_pairs_hook=unique_object)
canonical = json.dumps(
    receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=False
).encode() + b"\n"
if raw != canonical:
    raise SystemExit("alignment receipt is not canonical JSON")
if type(receipt) is not dict or set(receipt) != KEYS:
    raise SystemExit("alignment receipt schema mismatch")
for key, value in EXPECTED.items():
    if type(receipt.get(key)) is not type(value) or receipt[key] != value:
        raise SystemExit(f"alignment receipt {key} mismatch")
if not isinstance(receipt["accepted_commit"], str) or not re.fullmatch(
    r"[0-9a-f]{40}", receipt["accepted_commit"]
):
    raise SystemExit("alignment receipt accepted_commit is invalid")
if not isinstance(receipt["final_manifest_sha256"], str) or not re.fullmatch(
    r"[0-9a-f]{64}", receipt["final_manifest_sha256"]
):
    raise SystemExit("alignment receipt final_manifest_sha256 is invalid")
index_entries, index_root_tree = validate_full_index(index_path)
worktree_file_count, worktree_directory_count = validate_worktree(
    repo_path, index_entries
)

readback = {
    "alignment_receipt_sha256": expected_digest,
    "git_index_entry_count": len(index_entries),
    "index_root_tree": index_root_tree,
    "receipt": receipt,
    "worktree_directory_count": worktree_directory_count,
    "worktree_file_count": worktree_file_count,
}
readback_path.write_bytes(
    json.dumps(readback, sort_keys=True, separators=(",", ":")).encode() + b"\n"
)
print(receipt["branch"])
print(receipt["accepted_commit"])
print(receipt["baseline_commit"])
print(receipt["final_manifest_sha256"])
print(receipt["lock_sha256"])
print(index_root_tree)
PY
)"
mapfile -t RECEIPT_FIELDS <<< "$RECEIPT_OUTPUT"
test "${#RECEIPT_FIELDS[@]}" -eq 6
RECEIPT_BRANCH="${RECEIPT_FIELDS[0]}"
RECEIPT_ACCEPTED_COMMIT="${RECEIPT_FIELDS[1]}"
RECEIPT_BASELINE_COMMIT="${RECEIPT_FIELDS[2]}"
RECEIPT_MANIFEST_SHA256="${RECEIPT_FIELDS[3]}"
RECEIPT_LOCK_SHA256="${RECEIPT_FIELDS[4]}"
INDEX_ROOT_TREE="${RECEIPT_FIELDS[5]}"

test "$(
  /usr/bin/git "${GIT_TRUST_ARGS[@]}" branch --show-current
)" = "$RECEIPT_BRANCH"
test "$(
  /usr/bin/git "${GIT_TRUST_ARGS[@]}" rev-parse HEAD
)" = "$RECEIPT_ACCEPTED_COMMIT"
ACCEPTED_ROOT_TREE="$(
  /usr/bin/git "${GIT_TRUST_ARGS[@]}" \
    rev-parse "$RECEIPT_ACCEPTED_COMMIT^{tree}"
)"
test "$ACCEPTED_ROOT_TREE" = "$INDEX_ROOT_TREE"
/usr/bin/git "${GIT_TRUST_ARGS[@]}" merge-base \
  --is-ancestor "$RECEIPT_BASELINE_COMMIT" HEAD
read -r CURRENT_MANIFEST_SHA256 _ < <(
  /usr/bin/sha256sum docs/research_note/alpha_max_final_sha256_20260711.txt
)
test "$CURRENT_MANIFEST_SHA256" = "$RECEIPT_MANIFEST_SHA256"
read -r CURRENT_LOCK_SHA256 _ < <(/usr/bin/sha256sum uv.lock)
test "$CURRENT_LOCK_SHA256" = "$RECEIPT_LOCK_SHA256"

/usr/bin/git "${GIT_TRUST_ARGS[@]}" branch --show-current \
  | /usr/bin/tee "$RUNLOG/branch.txt"
/usr/bin/git "${GIT_TRUST_ARGS[@]}" rev-parse HEAD \
  | /usr/bin/tee "$RUNLOG/worktree-commit.txt"
/usr/bin/git "${GIT_TRUST_ARGS[@]}" rev-parse "$RECEIPT_BASELINE_COMMIT" \
  | /usr/bin/tee "$RUNLOG/frozen-baseline-commit.txt"
/usr/bin/sha256sum -c docs/research_note/alpha_max_final_sha256_20260711.txt \
  | /usr/bin/tee "$RUNLOG/source-sha256-check.txt"
uv sync --frozen --extra dev
uv run --frozen --extra dev python - <<'PY' | /usr/bin/tee "$RUNLOG/frozen-runtime-hashes.txt"
from lumina_quant.research.alpha_max_engine_runner import (
    ALPHA_MAX_CONFIG_FILE_SHA256,
    ALPHA_MAX_CONFIG_PAYLOAD_SHA256,
    ALPHA_MAX_RUNTIME_CONTRACT_SHA256,
)
print("runtime_contract", ALPHA_MAX_RUNTIME_CONTRACT_SHA256)
print("config_payload", ALPHA_MAX_CONFIG_PAYLOAD_SHA256)
print("config_file", ALPHA_MAX_CONFIG_FILE_SHA256)
PY
```

Expected frozen hashes:

```text
runtime_contract b3859443c842cf8b04d04ed32923e6c6a8207af18e26f68a717ba623b4edfef9
config_payload b062e3805d94087cc18cd22634918815503f94dd73f8fa8ac1979e7aef535f85
config_file 2f267451c4df6b6b7471d972b7756327e41c82522ae2ef4b9198fbf6aa8b5e9c
```
The following Rev5.15 files are normative and must match the final SHA-256
manifest:

```text
2f267451c4df6b6b7471d972b7756327e41c82522ae2ef4b9198fbf6aa8b5e9c  configs/research/alpha_max_portfolio_20260711_listing_aware.json
ae272f70f65797b4c8a87c29b7f8e64511617f8e0f2d4bd841b2d1addb7d1220  configs/research/alpha_max_contract_manifest_20260711_listing_aware.json
214e5da198307d8d32b30f69fb6b1f09002e0b31888dc476ed16060f79de9719  configs/research/alpha_max_official_availability_evidence_20260711.json
ea26b902bcec4458340e4c345fa648a3db9104e1b337fd42460d9a9461a738ac  scripts/research/prepare_alpha_max_phase_roots.py
```

`docs/research_note/alpha_max_checkpoint_sha256_20260711.txt` is a historical
mid-implementation checkpoint and is not the data-PC preflight manifest. Only
`alpha_max_final_sha256_20260711.txt` is normative for this handoff.
Rev5.14-named files retained in that manifest are historical audit inputs only;
they are never operational config or contract inputs.

Record the exact environment after removing every forbidden `LQ_*` key. No
profile, YAML, environment fallback, response file, runtime merge, or additional
CLI option is accepted.

```bash
while IFS='=' read -r name _; do
  case "$name" in LQ_*) unset "$name" ;; esac
done < <(env)
test -z "$(env | sed -n 's/^\(LQ_[^=]*\)=.*/\1/p')"
env -0 | sort -z | tr '\0' '\n' > "$RUNLOG/environment.txt"
printf '%q\n' "$REPO" "$DATA" "$PRELOCK_OUT" "$HISTORICAL_OUT" \
  > "$RUNLOG/explicit-paths.txt"
test ! -e "$PRELOCK_OUT"
test ! -L "$PRELOCK_OUT"
test ! -e "$HISTORICAL_OUT"
test ! -L "$HISTORICAL_OUT"
```

## Phase-root contract

Every root is an explicit absolute, read-only, phase-owned tree containing the
frozen ten-symbol declaration. Physical rows are only official phase
intersections; operators must not preselect, substitute, shorten, backfill
symbols/dates, or add post-delivery rows.

| Phase | Start UTC inclusive | End UTC exclusive |
|---|---:|---:|
| warmup | 2022-12-31 00:00:00 | 2024-01-01 00:00:00 |
| train | 2024-01-01 00:00:00 | 2025-06-01 00:00:00 |
| purge | 2025-06-01 00:00:00 | 2025-06-08 00:00:00 |
| validation | 2025-06-08 00:00:00 | 2025-08-31 00:00:00 |
| embargo | 2025-08-31 00:00:00 | 2025-09-07 00:00:00 |
| historical exposed evaluation | 2025-09-07 00:00:00 | 2026-07-01 00:00:00 |

Raw files use canonical monthly partitions such as
`market_ohlcv_1s/binance/BTCUSDT/2024-01.parquet`. Feature roots use
`feature_points/exchange=binance/symbol=.../date=.../part-*.parquet` and must
provide causal funding coverage. Sparse market events are allowed; synthetic
seconds are forbidden. Extra interval ownership, missing required partitions,
unsafe links, multi-linked files, changing content, duplicate/nonmonotone rows,
or incomplete native/funding boundaries fail closed.
TONUSDT raw coverage is exactly `[2024-03-01T12:31:10Z, 2026-06-23T09:00:00Z)`
and feature coverage is exactly `[2024-03-01T16:00:00Z, 2026-06-23T09:00:00Z)`.
Missing TONUSDT warmup or train history rejects TONUSDT admission. GRAMUSDT
substitution, synthetic warmup, synthesized listing-transition funding, date
shifts, and post-delivery rows are forbidden.

## No-discretion phase-root preparation

Prepare phase roots only from existing authorized canonical `market_ohlcv_1s`
and `feature_points` roots. The roots named below must be complete, canonical,
and read-only; never use 1m, synthetic, substitute, or shortened input. The
output root must be absent. Record the manifest and its SHA-256 before either
process.

```bash
ALPHA_SOURCE="/absolute/path/to/authorized/alpha-max-source"
test -d "$ALPHA_SOURCE/market_ohlcv_1s"
test -d "$ALPHA_SOURCE/feature_points"
test ! -e "$DATA"
test ! -L "$DATA"
uv run --frozen --extra dev python scripts/research/prepare_alpha_max_phase_roots.py \
  --raw-root "$ALPHA_SOURCE/market_ohlcv_1s" \
  --feature-root "$ALPHA_SOURCE/feature_points" \
  --contract-manifest "$REPO/configs/research/alpha_max_contract_manifest_20260711_listing_aware.json" \
  --output-root "$DATA"
test -f "$DATA/preparation_manifest.json"
sha256sum "$DATA/preparation_manifest.json" \
  | tee "$RUNLOG/preparation-manifest.sha256"
```

Capture the input inventory before either process:

```bash
find -P "$DATA" -xdev -printf '%m\t%y\t%s\t%p\n' \
  | LC_ALL=C sort > "$RUNLOG/input-inventory-before.tsv"
find -P "$DATA" -xdev -type f -print0 | LC_ALL=C sort -z \
  | xargs -0 sha256sum > "$RUNLOG/input-sha256-before.txt"
```

## Independent sealed-bundle auditor

Create this verifier once. It rejects noncanonical seals, missing/extra files,
symlinks, nonregular/multi-linked files, byte-count/hash drift, or modes other
than read-only files (`0444`) and directories (`0555`).

```bash
cat > "$RUNLOG/verify_sealed_bundle.py" <<'PY'
from __future__ import annotations
import hashlib, json, os, stat, sys
from pathlib import Path, PurePosixPath

def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result

requested_root = Path(sys.argv[1])
if requested_root.is_symlink():
    raise SystemExit("bundle root symlink forbidden")
root = requested_root.resolve(strict=True)
seal_path = root / "SEALED.json"
raw = seal_path.read_bytes()
seal = json.loads(raw, object_pairs_hook=_unique)
canonical = json.dumps(seal, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode() + b"\n"
if raw != canonical:
    raise SystemExit("noncanonical SEALED.json")
key = "artifacts" if "artifacts" in seal else "historical_artifacts"
entries = seal.get(key)
if type(entries) is not list:
    raise SystemExit("missing seal inventory")
expected: dict[str, tuple[int, str]] = {}
for entry in entries:
    if type(entry) is not dict or set(entry) != {"byte_count", "relative_path", "sha256"}:
        raise SystemExit("invalid seal entry")
    rel = entry["relative_path"]
    pure = PurePosixPath(rel)
    if pure.is_absolute() or not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
        raise SystemExit(f"unsafe relative path: {rel!r}")
    if rel in expected:
        raise SystemExit(f"duplicate inventory path: {rel}")
    expected[rel] = (entry["byte_count"], entry["sha256"])
observed: set[str] = set()
for path in root.rglob("*"):
    status = path.lstat()
    if stat.S_ISLNK(status.st_mode):
        raise SystemExit(f"symlink forbidden: {path}")
    if stat.S_ISDIR(status.st_mode):
        if stat.S_IMODE(status.st_mode) != 0o555:
            raise SystemExit(f"directory mode mismatch: {path}")
        continue
    if not stat.S_ISREG(status.st_mode) or status.st_nlink != 1:
        raise SystemExit(f"file identity invalid: {path}")
    if stat.S_IMODE(status.st_mode) != 0o444:
        raise SystemExit(f"file mode mismatch: {path}")
    rel = path.relative_to(root).as_posix()
    if rel == "SEALED.json":
        continue
    observed.add(rel)
    try:
        byte_count, digest = expected[rel]
    except KeyError as exc:
        raise SystemExit(f"unsealed extra file: {rel}") from exc
    payload = path.read_bytes()
    if len(payload) != byte_count or hashlib.sha256(payload).hexdigest() != digest:
        raise SystemExit(f"inventory mismatch: {rel}")
if observed != set(expected):
    raise SystemExit(f"inventory set mismatch: missing={sorted(set(expected)-observed)}")
if stat.S_IMODE(root.stat().st_mode) != 0o555:
    raise SystemExit("root mode mismatch")
print(json.dumps({"inventory_count": len(expected), "root": str(root), "seal_sha256": hashlib.sha256(raw).hexdigest()}, sort_keys=True))
PY
```

## 1. Prelock selection process

The exact command below is the run record. `/usr/bin/time -v` records wall time
and peak RSS. The target must not exist. Do not add or remove an argument.

```bash
cat > "$RUNLOG/prelock-command.txt" <<EOF_CMD
uv run --frozen --extra dev python scripts/research/run_alpha_max_prelock.py --config $REPO/configs/research/alpha_max_portfolio_20260711_listing_aware.json --contract-manifest $REPO/configs/research/alpha_max_contract_manifest_20260711_listing_aware.json --exchange binance --output-root $PRELOCK_OUT --warmup-raw-root $DATA/warmup/raw --warmup-feature-root $DATA/warmup/feature --train-raw-root $DATA/train/raw --train-feature-root $DATA/train/feature --purge-raw-root $DATA/purge/raw --purge-feature-root $DATA/purge/feature --validation-raw-root $DATA/validation/raw --validation-feature-root $DATA/validation/feature --embargo-raw-root $DATA/embargo/raw --embargo-feature-root $DATA/embargo/feature
EOF_CMD
/usr/bin/time -v -o "$RUNLOG/prelock-time.txt" \
  uv run --frozen --extra dev python scripts/research/run_alpha_max_prelock.py \
  --config "$REPO/configs/research/alpha_max_portfolio_20260711_listing_aware.json" \
  --contract-manifest "$REPO/configs/research/alpha_max_contract_manifest_20260711_listing_aware.json" \
  --exchange binance \
  --output-root "$PRELOCK_OUT" \
  --warmup-raw-root "$DATA/warmup/raw" \
  --warmup-feature-root "$DATA/warmup/feature" \
  --train-raw-root "$DATA/train/raw" \
  --train-feature-root "$DATA/train/feature" \
  --purge-raw-root "$DATA/purge/raw" \
  --purge-feature-root "$DATA/purge/feature" \
  --validation-raw-root "$DATA/validation/raw" \
  --validation-feature-root "$DATA/validation/feature" \
  --embargo-raw-root "$DATA/embargo/raw" \
  --embargo-feature-root "$DATA/embargo/feature" \
  > "$RUNLOG/prelock-stdout.txt" 2> "$RUNLOG/prelock-stderr.txt"

test -f "$PRELOCK_OUT/SEALED.json"
uv run --frozen --extra dev python "$RUNLOG/verify_sealed_bundle.py" "$PRELOCK_OUT" \
  | tee "$RUNLOG/prelock-seal-audit.json"
find -P "$PRELOCK_OUT" -xdev -printf '%m\t%y\t%s\t%p\n' \
  | LC_ALL=C sort > "$RUNLOG/prelock-inventory-before.tsv"
find -P "$PRELOCK_OUT" -xdev -type f -print0 | LC_ALL=C sort -z \
  | xargs -0 sha256sum > "$RUNLOG/prelock-before.sha256"
sha256sum "$PRELOCK_OUT/SEALED.json" > "$RUNLOG/prelock-seal.sha256"
```

Required prelock artifacts include:

```text
SEALED.json
admission/train.json
admission/train_computation.json
admission/train_liquidity_buckets.json
allocation/train_fit.json
allocation/train_validation_refit.json
diagnostics/validation/trend_liquidity_falsifier.json
inputs/config.json
inputs/contract_manifest.json
inputs/prior_trial_inventory.json
run/prelock_result.json
selection/prelock.json
status/matrix.json
terminal/prelock.json
trial/ledger.json
manifests/validation_train_fit/*.json
manifests/prelock_final_refit/*.json
capsules/validation_train_fit/*/*.json
capsules/prelock_final_refit/*/*.json
evidence/validation/cells/*/*.json
evidence/validation/rows/*.json
```

There must be exactly 17 manifests in each manifest phase, 68 actual-engine
row/cost cells, and 816 physical fold runs. The wildcard inventories above are
normative: they contain the exact child strategy classes and parameters,
allocation weights and gross caps, causal capsules, effective cost
configurations, attribution receipts, capacity observations, and terminal
evidence. Do not reduce them to a hand-entered summary.

Read back the non-performance control fields and hashes:

```bash
uv run --frozen --extra dev python - "$PRELOCK_OUT" <<'PY' \
  | tee "$RUNLOG/prelock-readback.json"
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1])
def read(path):
    raw = (root / path).read_bytes()
    return hashlib.sha256(raw).hexdigest(), json.loads(raw)
run_sha, run = read("run/prelock_result.json")
sel_sha, selection = read("selection/prelock.json")
diag_sha, diagnostic = read("diagnostics/validation/trend_liquidity_falsifier.json")
print(json.dumps({
    "diagnostic_report_only": diagnostic["report_only"],
    "diagnostic_selection_influence": diagnostic["selection_influence"],
    "diagnostic_sha256": diag_sha,
    "engine_cell_count": run["engine_cell_count"],
    "physical_fold_run_count": run["physical_fold_run_count"],
    "prelock_champion": run["prelock_champion"],
    "run_sha256": run_sha,
    "selection_sha256": sel_sha,
    "status": run["status"],
    "terminal_outcome": run["terminal_outcome"],
}, sort_keys=True))
PY
```

Expected structural values are `engine_cell_count=68`,
`physical_fold_run_count=816`, `status=complete`, and both diagnostic booleans
`report_only=true`, `selection_influence=false`. A null champion and
`no_demonstrated_alpha` are valid scientific outcomes.

Create the compact operator export from the sealed evidence. The exporter
revalidates canonical JSON, requires both 17-row manifest phases and all 68/816
actual-engine cells/runs, preserves every child class/parameter/weight/gross
cap, and records the exact effective configuration and cost-reconciliation
totals for every fold. It keeps the full capacity observations in their sealed
source artifacts while exporting their count, canonical set hash, and
finite-positive summary so the run record does not duplicate a potentially
large order-level ledger.

```bash
uv run --frozen --extra dev python \
  scripts/research/export_alpha_max_observability.py \
  --bundle-root "$PRELOCK_OUT" \
  --manifest-root "$PRELOCK_OUT" \
  --domain validation \
  --output "$RUNLOG/prelock-observability.json" \
  | tee "$RUNLOG/prelock-observability-receipt.json"
sha256sum "$RUNLOG/prelock-observability.json" \
  > "$RUNLOG/prelock-observability.sha256"
```

## 2. Physically separate one-touch historical process

Only after the audited prelock command returns may this process see the exposed
historical roots. Preserve the prelock bundle byte-for-byte. The command has no
config, validation-root, champion, selection, threshold, seed, or override
argument. A successful completion identity cannot be reused; do not rerun after
observing results or tune against this interval.

```bash
cat > "$RUNLOG/historical-command.txt" <<EOF_CMD
uv run --frozen --extra dev python scripts/research/run_alpha_max_historical_evaluation.py --sealed-prelock-directory $PRELOCK_OUT --embargo-feature-root $DATA/embargo/feature --historical-evaluation-raw-root $DATA/historical_exposed_evaluation/raw --historical-evaluation-feature-root $DATA/historical_exposed_evaluation/feature --exchange binance --output-root $HISTORICAL_OUT
EOF_CMD
/usr/bin/time -v -o "$RUNLOG/historical-time.txt" \
  uv run --frozen --extra dev python \
  scripts/research/run_alpha_max_historical_evaluation.py \
  --sealed-prelock-directory "$PRELOCK_OUT" \
  --embargo-feature-root "$DATA/embargo/feature" \
  --historical-evaluation-raw-root "$DATA/historical_exposed_evaluation/raw" \
  --historical-evaluation-feature-root "$DATA/historical_exposed_evaluation/feature" \
  --exchange binance \
  --output-root "$HISTORICAL_OUT" \
  > "$RUNLOG/historical-stdout.txt" 2> "$RUNLOG/historical-stderr.txt"

test -f "$HISTORICAL_OUT/SEALED.json"
uv run --frozen --extra dev python "$RUNLOG/verify_sealed_bundle.py" "$HISTORICAL_OUT" \
  | tee "$RUNLOG/historical-seal-audit.json"
find -P "$PRELOCK_OUT" -xdev -printf '%m\t%y\t%s\t%p\n' \
  | LC_ALL=C sort > "$RUNLOG/prelock-inventory-after.tsv"
find -P "$PRELOCK_OUT" -xdev -type f -print0 | LC_ALL=C sort -z \
  | xargs -0 sha256sum > "$RUNLOG/prelock-after.sha256"
diff -u "$RUNLOG/prelock-inventory-before.tsv" "$RUNLOG/prelock-inventory-after.tsv"
diff -u "$RUNLOG/prelock-before.sha256" "$RUNLOG/prelock-after.sha256"
find -P "$DATA" -xdev -type f -print0 | LC_ALL=C sort -z \
  | xargs -0 sha256sum > "$RUNLOG/input-sha256-after.txt"
diff -u "$RUNLOG/input-sha256-before.txt" "$RUNLOG/input-sha256-after.txt"
sha256sum "$HISTORICAL_OUT/SEALED.json" > "$RUNLOG/historical-seal.sha256"
```

Required historical artifacts include:

```text
SEALED.json
admission/train_liquidity_buckets.json
binding/prelock_seal.json
diagnostics/historical_exposed_evaluation/trend_liquidity_falsifier.json
report/historical_result.json
selection/historical_ranking.json
status/matrix.json
terminal/historical.json
evidence/historical_exposed_evaluation/cells/*/*.json
evidence/historical_exposed_evaluation/rows/*.json
```

The historical evidence inventory must contain the same 68 actual-engine
row/cost cells and exactly 680 physical fold runs. Its manifests and initial
causal capsules remain byte-owned by the sealed prelock bundle; the historical
receipts bind to those exact hashes rather than rematerializing them.

Read back the final structural outcome:

```bash
uv run --frozen --extra dev python - "$HISTORICAL_OUT" <<'PY' \
  | tee "$RUNLOG/historical-readback.json"
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1])
def read(path):
    raw = (root / path).read_bytes()
    return hashlib.sha256(raw).hexdigest(), json.loads(raw)
report_sha, report = read("report/historical_result.json")
terminal_sha, terminal = read("terminal/historical.json")
diag_sha, diagnostic = read("diagnostics/historical_exposed_evaluation/trend_liquidity_falsifier.json")
print(json.dumps({
    "confirmation_status": report["confirmation_status"],
    "diagnostic_report_only": diagnostic["report_only"],
    "diagnostic_selection_influence": diagnostic["selection_influence"],
    "diagnostic_sha256": diag_sha,
    "historical_evaluation_leader": report["historical_evaluation_leader"],
    "physical_fold_run_count": report["physical_fold_run_count"],
    "prelock_champion": report["prelock_champion"],
    "report_sha256": report_sha,
    "requires_fresh_confirmation": report["requires_fresh_confirmation"],
    "terminal_outcome": report["terminal_outcome"],
    "terminal_sha256": terminal_sha,
}, sort_keys=True))
PY
```

Expected structural values are `physical_fold_run_count=680`,
`requires_fresh_confirmation=true`, `confirmation_status=not_run`, and the same
report-only/non-selection diagnostic booleans. The historical leader is never a
selected or deployable id.

Export the historical fold observability while explicitly sourcing both
manifest phases from the still-byte-identical prelock bundle:

```bash
uv run --frozen --extra dev python \
  scripts/research/export_alpha_max_observability.py \
  --bundle-root "$HISTORICAL_OUT" \
  --manifest-root "$PRELOCK_OUT" \
  --domain historical_exposed_evaluation \
  --output "$RUNLOG/historical-observability.json" \
  | tee "$RUNLOG/historical-observability-receipt.json"
sha256sum "$RUNLOG/historical-observability.json" \
  > "$RUNLOG/historical-observability.sha256"
```

For every fold, the export includes the row id, nominal cost, seed, complete
effective configuration and hash, runtime/config/universe/root bindings, event
counts, ending cash/equity and ruin state, native finalization, plus these exact
cost fields: `model_commission_total`, `applied_commission_total`,
`portfolio_fee_total`, `funding_payment_total`, `portfolio_funding_total`,
`liquidation_cost_total`, and `portfolio_liquidation_total`. It also carries
pricing/application/no-fill counts and set hashes, all reconciliation booleans,
turnover/RPT, capacity count/summary/hash, target and realized gross, clip
counts, and per-symbol contribution totals/residuals. Missing fields or an
incorrect 68/680 structure make the exporter fail nonzero.

## Local process-control coverage before transfer

The repository-side child-process suite covers P01-P26 at the public CLI,
filesystem, seal, constructor, and activation boundaries. In particular, P23
uses transient manifest/config bytes during the actual consumer descriptor
open; P24 and P25 mutate the actual funding lookup, resolver, raw accessor, and
portfolio identities after construction; and P26 crosses the public prelock CLI
into the incumbent-audit preflight. Every hostile case is rejected before a
market/funding/order/fill/trade event.

P11 locally proves that the production row/cost/fold control invokes each of
the 816 validation and 680 historical schedules exactly once and never invokes
an unavailable incumbent or diagnostic row. Its replay payload is deterministic
test data, not physical market replay. Therefore only the commands in this
runbook, with all complete frozen data roots, can supply the performance-bearing
P11 replay evidence. Do not replace those roots with synthetic data or infer a
performance result from the local control-flow test.

## Failure taxonomy and recovery boundary

Input/schema/root/hash/identity/coverage/admission/capsule/manifest/config/runtime
failures occur before a valid final bundle. Engine/statistical/funding/cost/
reconciliation/inventory failures also fail closed. A target directory without
`SEALED.json` is invalid and must never be read as a result; the process attempts
to remove its entire owned partial tree. Never repair an output artifact or
resume inside it. Correct only an objectively invalid external input, choose a
new absent output path, preserve the failed logs, and rerun the same frozen
command. Never change dates, membership, thresholds, costs, seed, gates, or
code to obtain a survivor.

Missing coverage is reported as missing; it is not replaced with another symbol,
shorter interval, synthetic bars, or an ambient feature path. A historical
completion-claim conflict means the one-touch identity was already consumed;
it is not permission to create a new identity after viewing the result.

## Mandatory interpretation and no-claim boundary

- `no_demonstrated_alpha`: no validation row survived the frozen gates.
- `historical_evaluation_incomplete`: a champion exists but its exposed report
  is missing or invalid.
- `prelock_champion_historical_robustness_failed`: the immutable champion failed
  at least one exposed historical gate.
- `prelock_champion_historical_robustness_passed`: the immutable champion passed
  the fixed exposed gates, but this is not confirmation or deployment evidence.

DSR, SPA, and PBO answer different questions and are not interchangeable: DSR
corrects Sharpe significance for multiple trials/non-normality, SPA tests
relative predictive superiority under the frozen comparison set, and PBO
measures selection-overfit risk. Failure of any required gate remains failure.

Turnover/RPT, capacity, target-vs-realized gross, per-symbol contribution, and
train-frozen liquidity buckets are report-only diagnostics. A liquidity
falsifier pass is only `liquidity_falsifier_not_triggered`; it does not support a
causal or broad-momentum claim. `trend_mechanism_not_supported` is mandatory if
the liquid bucket is nonpositive or positive edge is confined to the weakest
bucket.

Scaled-vs-1x improvements are labeled `risk_transform_not_alpha`. The passive
scaled counterfactual is absent, so scaling cannot be described as a distinct
alpha source. Component/portfolio/control/LOO collisions remain separate rows;
unavailable incumbents and diagnostic evidence tiers cannot select or enter the
MDD comparator.

All results use exposed 2025-09-07 through 2026-07-01 historical data. Even a
passing champion remains research-only with `confirmation_status=not_run` and
requires a genuinely fresh, uninspected future/withheld interval under a new
predeclared protocol. No output from this run is “best,” confirmed, prospective,
deployable, or approved for capital.
