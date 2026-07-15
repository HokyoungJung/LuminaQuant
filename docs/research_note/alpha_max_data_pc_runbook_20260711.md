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
config, dates, symbols, thresholds, costs, seeds, or output artifacts on the
data PC.

```bash
set -euo pipefail
umask 077
REPO="$(pwd -P)"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
RUNLOG="/absolute/path/to/alpha-max-run-record-$RUN_ID"
DATA="/absolute/path/to/alpha-max-phase-roots"
PRELOCK_OUT="/absolute/path/to/new/alpha-max-prelock-$RUN_ID"
HISTORICAL_OUT="/absolute/path/to/new/alpha-max-historical-$RUN_ID"
mkdir -p "$RUNLOG"

command -v /usr/bin/time
command -v sha256sum
git cat-file -e 629d91e5d4aac26911af65a4a5e15ebdcbded30f^{commit}
git branch --show-current | tee "$RUNLOG/branch.txt"
git rev-parse HEAD | tee "$RUNLOG/worktree-commit.txt"
git rev-parse 629d91e5d4aac26911af65a4a5e15ebdcbded30f \
  | tee "$RUNLOG/frozen-baseline-commit.txt"
git status --porcelain=v1 | tee "$RUNLOG/worktree-status.txt"
test ! -s "$RUNLOG/worktree-status.txt"
sha256sum -c docs/research_note/alpha_max_final_sha256_20260711.txt \
  | tee "$RUNLOG/source-sha256-check.txt"
uv sync --frozen --extra dev
uv run --frozen --extra dev python - <<'PY' | tee "$RUNLOG/frozen-runtime-hashes.txt"
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
