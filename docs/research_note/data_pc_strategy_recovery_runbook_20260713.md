# Data-PC Strategy Recovery Runbook

작성일: 2026-07-13
상위 계획: [`strategy_recovery_master_plan_20260713.md`](strategy_recovery_master_plan_20260713.md)
권한: 데이터 인벤토리, 결손 복구와 research-only 실행 준비. 주문·paper·testnet·live·실자본 권한 없음.

## 0. 다른 PC에서 한 문장으로 시작하기

다른 PC의 코딩 에이전트에는 다음 한 문장만 전달한다.

> 이 저장소를 최신 `origin/main`으로 안전하게 동기화하고, `docs/research_note/data_pc_strategy_recovery_runbook_20260713.md` 0절의 실행 계약에 따라 상위 master plan을 의존성 순서대로 구현·테스트·실행해라.

에이전트는 아래 계약을 별도 확인 없이 수행한다.

1. **안전한 최신화**
   - 현재 작업을 삭제하거나 덮어쓰지 않는다.
   - worktree가 clean이면 `git fetch origin --prune`, `git switch main`, `git pull --ff-only origin main`을 수행한다.
   - dirty이면 기존 파일을 건드리지 말고 `origin/main`에서 새 clean worktree/branch를 만든다.
   - `git merge-base --is-ancestor 09e9bee origin/main`과 `HEAD == origin/main`을 확인하고 실제 commit을 실행 기록에 남긴다.
2. **경로와 환경 자동 발견**
   - `REPO="$(git rev-parse --show-toplevel)"`를 사용하고 `/home/hoky`를 가정하지 않는다.
   - `RUN_BASE`가 주어지지 않으면 repo 밖의 `$HOME/quants-recovery-runs`를 사용한다.
   - `SOURCE_ROOT`, `MARKET_ROOT`, `ALPHA_SOURCE`는 이 runbook의 bounded inventory로 찾는다. 여러 개면 provenance와 coverage가 가장 완전한 root를 선택하고 근거를 기록한다.
   - `uv sync --frozen --extra optimize --extra dev`가 실패하면 원인을 수정하거나 blocker receipt를 남기며 dependency version을 임의로 바꾸지 않는다.
3. **권위 문서와 실행 순서**
   - 먼저 [`strategy_recovery_master_plan_20260713.md`](strategy_recovery_master_plan_20260713.md), 이 runbook, [`strategy_reality_audit_20260713.md`](../audits/strategy_reality_audit_20260713.md)를 읽는다.
   - 첫 묶음 `D-01/D-01A/D-05/A-01`, 다음 `D-02/D-03/D-04/A-02`, 이어서 `R-01 -> R-02 -> R-03 -> R-04`와 `A-03` 순서를 지킨다. `C-00` 초안은 D-01 뒤 병렬 준비할 수 있지만 `C-01` 이후는 R-04/A-03 판정 뒤에만 실행하고, F-01도 R-04/A-03 뒤에 실행한다.
   - 문서에 없는 명령, 데이터, symbol, 날짜, parameter를 추측하지 않는다.
4. **미구현 blocker 처리**
   - D-01A, D-04, D-05, R-01~R-03처럼 코드가 없는 항목은 단순 STOP으로 끝내지 않는다. 기존 validator/runner/registry를 재사용하는 최소 구현과 targeted regression test로 blocker를 닫고 다음 단계로 진행한다.
   - 테스트가 실패하면 수정 후 재실행한다. gate를 완화하거나 locked OOS를 보고 parameter를 바꾸지 않는다.
   - 데이터·디스크·credential처럼 코드로 해결할 수 없는 blocker는 immutable receipt와 필요한 정확한 경로/기간을 남기고, 실행 가능한 독립 task를 계속한다.
5. **데이터와 Git 경계**
   - 원본 data root는 read-only로 취급한다. synthetic fill, symbol 대체, 상장 전 backfill, interior-gap 무단 append를 금지한다.
   - market parquet, credential, host, secret과 대용량 run artifact는 Git에 넣지 않는다. Git에는 최소 코드, 테스트, config/manifest, 작은 evidence와 문서만 넣는다.
   - source 변경은 새 recovery branch에 작은 단위로 commit하고 force-push하지 않는다. 검증 후 branch를 push한다.
6. **완료 조건**
   - 각 task에 command, Git/config/data hash, validation receipt, 테스트 결과와 PASS/STOP/KILL 판정을 남긴다.
   - 가능한 task가 모두 terminal이고 미해결 외부 blocker가 정확히 문서화될 때까지 계속한다.
   - 최종 보고에는 완료 task, scientific reject를 포함한 전 후보, 미해결 blocker, branch/commit과 artifact 경로를 적는다.
   - 모든 단계에서 주문·paper·testnet·live·실자본 배분은 `0%`다.

## 1. 이 runbook에서 지금 실행할 범위

데이터 PC에서 바로 수행할 것은 다음뿐이다.

1. 기존 데이터 root 발견과 read-only 인벤토리
2. core 10의 원래 Router 기간 기준 1m OHLCV 결손 확인
3. 결손이 있을 때만 공식 archive로 append
4. funding/support coverage 확인과 최소 복구
5. Alpha-Max canonical 1s/funding source가 있으면 Rev5.15 phase roots 준비

고-CAGR final replay, Alpha-Max prelock/historical, fresh-forward는 각각의 선행 blocker가 닫히기 전 실행하지 않는다.

상위 계획 6장의 후속 alpha/volatility 프로그램도 지금은 비활성이다. 아래의 조건부 데이터 계약 초안만 준비할 수 있으며, candidate 성과 실행과 V-DIAG는 `R-04`, `A-03`, `C-00`, `D-01A`, `D-04`가 모두 완료되기 전 시작하지 않는다.

## 2. 안전 규칙

- `set -euo pipefail`, UTC, 새 `RUN_DIR`를 사용한다.
- 기존 데이터 root는 원본으로 취급하고 먼저 inventory를 남긴다.
- `--no-resume`, synthetic fill, symbol 대체, 날짜 변경을 사용하지 않는다.
- collector는 항상 고정 symbol 목록과 명시한 기간으로 실행한다.
- broad current universe discovery는 coverage 조사에만 사용하고 frozen 성과 실험에는 사용하지 않는다.
- collector/report output과 market data root를 같은 디렉터리에 두지 않는다.
- 실패한 실행의 output을 고치지 말고 새 `RUN_ID`로 재실행한다.
- 현재 main의 `scripts/materialize_market_windows.py --help`는 실패한다. D-05가 수정되기 전 final raw-first materialization 명령을 실행하지 않는다.

## 3. 저장소와 실행 기록 준비

```bash
set -euo pipefail
umask 077
export TZ=UTC

REPO="$(git rev-parse --show-toplevel)"
RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
RUN_BASE="${RUN_BASE:-$HOME/quants-recovery-runs}"
RUN_DIR="$RUN_BASE/$RUN_ID"
mkdir -p "$RUN_DIR"

cd "$REPO"
git fetch origin --prune
git status --porcelain=v1 | tee "$RUN_DIR/main-status.txt"
test ! -s "$RUN_DIR/main-status.txt"
git rev-parse HEAD | tee "$RUN_DIR/main-commit.txt"
uv sync --frozen --extra optimize --extra dev
uv run python --version | tee "$RUN_DIR/python-version.txt"
env | sed 's/=.*//' | LC_ALL=C sort -u > "$RUN_DIR/environment-variable-names.txt"
uv pip freeze > "$RUN_DIR/python-packages.txt"
```

`git status`가 비어 있지 않으면 다른 clean checkout/worktree를 만들고 다시 시작한다. 데이터 파일 자체는 Git에 추가하지 않는다.

## 4. 기존 데이터 root 찾기

먼저 알려진 위치만 검사한다.

```bash
for candidate in \
  "$REPO/data/market_parquet" \
  "$REPO/LuminaQuant/data/market_parquet" \
  "/data/market_parquet" \
  "/mnt/d/market_parquet"
do
  if test -d "$candidate"; then
    printf '%s\n' "$(readlink -f "$candidate")"
  fi
done | tee "$RUN_DIR/data-root-candidates.txt"
```

없으면 bounded search를 한 번 수행한다.

```bash
for root in /home /data /mnt; do
  test ! -d "$root" || find "$root" -maxdepth 7 -type d -name market_parquet -print
done 2>/dev/null | LC_ALL=C sort -u \
  | tee -a "$RUN_DIR/data-root-candidates.txt"
```

실제 과거 보고서가 사용한 후보는 `.../LuminaQuant/data/market_parquet`였다. 디렉터리가 있다는 이유만으로 선택하지 말고 아래 repository coverage가 가장 완전하고 provenance를 설명할 수 있는 root를 사용한다.

```bash
: "${SOURCE_ROOT:?set SOURCE_ROOT to the selected existing market_parquet root}"
export SOURCE_ROOT
export MARKET_ROOT="${MARKET_ROOT:-$HOME/quants-recovery-market/$RUN_ID/market_parquet}"
test -d "$SOURCE_ROOT"
test ! -L "$SOURCE_ROOT"
test ! -e "$MARKET_ROOT"
printf '%s\n' "$(readlink -f "$SOURCE_ROOT")" | tee "$RUN_DIR/source-root.txt"
find -P "$SOURCE_ROOT" -xdev -printf '%m\t%y\t%s\t%T@\t%p\n' \
  | LC_ALL=C sort > "$RUN_DIR/source-file-inventory.tsv"
mkdir -p "$MARKET_ROOT"
cp -a --reflink=auto "$SOURCE_ROOT"/. "$MARKET_ROOT"/
test -z "$(find -P "$MARKET_ROOT" -xdev -type l -print -quit)"
test -z "$(find -P "$MARKET_ROOT" -xdev -type f -links +1 -print -quit)"
printf '%s\n' "$(readlink -f "$MARKET_ROOT")" | tee "$RUN_DIR/market-root.txt"
```

`SOURCE_ROOT`는 이후 수정하지 않는다. `MARKET_ROOT`는 별도 writable filesystem snapshot으로 준비해도 되며, 위 reflink/copy에 필요한 공간이 없으면 STOP한다.

## 5. 변경 전 인벤토리

core 10은 새 전략 목록이 아니라 고정된 최소 데이터 검증 묶음이다.

```bash
CORE=(BTCUSDT ETHUSDT BNBUSDT SOLUSDT XRPUSDT ADAUSDT DOGEUSDT AVAXUSDT LINKUSDT LTCUSDT)
ROUTER_SINCE="2025-01-01T00:00:00Z"
ROUTER_UNTIL="2026-06-30T23:59:59.999Z"
```

물리 layout과 실제 repository loader를 별도로 검사한다. `--scan-dir`과 `--root`는 서로 다른 모드이므로 한 명령에 같이 쓰지 않는다.

```bash
uv run python scripts/research/report_data_coverage.py \
  --scan-dir "$MARKET_ROOT" \
  --exchange binance \
  --symbols "${CORE[@]}" \
  --timeframes 1m 1h 4h 1d \
  --min-bars 360 \
  --json "$RUN_DIR/physical-coverage-before.json" \
  | tee "$RUN_DIR/physical-coverage-before.txt"

uv run python scripts/research/report_data_coverage.py \
  --root "$MARKET_ROOT" \
  --exchange binance \
  --symbols "${CORE[@]}" \
  --timeframes 1m 1h 4h 1d \
  --min-bars 360 \
  --json "$RUN_DIR/repository-coverage-before.json" \
  | tee "$RUN_DIR/repository-coverage-before.txt"

uv run python scripts/build_strategy_support_inventory.py \
  --symbols "${CORE[@]}" \
  --db-path "$MARKET_ROOT" \
  --exchange-id binance \
  --json-path "$RUN_DIR/strategy-support-before.json" \
  --csv-path "$RUN_DIR/strategy-support-before.csv"
```

이 세 결과는 root와 loader의 **triage coverage**일 뿐 데이터 무결성 영수증이 아니다. 현재 CLI만으로는 duplicate, nonmonotone, nonfinite, expected 1m grid의 interior gap, funding expected settlement gap을 fail-close할 수 없다. master plan의 D-01A validator가 별도 JSON receipt를 만들기 전에는 이를 통과 증거로 사용하지 않는다.

파일명, 크기, mode와 mtime을 기록한다. 전체 대용량 root SHA는 초기 단계에서 강제하지 않고, 실제 frozen experiment가 소유할 subset을 정한 뒤 그 subset만 hash한다.

```bash
find -P "$MARKET_ROOT" -xdev -printf '%m\t%y\t%s\t%T@\t%p\n' \
  | LC_ALL=C sort > "$RUN_DIR/market-file-inventory-before.tsv"
```

여기서 멈추고 JSON을 확인한다. 이미 원래 Router 기간을 덮는 series는 다시 다운로드하지 않는다.

## 6. Router용 1m 결손 복구

Router의 원래 train start는 `2025-01-01`이다. 이 트랙 때문에 2024년 데이터를 강제로 만들지 않는다. 위 `ROUTER_UNTIL`은 2026-06 월말로 동결해 7월 이후를 historical 재선택에 섞지 않는다.

먼저 dry-run을 수행한다. 이 dry-run은 market root에 쓰지 않지만 archive를 조회할 수 있다. 이 collector는 각 series의 마지막 timestamp 다음부터 재개하는 **tail append 전용**이다. existing interval 내부의 gap은 찾거나 고치지 않는다.

```bash
uv run python scripts/collect_binance_1m_research_universe.py \
  --source data-vision \
  --db-path "$MARKET_ROOT" \
  --exchange binance \
  --since "$ROUTER_SINCE" \
  --until "$ROUTER_UNTIL" \
  --symbols "${CORE[@]}" \
  --workers 4 \
  --global-request-interval-sec 1.0 \
  --dry-run \
  --report-dir "$RUN_DIR/ohlcv-1m-dry-run"
```

dry-run report는 마지막 timestamp 이후의 tail 계획만 보여 주며 interior gap 부재를 증명하지 않는다. D-01A validator의 pre-append receipt가 모든 대상 series에 대해 `interior_gap_count=0`, `missing_prefix_count=0`, `safe_tail_append=true`를 기록하기 전에는 `--dry-run`을 제거하지 않는다.

interior 또는 prefix gap이 하나라도 있으면 **STOP**한다. 아래 일반 collector를 실행하지 말고, 누락 interval을 명시적으로 소유하는 bounded gap-repair task와 immutable plan을 별도로 만든 뒤 validator를 다시 통과시킨다. 아래 실행 명령은 receipt가 허용한 연속 tail에만 사용한다.

```bash
uv run python scripts/collect_binance_1m_research_universe.py \
  --source data-vision \
  --db-path "$MARKET_ROOT" \
  --exchange binance \
  --since "$ROUTER_SINCE" \
  --until "$ROUTER_UNTIL" \
  --symbols "${CORE[@]}" \
  --workers 4 \
  --global-request-interval-sec 1.0 \
  --report-dir "$RUN_DIR/ohlcv-1m-executed"
```

최근 완전한 archive 이후의 작은 tail이 성과 window에 꼭 필요한 경우에만 `--source fapi`를 별도 `RUN_ID`로 사용한다. 과거 전구간을 FAPI로 긁지 않는다. 429/418이 발생하면 STOP하고 기다린 뒤 새 run record로 재개한다.

현재 static 또는 `static-plus-fapi-tradfi` universe를 frozen 실험 입력으로 사용하지 않는다. R-01이 만든 exact router manifest의 symbol tuple이 준비되면 같은 collector를 그 tuple에만 다시 적용한다.

## 7. Funding 최소 복구

먼저 funding만 plan-only로 확인한다. OI와 liquidation은 초기 Router/trend gate의 blocker가 아니며 수집하지 않는다.

```bash
uv run python scripts/collect_strategy_support_data.py \
  --symbols "${CORE[@]}" \
  --db-path "$MARKET_ROOT" \
  --exchange-id binance \
  --since "$ROUTER_SINCE" \
  --until "$ROUTER_UNTIL" \
  --include-funding \
  --skip-mark-index \
  --skip-open-interest \
  --skip-liquidations \
  --retries 3 \
  --backend parquet \
  | tee "$RUN_DIR/funding-plan.json"
```

plan이 정확하면 `--execute`만 추가한다. 수집 자체는 허용하지만 이 report는 settlement 완전성을 증명하지 않는다. 이후 D-01A post-append receipt가 expected funding cadence gap 0을 증명하기 전에는 전략 실행으로 넘어가지 않는다.

```bash
uv run python scripts/collect_strategy_support_data.py \
  --symbols "${CORE[@]}" \
  --db-path "$MARKET_ROOT" \
  --exchange-id binance \
  --since "$ROUTER_SINCE" \
  --until "$ROUTER_UNTIL" \
  --include-funding \
  --skip-mark-index \
  --skip-open-interest \
  --skip-liquidations \
  --retries 3 \
  --backend parquet \
  --execute \
  | tee "$RUN_DIR/funding-executed.json"
```

진짜 spot-perp basis/carry 트랙이 별도 preregistration을 마친 뒤에만 mark/index와 spot 양 leg를 추가한다. `--feature-profile strategy-used`는 mark/index를 제외하므로 basis proof에는 사용할 수 없다.

### 7.1 조건부 TradFi/precious-metal 데이터 계약

이 계약은 상위 계획의 C-00 manifest 초안을 위한 inventory 규칙이다. `CORE` 배열이나 Router의 정확히 두 후보 manifest에 TradFi/금속 symbol을 추가하지 않는다. 후보 데이터는 별도 `ALPHA_CANDIDATE_MANIFEST`와 SHA-256으로만 고정한다.

각 series는 최소한 다음 필드를 가진다.

- `canonical_symbol`, venue symbol, venue, instrument type, quote currency
- contract multiplier, price/tick unit, session/calendar, timezone
- first/last tradable timestamp와 point-in-time listing/delist provenance
- perpetual이면 mark/index와 실제 funding settlement cadence
- futures면 expiry, front/next contract mapping, roll rule과 실제 roll cost
- spot/CFD면 executable venue와 financing/borrow 조건

실제 완결된 raw/1m OHLCV와 executable BBO 또는 동등한 비용 provenance를 우선 사용한다. 1m가 있는 경우 4h/1d는 결정적으로 파생한다. daily-only 후보는 point-in-time 공식 daily OHLCV를 별도 계약으로 허용하지만 이를 intraday realized volatility나 체결비용의 대용물로 쓰지 않는다. daily realized volatility는 intraday log return 제곱합으로 계산한다. 가격 수준 rolling standard deviation, 월요일 label에 그 주 미래 평균을 넣는 `dat_ave(..., 'WS')`, current universe의 과거 역적용은 금지한다.

조건부 결손 복구 순서는 다음과 같다.

1. candidate manifest의 첫 평가 신호와 동결한 class/timeframe warmup으로 series별 owned interval을 계산한다.
2. source root를 read-only inventory하고 exact venue/contract series의 overlap과 gap을 기록한다.
3. 연속 tail만 공식 archive/API로 append한다. interior gap, expiry/roll gap 또는 prefix gap은 bounded repair manifest를 따로 만들고 symbol 대체나 synthetic fill을 하지 않는다.
4. pre/post D-01A receipt가 기존 구간 불변, expected-grid/lifecycle/session/funding gap 0을 증명한 뒤 frozen subset만 hash한다.

Router의 `2025-01-01` train start는 이 절 때문에 바뀌지 않는다. 예를 들어 C-ANCHOR를 2025-01-01부터 계산하려면 그 전에 실제 52-week-equivalent completed bars가 필요하다. 정확한 bar 수는 TradFi session calendar와 24/7 crypto calendar를 구분해 manifest에 고정하며, 데이터가 없으면 그 candidate의 최초 eligible signal만 늦추고 2024년 가격을 만들어 내지 않는다.

XAU/XAG/XPT/XPD 이름만으로 spot benchmark, CFD, CME futures와 Binance TradFi perpetual을 치환하지 않는다. `static-plus-fapi-tradfi` 같은 broad discovery 결과는 inventory용일 뿐 frozen experiment 입력이 아니다. option IV/RV는 executable chain/surface snapshot이, commodity curve carry는 expiry별 settle/BBO와 roll calendar가 생길 때까지 BLOCKED다. auxiliary `precious_metal` 저장소의 credential, host 또는 secret은 복사하거나 bundle/Git에 넣지 않는다.

D-01A receipt는 공통 overlap, expected grid gap, duplicate/nonfinite, session/calendar, lifecycle, funding cadence와 source provenance를 fail-close해야 한다. 한 항목이라도 불명확하면 해당 candidate만 STOP하고 Router/Alpha-Max 복구 범위를 넓히지 않는다.

## 8. 변경 후 인벤토리와 STOP 판정

```bash
uv run python scripts/research/report_data_coverage.py \
  --root "$MARKET_ROOT" \
  --exchange binance \
  --symbols "${CORE[@]}" \
  --timeframes 1m 1h 4h 1d \
  --min-bars 360 \
  --json "$RUN_DIR/repository-coverage-after.json" \
  | tee "$RUN_DIR/repository-coverage-after.txt"

uv run python scripts/build_strategy_support_inventory.py \
  --symbols "${CORE[@]}" \
  --db-path "$MARKET_ROOT" \
  --exchange-id binance \
  --json-path "$RUN_DIR/strategy-support-after.json" \
  --csv-path "$RUN_DIR/strategy-support-after.csv"

find -P "$MARKET_ROOT" -xdev -printf '%m\t%y\t%s\t%T@\t%p\n' \
  | LC_ALL=C sort > "$RUN_DIR/market-file-inventory-after.tsv"
```

현재 repository에는 아래 계약을 완전히 증명하는 실행 CLI가 없다. 따라서 이 단계의 첫 구현 작업은 master plan D-01A이며, 기존 `src/lumina_quant/compute/ohlcv_validation.py::validate_ohlcv_frame`을 재사용하고 expected-grid 및 funding-settlement 검사를 추가한다. funding은 고정 8시간을 가정하지 않고 해당 symbol/time의 거래소 원본 cadence/next-funding provenance를 기준으로 검사한다. 전략 gate가 소비할 JSON receipt의 최소 계약은 다음과 같다.

```json
{
  "artifact_kind": "research_data_contract_validation",
  "mode": "post_append_strict",
  "passed": true,
  "series": [{
    "duplicate_count": 0,
    "nonmonotone_count": 0,
    "nonfinite_count": 0,
    "expected_grid_gap_count": 0
  }],
  "funding": [{
    "unexpected_settlement_gap_count": 0
  }]
}
```

pre-append mode는 `interior_gap_count`, `missing_prefix_count`, `missing_tail_count`, `safe_tail_append`를 추가로 출력한다. post-append strict receipt의 `passed=true`가 없으면 아래 STOP 목록을 사람이 눈으로 확인했더라도 전략 실행으로 넘어가지 않는다.

다음 중 하나면 전략 실행으로 넘어가지 않는다.

- required interval 안의 설명되지 않는 gap/duplicate/nonmonotone row
- funding settlement 누락
- source 또는 listing provenance 부재
- current static universe만 있고 fold별 membership이 없음
- synthetic CSV가 loader에 섞임
- append 전후 기존 구간 값이 바뀜

## 9. 고-CAGR Router 실행 전 blocker

현 runner의 full rerun은 discovery용이며 exact R1/R2만 재생하는 CLI가 아니다. 또한 비용 stress가 proxy 10/15/20bp이고 final actual-engine proof가 아니다. 다음 네 산출물이 main에 merge되고 검증되기 전 고-CAGR 성과 명령을 실행하지 않는다.

1. exact R1/R2 frozen manifest replay seam
2. point-in-time symbol lifecycle manifest
3. `generic_fallback_proxy=0` fail-close receipt
4. 단일 strict + cost-realistic replacement profile과 10/15/20/30bp proof path

준비가 끝난 후에도 original Router 경계는 그대로 사용한다.

```text
train_start=2025-01-01
first_oos_start=2025-09-01
candidate_count=2
new_grid_search=false
recompute_from_json=false
post_oos_augment=false
```

## 10. Alpha-Max Rev5.15 phase roots 준비

Alpha-Max는 main 데이터와 실행 트리를 섞지 않는다.

```bash
ALPHA_REPO="$(dirname "$REPO")/Quants-agent-alpha-max-data-pc"
ALPHA_COMMIT="629d91e5d4aac26911af65a4a5e15ebdcbded30f"

cd "$REPO"
git fetch origin feat/alpha-max-20260710
test ! -e "$ALPHA_REPO"
git worktree add --detach "$ALPHA_REPO" "$ALPHA_COMMIT"

cd "$ALPHA_REPO"
test "$(git rev-parse HEAD)" = "$ALPHA_COMMIT"
test -z "$(git status --porcelain=v1)"
uv sync --frozen --extra dev
printf '%s  %s\n' \
  '2f267451c4df6b6b7471d972b7756327e41c82522ae2ef4b9198fbf6aa8b5e9c' \
  'configs/research/alpha_max_portfolio_20260711_listing_aware.json' \
  'ae272f70f65797b4c8a87c29b7f8e64511617f8e0f2d4bd841b2d1addb7d1220' \
  'configs/research/alpha_max_contract_manifest_20260711_listing_aware.json' \
  '214e5da198307d8d32b30f69fb6b1f09002e0b31888dc476ed16060f79de9719' \
  'configs/research/alpha_max_official_availability_evidence_20260711.json' \
  'ea26b902bcec4458340e4c345fa648a3db9104e1b337fd42460d9a9461a738ac' \
  'scripts/research/prepare_alpha_max_phase_roots.py' \
  | sha256sum -c - \
  | tee "$RUN_DIR/alpha-max-rev515-sha256-check.txt"
```

canonical source는 이미 존재하는 실제 1s monthly parquet와 funding feature root여야 한다. 일반 1m collector 결과를 Alpha-Max 입력으로 쓰지 않는다.

```bash
: "${ALPHA_SOURCE:?set ALPHA_SOURCE to the discovered canonical alpha source}"
ALPHA_PHASES="${ALPHA_PHASES:-$HOME/quants-alpha-phase-roots/v515-$RUN_ID}"
test -d "$ALPHA_SOURCE/market_ohlcv_1s"
test -d "$ALPHA_SOURCE/feature_points"
test ! -e "$ALPHA_PHASES"

uv run --frozen --extra dev python scripts/research/prepare_alpha_max_phase_roots.py \
  --raw-root "$ALPHA_SOURCE/market_ohlcv_1s" \
  --feature-root "$ALPHA_SOURCE/feature_points" \
  --contract-manifest "$ALPHA_REPO/configs/research/alpha_max_contract_manifest_20260711_listing_aware.json" \
  --output-root "$ALPHA_PHASES" \
  | tee "$RUN_DIR/alpha-max-phase-preparation.json"

test -f "$ALPHA_PHASES/preparation_manifest.json"
find -P "$ALPHA_PHASES" -xdev -printf '%m\t%y\t%s\t%p\n' \
  | LC_ALL=C sort > "$RUN_DIR/alpha-max-phase-inventory.tsv"
sha256sum "$ALPHA_PHASES/preparation_manifest.json" \
  > "$RUN_DIR/alpha-max-phase-manifest.sha256"
```

이 preparer는 원래 여섯 phase의 half-open interval과 symbol별 official availability를 교차해 decode/clip/rewrite하고 새 output root에 atomic publish한다. 날짜를 줄이지 않는다. TONUSDT의 짧은 공식 availability는 root에 기록되고 admission에서 탈락한다.

**중요:** branch의 기존 `alpha_max_data_pc_runbook_20260711.md`는 Rev5.14 config/manifest/hash를 지시한다. A-01의 Rev5.15 runbook 정합화와 push가 끝나기 전에는 위 phase-root preparation까지만 수행하고 `run_alpha_max_prelock.py`와 historical CLI는 실행하지 않는다.

## 11. 데이터 PC에서 회수할 handoff bundle

실제 market parquet를 Git이나 run bundle에 복사하지 않는다. 다음 작은 evidence만 회수한다.

- `main-commit.txt`, Alpha commit
- environment와 실행 명령
- before/after coverage JSON과 support inventory
- collector report와 오류 로그
- file inventory와 frozen subset hash
- point-in-time lifecycle provenance
- Alpha `preparation_manifest.json`과 hash
- 활성화된 경우 candidate/data/trial manifest와 hash
- 활성화된 경우 feature-admission, overlay-ablation, cost-grid와 selection-gate report
- data gap과 미해결 blocker 목록

handoff가 끝나도 실자본 배분은 0%다. 다음 단계는 master plan의 D-04, D-05, R-01부터이며 결과가 아니라 실행 경로를 먼저 고친다.

## 12. 후속 alpha/volatility 데이터-PC 실행 계약

이 절은 상위 계획 6장의 C-00~C-06이 활성화된 뒤의 실행 계약이다. 다음 조건을 모두 충족하지 않으면 inventory 초안에서 STOP한다.

- `candidate-manifest.json`, `data-contract.json`, `trial-ledger.json`과 SHA-256 동결
- D-01A strict validator와 D-04 point-in-time lifecycle 통과
- R-04와 A-03의 기존 R1/R2·Alpha-Max 판정 완료
- single combined strict/cost profile, actual registry route와 `generic_fallback_proxy=0`
- clean worktree, 별도 run ID, untouched lockbox와 actual cost/funding source

### 12.1 실행 matrix

| 묶음 | 비교 | 고정 조건 |
|---|---|---|
| standalone | 상위 계획 6.2의 각 candidate default 한 세트 | validation-only ranking, locked OOS report-only, 전 reject 보존 |
| V-DIAG | own-vol HAR/EWMA 또는 univariate-GARCH baseline 대 preregistered leader-vol 추가 | 비거래 진단, untouched lockbox 제외, `RV(t+1)` 한 horizon, validation-forward QLIKE/MSE/FDR |
| V-PAIR | 공통 `lookback_window=120`, `hedge_window=240`, `vol_lag_bars=2`; P0 `min_vol_convergence=0` 대 P1 `.60` | 기존 `state_volconv`의 나머지 parameter, pair residual, episode, execution, funding과 cost 동일 |
| V-OVERLAY | O0 child, O1 close-to-close vol-managed, O2 complete TradFi OHLC의 Yang-Zhang, O3 correlation crash guard, 각 dynamic arm에 gross-matched O4 static control | 각 O4는 별도 trial, 첫 cycle stack 금지, 같은 pre-overlay child signal/target·market data·cost/funding model |
| V-COV | equal-notional, inverse-vol, existing shrunk-covariance allocator | 다음 bar weight, alpha가 아닌 risk allocation 비교 |

V-DIAG가 상위 계획의 admission gate를 통과하기 전에는 새 DCC/BEKK/TVP-VAR 또는 direct GARCH/volatility 방향 alpha를 구현하지 않는다. V-PAIR/V-OVERLAY/V-COV는 standalone binding gate를 통과한 child에만 실행한다.

실행 전에 candidate manifest의 각 row를 실제 repository CLI, registry class와 exact parameter key에 매핑한다. producer가 없으면 명령을 추측하지 말고 STOP한 뒤, 활성화된 task에서 기존 runner를 재사용하는 최소 seam과 targeted test를 먼저 만든다.

### 12.2 필수 산출물과 STOP/KILL

각 run은 다음 파일 또는 동등한 immutable artifact를 회수한다.

- `candidate-manifest.json`, `data-contract.json`, `trial-ledger.json`
- `feature-admission.json`, `standalone-walkforward.json`, `overlay-ablation.json`
- `cost-grid.json`, `selection-gate.json`, `all-candidates.csv`, `decision.md`
- Git/config/data/candidate hash, resolved command와 environment receipt

trial ledger는 candidate, universe, timeframe, estimator, threshold, allocator, overlay arm과 cost cell 전부를 센다. DSR `>=0.90`, SPA `<=0.05`, PBO `<=0.50`, 20bp net `>0`, MDD `<=30%`, leave-best-fold-out net `>0`, active fold ratio `>=0.60`을 모두 통과해야 하며 PBO missing은 fail-close다. low-turnover/lead-lag는 RPT `>=10bp`도 필요하다.

source/lifecycle/cost provenance 부재, unexplained gap, synthetic/fallback route, locked-OOS 기반 선택, 한 fold·symbol 지배, liquidation/ruin 또는 20bp 실패가 하나라도 있으면 해당 candidate를 KILL한다. 결과표에는 survivor뿐 아니라 모든 reject와 사유를 남긴다. 통과해도 alpha leaf 최대 1개와 risk overlay 최대 1개만 60일 frozen shadow로 보내며 실자본은 계속 0%다.
