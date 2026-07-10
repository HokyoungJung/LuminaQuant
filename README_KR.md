[English Version](README.md)

# LuminaQuant

**LuminaQuant**는 전문적인 백테스팅, 워크포워드 최적화, 실거래를 위한 고성능 config 기반 퀀트 트레이딩 엔진입니다. 아키텍처 전면 개편(Phase 4–7)을 통해 레거시 이벤트 루프를 Rust 가속 커널과 통합 ExecutionModel로 교체했으며, Next.js 15 대시보드와 단일 `config.yaml` 기반 사용자 제어 인터페이스를 갖추고 있습니다.

## 저장소 역할 (Source of Truth)

- **Private 원본 저장소** (유지보수/내부): `https://github.com/hoky1227/Quants-agent.git`
- **Public 배포 저장소** (외부/읽기 중심): `https://github.com/HokyoungJung/LuminaQuant.git`
- Python 패키지/임포트 네임스페이스: `lumina_quant` (배포명: `lumina-quant`)

---

## 빠른 시작

### 필수 요구사항

| 항목 | 비고 |
| :--- | :--- |
| Python >=3.14 | `uv`로 관리 |
| [uv](https://docs.astral.sh/uv/) | 의존성 및 런타임 관리 |
| ta-lib 시스템 라이브러리 | `apt install libta-lib-dev` 또는 동등한 패키지 |
| Rust + maturin | `native/lumina_compute` pyo3 확장 빌드에 필요 |
| Node 20+ | 대시보드 프런트엔드 전용 |

### 설치

```bash
# 비공개 저장소 (이 repo)
git clone https://github.com/hoky1227/Quants-agent.git
cd Quants-agent

# 공개 미러 (LuminaQuant)
git clone https://github.com/HokyoungJung/LuminaQuant.git
cd LuminaQuant

# 코어 + 주요 extras
uv sync --extra optimize --extra live-binance --extra dashboard --extra dev

# Rust pyo3 확장 빌드 (백테스트/최적화/실거래에 필수)
python scripts/build_native_backends.py

# 대시보드 프런트엔드 (최초 1회)
cd apps/dashboard_web && npm install && cd ../..

# 선택 사항: GPU 런타임 (Linux x86_64 + CUDA 12 전용)
# 핵심 pin: polars>=1.35.2, GPU 엔진 cudf-polars-cu12>=26.6
uv sync --extra gpu
```

**제공 extras:** `backtest` · `optimize` · `gpu` · `live-binance` · `live-mt5` · `live-polymarket` · `dashboard` · `dev`

### 스모크 테스트 (DB·API 키 불필요)

```bash
uv run python scripts/minimum_viable_run.py
```

### 백테스트 실행

```bash
uv run lq backtest
```

### 워크포워드 최적화

```bash
uv run lq optimize
```

### 대시보드 실행

```bash
uv run lq dashboard --run
```

---

## CLI 참조

`uv run lq <command>`가 유일하게 지원되는 진입점입니다. 루트 호환 shim은 제거되었습니다.

| 명령 | 기능 |
| :--- | :--- |
| `lq backtest` | 설정된 전략으로 백테스트 실행 |
| `lq optimize` | Optuna 기반 워크포워드 최적화 |
| `lq live` | 실거래 시작 (기본값: paper/testnet) |
| `lq data` | 데이터 수집 및 materialization 헬퍼 |
| `lq exact-window` | 틱 재현 윈도우 기준 exact-window 평가 |
| `lq autonomous-research` | 자율 전략 리서치 파이프라인 |
| `lq dashboard` | Next.js 대시보드 관리 (`--run`, `--print-contract`) |
| `lq config show` | 해석된 `RuntimeConfig`를 JSON으로 출력 |
| `lq config validate` | config YAML을 전체 정규화 파이프라인으로 검증 |
| `lq registry list` | 등록된 전략·지표·포트폴리오 최적화기 목록 출력 |

---

## 구성 (Configuration)

모든 사용자 설정은 **`config.yaml`**(루트)과 활성 프로필 **`configs/profiles/{paper,real,research}.yaml`**에 집중되어 있습니다. 소스 코드를 수정할 필요가 거의 없습니다.

주요 설정 섹션과 주요 항목:

```yaml
# 거래소 / 드라이버 선택
live:
  exchange:
    driver: "binance_futures"   # binance_futures | mt5 | polymarket

# 심볼 및 데이터 종류
trading:
  symbols: ["BTC/USDT", "ETH/USDT"]
data:
  kinds: [ohlcv, funding, feature_points]  # ohlcv | funding | feature_points | aggtrades_tick

# 전략 선택
optimization:
  strategy: "RsiStrategy"       # 플러그인 레지스트리의 클래스명

# 메모리 한도 및 황금 회귀 허용 오차
memory:
  cap_gb: 8.0
validation:
  golden_rtol: 1.0e-8

# 실거래 안전 파이프라인
live:
  go_live_stage: "testnet"      # testnet | shadow | canary | full
  kill_switch_enabled: true     # 항상 활성 — false로 설정하면 로드 시 거부됨
  canary_position_fraction: 0.10
```

전체 스키마는 [`AGENTS.md`](AGENTS.md)와 [`docs/CONFIG_SPEC.md`](docs/CONFIG_SPEC.md)를 참고하세요.

---

## 아키텍처

### 기술 스택

| 계층 | 기술 |
| :--- | :--- |
| 언어 | Python >=3.14 |
| 패키지/런타임 | uv |
| 네이티브 가속 | Rust pyo3 확장 `native/lumina_compute` (maturin) |
| 연산 | Polars Lazy + 선택적 GPU (cudf-polars) |
| 저장소 | Parquet (ZSTD, exchange/symbol/date 파티션) + PostgreSQL 감사 |
| 대시보드 | Next.js 15 (`apps/dashboard_web`) |

### Rust 네이티브 확장 — `native/lumina_compute`

단일 pyo3 cdylib(`lumina_quant._compute`)으로 maturin을 통해 빌드합니다. 기존 5개의 ctypes 로더를 대체하는 7개의 커널을 제공합니다:

| 커널 | 기능 |
| :--- | :--- |
| `evaluate_metrics` | Sharpe, Sortino, MDD 등 성과 지표 계산 |
| `simulate_symbol_fold` | 폴드 전체를 벡터화한 백테스트 (내부 루프) |
| `debounced_state_signal` | 실거래 신호 상태 머신 |
| `trailing_state_signal` | 실거래 트레일링 스탑 상태 머신 |
| `evaluate_hybrid_optuna_portfolio` | Optuna 하이브리드 포트폴리오 평가기 |
| `aggregate_raw_aggtrades_to_1s` | 원시 aggTrades → 1초 OHLCV 집계 |
| `append_ohlcv_1s_wal` | 1초 시장 데이터 WAL 추가 |

빌드: `python scripts/build_native_backends.py` (`maturin develop --release` 실행).

### 저장소 구조

```
LuminaQuant/
├── config.yaml                  ← 단일 사용자 설정 파일
├── configs/profiles/            ← paper.yaml / real.yaml / research.yaml
├── pyproject.toml
├── src/lumina_quant/
│   ├── cli/                     ← lq 진입점
│   ├── configuration/           ← RuntimeConfig 스키마 + 유효성 검사
│   ├── core/                    ← 엔진, 이벤트, plugin_registry
│   ├── compute/                 ← _compute 커널용 Python 래퍼
│   ├── data/                    ← DataCollector, 로더
│   ├── storage/                 ← Parquet, PostgreSQL, WAL
│   ├── exchanges/               ← Binance futures, MT5, Polymarket 어댑터
│   ├── backtesting/             ← ExecutionModel, 백테스트 엔진
│   ├── optimization/            ← 워크포워드, Optuna, search_policy
│   ├── live/                    ← LiveTrader, 준비 상태 점검, paper exchange
│   ├── strategies/              ← 전략 레지스트리 + 내장 전략
│   ├── indicators/              ← 지표 레지스트리 (alpha101 등)
│   ├── portfolio/               ← 포트폴리오 최적화기
│   ├── dashboard/               ← bridge contract, 백엔드 서비스
│   └── workflows/               ← 리서치 / 자율 파이프라인
├── native/lumina_compute/       ← Rust pyo3 cdylib (maturin)
├── apps/dashboard_web/          ← Next.js 15 대시보드
├── baseline/                    ← 동결된 성능 기준 산출물
├── docs/perf/                   ← 단계별 벤치마크 결과
├── docs/divergences/            ← 설계 결정 기록
└── scripts/                     ← ci/, ops/, dev/, research/
```

### 백테스트 정밀도

백테스트 엔진은 시뮬레이션 백테스트와 실거래 모두에서 공유하는 **통합 `ExecutionModel`**(`backtesting/execution_model.py`)을 사용합니다:

- 수수료(maker/taker), 펀딩 비용, 레버리지, 청산 임계값
- 슬리피지 모델, 바 내 거래량 상한을 적용한 부분 체결
- LMT strict-cross 체결 규칙 (BUY는 `bar_low < limit_price`일 때만 체결; SELL은 `bar_high > limit_price`일 때만 체결)
- CI에서 `rtol=1e-8`(`validation.golden_rtol`) 기준 황금 회귀 검증
- 체결 가격 동등성 검증용 틱 재현 검증기(`TickReplayValidator`)
- 설정 가능한 폴드 수와 워밍업 구간이 있는 워크포워드

### 비용 현실성 & 엣지 재측정

헤드라인 백테스트 수치는 **낙관적 기본값**(플랫·주문크기 무시 슬리피지, 펀딩 0)으로 산출됩니다.
여러 현실성 제어 플래그가 **config-gated OFF**로 출하되어(황금 회귀 byte-identical 유지),
백테스트 PC에서 켜서 현실적 체결 비용 하에 엣지가 얼마나 살아남는지 측정해야 합니다:

```yaml
execution:
  slippage_impact_model: "sqrt_impact"   # 주문크기/시장충격 반영 슬리피지 (기본 "flat")
  slippage_impact_coefficient: 0.10      # 충격 강도 (보정 필요)
  require_funding_coverage: true         # 레버리지인데 펀딩 데이터 없으면 명시적 실패
risk:
  allow_metadata_risk_override: false    # metadata를 config 캡으로 클램프 (이미 기본값)
  attach_default_protective_stop: true   # 무방비 포지션 금지
  enforce_order_risk_gate_in_backtest: true  # live와 동일한 RiskManager 게이트
```

이후 `lq backtest` / `lq optimize`를 재실행해 플랫 기준선과 A/B 비교(+ 10/15/20 bps 비용
스트레스 그리드). 전체 절차: **[`docs/COST_REALISM_REMEASUREMENT.md`](docs/COST_REALISM_REMEASUREMENT.md)**
(한국어: [`docs/kr/COST_REALISM_REMEASUREMENT.md`](docs/kr/COST_REALISM_REMEASUREMENT.md)).

### 성능

Phase 4 리팩터 트리를 Phase 0 순수 Python 기준선과 비교한 측정 결과 (출처: [`docs/perf/phase4-results.md`](docs/perf/phase4-results.md)):

| 축 | 기준선 | Phase 4 | 속도 향상 |
| :--- | :--- | :--- | :--- |
| 백테스트 bars/sec (RsiStrategy, 1,268 bars, 14 심볼) | 22.44 | 6,632 | **295×** |
| 워크포워드 E2E (27회 실행: 3폴드 × 9 조합) | 170.71초 | 1.768초 | **97×** |

주요 동인: `simulate_symbol_fold` Rust 커널이 순수 Python 바별 이벤트 루프를 대체합니다.

---

## 대시보드

`uv run lq dashboard --run`으로 Next.js 15 프런트엔드를 시작합니다. Python 백엔드는 13개의 Next.js 라우트가 소비하는 `DashboardBridgeContractV2` JSON 계약을 노출합니다:

`/`(홈) · `/performance-price` · `/market-data` · `/optimization-insights` · `/exact-window` · `/factor-insights` · `/alpha-evidence` · `/execution-analytics` · `/risk-health` · `/workflows` · `/raw-data` · `/report-export` · `/system`

workflows 라우트는 관리형 백테스트·최적화·실거래 잡을 폴링 상태와 2단계 stop/kill 컨트롤과 함께 보여줍니다.

---

## 실거래 안전 모델

실거래는 `live.go_live_stage`로 제어되는 4단계 프로모션 파이프라인을 따릅니다:

1. **testnet** — 거래소 테스트넷, 실제 자금 없음
2. **shadow** — 실시간 시장 데이터, 시뮬레이션 주문
3. **canary** — 소규모 실제 포지션 비율 (`canary_position_fraction`, 기본 10%)
4. **full** — 전체 포지션 사이징

**킬 스위치는 항상 활성화됩니다.** config에서 `kill_switch_enabled: false`를 설정하면 로드 시 구조적으로 거부됩니다. 실거래 모드는 추가로 `LUMINA_ENABLE_LIVE_REAL` 환경 변수와 체결/슬리피지/BBO 패리티를 증명하는 준비 산출물이 필요합니다.

```bash
# Paper/testnet (기본값)
uv run lq live

# 실거래 모드 (환경 변수 + 준비 산출물 필요)
LUMINA_ENABLE_LIVE_REAL=true uv run lq live --enable-live-real
```

운영자 체크리스트는 [`docs/live-readiness/04-paper-trading-runbook.md`](docs/live-readiness/04-paper-trading-runbook.md)를 참고하세요.

---

## 전략·지표·포트폴리오 최적화기 추가

플러그인 시스템은 `lumina_quant.core.plugin_registry`의 `@register` 데코레이터를 사용합니다. 단계별 가이드는 [`AGENTS.md`](AGENTS.md)에 있습니다. 요약:

1. `src/lumina_quant/strategies/<name>.py`에 전략 인터페이스를 구현하는 클래스를 생성합니다.
2. `@register("strategy", "ClassName", interface="event_driven"|"polars_batch")`로 장식합니다.
3. `src/lumina_quant/strategies/registry.py`에 임포트 항목을 추가합니다.
4. `src/lumina_quant/tuning/param_registry.py`에 파라미터 스키마 항목을 추가합니다.
5. `config.yaml`에서 `optimization.strategy: "ClassName"`으로 활성화합니다.

지표(`"indicator"`)와 포트폴리오 최적화기(`"portfolio"`)에도 동일한 `@register` 패턴이 적용됩니다. 자세한 내용은 [`AGENTS.md`](AGENTS.md)를 참고하세요.

---

## 문서 목차

| 문서 | 설명 |
| :--- | :--- |
| [`AGENTS.md`](AGENTS.md) | 아키텍처 노트, 소유권 맵, 방법 가이드 |
| [`docs/CONFIG_SPEC.md`](docs/CONFIG_SPEC.md) | RuntimeConfig 전체 스키마 참조 |
| [`docs/DEPLOYMENT.md`](docs/DEPLOYMENT.md) | 배포 노트 및 운영 체크리스트 |
| [`docs/live-readiness/04-paper-trading-runbook.md`](docs/live-readiness/04-paper-trading-runbook.md) | Paper/testnet 실거래 핸드오프 런북 |
| [`docs/perf/phase4-results.md`](docs/perf/phase4-results.md) | Phase 4 벤치마크 결과 |
| [`docs/EXCHANGES.md`](docs/EXCHANGES.md) | Binance USDⓈ-M 선물, MetaTrader 5, Polymarket 설정 |
| [`docs/EXTERNAL_DATA.md`](docs/EXTERNAL_DATA.md) | 사용자 보유 데이터 연결 canonical contract |
| [`docs/METRICS.md`](docs/METRICS.md) | Sharpe, Sortino, Alpha, Beta, Calmar 정의 |
| [`docs/COST_REALISM_REMEASUREMENT.md`](docs/COST_REALISM_REMEASUREMENT.md) | 현실적 슬리피지/펀딩/리스크 플래그를 켜고 엣지 잔존량 재측정 ([한국어](docs/kr/COST_REALISM_REMEASUREMENT.md)) |
| [`docs/RUST_NATIVE_ACCELERATION.md`](docs/RUST_NATIVE_ACCELERATION.md) | hotspot 전용 Rust 정책, 빌드 명령, 벤치마크 |
| [`docs/QUICKSTART_8GB_BASELINE.md`](docs/QUICKSTART_8GB_BASELINE.md) | 8 GB RAM 최소 설치 및 스모크 절차 |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | 로컬 검사, CI parity 명령, PR 기준 |
| [`SECURITY.md`](SECURITY.md) | 취약점 제보 및 자격증명 관리 정책 |

---

## 환경 변수

API 키는 절대 커밋하지 마세요. 저장소 루트에 `.env` 파일을 생성하세요 (`.env.example` 참고):

```ini
BINANCE_API_KEY=your_api_key
BINANCE_SECRET_KEY=your_secret_key
LQ_POSTGRES_DSN=postgresql://localhost:5432/luminaquant
```

PostgreSQL은 백테스팅에서 선택 사항입니다. `LQ_POSTGRES_DSN`이 없으면 감사 저장이 건너뜁니다.

---

## 라이선스 및 면책 조항

이 소프트웨어는 연구 및 교육 목적으로 제공됩니다. 과거 백테스트 성과가 미래 수익을 보장하지 않습니다. 실거래는 상당한 손실 위험을 수반합니다. 킬 스위치와 go_live_stage 파이프라인은 프로모션 속도를 늦추기 위한 것이며 리스크를 제거하지 않습니다. 유지보수팀은 어떠한 종류의 보증도 제공하지 않습니다.
