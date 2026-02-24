# LuminaQuant 문서

**LuminaQuant**는 전문적인 백테스팅 및 실거래를 위해 설계된 고급 이벤트 기반 퀀트 트레이딩 시스템입니다. 다중 거래소 지원, 강력한 상태 관리, 정교한 전략 최적화 기능을 갖춘 모듈식 아키텍처를 특징으로 합니다.

[English Version](README.md)

---

## 📚 문서 목차 (Documentation Index)

| 섹션 | 설명 |
| :--- | :--- |
| **[설치 및 설정](#설치-installation)** | LuminaQuant 시작하기. |
| **[운영 워크플로우](docs/kr/WORKFLOW.md)** | Private/Public 브랜치 운영 및 공개 배포 체크리스트. |
| **[마이그레이션 가이드](docs/MIGRATION_GUIDE_POSTGRES_PARQUET.md)** | 레거시 저장소 제거 후 Parquet + PostgreSQL 전환 가이드. |
| **[GPU 자동 실행 설계](docs/DESIGN_NOTES_GPU_AUTO.md)** | Polars GPU/CPU 자동 선택 및 fallback 전략 설명. |
| **[선물 전략 팩토리](docs/kr/FUTURES_STRATEGY_FACTORY.md)** | 후보 생성, 가중치 기반 숏리스트, 단일-자산 조합 정책. |
| **[대시보드 실시간 분석 리포트](docs/DASHBOARD_REALTIME_ANALYSIS_REPORT.md)** | 실시간 갱신 동작 개선 분석 및 구현 결과. |
| **[거래소 가이드](docs/kr/EXCHANGES.md)** | **바이낸스(Binance)** (CCXT) 및 **MetaTrader 5 (MT5)** 상세 설정법. |
| **[거래 매뉴얼](docs/kr/TRADING_MANUAL.md)** | **실전 운용법**: 매수/매도, 레버리지, TP/SL, 트레일링 스탑. |
| **[성과 지표](docs/kr/METRICS.md)** | Sharpe, Sortino, Alpha, Beta 등 지표에 대한 설명. |
| **[개발자 API](docs/kr/API.md)** | 전략 작성법 및 시스템 확장 가이드. |
| **[구성 (Configuration)](#구성-configuration)** | `config.yaml` 빠른 참조. |

---

## 🏗 아키텍처 (Architecture)

LuminaQuant는 모듈식 **이벤트 기반 아키텍처(Event-Driven Architecture)**를 따릅니다:

```mermaid
graph TD
    Data[Data Handler] -->|MarketEvent| Engine[Trading Engine]
    Engine -->|MarketEvent| Strategy[Strategy]
    Strategy -->|SignalEvent| Portfolio[Portfolio]
    Portfolio -->|OrderEvent| Execution[Execution Handler]
    Execution -->|FillEvent| Portfolio
```

- **DataHandler**: 과거(CSV) 또는 실시간(WebSocket) 데이터 피드를 관리합니다.
- **Strategy**: 시장 데이터를 기반으로 `SignalEvent`를 생성합니다 (예: RSI < 30).
- **Portfolio**: 상태, 포지션, 리스크를 관리하며, 신호를 `OrderEvent`로 변환합니다.
- **ExecutionHandler**: 체결을 시뮬레이션(백테스트)하거나 API를 통해 실행(실거래)합니다.

현재 기본 로컬 스택:
- **1초 캔들 저장소**: Parquet(ZSTD, exchange/symbol/date 파티션)
- **상태/감사/잡 관리**: PostgreSQL(local)
- **백테스트/최적화 계산**: Polars Lazy + GPU 자동(`LQ_GPU_MODE=auto`)

---

## ⚙️ 설정 및 구성 (Setup & Configuration)

### 필수 요구사항 (Prerequisites)
- Python 3.11 이상 3.14 미만
- [uv](https://docs.astral.sh/uv/) (의존성/실행 환경 관리)
- [Polars](https://pola.rs/) (고성능 데이터 처리를 위해 사용)
- [Talib](https://github.com/TA-Lib/ta-lib-python) (기술적 지표 계산을 위해 사용)

### 환경 변수 (Environment Variables)
보안을 위해 **API 키를 절대 커밋하지 마세요**. 루트 디렉토리에 `.env` 파일을 생성하여 관리합니다:

```ini
# .env 파일 예시
BINANCE_API_KEY=your_api_key
BINANCE_SECRET_KEY=your_secret_key
LQ_POSTGRES_DSN=postgresql://localhost:5432/luminaquant
LQ_GPU_MODE=auto
LQ_GPU_DEVICE=0
LOG_LEVEL=INFO
```

*템플릿은 `.env.example` 파일을 참고하세요.*

---

## 🚀 빠른 시작 (Quick Start)

### 1. 설치 (Installation)

```bash
# 저장소 복제
git clone https://github.com/HokyoungJung/LuminaQuant.git
cd lumina-quant

# 프로젝트 Python 버전 고정 (< 3.14)
uv python pin 3.13

# 의존성 설치
uv sync --all-extras  # 또는 pip install ".[live,optimize,dashboard]"

# (선택 사항) MT5 지원을 위한 설치
uv sync --extra mt5
```

### 2. 구성 (Configuration)

LuminaQuant는 `config.yaml` 파일로 모든 설정을 관리합니다.

**일반 설정:**
```yaml
trading:
  symbols: ["BTC/USDT", "ETH/USDT"]
  timeframe: "1h"
  initial_capital: 10000.0
```

**거래소 선택:**

*   **바이낸스 (암호화폐)**: `driver: "ccxt"` 설정
*   **MetaTrader 5 (FX/주식)**: `driver: "mt5"` 설정

*👉 상세한 인증 설정 방법은 [거래소 가이드](docs/kr/EXCHANGES.md)를 참고하세요.*

### Public / Private 저장소 범위

- Public 저장소에서는 아래 연구 IP를 의도적으로 제외합니다.
  - `lumina_quant/indicators/`
  - `strategies/`
  - 전략/지표 전용 테스트 파일
- Public 저장소에서는 DB 구축/동기화 코드도 제외합니다.
  - `lumina_quant/data_sync.py`
  - `lumina_quant/data_collector.py`
  - `scripts/sync_binance_ohlcv.py`
  - `scripts/collect_market_data.py`
  - `tests/test_data_sync.py`
- 전략/지표 전체 구현 및 AGENTS 가이드는 Private 저장소에서 관리합니다.
- DB/런타임 산출물은 게시하지 않습니다 (`data/`, `logs/`, `.omx/`, `.sisyphus/`).

### 3. 시스템 실행 (Running the System)

**(Private 저장소 전용) 바이낸스 OHLCV 전체 수집 + Parquet 업데이트 (+CSV 미러):**
```bash
uv run python scripts/sync_binance_ohlcv.py \
  --symbols BTC/USDT ETH/USDT \
  --timeframe 1m \
  --db-path data/market_parquet \
  --force-full
```

Public 저장소에는 DB 동기화/구축 헬퍼를 의도적으로 포함하지 않습니다. 사전 구축된 DB 파일 또는 CSV 데이터를 사용하세요.

**전략 백테스트:**
```bash
uv run python run_backtest.py

# DB 데이터만 사용
uv run python run_backtest.py --data-source db --market-db-path data/market_parquet
```

**워크포워드 최적화:**
```bash
uv run python optimize.py

# DB 우선, 부족하면 CSV fallback
uv run python optimize.py --data-source auto --market-db-path data/market_parquet
```

**전략 팩토리 파이프라인 (후보 + 숏리스트):**
```bash
# dry-run
uv run python scripts/run_strategy_factory_pipeline.py --dry-run

# 후보/숏리스트 생성
uv run python scripts/run_strategy_factory_pipeline.py \
  --db-path data/market_parquet \
  --mode standard \
  --timeframes 1m 5m 15m \
  --seeds 20260221 \
  --single-min-score 0.0 \
  --single-min-return 0.0 \
  --single-min-sharpe 0.7 \
  --single-min-trades 20 \
  --drop-single-without-metrics
```

포트폴리오 숏리스트 기본 정책:
- **단일 전략**은 score/return/sharpe/trades 기준을 통과하지 못하면 제외
- `--allow-multi-asset`을 명시하지 않으면 **직접 multi-asset 전략은 포트폴리오 숏리스트에서 제외**
- 최종 포트폴리오 후보는 성과가 검증된 단일 전략을 자산별로 묶은 **`portfolio_sets`**(가중치 `portfolio_weight`)로 생성

**아키텍처/린트 검증:**
```bash
uv run python scripts/check_architecture.py
uv run ruff check .
```

**PostgreSQL 스키마 초기화:**
```bash
uv run python scripts/init_postgres_schema.py --dsn "$LQ_POSTGRES_DSN"
```

**백테스트 성능 벤치마크/회귀 비교:**
```bash
uv run python scripts/benchmark_backtest.py --output reports/benchmarks/baseline_snapshot.json

# 이전 스냅샷과 비교
uv run python scripts/benchmark_backtest.py \
  --output reports/benchmarks/current_snapshot.json \
  --compare-to reports/benchmarks/baseline_snapshot.json
```

**전략 팩토리 파이프라인 (manifest + shortlist):**
```bash
# Dry run
uv run python scripts/run_strategy_factory_pipeline.py --dry-run

# 단일 전략 성과 필터 + 가중치 + portfolio_sets 생성
uv run python scripts/run_strategy_factory_pipeline.py \
  --db-path data/market_parquet \
  --mode standard \
  --timeframes 1m 5m 15m \
  --seeds 20260221 \
  --single-min-score 0.0 \
  --single-min-return 0.0 \
  --single-min-sharpe 0.7 \
  --single-min-trades 20 \
  --drop-single-without-metrics
```

기본 shortlist 정책:
- 단일 전략은 score/return/sharpe/trades 기준을 통과해야 포함
- direct multi-asset 행은 기본 제외 (`--allow-multi-asset`으로 허용)
- 성공한 단일-자산 전략 조합으로 `portfolio_sets`가 생성되고 각 멤버에 `portfolio_weight`가 부여됨

**결과 시각화 (대시보드):**
```bash
uv run streamlit run dashboard.py
```

대시보드 개선 사항:
- 전략별 Run 필터(`Filter Run IDs By Strategy`) 및 전략 변경 시 Run 자동 재선택
- 감사 상태(PostgreSQL)와 시장 OHLCV(Parquet) 소스를 분리하여 표시
- 런타임 데이터가 없을 때 CSV fallback 상태를 명시적으로 경고

**대시보드 실시간 스모크 체크 (equity row 증가 확인):**
```bash
uv run python -m streamlit run dashboard.py --server.headless true
```

**Ghost RUNNING 정리 (PostgreSQL):**
```bash
# dry-run
uv run python scripts/cleanup_ghost_runs.py \
  --dsn "$LQ_POSTGRES_DSN" \
  --stale-sec 300 \
  --startup-grace-sec 90

# apply
uv run python scripts/cleanup_ghost_runs.py \
  --dsn "$LQ_POSTGRES_DSN" \
  --stale-sec 300 \
  --startup-grace-sec 90 \
  --apply
```

**실거래 실행:**
```bash
uv run python run_live.py
```

---

## 🌟 주요 기능 (Key Features)

- **이벤트 기반 코어**: 이벤트(`Market`, `Signal`, `Order`, `Fill`)를 순차적으로 처리하여 현실적인 체결을 시뮬레이션합니다.
- **다중 자산 & 다중 거래소**:
    - CCXT를 통한 바이낸스, 업비트 등 **암호화폐** 거래.
    - MetaTrader 5를 통한 **FX, CFD, 주식** 거래.
- **고급 백테스팅**: 슬리피지, 수수료 모델, 트레일링 스탑 로직 포함.
- **최적화**: **Optuna**(베이지안 최적화)를 내장하여 최적의 전략 파라미터를 탐색.
- **실거래 안정성**:
    - **상태 복구**: 재시작 시 포지션 동기화.
    - **서킷 브레이커**: 일일 손실 한도 초과 시 거래 중단.

---

## 📊 대시보드 미리보기

내장된 Streamlit 대시보드는 전문가 수준의 분석을 제공합니다:

- **자산 곡선 & 낙폭**: 포트폴리오 성장과 리스크 시각화.
- **매매 분석**: 차트상에서 매수/매도 타점 확인.
- **포괄적 지표**: Sharpe Ratio, Sortino, Calmar, Alpha, Beta 등.

*👉 모든 통계의 정의는 [성과 지표](docs/kr/METRICS.md)를 참고하세요.*
