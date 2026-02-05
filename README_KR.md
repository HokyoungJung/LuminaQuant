[English Version](README.md)

# Quants Agent - 이벤트 기반 퀀트 트레이딩 시스템 (Event-Driven Quantitative Trading System)

바이낸스(Binance)에서의 백테스팅(Backtesting) 및 실전 매매(Live Trading)를 위해 설계된 고성능 이벤트 기반 거래 파이프라인입니다. 시뮬레이션과 실전 실행 간의 일관성을 보장하기 위해 모듈식 아키텍처로 구축되었습니다.

## 주요 기능 (Features)

- **이벤트 기반 아키텍처**: 실제 거래 메커니즘을 모방하기 위해 중앙 이벤트 루프(`MarkteEvent`, `SignalEvent`, `OrderEvent`, `FillEvent`)를 사용합니다.
- **정확한 백테스팅**:
  - **Trailing Stops**, Stop Loss, Take Profit 지원.
  - 체결 지연(latency) 및 수수료(commission) 시뮬레이션.
- **실전 매매 (Live Trading)**:
  - `ccxt`를 통한 **Binance** 연동.
  - **강력한 상태 관리**: 시작 시 거래소의 실제 포지션과 동기화.
  - **부분 체결 처리**: 유동성 부족 시 잔량을 대기 주문(Pending Order)으로 처리하여 현실적인 체결 보장.
- **최적화 (Optimization)**:
  - **Optuna**(베이지안 최적화) 또는 그리드 탐색(Grid Search)을 이용한 파라미터 튜닝.
  - Walk-Forward Analysis (Train/Validation/Test 분할 검증).

## 프로젝트 구조

```
quants-agent/
├── quants_agent/           # 핵심 패키지
│   ├── backtest.py         # 백테스트 엔진
│   ├── live_trader.py      # 실전거래 엔진
│   ├── ...
├── generate_data.py        # 테스트용 데이터 생성기
├── run_backtest.py         # 백테스트 실행 스크립트
├── optimize.py             # 전략 최적화 (Optuna)
├── run_live.py             # 실전거래 실행 스크립트
└── README.md
```

## 설치 (Setup)

1.  **의존성 설치**
    ```bash
    uv sync
    # 또는
    pip install .
    ```

2.  **설정 (Configuration)**
    `.env.example` 파일을 복사하여 `.env`를 생성하고 키를 설정합니다:
    ```bash
    cp .env.example .env
    ```
    `.env` 파일 수정:
    ```ini
    BINANCE_API_KEY=your_key
    BINANCE_SECRET_KEY=your_secret
    IS_TESTNET=True  # 실전 투자 시 False 로 변경
    ```

## 사용 방법 (Usage)

### 1. 데이터 생성 (Data Generation)
테스트를 위한 가상 데이터를 생성합니다 (기본 1000일 치):
```bash
uv run generate_data.py
```

### 2. 백테스팅 (Backtesting)
기본 백테스트 시뮬레이션을 실행합니다:
```bash
uv run run_backtest.py
```
결과(샤프 지수, 최대 낙폭 등)가 콘솔에 출력되며, `equity.csv` 파일로 자산 곡선이 저장됩니다.

### 3. 최적화 (Optimization)
Optuna를 사용하여 전략에 가장 적합한 파라미터를 찾습니다:
```bash
uv run optimize.py
```
Train(학습), Validation(검증), Test(테스트) 단계를 거쳐 최적의 파라미터를 `best_optimized_parameters/` 폴더에 저장합니다.

### 4. 실전 매매 (Live Trading)
실전 봇을 실행합니다 (`.env` 설정 필요):
```bash
uv run run_live.py
```
*주의: 계좌(혹은 테스트넷 계좌)에 충분한 USDT가 있는지 확인하세요.*

## 데이터 분석 (Data Analysis)
제공되는 성과 지표는 다음과 같습니다:
- **Total Return** (총 수익률)
- **Sharpe Ratio** (샤프 지수)
- **Max Drawdown** (최대 낙폭)
- **Drawdown Duration** (낙폭 지속 기간)
