# 거래소 설정 가이드

LuminaQuant는 통합 인터페이스를 통해 여러 거래소 드라이버를 지원합니다.

암호화폐의 production-critical 경로는 이제 **native Binance USDⓈ-M Futures only** 입니다.
Binance 라이브 트레이딩 / 유저 스트림 / 히스토리컬 시장데이터 수집에 CCXT가 필요하지 않습니다.

## 1. Binance USDⓈ-M Futures (native)

### 지원 범위
- native Futures REST market data
- native Futures aggTrade historical ingestion
- native Futures websocket aggTrade live market data
- native Futures order placement / cancel / query
- native Futures balances / positions / open orders
- native Futures user data stream
- native Futures leverage / margin / position-mode controls

### 설정 (`config.yaml`)

```yaml
live:
  mode: "paper"
  market_data_source: "binance_futures"
  order_state_source: "user_stream"
  exchange:
    driver: "binance_futures"
    name: "binance"
    market_type: "future"
    position_mode: "HEDGE"
    margin_mode: "isolated"
    leverage: 3
```

### 환경 변수 (`.env`)

```ini
BINANCE_API_KEY=your_actual_api_key
BINANCE_SECRET_KEY=your_actual_secret_key
```

### 메모
- Binance에서는 `live.exchange.market_type: future`만 허용됩니다.
- canonical raw market data는 **aggTrades** 입니다.
- 1초 봉은 raw aggTrades에서 파생됩니다.
- 상위 타임프레임은 실제 lower-timeframe 데이터를 deterministic resample 해서 만듭니다.
- final validation은 real-data-only + latest-anchored 방식입니다. `docs/FINAL_VALIDATION.md` 참고.

---

## 2. MetaTrader 5 (MT5)

MetaTrader 5 터미널과 직접 연동하여 FX / CFD / 주식 / 선물을 거래할 수 있습니다.

---

## 3. Polymarket (Phase 1)

현재 지원 범위:
- market-data ingestion
- signal generation
- paper/shadow execution
- `allow_real_execution: true`일 때 실험적 real order path
