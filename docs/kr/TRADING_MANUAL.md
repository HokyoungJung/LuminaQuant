# 거래 매뉴얼: 실전 운용 (Trading Manual)

이 가이드는 LuminaQuant에서 수행하는 공통적인 거래 운용에 대한 구체적인 지침과 코드 예시를 제공합니다.

## 0. 실거래 안전 체크리스트 (Ops)

`live.mode=real` 실행 전:

1. `config.yaml`에서 `live.require_real_enable_flag: true` 유지
2. 명시적 arming 후 실행:
   - `LUMINA_ENABLE_LIVE_REAL=true uv run lq live --enable-live-real`
3. stop-file 기반 정상 종료 권장:
   - 실행 시 `--stop-file /tmp/lq.stop`
   - 종료 시 `touch /tmp/lq.stop`
4. 비상 시 systemd stop / 프로세스 kill 절차를 운영 문서에 사전 정의

## 1. 매수 및 매도 (Buying and Selling)

LuminaQuant에서는 반복문 안에서 직접 "주문을 넣는" 방식이 아닙니다. 대신 **전략(Strategy)**이 **시그널(Signal)**을 생성하면, 시스템이 이를 주문으로 변환합니다.

### 시그널 흐름 (Signal Flow)
1.  **Strategy** (`calculate_signals`)가 조건을 감지합니다.
2.  **Strategy**가 `SignalEvent` ("LONG", "SHORT", 또는 "EXIT")를 방출합니다.
3.  **Portfolio**가 시그널 수신 -> 리스크 체크 -> `OrderEvent` 생성.
4.  **ExecutionHandler**가 주문 수신 -> 거래소(Binance/MT5)로 전송.

### 코드 예시: 기본 매수/매도
전략의 `calculate_signals(self, event)` 메서드 내부:

```python
# LONG (매수)
# signal_type="LONG" -> 시스템이 양(+)의 수량을 매수
self.events.put(
    SignalEvent(
        strategy_id=1, symbol="BTC/USDT", datetime=event.time, signal_type="LONG", strength=1.0
    )
)

# SHORT (매도/공매도)
# signal_type="SHORT" -> 시스템이 매도(공매도 오픈)
self.events.put(
    SignalEvent(
        strategy_id=1, symbol="BTC/USDT", datetime=event.time, signal_type="SHORT", strength=1.0
    )
)

# EXIT (포지션 청산)
# signal_type="EXIT" -> 시스템이 현재 보유량을 계산하여 청산(0으로 만듦)
self.events.put(
    SignalEvent(
        strategy_id=1, symbol="BTC/USDT", datetime=event.time, signal_type="EXIT", strength=1.0
    )
)
```

---

## 2. 익절 (TP) 및 손절 (SL)

TP/SL을 구현하는 두 가지 방법이 있습니다: **Hard** (거래소 측) 및 **Soft** (전략 측).

### 방법 A: Hard TP/SL (거래소 측)
*신뢰성이 가장 높습니다. 주문이 거래소 서버에 저장됩니다.*

#### MetaTrader 5 (MT5)
주문 실행 시 `params`를 통해 `sl`과 `tp` 값을 전달하면 됩니다.
`stop_loss` / `take_profit` 값을 넣고도 거래소 측 보호주문 파라미터를 명시하지 않으면, 현재 `LiveExecutionHandler` 는 **real mode 에서는 즉시 실패(fail-fast)** 하고, paper/testnet 계열 모드에서는 경고만 남기고 거래소 측 보호주문 없이 진행합니다.

**고급(Advanced)**: 전략 내에서 거래소 객체를 직접 호출하거나, 커스텀 실행 로직을 구현해야 할 수 있습니다.

```python
# 예시: 직접 거래소 호출 (실거래 전용 라이브러리 사용 시)
if self.mode == "LIVE":
    self.exchange.execute_order(
        symbol="EURUSD",
        type="market",
        side="buy",
        quantity=0.1,
        params={
            "sl": 1.0500,  # 손절가 (Stop Loss Price)
            "tp": 1.0700,  # 익절가 (Take Profit Price)
        },
    )
```

#### 바이낸스 USDⓈ-M Futures (Native)
native Binance USDⓈ-M Futures 주문 파라미터(`stopPrice`, `activationPrice`, `callbackRate`, `reduceOnly`, `positionSide` 등)를 명시적으로 전달해야 합니다. 문서화되지 않은 Spot/OCO 동작은 사용하지 않습니다.

### 방법 B: Soft TP/SL (전략 측)
*범용적인 방법입니다 (백테스트 및 실거래 모두 작동).*
전략 내에서 가격을 추적하다가 제한선에 도달하면 `EXIT` 시그널을 보냅니다.

```python
class MyStrategy(Strategy):
    def __init__(self, ...):
        self.entry_price = {}

    def calculate_signals(self, event):
        if event.type == "MARKET":
            price = self.bars.get_latest_bar_value(s, "close")
            
            # 매수 포지션 체크
            if self.bought[s] == "LONG":
                entry = self.entry_price[s]
                
                # 손절 (예: 2% 하락 시)
                if price < entry * 0.98:
                    self.events.put(SignalEvent(..., signal_type="EXIT"))
                
                # 익절 (예: 5% 상승 시)
                elif price > entry * 1.05:
                    self.events.put(SignalEvent(..., signal_type="EXIT"))
```

---

## 3. 트레일링 스탑 (Trailing Stop)

트레일링 스탑은 백테스트와 실거래의 일관성을 위해 **Soft Stop (전략 측)**으로 구현하는 것이 가장 좋습니다.

### 구현 가이드
1.  진입 이후 `highest_price` (최고가)를 추적합니다 (매수 포지션 기준).
2.  `stop_price`를 동적으로 업데이트합니다: `stop_price = highest_price * (1 - trail_percent)`.
3.  현재 가격 < `stop_price`이면, EXIT 시그널을 발생시킵니다.

```python
class TrailingStopStrategy(Strategy):
    def __init__(self, ..., trail_pct=0.02):
        self.trail_pct = trail_pct
        self.high_water_mark = {} # 진입 후 최고가 추적

    def calculate_signals(self, event):
        price = ...
        
        # 매수 포지션 로직
        if self.bought[s] == "LONG":
            # 1. 고점 갱신 (High Water Mark)
            if price > self.high_water_mark[s]:
                self.high_water_mark[s] = price
            
            # 2. 트레일링 스탑 가격 계산
            stop_price = self.high_water_mark[s] * (1 - self.trail_pct)
            
            # 3. 이탈(Breach) 확인
            if price < stop_price:
                self.events.put(SignalEvent(..., signal_type="EXIT"))
```

---

## 4. 레버리지 설정 (Leverage Settings)

### 바이낸스 (선물)
레버리지는 보통 계정 설정이지만, API로 설정할 수도 있습니다.
LuminaQuant는 공식 Binance USDⓈ-M Futures API를 통해 레버리지/마진/포지션 모드 설정을 적용합니다.

**설정 방법:**
전략의 `__init__`이나 `live_trader.py`에 다음을 추가하세요:

```python
# 전략 또는 초기화 코드 내부
if self.exchange.name == "binance":
    # BTC/USDT에 대해 레버리지 10배 설정
    self.exchange.exchange.set_leverage(10, "BTC/USDT")
```

### 메타트레이더 5 (MT5)
레버리지는 **브로커 계정 설정**에 따릅니다. 대부분의 경우 API로 변경할 수 없습니다. 브로커 홈페이지나 포털에 로그인하여 계정 레버리지를 변경해야 합니다 (예: 1:100, 1:500).

---

## 5. 주문 유형 (Order Types)

기본적으로 LuminaQuant live execution은 **limit-first**입니다. 시장가 주문은 명시적
operator override로만 켤 수 있습니다.

**현재 기본값**:
- Signal `LONG`/`SHORT`/`EXIT` -> `OrderEvent(type='LMT')` -> `execute_order(type='limit')`.
- `live.limit_price_mode: "one_tick_worse"`는 빠른 bounded execution을 위해 BUY를 기준가보다 1 exchange tick 위, SELL을 1 tick 아래로 냅니다.
- `live.limit_price_mode: "same_price"`는 기준가 그대로, `one_tick_better`는 passive 옵션입니다.
- 시장가 주문은 `live.default_order_type: "MKT"`와 `live.allow_market_orders: true`를 둘 다 명시해야 합니다.
- paper/testnet protective algo 주문도 기본은 conditional limit `STOP` / `TAKE_PROFIT`입니다. `STOP_MARKET` / `TAKE_PROFIT_MARKET` 스타일은 같은 market opt-in review가 필요합니다.
