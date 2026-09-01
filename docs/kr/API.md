# 개발자 API 참조 (Developer API Reference)

이 가이드는 커스텀 전략을 생성하고 시스템 기능을 확장하는 방법을 설명합니다.

## 1. 전략 생성 (Creating a Strategy)

모든 전략은 `Strategy` 추상 기본 클래스를 상속받아야 합니다.

### 클래스 구조

```python
from lumina_quant.strategy import Strategy
from lumina_quant.core.events import SignalEvent


class MyStrategy(Strategy):
    def __init__(self, bars, events, my_param=10):
        """
        bars: DataHandler 인스턴스 (과거 데이터 접근 가능)
        events: EventQueue (여기에 SignalEvent를 넣음)
        **kwargs: config.yaml에 정의된 임의의 파라미터들
        """
        self.bars = bars
        self.events = events
        self.my_param = my_param
        self.symbol_list = self.bars.symbol_list

    def calculate_signals(self, event):
        """
        모든 'MARKET' 이벤트(새로운 봉 마감)마다 호출됩니다.
        """
        if event.type == "MARKET":
            for s in self.symbol_list:
                # 1. 데이터 가져오기
                # get_latest_bars_values(symbol, "close", N)는 float 리스트를 반환
                closes = self.bars.get_latest_bars_values(s, "close", N=self.my_param)

                # 2. 로직 수행
                if len(closes) < self.my_param:
                    continue

                # 3. 시그널 생성
                if closes[-1] > closes[0]:
                    # SignalEvent(strategy_id, symbol, datetime, signal_type, strength)
                    # signal_type: "LONG", "SHORT", "EXIT"
                    signal = SignalEvent(1, s, event.time, "LONG", 1.0)
                    self.events.put(signal)
```

### 선택적 고급 전략 계약

전략은 이제 provider raw payload 대신 자신이 필요한 canonical input 을 선언할 수 있습니다.

```python
class MyContextStrategy(Strategy):
    required_inputs = ("market_window",)
    required_features = ("feature_points",)
    preferred_contract = "context"  # market_event | market_window | context

    def calculate_signals_context(self, context):
        event = context.event
        aggregator = context.aggregator
        feature_lookup = context.feature_lookup
        _ = (event, aggregator, feature_lookup)
```

호출 순서:
1. `preferred_contract == "context"` 이고 `calculate_signals_context(context)`가 있으면 우선 호출
2. 아니면 `calculate_signals_window(event, aggregator)`
3. 아니면 기존 `calculate_signals(event)`

## 2. 데이터 핸들러 API (Data Handler API)

`DataHandler`는 백테스트 및 실거래 중에 시장 데이터에 접근할 수 있는 메서드를 제공합니다.

- `get_latest_bar(symbol)`: 가장 최근의 전체 OHLCV 튜플을 반환합니다.
- `get_latest_bars(symbol, N=1)`: 최근 N개의 튜플 리스트를 반환합니다.
- `get_latest_bar_value(symbol, val_type)`: 단일 float 값(예: "close", "high")을 반환합니다.
- `get_latest_bars_values(symbol, val_type, N=1)`: float 값들의 리스트를 반환합니다.

외부 데이터는 canonical OHLCV CSV/parquet 루트(백테스트/최적화) 또는 canonical `MARKET_WINDOW` / 1초 OHLCV 외부 라이브 adapter를 통해 공급할 수 있습니다. 자세한 내용은 `docs/kr/EXTERNAL_DATA.md`를 참고하세요.

## 3. 거래소 인터페이스 (`ExchangeInterface`)

새로운 거래소 드라이버를 추가하려면 이 인터페이스를 구현해야 합니다.

```python
class ExchangeInterface(ABC):
    @abstractmethod
    def connect(self): pass

    @abstractmethod
    def get_balance(self, currency: str) -> float: pass

    @abstractmethod
    def get_all_positions(self) -> Dict[str, float]: pass

    @abstractmethod
    def execute_order(self, ...): pass
    
    @abstractmethod
    def fetch_open_orders(self, symbol=None): pass
    
    @abstractmethod
    def cancel_order(self, order_id, symbol=None): pass
```

구현 예시는 `lumina_quant/exchanges/` 폴더 내의 `BinanceFuturesExchange`, `MT5Exchange`, 그리고 Phase 1 `PolymarketExchange`를 참고하세요.
