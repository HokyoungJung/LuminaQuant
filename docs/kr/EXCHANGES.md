# 거래소 설정 및 구성 (Exchange Setup)

LuminaQuant는 통합 인터페이스를 통해 여러 거래소를 지원합니다. 현재 지원되는 드라이버는 **CCXT** (암호화폐용)와 **MetaTrader 5** (FX/주식용)입니다.

## 1. 바이낸스 (Binance, via CCXT)

LuminaQuant는 `ccxt` 라이브러리를 사용하여 바이낸스를 포함한 100개 이상의 암호화폐 거래소에 연결합니다.

### 필수 조건 (Prerequisites)
- Python `ccxt` 패키지 (`uv sync`로 설치).
- 바이낸스 계정 (실계좌 또는 테스트넷).
- API Key 및 Secret Key.

### 구성 (`config.yaml`)

```yaml
live:
  exchange:
    driver: "ccxt"
    name: "binance"
  testnet: true  # 실거래 시 false로 설정
```

### 환경 변수 (`.env`)
프로젝트 루트에 `.env` 파일을 생성하고 키를 입력합니다:

```ini
BINANCE_API_KEY=your_actual_api_key
BINANCE_SECRET_KEY=your_actual_secret_key
```

### 고급 사용법 (파라미터)
전략에서 주문을 실행할 때, `params` 딕셔너리를 통해 CCXT 전용 파라미터를 전달할 수 있습니다.

```python
# 예시: Time-In-Force 'GTC' 지정가 주문 전송
self.execution_handler.execute_order(
    OrderEvent(
        symbol="BTC/USDT",
        order_type="LMT",
        quantity=0.1,
        headers={"price": 50000}, # Config 의존
        direction="BUY"
    ),
    params={"timeInForce": "GTC"} # ccxt.create_order로 직접 전달됨
)
```

---

## 2. 메타트레이더 5 (MetaTrader 5, MT5)

MetaTrader 5 터미널과 직접 연동하여 FX, CFD, 주식, 선물 등을 거래할 수 있습니다.

### 필수 조건
1.  **OS**: Windows (MT5 Python API는 윈도우 전용입니다).
2.  **소프트웨어**: MetaTrader 5 터미널이 설치되어 있고 실행 중이어야 합니다.
3.  **계정**: 터미널에서 데모 또는 실계좌에 로그인된 상태여야 합니다.
4.  **설정**:
    *   메뉴에서 **도구(Tools) -> 옵션(Options) -> 전문가 조언자(Expert Advisors)**로 이동.
    *   ✅ **알고리즘 트레이딩 허용 (Allow algorithmic trading)** 체크.
    *   ✅ **DLL 가져오기 허용 (Allow DLL imports)** 체크 (환경에 따라 필요할 수 있음).

### 구성 (`config.yaml`)

```yaml
live:
  exchange:
    driver: "mt5"
    name: "metatrader" # mt5 드라이버에서는 이 이름이 무시됨
  
  # 기본 주문 설정
  mt5_magic: 234000      # 이 봇의 주문 고유 ID (매직 넘버)
  mt5_deviation: 20      # 최대 허용 슬리피지 (포인트 단위)
  mt5_comment: "LuminaBot"
```

### 고급 사용법 (파라미터)
주문마다 전역 설정을 덮어쓸(Override) 수 있습니다.

```python
# 예시: 커스텀 매직 넘버를 사용한 주문 전송
signal = SignalEvent(...)
# Execution Handler 또는 커스텀 주문 생성 시:
exchange.execute_order(
    symbol="EURUSD",
    type="market",
    side="buy",
    quantity=0.1,
    params={
        "magic": 999999,        # 설정된 매직 넘버 덮어쓰기
        "deviation": 50,        # 슬리피지 허용치 변경
        "comment": "SuperStrat" # 커스텀 코멘트
    }
)
```

### MT5 문제 해결
- **"Not Connected"**: MT5 터미널이 켜져 있고 계정에 로그인되었는지 확인하십시오.
- **"IPC Error"**: MT5를 관리자 권한으로 실행했는데 파이썬은 아닐 경우(또는 그 반대) 발생할 수 있습니다. 둘 다 동일한 권한으로 실행하십시오.
- **Symbol Not Found**: 거래하려는 종목(예: "EURUSD")이 MT5의 **종합시세(Market Watch)** 창에 추가되어 있는지 확인하십시오.
