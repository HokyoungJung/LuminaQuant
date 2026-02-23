# 성과 지표 (Performance Metrics)

LuminaQuant는 벤치마크 대비 전략 성과를 평가하기 위해 포괄적인 금융 지표들을 계산합니다.

## 핵심 지표 (Core Metrics)

### 총 수익률 (Total Return)
전체 백테스트 기간 동안 포트폴리오의 절대적인 수익 또는 손실 비율입니다.
$$ \text{Total Return} = \frac{\text{최종 자산} - \text{초기 자산}}{\text{초기 자산}} $$

### 벤치마크 수익률 (Benchmark Return)
심볼 리스트의 첫 번째 자산을 "매수 후 보유(Buy & Hold)"했을 때의 수익률입니다. 활성 전략과 수동적 투자의 성과를 비교할 때 유용합니다.

### 연평균 성장률 (CAGR)
1년 이상의 기간 동안 투자의 평균 연간 성장률입니다. 주기적인 수익률의 변동성을 평탄화하여 보여줍니다.
$$ \text{CAGR} = \left( \frac{\text{기말 가치}}{\text{기초 가치}} \right)^{\frac{1}{n}} - 1 $$
*여기서 $n$은 연수(years)입니다.*

## 리스크 지표 (Risk Metrics)

### 연환산 변동성 (Annualized Volatility)
일일 수익률의 표준편차를 연환산한 값입니다. 전략 수익의 리스크 또는 변동성을 나타냅니다.
$$ \sigma_{ann} = \sigma_{daily} \times \sqrt{\text{연간 기간 수}} $$
*(기간 수: 일간 데이터 252, 시간당 데이터 8760 등)*

### 최대 낙폭 (Max Drawdown, MDD)
포트폴리오가 고점에서 저점까지 기록한 최대 손실폭입니다. 해당 기간 동안 겪을 수 있는 최악의 하락 위험을 나타냅니다.

### 낙폭 지속 기간 (Drawdown Duration)
전고점을 회복하는 데 걸린 가장 긴 시간(봉 개수 또는 일수)입니다.

## 위험 조정 수익률 (Risk-Adjusted Returns)

### 샤프 비율 (Sharpe Ratio)
리스크(변동성) 한 단위당 초과 수익을 얼마나 냈는지 측정합니다. 높을수록 좋습니다.
$$ \text{Sharpe} = \frac{R_p - R_f}{\sigma_p} $$
*참고: 암호화폐/FX 컨텍스트에서는 편의상 무위험 수익률($R_f$)을 0으로 가정합니다.*

### 소르티노 비율 (Sortino Ratio)
샤프 비율과 유사하지만, 전체 변동성 대신 **하락(음수)** 수익률의 표준편차(하방 편차)만을 리스크로 사용하여 계산합니다. 유해한 변동성만 고려한다는 장점이 있습니다.
$$ \text{Sortino} = \frac{R_p - R_f}{\sigma_d} $$

### 칼마 비율 (Calmar Ratio)
연평균 수익률(CAGR)을 최대 낙폭(MDD)으로 나눈 값입니다.
$$ \text{Calmar} = \frac{\text{CAGR}}{\text{Max Drawdown}} $$

## 비교 지표 (vs 벤치마크)

### 알파 ($\alpha$)
시장 지수(벤치마크) 대비 투자의 초과 수익률을 나타냅니다. 양(+)의 알파는 전략이 "시장을 이겼음"을 의미합니다.

### 베타 ($\beta$)
시장 전체 대비 개별 증권이나 포트폴리오의 민감도(또는 체계적 위험)를 나타냅니다.
*   $\beta = 1$: 시장과 동일하게 움직임.
*   $\beta > 1$: 시장보다 변동성이 큼.
*   $\beta < 1$: 시장보다 변동성이 작음.
*   $\beta < 0$: 시장과 반대로 움직임(역상관).

### 정보 비율 (Information Ratio, IR)
벤치마크 대비 초과 수익(Active Return)을 추적 오차(Tracking Error, 초과 수익의 변동성)로 나눈 값입니다.
$$ \text{IR} = \frac{\text{Active Return}}{\text{Tracking Error}} $$

## 트레이딩 통계

### 일일 승률 (Daily Win Rate)
포트폴리오 수익률이 양(+)수였던 날(또는 기간)의 비율입니다.
