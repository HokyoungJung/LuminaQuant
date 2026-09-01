# Quants-agent 전략 현실성 감사

작성일: 2026-07-13
범위: 저장소의 데이터 → 신호 → 선택 → 백테스트 → 승격 경로, 기존 성과 아티팩트, 외부 1차 문헌
판정 기준: 미래 수익 보장이 아니라 **수정된 평가 경로에서 비용 후 재현되는 엣지**가 있는가

## 1. 결론

현재 저장소에는 **실제 시장 데이터, 현실적 비용, 수정된 전략 라우팅, untouched OOS를 모두 통과한 수익 전략이 없다.** 실투자 배분은 0%가 맞다.

다만 이는 모든 전략 아이디어가 죽었다는 뜻이 아니다. 2026-07-10 결함 감사에 따르면 84/111개 전략 클래스와 2,833/3,674개 후보, 신규 research-only 33개 클래스 전부가 실제 전략이 아니라 하나의 64-bar 모멘텀 프록시로 평가됐다. 따라서 이 범위의 과거 양성·음성 판정은 모두 무효이며, 정확한 표현은 **“검증된 전략이 없다”**이지 **“유효한 전략이 존재하지 않는다”**가 아니다.

가장 짧은 해결 경로는 새 전략을 더 만드는 것이 아니라 다음 세 패밀리만 동결해 다시 측정하는 것이다.

1. 코어 크립토 6~10개의 저회전 시계열 추세
2. 실제 현물-무기한 선물 양 leg를 갖춘 funding/basis carry
3. point-in-time 크립토 유니버스의 beta-residual 횡단면 모멘텀

## 2. 가장 강한 증거

### 2.1 기존 결과는 OOS에서 반복적으로 붕괴했다

| 실험 | 검증 구간 | OOS | 판정 |
|---|---:|---:|---|
| Curated portfolio | +5.47%, Sharpe 0.791 | **-15.22%, Sharpe -0.422** | 실제 선택 실패 |
| BTC/BNB pair-spread clones | Sharpe 5~6 | **Sharpe -20.9~-24.0** | clone cluster 14/22, hard reject |
| Teacher blend | 약 +85.5% | **-28.1~-42.5%** | 붕괴 |
| Alpha101 | +10.68%, Sharpe 2.944 | +3.07%, Sharpe **-2.576** | 경제적 일관성 없음 |
| Lead-lag incumbent | — | +2.51%, Sharpe 0.24 | 20~30bp에서 음수 전환 |
| July-4 strict gate | — | 선택 행 **22/22 탈락** | DSR/SPA/PBO 불충족 |

근거: `var/reports/portfolio_superiority_curated/portfolio_optimization_latest.json`, `docs/research_note/research_note.md:64-101,120-121,2441-2445`, `docs/research_note/overfit_selection_gate_integration.md:40-45`.

2026-07-08/09의 110-symbol walk-forward headline도 신뢰할 수 없다. 예를 들어 lagged router의 +63.36%, Sharpe 1.50은 결함 수정 전 결과이며, clean dynamic conviction의 +7.99%는 11개 fold 중 양수 fold가 3개뿐이고 사실상 2025-11의 +28.71% 한 번에 의존한 뒤 마지막 fold에서 -16.71%를 기록했다. 기존 보고서에는 실제 전략 실행 여부를 나타내는 `evaluation_mode`도 없다.

### 2.2 성과보다 측정 경로가 먼저 고장 나 있었다

1. **실제 전략 대신 공통 프록시 평가**
   `src/lumina_quant/strategy_factory/research_runner.py:6543-6564`에서 실제 registry class 라우팅은 opt-in이고, `strategy_signal_dispatch.py:68-112`의 미매핑 fallback은 64-bar rolling-z momentum이다. `configs/profiles/research.yaml:57-64`에서만 실제 registry simulator 라우팅을 켠다.

2. **기본 OOS가 선택에도 사용됨**
   `src/lumina_quant/configuration/schema.py:653-725`의 기본값은 lockbox, purge/embargo, HAC, CSCV PBO, exposure normalization, hard reject를 모두 끈다. 기본 3-way 경로에서는 reported OOS가 binding selection에 다시 쓰인다. 엄격한 값은 `configs/profiles/research.yaml:21-64`에만 있다.

3. **`lq alpha promote`는 실데이터 탐색기가 아님**
   `src/lumina_quant/cli/alpha.py:18-21,288-308,380-400`은 항상 의도적으로 close-momentum edge를 심은 synthetic panel을 만든다. 이 명령의 승격 결과는 실전 알파 증거로 사용할 수 없다.

4. **locked-OOS 내부 미래 분포 사용**
   `src/lumina_quant/research/crypto_fx_alpha_zoo_real_data.py:403-521`은 split 전체 factor quantile로 해당 split 진입 threshold를 정하면서도 `uses_locked_oos_for_selection=False`를 기록한다. locked-OOS의 미래 분포를 미리 아는 look-ahead다.

5. **point-in-time 유니버스가 아님**
   `src/lumina_quant/research_universe.py:14-160`은 2026-06-13 현재 상장 스냅샷이다. 역사적 상장·폐지 membership이 없다. 최근 full-universe 보고서의 110개 중 2025-01-01부터 1d 이력이 있는 것은 10개뿐이고, 69개는 60일 이하, 88개는 120일 이하 이력만 가진다. 동일 횡단면으로 비교하면 listing-season 효과와 survivorship이 섞인다.

6. **기본 비용과 live parity가 낙관적**
   `config.yaml:64,68`은 기본 보호 stop과 백테스트 order risk gate를 끈다. `config.yaml:80-105`는 flat size-blind slippage, funding 0, missing funding 허용, 실제 UTC settlement 미사용이다. sqrt impact와 funding coverage는 `configs/profiles/backtest_cost_realistic.yaml:86-115`에서만 강제된다. 현재 구현의 시장가 기대비용은 편도 10bp(taker 4bp + half-spread 1bp + slippage 5bp), round-trip 20bp이며 impact와 funding은 별도다(`cost_realism.py:100-115`, `quality_gated_allocation.py:67-76`).

7. **실데이터가 없어 현재 머신에서 알파를 재현할 수 없음**
   기본 `data/market_parquet`에는 parquet가 0개다. `data/BTCUSDT.csv`, `data/ETHUSDT.csv`는 `generate_data.py:8-60`이 만든 1,000일 결정론적 random walk다. 별도 audit backup에 BTC 하루치 1s/1h 조각 두 개가 있지만 전체 유니버스 검증에는 쓸 수 없다. 즉 현재 로컬 데이터는 기능 smoke test용이지 수익성 검증용이 아니다.

8. **연구량이 표본보다 너무 큼**
   저장소에는 약 148개 전략 클래스와 G005 기준 1,404개 후보가 있으며, 개별 실험은 18,450~36,000 trial까지 수행했다. G005의 964개 집계 후보에서도 1d 105/105, 30m 107/107이 탈락했고 4h 최대 OOS DSR은 약 0.026이었다. raw Sharpe 6.64가 PBO 1과 공존한 것은 좋은 전략보다 selection artifact의 전형에 가깝다.

9. **엄격 프로필 두 개를 단순히 이어 쓸 수 없음**
   `LQ_CONFIG_PATH` 프로필은 root config를 merge하지 않고 대체한다. `configs/profiles/research.yaml:64`에는 `route_unmapped_registered_strategies: true`가 있지만 `backtest_cost_realistic.yaml`에는 이 flag가 없다. 따라서 비용 프로필만 final proof에 사용하면 미매핑 전략이 다시 generic fallback으로 갈 수 있다. 실제 최종 실험에는 비용 프로필을 바탕으로 이 routing flag까지 포함한 **하나의 동결된 결합 프로필**이 필요하다.

### 2.3 수정 코드와 수익성은 별개다

현재 main에서 아래 라우팅·엄격 gate 표적 테스트를 한 번에 재실행해 **166 passed in 2.84s**를 확인했다.

```bash
uv run pytest -q \
  tests/test_research_profile_activation.py \
  tests/test_overfit_selection_engine_gate.py \
  tests/test_research_selection_flags_config.py \
  tests/test_backtest_audit_gates.py \
  tests/test_strategy_signal_dispatch_routing.py \
  tests/test_strategy_signal_dispatch.py \
  tests/test_research_runner_feature_support.py \
  tests/test_alpha_zoo_69_asset_efficiency_repair_optuna.py
```

이는 결함 수정과 설정 wiring이 동작한다는 증거일 뿐, 수정 후 실제 시장 replay가 없으므로 수익성 증거가 아니다.

## 3. 전략 후보 우선순위

### A. 저회전 시계열 추세 — 1순위

**가설:** BTC/ETH 및 고유동성 major에서 1d 또는 4h 추세를 30~180일 horizon으로 측정하고, 5일 또는 주간 리밸런싱과 hysteresis/min-hold로 turnover를 제한한다.

**왜 남기는가:** crypto의 일·주간 momentum과 전통 선물의 1~12개월 time-series momentum은 별도 표본에서 반복 보고됐다. [NBER의 crypto risk/return 연구](https://www.nber.org/system/files/working_papers/w24877/w24877.pdf), [Moskowitz·Ooi·Pedersen의 Time Series Momentum](https://w4.stern.nyu.edu/facdir/lpederse/papers/TimeSeriesMomentum.pdf).

**레포 적합성:** 이미 `TopCapTimeSeriesMomentumStrategy`, `CrossAssetDiversifiedTrendStrategy` 계열이 있어 새 추상화가 필요 없다. Alpha-Max 브랜치의 daily trend도 같은 방향이다.

**반증 조건:** 소수 horizon을 사전등록하고, 한 bull fold를 제거해도 순수익이 양수이며, 10/15/20/30bp와 실제 funding/sqrt impact를 견뎌야 한다. gross leverage가 아닌 exposure-normalized 성과로 통과해야 한다.

### B. 진짜 funding/basis carry — 2순위, 인프라 선행

**가설:** BTC/ETH/소수 major에서 long spot + short perpetual/futures로 basis와 funding을 수취하는 market-neutral book을 구축한다.

**왜 남기는가:** BIS는 BTC/ETH cash-and-carry의 높은 역사적 carry와 차익거래 자본 제약을 분석한다. 동시에 높은 carry가 향후 crash와 연결될 수 있어 위험 조정이 필수다. [BIS Working Paper 1087, revised 2025](https://www.bis.org/publ/work1087.htm).

**중요한 불일치:** 현재 기본 live 경로는 futures-only이며, `cross_sectional_funding_momentum_carry.py`의 기본 `true_carry_sign=false`는 진짜 carry가 아니라 funding 변화/기울기 추종이다. BIS의 근거를 이 클래스의 수익 근거로 전용하면 안 된다.

**필수 데이터:** 실제 `fundingTime`, symbol별 funding interval, mark/index/spot, 양 leg 수수료와 체결, borrow, basis loss, margin과 liquidation. Binance 공식 API는 실제 funding timestamp와 rate를 제공한다. [Binance funding-rate history](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Get-Funding-Rate-History).

**반증 조건:** 모든 비용 후 순carry가 양수이고 beta가 0에 가깝고, 음의 funding·basis 급변·margin shock에서도 생존해야 한다.

### C. beta-residual 횡단면 모멘텀 — 3순위

**가설:** 110개 혼합자산을 rank하지 말고 point-in-time crypto-only 유니버스에서 유동성·상장연령 필터를 적용한 뒤 BTC beta를 제거한 주간 residual momentum을 운용한다.

**왜 남기는가:** 1,700개 이상 crypto 횡단면에서 market, size, momentum 요인이 기대수익 차이를 설명한다는 근거가 있다. [NBER Working Paper 25882](https://www.nber.org/papers/w25882).

**반증 조건:** delisted 자산을 포함한 point-in-time membership, beta-neutral/equal-weight/vol-weight 모두에서 생존하고, test set에서 rank rule이나 universe를 다시 고르지 않아야 한다.

### D. near-52-week-high — 보조 가설만

Alpha-Max의 daily near-52-week-high는 외부 prior가 있지만 crypto 직접 근거가 trend/momentum보다 약하다. 독립 패밀리로 trial 수를 늘리기보다 trend와 동일한 동결 번들에서 한 개 사전등록 variant만 측정한다. [52-week high momentum 원 논문](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1104491).

## 4. 당분간 폐기할 것

- **minute lead-lag:** 문헌의 gross spread 1.54~3.34bp가 현재 기본 시장가 편도비용 10bp보다 작다. 느린 horizon에서 break-even cost가 실제 비용을 넘는다는 증거가 없으면 경제적으로 무효다. [JEDC 2024](https://www.sciencedirect.com/science/article/abs/pii/S0165188924000551).
- **고빈도 reversal:** reversal은 비유동 crypto에 집중된다는 연구가 있고, 이 영역은 spread·impact·capacity가 가장 나쁘다. midquote, queue fill, BBO/depth가 없으면 평가하지 않는다. [International Review of Financial Analysis 연구](https://www.sciencedirect.com/science/article/pii/S1057521921002349).
- **새 indicator/ML/parameter clone:** 기존 탐색 규모가 이미 표본을 압도한다. 검증 인프라를 통과한 frozen family가 나오기 전에는 후보 수를 추가하지 않는다.
- **110개 현재시점 혼합 유니버스:** crypto, 금속, ETF/지수 프록시, 단기 상장 주식 perpetual을 한 횡단면으로 rank하지 않는다.
- **기존 headline 재활용:** +60~200% router, synthetic alpha promotion, pre-fix research-only scoreboard는 투자 근거에서 제외한다.

## 5. 한 번만 제대로 돌릴 검증 프로토콜

### 단계 0 — 데이터

Binance 공식 public archive/API에서 코어 6~10개의 실제 1h/4h/1d OHLCV, mark/index, funding을 먼저 백필한다. 장기 OI는 공식 REST의 보존 한계가 있으므로 최초 실험의 필수 조건에서 제외한다. [Binance public data](https://github.com/binance/binance-public-data), [Binance exchange info](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Exchange-Information).

### 단계 1 — 후보 동결

- 전략 패밀리 3개, 각 패밀리의 horizon/weight variant를 최소화한다.
- branch `feat/alpha-max-20260710`의 trend / near-52-week-high / 4h funding 번들을 참고하되, 그 브랜치의 `alpha_max_final_delivery_20260711.md`도 **실제 market replay와 수익 주장을 하지 않는다**고 명시한다.
- current main에 무작정 merge하지 말고, 실행할 commit·manifest·seed·데이터 해시를 먼저 동결한다.

### 단계 2 — 측정 경로 강제

- discovery: `configs/profiles/research.yaml`
- final proof: `configs/profiles/backtest_cost_realistic.yaml`을 바탕으로 `research.route_unmapped_registered_strategies: true`까지 명시한 **단일 동결 결합 프로필**의 windowed/live-parity 경로. 두 프로필은 root-replacement 방식이므로 단순히 연속 적용하지 않는다.
- `evaluation_mode`를 모든 후보 행에 기록하고 `registry_simulator` 또는 명시적 handler가 아니면 무효 처리
- `best_params.json` provenance가 validation-only가 아니면 경고가 아니라 reject
- 실제 funding settlement와 sqrt impact를 필수로 하고, 10/15/20/30bp를 같은 frozen signal에 적용

### 단계 3 — 선택 규칙

- validation에서만 ranking과 gate를 수행하고 lockbox는 마지막 1회만 연다.
- purge/embargo, HAC, CSCV PBO, single correlation discount, exposure normalization을 모두 켠다.
- 최소 gate: **DSR >= 0.90, SPA p <= 0.05, PBO <= 0.50**.
- 한 fold가 전체 수익을 지배하거나, funding 누락, point-in-time membership 누락, 비용 20bp에서 순수익 <= 0이면 즉시 폐기한다.
- 여러 후보 중 최고값을 OOS에서 다시 고르지 않는다. DSR/PBO를 쓰는 이유는 다중검정과 backtest overfitting을 직접 통제하기 위해서다. [Deflated Sharpe Ratio 원 논문](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf), [Probability of Backtest Overfitting](https://escholarship.org/uc/item/4hn4t174).

### 단계 4 — fresh forward

lockbox 생존 후보만 1~2개월 frozen shadow/paper-forward로 보낸다. 이 기간 동안 parameter, universe, risk scaling을 바꾸면 시계를 다시 시작한다. fresh-forward 전 실제 자본 배분은 0%다.

## 6. 의사결정 규칙

| 결과 | 행동 |
|---|---|
| 세 패밀리 모두 gate 또는 20bp 비용에서 탈락 | 현재 전략군 종료, 새 코드 추가 금지, 데이터/시장 가설부터 재정의 |
| validation만 통과하고 lockbox 실패 | 과최적화로 폐기 |
| lockbox 통과, forward 실패 | 구조적 decay 또는 체결 모델 오차로 폐기 |
| lockbox와 1~2개월 forward 모두 통과 | 소액 testnet/shadow risk review로만 이동; 수익 보장으로 표현 금지 |

## 7. 최종 판단

사용자가 유효한 전략을 못 찾은 것은 자연스럽다. 현재 저장소는 전략 생성 능력에 비해 **데이터, point-in-time universe, 비용 현실성, untouched OOS, 결과 provenance**가 뒤처져 있다. 지금 필요한 것은 149번째 전략이 아니라, 세 개의 동결된 경제 가설을 실제 데이터로 한 번 정확히 죽이거나 살리는 실험이다.
