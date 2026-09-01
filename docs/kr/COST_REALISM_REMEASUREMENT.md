# 비용 현실성 엣지 재측정 가이드

**대상:** 백테스트 머신에서 백테스트/워크포워드를 돌리는 사람.
**목표:** 전략의 헤드라인 엣지가 *현실적인* 체결 비용(주문크기/시장충격 반영 슬리피지,
실제 부과되는 펀딩, 강제되는 리스크 캡, 보호 스탑) 하에서 얼마나 살아남는지를, 헤드라인
수치가 산출된 낙관적 기본값 대신 측정한다.

> **왜 필요한가.** 감사-하드닝 작업(`main` 머지, 커밋 `2c8f685`)이 여러 비용-현실성
> 제어를 추가했지만, 골든 회귀를 byte-identical로 유지하고 과거 수치를 재현 가능하게
> 두기 위해 **config-gated OFF**(기본 비활성)로 두었다. 공개된 헤드라인 수치(예: lagged
> leaf router의 85자산 리플레이 `+197%` OOS → 확장 유니버스에서 `+9.75%`로 감쇠)는
> **플랫·주문크기 무시 슬리피지 + 펀딩 0**으로 측정됐다. 이는 한 방향(낙관) 편향 원천이다.
> 이 가이드는 현실성을 켜고 재실행해 실제 투자 가능성을 확인한다.
>
> 아래 플래그는 모두 *과거* 동작이 기본값이며, 켜면 백테스트 수치가 의도적으로 바뀐다.
> CI가 아니라 백테스트 PC에서 수행하라.

---

## 1. 제어 플래그

모든 노브는 `config.yaml`에 있다(별도 CLI 플래그 불필요). 스키마:
`src/lumina_quant/configuration/schema.py` (`RiskConfig`, `ExecutionConfig`).

### 체결/슬리피지 현실성 — `execution:`

| 키 | 기본값 | 현실값 | 효과 |
| :-- | :-- | :-- | :-- |
| `slippage_impact_model` | `"flat"` | `"sqrt_impact"` | `flat` = 레거시 주문크기 무시 슬리피지(골든 byte-identical). `sqrt_impact` = 제곱근 시장충격 항 추가 → 크고 레버리지 큰 주문이 더 많이 지불. |
| `slippage_impact_coefficient` | `0.0` | 보정(§5) | 충격 강도. 페널티 ∝ `coefficient * sqrt(participation)`. `0.0`이면 `sqrt_impact`라도 충격 0. |
| `slippage_adv_quote` | `0.0` | 심볼별 ADV(quote) | participation 분모. `0.0`이면 bar별 quote 거래량 사용. 의도하는 규모의 일평균 거래대금을 넣으면 participation 반영. |
| `require_funding_coverage` | `false` | `true`(레버리지) | `true`면 per-bar 펀딩 데이터가 없을 때 조용히 0.0을 부과하지 않고 **명시적으로 실패**. 펀딩이 실제로 부과되도록 데이터 보유를 강제. |

펀딩은 `execution.funding_rate_per_8h`(정적) 및/또는 per-bar 펀딩 피처 데이터에서 부과된다.
현실적 펀딩을 부과하려면 **펀딩 데이터를 수집**해야 한다(`data.kinds`에 `funding` 포함).
`require_funding_coverage: true`는 그것을 했는지 검증하는 가드다.

### 리스크 강제(어떤 거래/사이즈가 일어나는지를 바꿈) — `risk:`

| 키 | 기본값 | 현실값 | 효과 |
| :-- | :-- | :-- | :-- |
| `allow_metadata_risk_override` | `false` | `false` 유지 | **이미 기본 활성.** `false`면 슬리브 signal metadata가 리스크 상한을 *낮추기*만 가능 — leverage/exposure/order value/notional을 config 캡 위로 올리지 못함. 옛 무클램프 수치를 재현할 때만 `true`. |
| `max_leverage` | `0.0` | 하드 상한 | 클램프 활성 시 metadata leverage 오버라이드의 절대 상한. `0.0`이면 metadata가 설정된 run leverage 위로 못 올림. |
| `attach_default_protective_stop` | `false` | `true` | `stop_loss`가 없는 시그널에 `default_stop_loss_pct` 합성 스탑 부착 → 무방비 포지션 제거. 미스탑 슬리브의 PnL이 바뀜. |
| `enforce_order_risk_gate_in_backtest` | `false` | `true` | 백테스트 주문 경로에서도 live와 동일한 `RiskManager.check_order` 게이트 실행 → 캡 초과 주문 거부 가능. |
| `hard_drawdown_flatten_pct` | `0.0` | 선택, 예: `0.20` | `> 0`이면 `auto_flatten_on_breach`가 false여도 장중 DD가 이 비율 초과 시 전 포지션 청산. `0.0`이면 비활성. |

> **참고:** `live.max_bbo_age_seconds`는 *live 전용* 안전 플래그(기본 `2.0`)이며 백테스트에
> 영향 없음 — 재측정 대상 아님.

---

## 2. 권장 현실성 프로파일

시작점으로 좋은 "현실적" `config.yaml` 오버레이:

```yaml
execution:
  slippage_impact_model: "sqrt_impact"
  slippage_impact_coefficient: 0.10   # 여기서 시작, §5에서 보정
  slippage_adv_quote: 0.0             # 0 = bar별 거래량; 또는 심볼별 ADV
  require_funding_coverage: true      # 펀딩 데이터 수집된 경우만
  maker_fee_rate: 0.0002
  taker_fee_rate: 0.0004
  spread_rate: 0.0002
  slippage_rate: 0.0005

risk:
  allow_metadata_risk_override: false
  max_leverage: 0.0
  attach_default_protective_stop: true
  enforce_order_risk_gate_in_backtest: true
  hard_drawdown_flatten_pct: 0.0

data:
  kinds: [ohlcv, funding, feature_points]   # funding 반드시 포함
```

§3의 A/B를 위해 기본값 `config.flat.yaml`(기준선)을 따로 보관하라.

---

## 3. 재측정 절차 (A/B)

```bash
# 0) 유니버스/구간의 펀딩 + OHLCV 데이터 수집
uv run lq data collect            # (소스는 docs/EXTERNAL_DATA.md 참조)

# 1) 기준선 — 플랫 비용, 현실성 OFF (수치 기록)
uv run lq backtest --run-id baseline_flat
uv run lq optimize --folds 10 --oos-days 30 --validation-days 30 --run-id baseline_flat_wf

# 2) 현실 — 준비된 프로파일 사용 (config.yaml + 비용현실성 플래그 ON).
#    LQ_CONFIG_PATH는 config.yaml을 "대체"한다(병합 아님). 프로파일은 config.yaml
#    전체 복사본에서 플래그만 켠 것 — config.yaml을 수정하면 동기화 유지할 것.
LQ_CONFIG_PATH=configs/profiles/backtest_cost_realistic.yaml uv run lq backtest --run-id realistic
LQ_CONFIG_PATH=configs/profiles/backtest_cost_realistic.yaml \
  uv run lq optimize --folds 10 --oos-days 30 --validation-days 30 --run-id realistic_wf
```

> 기준선(1단계)은 루트 `config.yaml`(플래그 OFF), 현실 실행은
> `configs/profiles/backtest_cost_realistic.yaml`(플래그 ON) — 비용/리스크 현실성
> 블록만 다른 깨끗한 A/B. 직접 손튜닝하려면 §2 오버레이를 본인 config 복사본에 적용.

- 기준선과 현실 실행 간 `backtest.random_seed`, 유니버스, 구간, 폴드 수,
  `--oos-days`/`--validation-days`를 **동일하게** 유지하고 비용/리스크 플래그만 바꾼다.
- "엣지 잔존"의 의미 있는 비교는 워크포워드(`lq optimize`)다. 단일 `lq backtest`는 빠른 확인용.
- 아티팩트는 `--run-id`로 키된 `var/reports/...`에 저장된다 — 둘을 비교.

---

## 4. 비용 스트레스 그리드 (왕복 10 / 15 / 20 bps)

편도 비용 ≈ `taker_fee_rate + spread_rate/2 + slippage_rate` (+ `sqrt_impact` 항).
**왕복 ≈ 2 × 편도.** 왕복 `X` bps로 스트레스하려면 편도 ≈ `X/2` bps가 되도록 노브 설정:

| 왕복 목표 | 제안 (기본 fees+spread ≈ 5 bps/편도 기준) |
| :-- | :-- |
| 10 bps | 편도 ≈ 5 bps |
| 15 bps | 편도 ≈ 7.5 bps |
| 20 bps | 편도 ≈ 10 bps |

각 비용 수준에서 §3-2를 재실행하고 감쇠를 기록한다. 진짜 엣지는 그리드 전체에서 양수를
유지(그리고 §6 게이트 통과)해야 한다. 15–20 bps에서 음수로 뒤집히면 비용-취약 엣지다.
체계적 그리드는 `scripts/research/`의 리서치 러너(`docs/research_note/research_note.md`가
참조하는 `*_cost_stress_*` 아티팩트)가 자동화한다. 위의 수동 config 방식이 재현 기준선이다.

---

## 5. `sqrt_impact` 보정

충격 페널티 ≈ `slippage_impact_coefficient * sqrt(order_notional / 분모)`,
분모 = `slippage_adv_quote`(설정 시), 아니면 bar별 quote 거래량(`price * volume`).

- 보수적 1차 패스: `slippage_impact_coefficient: 0.10`, `slippage_adv_quote: 0.0`(bar 거래량).
- 캐파시티/규모 연구: `slippage_adv_quote`를 의도 규모의 심볼별 일평균 거래대금으로 설정해
  participation(따라서 충격)이 실제 규모를 반영하게 한다.
- 페널티는 `[0, 0.99]`로 클램프(음수 체결가 없음) — 계수 오설정 시 PnL을 부풀리지 않고 안전 실패.

---

## 6. 결과 해석 & go/no-go

기준선 vs 현실을 워크포워드 OOS 집계로 비교:

| 지표 | 관전 포인트 |
| :-- | :-- |
| OOS 복리/연율 | 헤드라인. 현실 비용에서 얼마나 살아남나 |
| Sharpe | 위험조정 잔존 |
| 최대 OOS DD | 현실 하의 꼬리 |
| Profit factor | PF가 비현실적(예 ~30)에서 ~1–3로 붕괴하면 헤드라인이 비용-낙관이었다는 신호 |
| 양수 폴드 (예 x/10) | 워크포워드 안정성 |
| 회전율 / 거래당 수익(RPT) | 고회전 엣지가 가장 비용 민감 |

이 재측정은 **리서치 증거일 뿐** 무엇도 실거래로 승격하지 않는다. 실거래 go-live는 여전히
거버넌스 게이트(`src/lumina_quant/live/readiness_policy.py`, `docs/live-readiness/`)를 요구한다:
train/validation 전용 클린 선택, locked-OOS 리포트 전용 워크포워드, 체결 텔레메트리가 있는
fresh-forward 섀도/페이퍼, 위 비용 스트레스 그리드, 회전율/RPT, BBO/슬리피지,
부분/거부/취소/리컨실 증거, `ready_for_real`/`clean_promotion_eligible` 플래그 전환,
그리고 사람 리뷰. 기본 실거래 배분은 `0%`로 유지된다.

---

## 7. 재현성 체크리스트

- [ ] A/B 간 `random_seed`, 유니버스, 타임프레임, 구간, 폴드, `--oos-days`/`--validation-days` 동일.
- [ ] `require_funding_coverage` 켜기 전 펀딩 데이터 수집(`data.kinds`에 `funding`).
- [ ] `validation.golden_rtol` 불변; 현실성 플래그 **OFF**(기본)에서 골든 스위트 통과 유지.
- [ ] 두 `--run-id`와 현실 실행에 쓴 정확한 `config.yaml` diff 기록.
- [ ] 각 아티팩트에 비용 수준(10/15/20 bps) 명기.

함께 보기: [`../CONFIG_SPEC.md`](../CONFIG_SPEC.md) · [`../MODEL_ASSUMPTIONS.md`](../MODEL_ASSUMPTIONS.md) ·
[`METRICS.md`](METRICS.md) · [`../FINAL_VALIDATION.md`](../FINAL_VALIDATION.md) ·
[`../research_note/research_note.md`](../research_note/research_note.md).
