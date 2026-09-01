# Hybrid live investability snapshot — 2026-04-17

## Bottom line
- **성능 관점:** 현재 reboot split 기준으로 `hybrid_guarded_mode`가 여전히 가장 강한 diversified / guarded live lane이다.
- **실투자 관점:** 아직 **real-money deployable 상태는 아님**.
- 결론적으로 현재 상태는:
  - **research / paper candidate로는 강함**
  - **실투자 preflight는 아직 block**

## 현재 성능 상태
### One-shot live decision
- current switch mode: `hybrid_guarded_mode`
- market state: `mixed / bullish / broad / calm / pair_liquidity=normal`

### Current performance-first override
- min OOS return edge: `+0.5000%`
- min OOS sharpe edge: `2.5000`
- min hybrid val return: `+6.0000%`
- min hybrid val sharpe: `3.0000`

### Current hybrid evidence
- hybrid OOS return: `+0.6868%`
- hybrid OOS sharpe: `3.2370`
- hybrid OOS max DD: `0.2573%`
- coverage-adjusted switch replay:
  - OOS return `+0.6839%`
  - sharpe `3.4091`
  - max DD `0.2406%`

## 실집행 가능성 판단
### 긍정 요소
- pair tactical은 계속 tactical-only로 제한되어 있고,
  hybrid 내부 observed pair max weight도 `24.68%` cap 아래에 머문다.
- OOS 34일 기준 allocation turnover proxy:
  - avg `0.0952`
  - median `0.0073`
  - p90 `0.2812`
- pair 평균 weight는 `10.24%`, 평균 cash weight는 `26.35%`라서
  과도한 full-risk / full-rotation 구조는 아니다.

### 비용 민감도
Hybrid sleeve 기준 one-way all-in cost stress:

| One-way cost | Adj. OOS return | Adj. Sharpe | Adj. Max DD |
| ---: | ---: | ---: | ---: |
| `5 bps` | `+0.5240%` | `2.5224` | `0.3758%` |
| `10 bps` | `+0.3614%` | `1.7447` | `0.5053%` |
| `20 bps` | `+0.0368%` | `0.1861` | `0.7639%` |
| `30 bps` | `-0.2867%` | `-1.3122` | `1.0219%` |

Interpretation:
- **낮은 비용/슬리피지(대략 5~10bps one-way)** 환경이면 여전히 의미 있는 edge가 남는다.
- **20bps+ one-way** 수준으로 밀리면 edge가 거의 사라지거나 음수로 뒤집힌다.
- 따라서 실투자 가능성은 **전략 자체보다도 집행 품질과 실제 fee tier 확보**에 크게 의존한다.

## 아직 real-money ready가 아닌 이유
Generated artifact:
- `var/reports/exact_window_backtests/followup_status/portfolio_incumbent_autoresearch_grouped/hybrid_live_investability_current/hybrid_live_investability_latest.md`

현재 block 요인:
1. refresh artifact가 live preflight 기준으로 stale
2. `portfolio_live_readiness_decision_latest.json` artifact를 최신 상태로 유지해야 함
3. `startup_reconciliation_hard_fail`가 아직 `false`
4. refresh artifact freshness가 live preflight 기준을 만족해야 함

추가로 현재 runtime config는:
- `live.mode = paper`
- `testnet = true`
- `market_data_source = committed`
- `order_state_source = polling`

즉 지금은 **paper/testnet 운영 준비도 완전 green이 아니고**, real mode는 더더욱 아니다.

## 실투자 전 필수 조건
1. fresh refresh + decision artifact 재생성
2. paper preflight를 green으로 만들기
3. `startup_reconciliation_hard_fail=true`로 승격
4. 실제 venue/account fee tier를 반영한 비용 검증
5. 실 fill 로그 기반 slippage summary 축적
6. 최소 수일~수주 paper burn-in 후 partial capital rollout

## Practical recommendation
- 지금 당장 full real capital 투입: **비권장**
- 다음 단계:
  1. current tuned hybrid gate 유지
  2. paper/live readiness artifact chain 복구
  3. 실제 fill/slippage 수집
  4. 비용이 10bps one-way 이하로 유지되는지 확인
  5. 그 뒤 small-capital staged rollout 검토
