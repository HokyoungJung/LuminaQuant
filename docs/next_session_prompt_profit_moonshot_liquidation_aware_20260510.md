# New session prompt — profit moonshot liquidation-aware 5x validation

Paste this into the next Codex/OMX session:

```text
$ralplan $team $ralph 이어서 진행해. 작업 디렉터리는 /home/hoky/Quants-agent/LuminaQuant 이고, 먼저 아래 파일들을 읽어:

- .omx/plans/profit_moonshot_integer_leverage_20260509.md
- .omx/plans/profit_moonshot_liquidation_aware_next_20260510.md
- docs/session_handoff_20260509_profit_moonshot_integer_leverage.md
- docs/session_handoff_20260510_profit_moonshot_liquidation_next.md
- var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/integer_leverage_audit_20260509.json
- var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/integer_leverage_alpha_v2_top40_20260509/fresh_portfolio_tuning_latest.json

현재 green baseline은 private/main 02f4520cf906f48089b8852c2651a0f1e4bd0c1c 이다. 이 baseline을 보존해.

목표는 기존 current-base sleeve tuple의 integer 5x leverage가 실제로 좋은지, 특히 청산 위험까지 고려했을 때 deployable improvement인지 검증하는 것:

1) 기존 current-base 2.3427x와 forced current-base 5x를 비교하되, 5x는 아직 조건부 best일 뿐 promoted/deployable로 보지 마.
2) 구현 전 테스트를 먼저 추가해서 liquidation-aware behavior를 잠가:
   - intrabar adverse high/low가 liquidation threshold를 넘으면 liquidation event 발생,
   - train/validation/OOS별 liquidations, minimum margin buffer, minimum margin ratio가 artifact에 기록,
   - locked-OOS는 selection에 절대 사용하지 않고 report-only/gate-only,
   - liquidation count > 0 또는 minimum margin buffer <= 0이면 promoted success 금지.
3) Binance USDT perpetual에 준하는 보수적 선물 margin 모델을 구현/적용해. 정확한 tier table이 없으면 conservative scalar fallback을 명시하고, cross/isolated margin 가정을 artifact에 남겨.
4) fees/slippage/funding 또는 stress buffer를 포함해. 최소한 기존 sleeve replay 비용 가정과 추가 liquidation stress buffer를 분리해서 기록해.
5) current-base sleeve tuple에 대해 integer 5x를 train/validation/OOS 전체에서 liquidation-aware replay하고, 필요하면 1x~6x integer grid에서 청산 없는 최고 성능 leverage를 찾되 selection은 train/validation only로 유지해.
6) 통과 조건:
   - train/validation/OOS liquidation count = 0,
   - 모든 split minimum margin buffer > 0,
   - OOS MDD <= 25%,
   - OOS return과 OOS return/MDD가 current base보다 개선,
   - Sharpe/Sortino/smart Sortino/Calmar가 충분히 높음,
   - memory < 8 GiB,
   - locked-OOS는 gate-only/report-only.
7) 5x가 청산 위험 없이 통과하면 current-base 5x를 새 best로 문서화하되, 그렇지 않으면 청산 없는 최적 integer leverage 또는 기존 current base 유지 결론을 내.
8) 결과와 handoff를 .omx/notepad.md, .omx/plans, docs/session_handoff_* 및 var/reports/.../alpha_v2/liquidation_aware_* 에 저장해.
9) targeted tests/full pytest/ruff/compileall/git diff --check를 돌리고, Lore commit으로 private/main에 push한 뒤 GitHub Actions ci/private-ci green까지 확인해.
```
