# Strategy Recovery Master Plan

작성일: 2026-07-13
상태: 실행 준비 계획, 실자본 배분 0%
근거 감사: [`docs/audits/strategy_reality_audit_20260713.md`](../audits/strategy_reality_audit_20260713.md)

## 1. 목적과 종료 조건

이 계획은 검증 없이 새 전략 수를 늘리는 계획이 아니다. 먼저 다음 세 트랙을 실제 데이터와 수정된 평가 경로로 한 번만 다시 측정하고, 통과하지 못하면 같은 표본에서 추가 튜닝하지 않고 종료한다. 그 판정 뒤에는 별도 preregistration과 run ID로 기존 구현을 재사용하는 후속 알파 후보 프로그램을 열 수 있다.

1. 고-CAGR router R1: `codex_lagged_leaf_router_grid:h4_avg1_tr-0.02_tmdd0.50_val0.00_vmdd0.25_lagged_plus_val025_exact_unscaled`
2. 고-CAGR router R2: `codex_lagged_leaf_router_grid:h4_avg1_tr-0.02_tmdd0.50_val0.00_vmdd0.25_lagged_plus_val025_fallback_mdd20_cap2`
3. Alpha-Max Revision 5.15 listing-aware 프로토콜

완료 조건은 데이터 PC가 재현 가능한 입력 묶음과 실행 기록을 만들고, 각 트랙이 과학적 통과 또는 종료 판정을 받은 뒤 최대 두 후보만 fresh-forward로 넘기는 것이다. 후속 알파 후보 프로그램도 한 cycle당 독립 알파 leaf 최대 1개와 risk overlay 최대 1개만 forward로 보낸다. 이 계획의 완료 자체는 paper, testnet, live 또는 실자본 승인을 뜻하지 않는다.

## 2. 현재 판단

### 2.1 고-CAGR 전략은 포함한다

기존 headline을 버리는 것이 아니라 가장 강했던 두 router만 보존한다.

| 후보 | 과거 진단 headline | 현재 지위 |
|---|---:|---|
| R1 exact-unscaled | 10-fold comp `+159.83%`, ann approx `+214.51%`, MDD `27.69%`, hit `4/10`; 별도 exposed replay는 comp `+197.37%` | post-OOS 연구 진단, 미검증 |
| R2 fallback-mdd20-cap2 | comp `+138.11%`, ann approx `+183.23%`, MDD `23.58%`, hit `4/10` | 위험절감 연구 진단, 미검증 |
| 최신 11-fold R1 계열 | comp `+63.36%`, ann approx `+70.81%` | `clean_promotion_eligible=false`, fresh-forward 필수 |

개선 목표는 과거 CAGR을 더 키우는 것이 아니다. 실제 전략 라우팅, point-in-time 유니버스, 실제 funding, 비용, 노출 정규화와 fold 안정성을 고쳐 그 CAGR 중 재현 가능한 부분이 있는지 확인한다.

### 2.2 날짜 정책

- Router는 기존 runner의 원래 경계인 train start `2025-01-01`, first OOS `2025-09-01`을 유지한다. Router 때문에 2024년부터 전량 백필하지 않는다.
- Alpha-Max는 원래 기간을 유지한다. warmup이나 train 시작을 뒤로 미루지 않는다.
- 데이터가 더 오래 존재하면 보존하지만, 존재한다는 이유만으로 discovery 기간을 넓히지 않는다.

### 2.3 백필의 정의

이 계획에서 백필은 **인벤토리로 확인된 owned interval의 실제 결손을 공식 원천으로 채우는 것**이다. 다음은 백필이 아니다.

- 현재 스냅샷 110개 전체를 무조건 오래 수집하기
- 없는 상장 전 구간을 0, forward-fill 또는 합성 가격으로 만들기
- Alpha-Max의 날짜를 바꾸거나 TONUSDT를 다른 symbol로 대체하기
- 기존 funding row가 없는데 파생 열만 계산해 실제 funding을 대체하기

## 3. 불변조건

1. R1/R2와 Alpha-Max 판정 전에는 신규 indicator, ML, overlay, router grid와 인접 후보 대체를 추가하지 않는다. 후속 알파 프로그램도 기존 class·indicator·wrapper로 검증 가능한 가설은 새 코드 없이 측정하고, feature admission을 통과했지만 기존 구현이 없는 경우에만 별도 preregistration으로 최소 구현한다.
2. 모든 단계의 실자본 배분은 0%다.
3. synthetic `data/BTCUSDT.csv`, `data/ETHUSDT.csv`는 성과 실험에서 제외한다.
4. operational STOP과 scientific KILL을 구분한다.
   - STOP: 디스크, 네트워크, checksum, 프로세스 실패처럼 입력을 고치고 새 output ID로 재실행할 수 있는 문제
   - KILL: 유효한 실행이 binding gate를 실패하거나 원본 전략을 유일하게 복원할 수 없는 문제
5. exposed historical 결과는 diagnostic일 뿐 selection이나 배분에 사용하지 않는다.
6. result가 좋아도 같은 과거 구간에서 파라미터, universe, risk cap을 다시 고르지 않는다.

## 4. 알려진 선행 결함

| ID | 결함 | 영향 | 필요한 조치 |
|---|---|---|---|
| G-01 | `LQ_CONFIG_PATH`는 root replacement이며 profile merge가 아님 | 비용 profile만 쓰면 registry routing flag가 사라질 수 있음 | cost-realistic 기준의 단일 strict combined profile과 hash 생성 |
| G-02 | 고-CAGR CLI는 family 단위만 선택하고 R1/R2 exact ID replay 입력이 없음 | full rerun이 다시 grid search가 될 수 있음 | exact 두 candidate만 받는 frozen manifest replay seam 추가 |
| G-03 | 과거 router artifact에 faithful actual-engine leaf manifest가 없음 | headline을 실제 전략 실행으로 유일하게 복원하지 못할 수 있음 | fold별 class/params/symbol/timeframe/weights/gross/decision-clock 복원; 불가능하면 router KILL |
| G-04 | point-in-time lifecycle registry가 없음 | survivorship와 listing-season 혼입 | symbol lifecycle registry와 fold별 membership manifest 구현 |
| G-05 | monthly router의 비용 stress는 선형 proxy 10/15/20bp | final cost proof가 아님 | frozen signal에 actual funding, sqrt impact, 10/15/20/30bp 적용 |
| G-06 | main의 `scripts/materialize_market_windows.py --help`가 config view 오류로 실패 | raw-first final proof 준비 불가 | CLI 회귀 테스트 후 최소 수정 |
| G-07 | Alpha-Max branch의 공식 data-PC runbook은 Rev5.14 파일과 hash를 지시 | 최신 Rev5.15를 그대로 실행할 수 없음 | runbook을 Rev5.15에 맞춰 수정·검증·push하기 전 prelock 실행 금지 |
| G-08 | 현재 coverage/inventory CLI는 row count와 first/last 중심이며 duplicate, nonmonotone, nonfinite, expected-grid gap, funding settlement gap을 fail-close하지 않음 | 단순 coverage JSON을 데이터 무결성 증명으로 오인할 수 있음 | 기존 `validate_ohlcv_frame`을 재사용하고 expected 1m grid와 실제 funding cadence를 추가한 단일 JSON validation receipt CLI 구현 |

## 5. 실행 단계

### Phase D0 — 데이터 PC 인벤토리와 원본 복구

**입력**

- main의 정확한 실행 commit
- Alpha-Max `feat/alpha-max-20260710` commit `629d91e5d4aac26911af65a4a5e15ebdcbded30f`
- 기존 parquet, raw aggTrades/1s, feature/funding root
- 거래소 listing/delivery와 source provenance

**작업**

1. 기존 데이터 root를 먼저 찾고 read-only 원본으로 동결한다.
2. 물리 파일 인벤토리와 실제 repository loader coverage를 각각 만든다.
3. coverage/inventory 결과는 triage로만 기록하고, D-01A validation receipt로 symbol/timeframe별 duplicate, nonmonotone, nonfinite, expected-grid gap과 funding settlement gap을 검증한다.
4. 현재 static universe와 별개로 onboard/delivery/delist provenance gap을 기록한다.
5. 복구 가능한 기존 백업을 새 canonical root에 복사하고 원본은 수정하지 않는다.

**산출물**

- `physical-coverage.json`
- `repository-coverage.json`
- `strategy-support-inventory.json`과 CSV
- `data-gap-ledger.md`
- source path, Git SHA, 명령, 환경, 파일 inventory를 포함한 run record

**Acceptance**

- 실제 repository loader가 필요한 timeframe을 읽는다.
- synthetic source가 0건이다.
- 원본 root는 read-only이며 run output과 분리된다.
- 각 fold에서 membership과 data availability를 판정할 수 있다.
- D-01A validator가 pre-append에서 interior/prefix gap 0과 tail-only 여부를 JSON으로 판정한다.

**STOP**: source hash drift, symlink/multilink, 부분 복구, 디스크 부족.
**KILL**: 없음. 이 단계는 결손을 기록한다.

### Phase D1 — 결손 기반 최소 백필

순서는 다음으로 고정한다.

1. Router frozen symbol의 공식 1m OHLCV를 original window에 맞춰 복구한다.
2. 실제 funding settlement와 listing interval을 복구한다.
3. 4h/1d는 1m에서 결정적으로 파생하고 직접 서로 다른 원천을 섞지 않는다.
4. bar-level gate를 통과한 finalist만 raw aggTrades/1s와 체결 현실성 데이터를 준비한다.
5. Alpha-Max는 별도의 canonical 1s와 funding source에서 공식 owned interval만 phase root로 materialize한다.

초기 단계에서 OI, L2, liquidation history 또는 110-symbol 전체 1s를 수집하지 않는다. `collect_all_strategy_support_data.py`는 1s layout만 OHLCV 경계로 인식하므로 1m-only root에는 사용하지 않는다. `backfill_funding_fee_features.py`는 기존 funding의 파생 열 도구이지 funding downloader가 아니다.

**Acceptance**

- D-01A post-append receipt에서 required owned interval의 expected-grid gap, duplicate, nonmonotone, nonfinite가 0이다.
- D-01A receipt에서 funding은 실제 settlement timestamp, expected cadence gap 0, source receipt를 가진다.
- append 전후 coverage와 collector report가 보존된다.
- point-in-time membership이 없는 full-universe 실험은 시작하지 않는다.

**STOP**: API rate limit/ban, 중단된 temporary output, checksum mismatch.
**KILL**: 공식 owned-interval 데이터가 존재하지 않거나 lifecycle provenance를 복구할 수 없는 해당 트랙.

### Phase R1 — 고-CAGR router 두 개의 실제 전략 복구

허용 후보는 R1과 R2 정확히 두 개다. `--recompute-from-json`, post-OOS selector augment, 새로운 grid search는 성과 증명에 사용하지 않는다.

각 fold에 다음을 가진 immutable leaf manifest를 만든다.

- strategy class와 params
- symbol tuple과 point-in-time membership receipt
- native timeframe
- weights와 gross exposure
- decision timestamp와 입력 cutoff
- evaluation mode와 source artifact hash

R2는 R1과 동일한 leaf 선택을 가져야 하고 strict fallback의 MDD20/cap2 차이만 허용한다.

**Acceptance**

- candidate 수와 hash가 정확히 2다.
- 모든 leaf의 `evaluation_mode`가 `handler` 또는 `registry_simulator`다.
- `generic_fallback_proxy=0`이다.
- current-fold OOS 입력이 0건이다.
- 원본 label을 유일한 actual-engine manifest로 복원한다.

**KILL**: proxy/inferred semantics가 필요하거나 leaf identity가 비유일적이면 router 전체 종료. 근사 후보로 교체하지 않는다.

### Phase R2 — strict + cost-realistic proof

최종 proof는 cost-realistic profile을 기준으로 strict research gate와 `research.route_unmapped_registered_strategies: true`를 포함하는 **하나의 frozen replacement profile**을 사용한다. profile 두 개를 순서대로 적용하지 않는다.

같은 frozen signal과 position에 비용만 바꾼 10/15/20/30bp 네 셀을 실행한다. 실제 funding, UTC settlement, sqrt impact, protective stop, exposure normalization과 purge/embargo를 적용한다.

**Binding gate**

- DSR `>= 0.90`
- SPA p-value `<= 0.05`
- PBO `<= 0.50`
- 20bp all-in net return `> 0`
- MDD `<= 30%`
- liquidation/ruin `0`
- leave-best-fold-out net return `> 0`
- complete funding와 point-in-time membership
- 한 fold의 이익이 전체 결론을 지배하지 않음

둘 다 통과하면 validation 20bp Calmar가 높은 후보를 선택하고, 동률이면 MDD가 낮은 후보 하나만 forward로 동결한다. 어느 binding gate든 실패하면 해당 후보를 종료한다. 둘 다 실패하면 router 트랙을 끝내며 새 변형을 만들지 않는다.

### Phase A1 — Alpha-Max Revision 5.15

Alpha-Max는 main과 별도의 detached worktree와 별도 데이터 root에서 실행한다.

**고정 입력**

- branch/commit: `feat/alpha-max-20260710` / `629d91e5d4aac26911af65a4a5e15ebdcbded30f`
- config: `configs/research/alpha_max_portfolio_20260711_listing_aware.json`
- contract: `configs/research/alpha_max_contract_manifest_20260711_listing_aware.json`
- provenance: `configs/research/alpha_max_official_availability_evidence_20260711.json`
- phase-root preparer: `scripts/research/prepare_alpha_max_phase_roots.py`

각 SHA-256은 순서대로 `2f267451c4df6b6b7471d972b7756327e41c82522ae2ef4b9198fbf6aa8b5e9c`, `ae272f70f65797b4c8a87c29b7f8e64511617f8e0f2d4bd841b2d1addb7d1220`, `214e5da198307d8d32b30f69fb6b1f09002e0b31888dc476ed16060f79de9719`, `ea26b902bcec4458340e4c345fa648a3db9104e1b337fd42460d9a9461a738ac`다.

**원래 기간 — 변경 금지**

| Phase | Start inclusive | End exclusive |
|---|---:|---:|
| warmup | 2022-12-31 | 2024-01-01 |
| train | 2024-01-01 | 2025-06-01 |
| purge | 2025-06-01 | 2025-06-08 |
| validation | 2025-06-08 | 2025-08-31 |
| embargo | 2025-08-31 | 2025-09-07 |
| exposed historical | 2025-09-07 | 2026-07-01 |

TONUSDT는 공식 raw `[2024-03-01T12:31:10Z, 2026-06-23T09:00:00Z)`, feature `[2024-03-01T16:00:00Z, 2026-06-23T09:00:00Z)`만 소유한다. chronology는 그대로이며 원 warmup/train gate에서 탈락해야 한다. GRAMUSDT 대체, synthetic warmup, 시작일 이동은 금지한다.

현재 branch의 `docs/research_note/alpha_max_data_pc_runbook_20260711.md`는 Rev5.14 지시이므로 **Rev5.15 정합화 commit이 push되기 전에는 phase-root preparation까지만 허용하고 prelock/historical 실행은 BLOCKED**다.

**Structural acceptance**

- prelock 68 actual-engine cells, 816 physical fold runs
- phase별 17 manifests
- historical 680 physical runs
- sealed bundle과 before/after prelock byte identity 통과
- TON availability-aware admission rejection 허용

유효 실행의 `no_demonstrated_alpha` 또는 historical robustness 실패는 Alpha-Max KILL이다. 날짜나 후보를 다시 조정하지 않는다.

### Phase F1 — fresh-forward

대상은 router champion 최대 1개와 Alpha-Max champion 최대 1개다. 서로 독립된 frozen commit/config/manifest/universe/risk/cost로 shadow-only 실행한다.

- 30일 checkpoint, 목표 60일
- 시작점은 final manifest freeze 이후이면서 이미 열어 본 마지막 bar보다 늦은 첫 complete bar다.
- parameter, universe, risk scaling 변경 시 forward clock을 0일부터 다시 시작
- 같은 forward interval을 사용해 새 후보를 추가 선택하지 않음
- actual funding, hypothetical order/fill, spread/slippage/impact와 position reconciliation을 일별 저장

**60일 acceptance**

- after-cost net return `> 0`
- coverage/reconciliation error `0`
- liquidation/kill-switch event `0`
- frozen MDD ceiling 준수
- implementation/config/data hash drift `0`

통과는 별도 canary risk review 자격만 뜻한다. 자동 자본 배분은 없다.

## 6. 복구 트랙 종료 후의 후속 알파 후보 프로그램

R1/R2와 Alpha-Max가 과학적 판정을 받기 전에는 아래 프로그램의 전략 성과 실행을 시작하지 않는다. 데이터 계약과 preregistration 초안은 병렬로 준비할 수 있지만, 기존 OOS를 보고 candidate·parameter·universe·overlay를 바꾸지 않는다. 활성화 시 별도 run ID, clean worktree, trial ledger와 untouched lockbox를 사용한다.

### 6.1 증거 상태와 역할 구분

| 영역 | 현재 증거 | 허용 역할 |
|---|---|---|
| Local auxiliary `precious_metal`의 `data.ipynb`, `anomaly_detection.ipynb` | 2026-07-13 snapshot에는 `dat_ave(..., 'WS')`, 가격수준 rolling std, 단변량 GARCH와 가격 MAPE가 있으나 causal cross-volatility 예측 또는 executable PnL 검증은 없음 | 가설 출처만; 성과 근거 금지 |
| Quants-agent 금속 pair | tracked [4금속 split report](../../var/reports/profit_moonshot_20260501/current_tail_20260505/precious_metal_pair_aggressive/precious_pair_available_window_split_backtest.md)의 OOS `-0.0478%`, Sharpe `-0.1641` | expected-null 재검증만; 같은 z-score parameter clone 금지 |
| Volatility-regime residual basket | 두 후보 모두 train/validation/OOS 음수, hard reject (`docs/session_handoff_20260412_broader_redesign_and_ralph_audit.md:36-49`) | 재실행 금지; 새 후보의 prior-of-death |
| TradFi discovery | symbol discovery와 일부 diagnostic은 있으나 point-in-time lifecycle·full refresh·cost-realistic proof가 없음 | 별도 data contract가 생길 때만 research-only |
| `VolManagedRiskOverlayStrategy`, `AvgCorrelationCrashGuardOverlayStrategy` | 코드와 회귀 테스트는 있으나 실제 금속/TradFi market replay 증거 없음 | alpha가 아닌 risk A/B |
| Cross-metal volatility spillover | 현재 tracked artifact에는 own-vol baseline 대 causal forecast 비교와 raw tradable return provenance가 없음 | prediction diagnostic부터; 곧바로 전략 구현 금지 |

외부 문헌은 volatility spillover가 존재할 수 있고 volatility-managed exposure가 위험조정 성과를 개선할 수 있다는 prior만 제공한다. 이 저장소의 수익 증거로 전용하지 않는다. [Diebold–Yilmaz directional spillover](https://www.econstor.eu/bitstream/10419/45422/1/638343968.pdf), [Moreira–Muir volatility management](https://www.nber.org/papers/w22208), [precious-metal asymmetric spillover](https://doi.org/10.1016/j.resourpol.2019.101509).

### 6.2 동결 후보 레지스트리

각 ID는 기본 parameter 한 세트만 manifest에 동결한다. 서로 다른 timeframe, estimator, threshold, universe 또는 allocator는 각각 별도 trial로 센다. 같은 ID 내부 grid search는 금지한다.

| ID | 경제 가설 | 기존 재사용 경로 | 최초 지위와 반증 조건 |
|---|---|---|---|
| C-TSMOM | 저회전 1d/4h time-series trend가 비용 후 지속된다 | `LowTurnoverTrendPersistenceStrategy`, TradFi/상품은 `CrossAssetDiversifiedTrendStrategy` 또는 `AdaptiveTrendRiderStrategy` 중 manifest에서 하나만 선택 | 1순위 alpha. RPT `<10bp`, 20bp net `<=0`, best fold 제거 후 net `<=0`이면 KILL |
| C-ANCHOR | 52주 고점 근접도가 단순 momentum과 다른 횡단면 under-reaction을 포착한다 | `CrossSectionalNearHighAnchoringStrategy` | point-in-time universe와 최소 52주 history가 없으면 STOP; incremental IC와 net long-short가 없으면 KILL |
| C-RESMOM | BTC beta를 제거한 residual momentum이 공통시장 노출과 다른 횡단면 수익을 가진다 | `TrendGatedResidualMomentumStrategy`와 기존 rolling-beta utility | lifecycle·delisting 포함 universe가 없으면 STOP; beta-neutral net alpha가 없으면 KILL |
| C-REBAL | 느린 diversity/equal-weight rebalancing premium이 고상관 시장에서도 비용을 넘는다 | `RebalancingPremiumHarvestStrategy` | buy-and-hold/inverse-vol basket 대비 net excess growth와 Calmar가 개선되지 않으면 KILL |
| C-METAL-RV | 실제 거래 가능한 금속 pair의 hedge-adjusted relative-value dislocation이 episode 단위로 회귀한다 | `PairSpreadZScoreStrategy`의 기존 `state_volconv` row와 아래 P0/P1 | XAU/XAG/XPT/XPD를 venue·contract별로 식별하고 fold-local trailing residual stationarity admission을 통과해야 한다. 기존 negative evidence를 이기지 못하면 KILL |
| C-VOLSQ | volatility compression 뒤 가격 breakout 방향의 expansion이 지속된다 | `VolatilitySqueezeBreakoutRiderStrategy` | volatility만으로 방향을 정하지 않고 completed-bar breakout 확인 필수. 20bp와 fold 안정성 실패 시 KILL |
| C-IDIOVOL | BTC beta 제거 후 low idiosyncratic-vol long/high-vol short premium이 존재한다 | `IdiosyncraticVolatilityStrategy` 기본값 한 세트 | 기존 소표본 무거래가 prior-of-death. eligible symbol/fold 폭 또는 IC가 부족하면 KILL |
| C-SLOW-LL | 넓은 횡단면의 느린 lead-lag loading이 단일 pair lead-lag보다 비용을 견딘다 | `SlowCrossSectionalLeadLagStrategy` | 가장 먼저 자를 conditional 후보. 20bp 또는 RPT gate 실패 시 즉시 KILL |
| C-TRUE-CARRY | 실제 spot long + perpetual/futures short의 funding/basis가 양 leg 비용 후 남는다 | 기존 funding scorer는 비교 feature만 사용; dual-leg execution은 별도 prerequisite | spot/perp 양 leg, basis, funding settlement, borrow, margin이 없으면 BLOCKED. futures-only funding slope를 carry 성과로 인정하지 않음 |

다음은 이 cycle에서 실행하지 않는다.

- GARCH lower-tail 가격 경로, 가격수준 rolling-std z-score와 full-sample anomaly를 방향 alpha로 사용하는 것
- options implied-vs-realized volatility: option chain, surface, executable quote가 없으므로 BLOCKED
- commodity curve carry/roll yield: 연속 front price가 아니라 개별 expiry chain과 roll 체결 데이터가 생기기 전까지 BLOCKED
- 새 DCC/BEKK/TVP-VAR 전략: 아래 volatility diagnostic이 먼저 통과하지 않으면 YAGNI
- 기존에 실패한 volatility-regime residual basket 또는 pair-spread parameter clone 부활
- `StationarityGatedResidualReversionStrategy`, `TimeframePairZScoreReversionStrategy`, `MetalsRelativeValueBasketStrategy`를 C-METAL-RV와 동등한 구현으로 바꿔 끼우는 것; 각각 다른 residual/ratio 가설이므로 재개 시 별도 candidate ID와 trial이 필요

### 6.3 Volatility connection 프로그램

Volatility는 `log return` 또는 intraday return 제곱합으로만 정의한다. 가격 수준의 rolling 표준편차는 volatility feature로 금지한다.

#### V-DIAG — cross-volatility prediction admission

거래하지 않는 진단이다.

- target: `RV(t+1)` 한 horizon만 사용한다.
- baseline: target 자산 자신의 lagged log-RV와 1/5/22일 trailing block을 사용한 고정 HAR/EWMA 또는 기존 univariate GARCH 중 manifest에서 하나만 선택한다.
- candidate: baseline에 사전등록한 leader의 동일한 lagged RV block만 추가한다.
- crypto 방향은 `BTC -> eligible asset`, metals는 실제 raw provenance가 있을 때만 `Au -> Ag/Pt/Pd`를 고정한다.
- untouched lockbox를 제외한 validation-forward folds의 QLIKE/MSE, coefficient sign stability와 paired block bootstrap을 기록한다.

**Admission**: median validation-forward QLIKE 개선 `>=5%`, non-overlapping fold의 `>=60%`에서 개선, whole-search/FDR 조정 `p<=0.05`를 모두 만족해야 한다. 실패하면 direct volatility-spillover 전략과 새 multivariate-vol 코드 전체를 KILL한다. 통과해도 이는 변동 크기 예측이며 spot/perp의 가격 방향 alpha가 아니다.

#### V-PAIR — pair volatility-convergence gate

`PairSpreadZScoreStrategy`가 상속하는 `PairTradingZScoreStrategy`의 `_vol_spread_zscore`, `vol_lag_bars`, `min_vol_convergence`를 재사용한다. 신규 class는 만들지 않는다. 두 arm 모두 기존 `state_volconv` row의 나머지 parameter를 같게 동결하고 `lookback_window=120`, `hedge_window=240`, `vol_lag_bars=2`를 공유한다.

- Arm P0: `min_vol_convergence=0.0`, volatility gate OFF
- Arm P1: `min_vol_convergence=0.60`, volatility gate ON
- signal, execution, cost, funding과 episode 정의는 동일하게 고정한다.

P1은 P0보다 strict alpha binding gate를 모두 통과하고 validation Calmar가 개선되어야 한다. 단순 trade 수 감소나 한 episode 반복표본으로 생긴 승률은 uplift로 인정하지 않는다.

#### V-OVERLAY — risk-only A/B

standalone binding gate를 통과한 child 최대 2개에만 다음 arm을 적용한다. 첫 cycle에는 overlay를 서로 stack하지 않는다.

| Arm | 구성 |
|---|---|
| O0 | child only |
| O1 | 동일 child + `VolManagedRiskOverlayStrategy`, `close_to_close`, class default 한 세트 |
| O2 | TradFi OHLC가 완전할 때만 동일 child + `yang_zhang`; crypto에는 생성하지 않음 |
| O3 | 동일 child + `AvgCorrelationCrashGuardOverlayStrategy`, class default 한 세트 |
| O4 | O1/O2/O3 각각에 대해 그 arm의 평균 gross scale과 같은 static-scale child control; 각 control은 별도 trial |

**Overlay promotion**은 alpha promotion과 분리한다. O1/O2/O3가 각각 대응하는 O4보다 Calmar가 높고, MDD와 ES95를 각각 `>=10%` 줄이고, O0의 20bp net return을 `>=90%` 보존하며, non-overlapping fold의 `>=60%`에서 risk-adjusted uplift가 있어야 한다. cost/funding coverage 누락 또는 liquidation이 하나라도 생기면 KILL하며 turnover와 비용 증가는 net 보존 gate에 포함해 보고한다. 두 overlay를 함께 쓰려면 별도 preregistration과 새 trial count가 필요하다.

#### V-COV — covariance allocation

동일 alpha signals에 equal-notional, inverse-vol, 기존 Ledoit-Wolf/shrunk-covariance allocator 세 개만 비교한다. 이는 return alpha가 아니라 allocation test다.

- trailing 데이터만 사용하고 다음 bar부터 weight를 적용한다.
- realized-vol target error, MDD, ES95, concentration, turnover와 비용을 보고한다.
- target-vol absolute error가 `>=20%` 줄고 Calmar가 개선되며 20bp net이 양수일 때만 allocator로 채택한다.

### 6.4 자산별 데이터 계약

#### Crypto

- point-in-time listing/delist membership, actual 1m OHLCV, mark/index, 실제 funding settlement, spot/perp mapping
- 4h/1d는 validated 1m에서 결정적으로 파생
- cross-sectional 후보는 상장 전 history를 만들거나 현재 universe를 과거에 역적용하지 않음

#### TradFi와 precious metals

- `canonical_symbol`, venue symbol, instrument type, contract multiplier, quote currency, session/calendar, timezone, first/last tradable timestamp를 frozen manifest에 기록
- candidate별 owned interval은 `첫 평가 신호 - 동결 warmup`부터 계산한다. Router를 위해 2024년을 채우지는 않지만, C-ANCHOR가 2025-01-01에 첫 신호를 내려면 그 전 실제 52-week-equivalent bars가 필요하다. bar 수는 manifest의 session calendar로 고정하고, 없으면 합성하지 않고 첫 eligible signal을 늦춘다.
- futures면 expiry·roll rule·front/next mapping, perpetual이면 mark/index/funding, spot/CFD면 financing과 executable venue를 분리
- XAU/XAG/XPT/XPD라는 이름만 같다고 CME futures, spot benchmark, CFD와 Binance TradFi perpetual을 같은 시계열로 대체하지 않음
- closed bar cutoff 이후 다음 bar에서만 실행하고, 월요일 label에 그 주 미래 평균을 넣는 `dat_ave(..., 'WS')` 자료는 성과 입력으로 사용하지 않음
- auxiliary `precious_metal` 저장소의 DB credential과 내부 host를 복사하거나 Git에 추가하지 않음

#### Blocked data families

- option IV/RV는 executable option chain과 surface snapshot이 생길 때까지 BLOCKED
- commodity curve carry는 expiry별 settle/BBO, roll calendar와 실제 roll cost가 생길 때까지 BLOCKED
- fundamentals/earnings 전략은 release timestamp를 가진 point-in-time vintage가 생길 때까지 BLOCKED

### 6.5 단계별 검증 순서

1. **C-00 — preregistration**
   - `candidate-manifest.json`, `data-contract.json`, `trial-ledger.json`과 SHA-256을 만든다.
   - trial ledger는 candidate, universe, timeframe, horizon, estimator, threshold, allocator, child-overlay arm과 cost cell 전부를 센다.
   - 모든 candidate의 prior-of-death, single falsifying measurement와 expected null을 기록한다.
2. **C-01 — data admission**
   - D-01A validator와 lifecycle/funding gate를 통과한다.
   - factor IC는 causal non-overlapping label, HAC/block-bootstrap, whole-search/FDR 조정 `p<=0.05`, 가설 방향 median rank-IC 절댓값 `>=0.02`, 올바른 부호 fold 비율 `>=0.60`을 모두 사용한다. `factor_ic>0` 하나만으로 admit하지 않는다.
3. **C-02 — volatility feature admission**
   - V-DIAG를 이 단계에서 종료한다.
4. **C-03 — standalone alpha walk-forward**
   - leaf별 default 한 세트, validation-only ranking, locked OOS report-only다.
   - actual registry/handler 실행만 허용하고 `generic_fallback_proxy=0`을 강제한다.
5. **C-04 — overlay/covariance ablation**
   - C-03 standalone gate를 통과한 child에만 V-PAIR/V-OVERLAY/V-COV를 적용한다.
   - OFF/ON/static-scale control은 같은 pre-overlay child signal/target, market data와 cost/funding model을 사용한다. realized position은 동결한 arm scaling 때문에만 달라질 수 있다.
6. **C-05 — strict proof**
   - 같은 signal에 actual spread/fees/sqrt-impact/funding과 10/15/20/30bp를 적용한다.
   - per-fold admit와 final merge 모두 selection/dedup gate를 통과해야 한다.
7. **C-06 — lockbox와 fresh-forward**
   - locked OOS는 마지막 1회 report-only다.
   - alpha leaf 최대 1개와 overlay 최대 1개만 60일 frozen shadow로 보낸다.
   - parameter, universe, estimator, allocator 또는 risk scale 변경 시 clock을 0일부터 다시 시작한다.

### 6.6 공통 binding gate와 보고 계약

Standalone alpha는 기존 R2 기준을 그대로 사용한다.

- DSR `>=0.90`, SPA p-value `<=0.05`, PBO `<=0.50`; PBO missing은 fail-close
- 20bp all-in net return `>0`, MDD `<=30%`, liquidation/ruin `0`
- leave-best-fold-out net return `>0`, active fold ratio `>=0.60`
- low-turnover/lead-lag 후보는 RPT `>=10bp`
- 한 fold 또는 한 symbol이 전체 결론을 지배하지 않음
- locked OOS 내부 quantile이나 current-fold OOS로 threshold·candidate·weight를 선택하지 않음

모든 실행은 reject를 포함한 전 후보를 보고한다. 결과표에 Git SHA, resolved config/data/candidate manifest hash, evaluation mode, trial count, cost/funding coverage, STOP/KILL 사유가 없으면 무효다. 다음 표현은 금지한다.

- volatility, GARCH, covariance 또는 correlation이 그 자체로 방향 alpha를 만든다
- 동시상관이 lead-lag 예측력이다
- TradFi full refresh 또는 metals data가 완전하다
- 기존 고-CAGR headline이 재현·검증됐다
- linear cost proxy가 cost-realistic proof다
- DSR/SPA/PBO가 whole-search trial count 또는 PBO 없이 통과했다

## 7. 작업 목록과 의존성

| Task | 작업 | 선행 | 완료 증거 |
|---|---|---|---|
| D-01 | data PC root 발견, read-only inventory | 없음 | coverage와 gap ledger |
| D-01A | fail-closed research data contract validator 구현 | 없음 | `validate_ohlcv_frame` 재사용, expected-grid/actual-funding-cadence tests, JSON receipt |
| D-02 | core/router 1m tail 복구; interior gap은 별도 bounded repair | D-01,D-01A | pre/post validation receipt, collector receipt, coverage diff |
| D-03 | funding와 support inventory 복구 | D-01,D-01A | settlement validation receipt |
| D-04 | symbol lifecycle registry 구현 | D-01 | fold membership manifests와 tests |
| D-05 | main materializer CLI 회귀 수정 | 없음 | `--help`와 targeted tests 통과 |
| R-01 | exact R1/R2 frozen manifest replay seam | D-04 | exactly-two replay test |
| R-02 | actual-routing/evaluation-mode fail-close | R-01 | fallback 0 receipt |
| R-03 | single combined strict/cost profile | R-02 | resolved config와 SHA |
| R-04 | 10/15/20/30bp proof와 gate report | D-02,D-03,R-03 | immutable decision bundle |
| A-01 | Rev5.15 Alpha-Max runbook 정합화와 push | 없음 | exact hashes/commands review |
| A-02 | canonical source에서 phase roots 생성 | D-01 | preparation manifest |
| A-03 | prelock/historical one-touch 실행 | A-01,A-02 | sealed bundles |
| F-01 | 최대 두 champion fresh-forward | R-04,A-03 | 60-day final report |
| C-00 | 후속 candidate/data/trial manifest preregistration | D-01; 성과 실행 gate는 R-04,A-03 | frozen JSON 3종과 SHA-256 |
| C-01 | Crypto/TradFi/metals 데이터 계약과 feature admission 입력 검증 | D-01A,D-04,C-00,R-04,A-03 | asset-contract receipt와 frozen subset hash |
| C-02 | V-DIAG cross-volatility prediction admission | C-01 | own-vol baseline 비교, FDR/QLIKE report |
| C-03 | standalone alpha walk-forward | C-01; V-DIAG feature를 직접 소비하는 미래 후보만 C-02 | 전 후보·전 reject와 locked-OOS report |
| C-04 | V-PAIR/V-OVERLAY/V-COV ablation | C-03 | OFF/ON/static-scale 동일-signal report |
| C-05 | 후속 후보 strict/cost proof와 최종 판정 | C-03,C-04 | selection gate와 immutable decision bundle |
| C-06 | alpha leaf 최대 1개 + overlay 최대 1개 fresh-forward | C-05 | 60-day frozen shadow report |

Critical path는 `D-01A -> D-02/D-03 -> R-04 -> F-01`, `D-01 -> D-04 -> R-01 -> R-02 -> R-03 -> R-04 -> F-01`, `D-01 -> A-02`, `A-01 -> A-03 -> F-01`이다. 후속 후보 경로는 `C-00 -> C-01 -> C-02/C-03 -> C-04 -> C-05 -> C-06`이며, `C-01` 이후의 성과·feature admission 실행은 `R-04`와 `A-03` 판정 뒤에만 열린다. 데이터 inventory와 validator가 준비되기 전에 전략 코드를 늘리지 않는다.

## 8. 최종 체크리스트

- [ ] 정확한 Git commit과 clean worktree 기록
- [ ] synthetic source 0건
- [ ] source provenance와 immutable inventory
- [ ] 연속 tail만 일반 collector로 복구; interior gap은 bounded repair
- [ ] point-in-time lifecycle와 actual funding 완전
- [ ] router 후보 정확히 2개
- [ ] faithful actual-engine leaf manifest
- [ ] `generic_fallback_proxy=0`
- [ ] combined strict/cost profile 하나와 hash
- [ ] 같은 signal의 10/15/20/30bp 결과
- [ ] DSR/SPA/PBO, MDD, leave-best-fold-out gate
- [ ] PBO missing은 fail-close
- [ ] Alpha-Max Rev5.15 runbook 갱신 후 실행
- [ ] Alpha-Max 원래 날짜 유지
- [ ] seals, readback, before/after identity
- [ ] 60일 forward와 변경 시 clock reset
- [ ] 후속 candidate/data/trial manifest와 whole-search trial count 동결
- [ ] Crypto/TradFi/metals별 venue·instrument·lifecycle·cost 계약 검증
- [ ] V-DIAG가 own-vol baseline을 이기기 전 cross-volatility 전략 구현 0건
- [ ] standalone alpha 통과 전 overlay/allocator 승격 0건
- [ ] O0/O1/O2/O3/O4가 동일 pre-overlay child signal/target·market data·funding·cost model 사용
- [ ] survivor만이 아니라 전 candidate와 reject 사유 보고
- [ ] 모든 단계 실자본 0%
