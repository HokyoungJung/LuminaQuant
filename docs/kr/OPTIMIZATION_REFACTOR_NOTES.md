# 최적화 리팩터링 노트

이 문서는 기존 동작을 보존하면서 최적화/리서치 코드의 중복 search-loop를 줄이는
최신 구조를 설명합니다.

## 범위

- 이벤트 흐름, 전략 의미, live/paper/testnet 안전 정책은 바꾸지 않습니다.
- 전략, 포트폴리오, 지표, artifact schema는 regression test로 보호한 뒤에만 수정합니다.
- 튜닝 가능한 최적화는 Optuna-first로 유지하고, grid는 작은 deterministic enumeration에만 허용합니다.

## 모듈 경계

### 공용 search policy 모듈

- 추가 모듈: `lumina_quant/optimization/search_policy.py`
- 목적:
  - Optuna study 생성/실행을 `run_optuna_study(...)`로 통일
  - canonical Optuna config schema에서 trial parameter를 만드는
    `suggest_params_from_optuna_config(...)` 제공
  - 작은 grid enumeration은 `build_bounded_grid_combinations(...)`로만 수행하고
    justification/cap metadata를 남김
  - artifact/log에 들어갈 selection/objective policy와 locked-OOS non-use flag를
    `optimization_search_policy_payload(...)`로 통일

### 사용 지점

- `src/lumina_quant/cli/optimize.py`
  - grid 조합 생성과 Optuna study 실행을 공용 search policy로 위임합니다.
- `scripts/research/optuna_tune_hybrid_online_portfolio.py`
  - script-local `optuna.create_study(...)` 루프 대신 `run_optuna_study(...)`를 사용합니다.

## 정책

- 새 리서치/튜닝 runner는 가능한 한 `run_optuna_study(...)`를 사용합니다.
- 직접 `itertools.product` 기반 grid를 추가하지 않습니다. 꼭 필요한 경우
  `build_bounded_grid_combinations(...)`를 통해 작은 deterministic enumeration임을
  명시하고 cap/metadata를 남깁니다.
- locked-OOS는 selection/objective/pruning/parameter fitting 입력으로 쓰지 않습니다.
  진단 목적 예외는 artifact에 diagnostic이라고 명시해야 합니다.
- search space는 `lumina_quant.tuning.param_registry` 또는 strategy registry 기본값을
  원천으로 삼고, script마다 더 나쁜 중복 domain을 만들지 않습니다.

## 검증

- `uv run pytest -q tests/test_optimization_search_policy.py tests/test_param_registry.py tests/test_portfolio_optimizer_core.py tests/test_strategy_alias_compat.py`
  → `19 passed`
- `uv run ruff check src/lumina_quant/optimization/search_policy.py src/lumina_quant/optimization/__init__.py src/lumina_quant/cli/optimize.py scripts/research/optuna_tune_hybrid_online_portfolio.py tests/test_optimization_search_policy.py`
  → 통과
