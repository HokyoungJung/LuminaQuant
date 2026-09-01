# 스코어 설정 가이드

`configs/score_config.example.json`은 전략 리서치/선정/최적화 스코어링에 공통으로 쓰는 템플릿입니다.

## 섹션 매핑

- `candidate_research`
  - 스크립트: `scripts/run_research_candidates.py --score-config ...`
- `portfolio_optimization`
  - 스크립트: `scripts/run_portfolio_optimization.py --score-config ...`
- `strategy_shortlist`
  - 스크립트: `scripts/select_research_shortlist.py --score-config ...`
- `research_hurdle`
  - 스크립트: `scripts/run_research_hurdle.py --score-config ...`

## 빠른 실행 예시

```bash
uv run python scripts/run_research_candidates.py \
  --score-config configs/score_config.example.json

uv run python scripts/run_portfolio_optimization.py \
  --score-config configs/score_config.example.json

uv run python scripts/select_research_shortlist.py \
  --score-config configs/score_config.example.json

uv run python scripts/run_research_hurdle.py \
  --score-config configs/score_config.example.json
```

## 참고

- 각 스크립트는 자기 섹션만 읽습니다.
- 모르는 키는 무시됩니다.
- 실험 재현성을 위해 하나의 파일에 섹션을 모아 두고 결과 리포트와 함께 버전 관리하는 것을 권장합니다.

## 최적화 검색 정책 가드레일

- 고차원/튜닝형 최적화는 `lumina_quant.optimization.search_policy`의 공유 Optuna runner를
  사용하거나 테스트된 예외를 명시해야 합니다.
- Bounded grid는 작은 deterministic policy/profile sweep에만 쓰며, artifact에 cap,
  justification, search-space provenance, skipped/truncated count를 남깁니다.
- artifact가 명시적인 diagnostic 용도가 아닌 한 locked-OOS는
  selection/objective/pruning/parameter-fitting metadata에서 제외합니다.
