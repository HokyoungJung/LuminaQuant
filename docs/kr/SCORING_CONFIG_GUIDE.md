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
