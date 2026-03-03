# Scoring Config Guide

`configs/score_config.example.json` is the shared template for strategy research/selection/optimization scoring.

## Section mapping

- `candidate_research`
  - script: `scripts/run_research_candidates.py --score-config ...`
- `portfolio_optimization`
  - script: `scripts/run_portfolio_optimization.py --score-config ...`
- `strategy_shortlist`
  - script: `scripts/select_research_shortlist.py --score-config ...`
- `research_hurdle`
  - script: `scripts/run_research_hurdle.py --score-config ...`

## Quick examples

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

## Notes

- Each script only reads its own section.
- Unknown keys are ignored.
- For portability, keep all scoring sections in one file and version it with your experiment reports.
