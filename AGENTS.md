# LuminaQuant Repository Layout (2026-03-03)

Preferred structure (high level):

```text
LuminaQuant/
├─ README.md
├─ README_KR.md
├─ pyproject.toml
├─ config.yaml
├─ configs/
│  ├─ score_config.example.json
│  ├─ config.example.yaml
│  └─ profiles/
├─ src/
│  └─ lumina_quant/
│     ├─ cli/
│     ├─ workflows/
│     ├─ backtesting/
│     ├─ live/
│     ├─ optimization/
│     ├─ strategies/
│     ├─ indicators/
│     ├─ data/
│     └─ storage/
│        ├─ wal/
│        ├─ parquet/
│        └─ postgres/
├─ apps/
│  └─ dashboard/
│     ├─ app.py
│     ├─ components/
│     └─ services/
├─ scripts/
│  ├─ ci/
│  ├─ ops/
│  ├─ dev/
│  └─ research/
├─ tests/
│  ├─ unit/
│  ├─ integration/
│  └─ fixtures/
└─ var/
   ├─ data/
   ├─ logs/
   ├─ reports/
   ├─ optimized_params/
   └─ cache/
```

Notes:
- Use `uv run lq ...` as the primary CLI.
- Root entrypoints are compatibility shims (`run_backtest.py`, `optimize.py`, `run_live.py`, `run_live_ws.py`, `dashboard.py`).
- Runtime artifacts should prefer `var/`.
