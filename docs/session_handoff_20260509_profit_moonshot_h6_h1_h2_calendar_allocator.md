# Profit moonshot H6/H1/H2 handoff — 2026-05-09

## Status
- Baseline preserved from `private/main` commit `87d66ed5c0e76a0cc671bf28fda175b8deb1ba1d` before edits.
- H6 quarantine is locked: high-return locked-OOS diagnostics that fail gates are labeled `diagnostic_not_promoted` and cannot be `success_candidate` / `improved_candidate`.
- H1 allocator added: `cluster_capped_validation_weight` uses train/validation curves and caps correlated calendar sleeve clusters; locked-OOS is not used for selection.
- H2 objective added: calendar parameter-neighborhood representatives are selected by train/validation stability before broad top-N fill.
- No standalone non-calendar family was forced/promoted.

## Bounded H1/H2 run
Artifacts: `var/reports/profit_moonshot_20260501/current_tail_20260508/h1_h2_calendar_allocator/`

Command:
```bash
uv run --extra dev python scripts/research/tune_profit_moonshot_fresh_portfolio.py \
  --candidate-csv var/reports/profit_moonshot_20260501/current_tail_20260508/all_family_expansion/fresh_start_overhaul_replay_candidates.csv \
  --output-dir var/reports/profit_moonshot_20260501/current_tail_20260508/h1_h2_calendar_allocator \
  --top-n 18 \
  --calendar-neighborhood-reps 10 \
  --max-sleeves 4 \
  --max-combos-per-size 6000
```

Results:
- Candidate sleeves: `18`; portfolio specs evaluated: `20,145`.
- Improved/promoted candidates: `0`.
- Peak RSS: `414.578 MiB` (`/usr/bin/time` max RSS `424,528 KiB`), below the <8GiB guard.
- Output JSON: `fresh_portfolio_tuning_latest.json`; markdown: `fresh_portfolio_tuning_latest.md`; CSV: `fresh_portfolio_tuning_candidates.csv`.

## Selection / quarantine evidence
Selected by validation:
- mode: `additive_sleeves`; promotion: `diagnostic_not_promoted`.
- train return `11.7292%`; validation return `13.5530%`.
- locked-OOS return `2.1374%`, MDD `0.6571%`, Sharpe `3.0229`.
- failed gates: `oos_return_risk_beats_current_champion`, `oos_mdd_beats_shadow`.

Diagnostic best OOS:
- locked-OOS return `3.9798%`, MDD `0.5173%`, Sharpe `5.7518`.
- promotion: `diagnostic_not_promoted`; failed gate: `oos_mdd_beats_shadow`.

Conclusion: although some candidates beat the locked-OOS return threshold (`>1.2181%`), none improved return/risk while passing the shadow-MDD gate, so no candidate is reported as improved.

## Policy locks
- `selection_basis`: train/validation only.
- locked-OOS: report-only / gate-only.
- current champion threshold: OOS return must exceed `1.2181%` and return/risk must beat champion.
- high-return OOS-ranked/MDD-failed rows remain diagnostics only.

## Verification
- Pre-edit behavior lock: `uv run --extra dev pytest -q tests/test_profit_moonshot_fresh_portfolio_tuning.py` -> `3 passed`.
- Post-edit targeted: `uv run --extra dev pytest -q tests/test_profit_moonshot_fresh_portfolio_tuning.py` -> `8 passed`.
- Targeted portfolio/memory subset: `36 passed`.
- Focused ruff on touched files: passed.
- Full suite: `uv run --extra dev pytest -q` -> `1198 passed in 388.85s` (`/usr/bin/time` wall `6:01.74`).
- Repo lint: `uv run --extra dev ruff check .` -> passed.
- Compileall: `python3 -m compileall -q src scripts tests` -> passed.
- Whitespace check: `git diff --check` -> passed.

## Remaining risk
- H1/H2 did not find an improved deployable candidate in the bounded run; it intentionally preserves the existing champion rather than promoting an MDD-failed diagnostic.
- Wider searches can be run later, but promotion criteria should remain unchanged.
