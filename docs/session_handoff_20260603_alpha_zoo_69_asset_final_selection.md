# Session Handoff — 69-Asset Alpha Zoo Monthly WF Final Selection (2026-06-03 KST)

## Scope

Finalized the 69-asset monthly clean-OOS walk-forward research using 30m through 1D timeframes, 10bps slippage, monthly day-1 refit, expanding train from 2025-01-01, previous 2 calendar months as validation, and next month as locked OOS. Latest available data ends at 2026-06-01T06:30:00Z, so June 2026 is a partial fold.

## Final performance snapshot

| Candidate | Clean | OOS Comp | Max OOS MDD | Monthly Eq MDD | Sharpe | Sortino | PF | Hit | Status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| `relaxed_efficiency:hybrid_v3_5` | Y | 156.03% | 19.75% | 15.66% | 1.69 | 10.48 | 7.04 | 5/10 | clean-only top, paper/testnet challenger |
| `fixed_relaxed_dynamic_blend:relaxed70_dynamic30` | N | 122.36% | 16.66% | 14.66% | 1.74 | 8.97 | 6.33 | 6/10 | high-return forward-shadow |
| `fixed_relaxed_dynamic_blend:relaxed60_dynamic40` | N | 111.75% | 16.19% | 14.35% | 1.75 | 8.43 | 6.00 | 6/10 | selected balanced forward-shadow |
| `dynamic_aware_hybrid:hybrid_v3_5_train_validation_fit` | Y | 54.76% | 16.75% | 12.76% | 1.53 | 4.43 | 3.87 | 5/10 | clean defensive sleeve |
| `asset_timeframe_leverage:hybrid_v3_6` | Y | 21.78% | 21.97% | 21.19% | 0.72 | 1.67 | 1.79 | 4/10 | monitor/diagnostic only |

## Decision

- Use `fixed_relaxed_dynamic_blend:relaxed60_dynamic40` as the best risk/return forward-shadow or paper candidate, not as same-window clean promotion.
- Use `fixed_relaxed_dynamic_blend:relaxed70_dynamic30` only when prioritizing compounded return over the marginally lower MDD of 60/40.
- Treat `relaxed_efficiency:hybrid_v3_5` as the clean-window top score, but still paper/testnet challenger because it is relaxed-repair dependent and has hit 5/10.
- Keep asset/timeframe leverage scaling as a monitored diagnostic axis; it did not improve the final core.
- Real-money remains disabled until fresh-forward shadow plus paper/testnet fill, BBO, slippage, and reconciliation telemetry pass.

## Artifacts

- Exact blend reports: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_69_asset_exact_blend_full_tuning_20260603/`
- Asset/timeframe leverage reports: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/alpha_zoo_69_asset_asset_tf_leverage_20260603/`
- Research note updated: `docs/research_note/research_note.md`
- Notepad updated: `.omx/notepad.md`

## Verification

Final targeted verification before note/git handoff included:

```
uv run python -m py_compile <changed 69-asset research scripts/tests>
uv run python -m ruff check <changed 69-asset research scripts/tests>
uv run python -m pytest -q tests/test_alpha_zoo_30m_plus_alpha_feedback_discovery.py tests/test_alpha_zoo_69_asset_*.py
uv run python scripts/verify_docs.py docs/research_note/research_note.md docs/session_handoff_20260603_alpha_zoo_69_asset_final_selection.md
git diff --check
```

Result: targeted pytest `66 passed in 1.31s`; lint/compile/docs/diff-check passed, including docs verification `118 markdown files checked`.
