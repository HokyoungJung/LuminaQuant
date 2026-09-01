# Next Session Plan — Profit Moonshot Continuation

Generated: `2026-05-01T13:23:00Z`
Repo: `/home/hoky/Quants-agent/LuminaQuant`
Branch: `private-main`
Remote target: `private main`
Code/evidence commit before this plan: `ba084d724de93401739e16680dc8dbf2f2e1d9b6`
Plan storage: this file is committed after the code/evidence commit; in a new session treat `private/main` HEAD as authoritative.

## Current conclusion

The current best **profit-return** candidate is:

- Mode: `profit_moonshot_adaptive_momentum_boost_mode`
- Source artifact: `var/reports/profit_moonshot_20260501/continuation/adaptive_boost/live_equivalent_revalidation_latest.json`
- Continuation summary: `var/reports/profit_moonshot_20260501/continuation/latest.md`
- Autoresearch gate artifact: `.omx/specs/autoresearch-profit-moonshot-continuation/result.json`

Live-equivalent train/val evidence:

| metric | value |
|---|---:|
| train return | `-2.994796%` |
| train MDD | `18.021085%` |
| train trades | `361` |
| val return | `0.509082%` |
| val MDD | `1.358270%` |
| val Sharpe | `0.014751` |
| val Sortino | `0.014527` |
| val trades | `56` |
| train/val liquidations | `0/0` |
| max RSS | `5,821,420 KB` |

Baseline comparison:

| mode | val return | val MDD | val Sharpe | val trades | liq |
|---|---:|---:|---:|---:|---:|
| `profit_moonshot_adaptive_momentum_mode` | `0.264933%` | `0.754391%` | `0.012417` | `52` | `0` |
| `profit_moonshot_adaptive_momentum_boost_mode` | `0.509082%` | `1.358270%` | `0.014751` | `56` | `0` |

Important caveat: `profit_moonshot_adaptive_momentum_boost_mode` is **not deployment-ready**. It is the best profit-return research candidate so far, but train return is close to the gate floor (`-3%`) and train MDD is high (`18.02%` vs `20%` cap). Treat it as a robustness target, not a live promotion.

## Start-of-session checklist

1. Sync the pushed state:

   ```bash
   cd /home/hoky/Quants-agent/LuminaQuant
   git fetch private
   git checkout private-main
   git reset --hard private/main
   git status --short
   ```

2. Confirm no duplicate backtest is running:

   ```bash
   ps -eo pid,ppid,etimes,rss,cmd | rg 'revalidate_live_equivalent|execute-backtests|profit_moonshot' | rg -v 'rg ' || true
   ```

3. Re-read current evidence:

   ```bash
   sed -n '1,220p' var/reports/profit_moonshot_20260501/continuation/latest.md
   sed -n '1,140p' var/reports/profit_moonshot_20260501/profit_moonshot_summary_latest.md
   ```

4. Keep memory below 8GB. Use one mode per backtest unless deliberately resuming a completed checkpoint.

## Recommended next work

### Phase 1 — Robustness-first exposure ladder

Implement fixed live-equivalent modes that reuse the same adaptive momentum signal but test smaller sizing steps around the boost candidate:

- `profit_moonshot_adaptive_momentum_120_mode` — gross exposure `0.0060`, max order `240`
- `profit_moonshot_adaptive_momentum_130_mode` — gross exposure `0.0065`, max order `260`
- `profit_moonshot_adaptive_momentum_140_mode` — gross exposure `0.0070`, max order `280`
- Existing `profit_moonshot_adaptive_momentum_boost_mode` is 150% sizing — gross exposure `0.0075`, max order `300`

Goal: find a candidate with better robustness than boost while still beating baseline.

Preferred pass criteria:

- `val_total_return > 0.0026493262` baseline, preferably `>= 0.0040`
- `train_total_return > -0.025`
- `train_max_drawdown < 0.15`
- `val_max_drawdown <= 0.02`
- `val_sharpe > 0`, `val_sortino > 0`
- `train_trades >= 20`, `val_trades >= 3`
- `liquidations == 0`
- max RSS `< 8GB`

Likely edit files:

- `src/lumina_quant/strategies/artifact_portfolio_mode.py`
- `src/lumina_quant/live_selection.py`
- `tests/unit/test_artifact_portfolio_mode.py`

Backtest command template, one mode at a time:

```bash
mode=profit_moonshot_adaptive_momentum_140_mode
out=var/reports/profit_moonshot_20260501/next/${mode}
/usr/bin/time -v uv run python scripts/research/revalidate_live_equivalent_candidates.py \
  --output-dir "$out" \
  --backtest-checkpoint-path "${out}_checkpoint.json" \
  --portfolio-modes "$mode" \
  --execute-backtests --chunk-days 7 --no-live-decision-update
```

After each completed backtest:

```bash
uv run python scripts/research/profit_moonshot_research.py \
  --input-dir var/reports/profit_moonshot_20260501 \
  --output-dir var/reports/profit_moonshot_20260501
uv run python scripts/research/validate_profit_moonshot_continuation.py || true
```

### Phase 2 — Add a train drawdown guard if ladder is still fragile

If all exposure ladder modes either fail to beat baseline or remain train-fragile, implement a risk governor inside adaptive-momentum mode rows or a new strategy variant. Focus on the October 2025 circuit-breaker period shown in boost logs.

Candidate ideas:

1. Lower risk when broad momentum/regime is weak but not fully risk-off.
2. Cap or disable shorts during high realized-volatility periods if they cause train drawdown.
3. Add a `max_realized_vol` or broad-score floor to the boost row rather than increasing exposure further.

Do **not** use unbounded leverage or raise exposure before drawdown is stable.

### Phase 3 — Run omitted existing modes individually only if needed

The broad `existing_modes` sweep was aborted because it started with `profit_moonshot_ensemble_mode` and was too slow. If needed, test omitted modes one by one:

- `profit_reboot_adaptive_momentum_defensive_mode`
- `profit_reboot_adaptive_momentum_short_bias_mode`
- `profit_reboot_panic_rebound_mode`
- `profit_reboot_session_pair_carry_mode`
- `profit_moonshot_panic_rebound_mode`
- `profit_moonshot_session_pair_carry_mode`

Do not run all together unless a checkpoint already covers the expensive train splits.

## Required final validation before next push

Run:

```bash
uv run ruff check
uv run pytest \
  tests/unit/test_artifact_portfolio_mode.py \
  tests/unit/test_profit_moonshot_research.py \
  tests/test_live_selection_infer.py \
  tests/unit/test_profit_moonshot_strategies.py \
  tests/test_strategy_factory_library.py \
  tests/unit/test_adaptive_regime_momentum.py \
  tests/unit/test_live_equivalent_revalidation.py -q
```

Then commit with Lore protocol and push:

```bash
git push private private-main:main
```

## Existing useful artifacts

- `var/reports/profit_moonshot_20260501/profit_moonshot_summary_latest.md`
- `var/reports/profit_moonshot_20260501/profit_moonshot_summary_latest.json`
- `var/reports/profit_moonshot_20260501/continuation/latest.md`
- `var/reports/profit_moonshot_20260501/continuation/adaptive_boost/live_equivalent_revalidation_latest.md`
- `var/reports/profit_moonshot_20260501/continuation/adaptive_boost/live_equivalent_revalidation_latest.json`
- `scripts/research/validate_profit_moonshot_continuation.py`

## Stop condition for the next session

Stop and report if one of these is true:

1. A mode beats baseline with materially safer train metrics than boost.
2. A mode beats boost return while still passing train/val gates and memory limit.
3. Multiple attempts fail, in which case keep `profit_moonshot_adaptive_momentum_boost_mode` as the best research candidate and document why the next alpha family is needed.
