# Verification summary — asset-diverse Alpha Zoo discovery (2026-05-23)

## Discovery run
- Command: `/usr/bin/time -v uv run python scripts/research/run_alpha_zoo_asset_diverse_strategy_discovery.py`
- Artifact: `alpha_zoo_asset_diverse_strategy_discovery_latest.json`
- Candidates evaluated: `97,560`
- Paper/testnet-only gate-pass rows: `4`
- Runner max RSS: `2,020,024 KiB` (`1,972.680 MiB` reported in artifact), below 8 GiB.
- Real-money flags: `ready_for_real=false`, `real_money_execution=false`.

## Best paper/testnet-only row
- Model: `a30fb_asset_diverse_residual_reclaim_2h_ethusdt_btcusdt_lb48_z1p0_hold6_4p0x_0p125_fa49c5d5`
- Strategy: ETHUSDT 2h `relative_residual_reclaim` vs BTCUSDT.
- Train / validation / locked-OOS return: `+16.8301% / +4.7367% / +4.8120%`.
- Validation MDD: `4.89%`.
- Trades train/validation/OOS: `184 / 32 / 26`.
- Return-per-turnover proxy bps train/validation/OOS: `18.29 / 29.60 / 37.02`.
- Locked-OOS liquidation/account-wipeout: `0 / 0`.

## Shadow discoveries
- Best train+validation-ranked shadow rows were XRPUSDT 1h cross-asset rank chandelier variants with train roughly `+36%` to `+55%`, validation roughly `+29%` to `+32%`, and RPT above 10bps on train/validation.
- They are rejected/no-promotion because local XRPUSDT data lacks locked-OOS coverage after 2026-03 (`locked_oos_trade_event_count=0`, OOS RPT missing), and shadow symbols are explicitly non-promotable.

## Governance checks
- Locked-OOS is marked report/gate-only after train+validation ranking freeze.
- `uses_locked_oos_for_discovery=false`, `uses_locked_oos_for_selection=false`, `uses_locked_oos_for_objective=false`, `uses_locked_oos_for_pruning=false`, `uses_locked_oos_for_parameter_fitting=false`.
- Promotion rows satisfy train trades >=80, validation >=30, OOS >=20, validation return >=2%, train>0, train>=validation, validation MDD<=12%, OOS>0, zero OOS liq/wipeout, RPT>10bps all splits, and replay/live notional parity.

## Local verification
- Artifact invariant check: passed.
- Targeted tests: `15 passed`.
- `uv run ruff check .`: passed.
- `uv run python -m compileall -q src scripts tests`: passed.
- `uv run python scripts/audit_hardcoded_params.py`: `total=567 new=0 baselined=567`.
- `git diff --check && git diff --cached --check`: passed.
- Full pytest: `1422 passed in 61.85s`; max RSS `2,769,980 KiB`, below 8 GiB.
