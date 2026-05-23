# Verification summary — 30m+ alpha booster discovery

- Runner: `scripts/research/run_alpha_zoo_30m_plus_alpha_booster_discovery.py`
- Test: `tests/test_alpha_zoo_30m_plus_alpha_booster_discovery.py`
- Artifact: `alpha_zoo_30m_plus_alpha_booster_discovery_latest.json`
- Candidates evaluated: `63,450`
- Strict paper/testnet candidates: `46`
- Preferred booster target candidates: `0`
- Best paper/testnet candidate: `a30fb_booster_rs_chandelier_2h_solusdt_lb12_atr0p25_rel0p005_trail3p0_hold18_4p0x_0p125_9eeb8c26`
- Metrics: train `+37.4602%`, validation `+16.0919%`, locked-OOS report-only `+4.2373%`, validation MDD `10.8554%`, trades `242/53/27`, RPT `30.96/60.72/31.39bps`, liq/wipeout `0/0`.
- `/usr/bin/time -v` max RSS: `2,302,580 KiB` (<8 GiB).
- All artifacts are paper/testnet-only: `ready_for_real=false`, `real_money_execution=false`.

Final local verification passed: artifact invariants; targeted Alpha Zoo tests `24 passed`; `ruff check .`; `python -m compileall -q src scripts tests`; hardcoded-parameter audit `new=0`; `git diff --check`; `git diff --cached --check`; full pytest `1418 passed in 78.40s`; full-pytest max RSS `2,773,236 KiB` (<8 GiB).
