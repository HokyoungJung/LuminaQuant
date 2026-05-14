# Session Handoff — Crypto/FX Alpha Zoo Real-Data Strict Policy Restoration (2026-05-14)

## Decision

Restored the Alpha Zoo deployability contract requested by the operator: OOS return **and** OOS return/MDD must both beat the invalid current-base/calendar reference before `deployable_success` can be true. The current-base/calendar tuple remains `hypothesis_reference_only` and was not used as a selector or promotion target.

Result: `deployable_success=false`. The train/validation-selected `CryptoFxAlphaZooStateStrategy` replay remains economically interesting, but strict `6.0x` return/MDD `3.007073` does not beat the current-base reference `6.916878`.

## Strategy / factors / calibration used

- Strategy: `CryptoFxAlphaZooStateStrategy`.
- Data: real current-tail crypto panel `var/cache/profit_moonshot_fresh_start/joined_panel_de62df511cec53df6ad39521.parquet` plus lagged FRED regime context `external_market_state_lagged.csv`.
- Factor screen: Alpha Zoo formulaic crypto/FX factor specs; no calendar/month/day/hour entry rules.
- Selected replay candidate: `alpha_zoo_conservative_exit` from `9` narrow, interpretable train/validation-only grid candidates.
- Candidate outcome ledger: triple-barrier outcomes from real factor entries.
- Calibration: `edge_calibration.py` over train+validation records only, with lower-confidence edge gates and blocked/downsize decisions for weak/tail-loss buckets.

## Selection provenance

- Selection inputs: `train`, `validation` only.
- Candidate freeze before locked-OOS gate: `true`.
- `uses_locked_oos_for_selection=false` in screen, replay, factor cards, strategy provenance, and summary.
- Locked-OOS calibration records: `0`.
- Current-base/calendar tuple role: `hypothesis_reference_only`; `selection_target=false`; `promotion_target=false`.

## Real-data screen / ledger / calibration

- Screen rows: `58,845`; factor count: `63`; selected factor cards: `20`.
- Symbols: `BNB/USDT`, `BTC/USDT`, `ETH/USDT`, `SOL/USDT`, `TRX/USDT`.
- Direct FX OHLCV trading remains blocked; lagged FRED state is regime context only.
- Ledger records: `67,259`; train+validation `45,311`; locked-OOS `21,948`.
- Edge calibration input records: `67,259`; calibration records: `45,311`; locked-OOS calibration records: `0`; excluded locked-OOS records: `21,948`.
- Calibrated strategy edge keys: `12`.

## Strict zero-liquidation lane

Strict lane highest zero-liquidation integer: `6.0x`.

| Split | Return | MDD | Return/MDD | Sharpe | Sortino | Smart Sortino | Liq / Min buffer |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| locked-OOS @ 6x | +41.0967% | 13.6667% | 3.007073 | 2.143209 | 2.841936 | 2.500237 | 0 / 9049.125962 overall min |

Strict gates:

- zero liquidation: pass
- positive margin buffer: pass
- OOS MDD <= 25%: pass
- OOS return beats current-base +6.4281%: pass
- OOS return/MDD beats current-base 6.916878: **fail**
- Sharpe/Sortino/smart Sortino/Calmar positive: pass

Decision: no live promotion.

## Diagnostic nonfatal 5x/6x lane

This lane is separate and non-promotional.

- `5.0x`: OOS return `+33.4851%`, MDD `11.5023%`, return/MDD `2.911176`, liquidations `0`, min buffer `9207.604968`, promotion_allowed `False`.
- `6.0x`: OOS return `+41.0967%`, MDD `13.6667%`, return/MDD `3.007073`, liquidations `0`, min buffer `9049.125962`, promotion_allowed `False`.

## Memory

Peak RSS from `/usr/bin/time -v` stage logs: `626.7266 MiB` (`641,768 KiB`), below the 8 GiB limit.

## Artifacts

- Screen: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/crypto_fx_alpha_zoo_real_data_20260514/crypto_fx_alpha_zoo_screen_latest.json`
- Candidate ledger: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/crypto_fx_alpha_zoo_real_data_20260514/candidate_outcome_ledger_latest.jsonl`
- Edge calibration: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/crypto_fx_alpha_zoo_real_data_20260514/edge_calibration_latest.json`
- Replay: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/crypto_fx_alpha_zoo_real_data_20260514/crypto_fx_alpha_zoo_state_replay_latest.json`
- Summary JSON: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/crypto_fx_alpha_zoo_real_data_20260514/crypto_fx_alpha_zoo_real_data_summary_latest.json`
- Summary MD: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/crypto_fx_alpha_zoo_real_data_20260514/crypto_fx_alpha_zoo_real_data_summary_latest.md`

## Research inventory/source ledger

Global research inventory/source ledger was not regenerated: no new external source family or global chronology/source-ledger class was introduced. This pass reused the existing current-tail crypto panel and 20260512 lagged FRED external-state artifact, and added only session-scoped Alpha Zoo strict-policy artifacts.

## Next step

Do not promote the current Alpha Zoo 6x result live. If continuing, broaden only train/validation-selected formulaic Alpha Zoo candidates or calibration buckets, freeze the candidate, and then open locked-OOS gate/report-only. A future deployable candidate must clear both OOS return and return/MDD reference gates.
