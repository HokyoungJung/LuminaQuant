# Paper/testnet handoff — debounced efficiency repair

- Status: `paper_testnet_candidates_available`
- Candidate count: `82`
- `ready_for_paper=true`
- `ready_for_real=false`
- `real_money_execution=false`
- Real-money execution remains prohibited; this handoff is paper/testnet-only.

## Preflight contract

- Required mode: paper/testnet only.
- Confirm replay/live notional parity before observation.
- Confirm liquidation/account-wipeout telemetry fields are wired into monitoring.
- Record realized fee, slippage, all-in round-trip cost, BBO spread at submit, and notional.

## Top candidates

| Rank | Model | Symbol | TF | Side | Train | Val | OOS | RPT train/val/OOS |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |
| 180 | `debrepair_debounced_efficiency_repair_1h_solusdt_short_only_lb12_e0p02_x0p005_hold36_cool0_none_3p0x_0p15_1e40357d` | SOLUSDT | 1h | short_only | 28.2198% | 16.9294% | 2.4704% | 24.69/59.72/21.11 |
| 283 | `debrepair_debounced_efficiency_repair_1h_solusdt_short_only_lb12_e0p02_x0p005_hold36_cool0_none_4p0x_0p1_edcf6277` | SOLUSDT | 1h | short_only | 25.3406% | 15.0157% | 2.2043% | 24.94/59.59/21.19 |
| 350 | `debrepair_debounced_efficiency_repair_1h_solusdt_long_short_lb12_e0p03_x-0p005_hold48_cool0_none_3p0x_0p15_d6eac828` | SOLUSDT | 1h | long_short | 26.3570% | 15.1313% | 1.7643% | 23.43/55.12/17.82 |
| 395 | `debrepair_debounced_efficiency_repair_1h_solusdt_short_only_lb12_e0p02_x0p005_hold36_cool12_none_3p0x_0p15_6ff58bc8` | SOLUSDT | 1h | short_only | 24.5154% | 13.6239% | 2.4704% | 23.69/51.31/21.11 |
| 485 | `debrepair_debounced_efficiency_repair_1h_solusdt_short_only_lb12_e0p02_x0p005_hold36_cool6_none_3p0x_0p15_ef4179cd` | SOLUSDT | 1h | short_only | 23.3233% | 13.8100% | 2.4704% | 21.07/48.71/21.11 |
| 497 | `debrepair_debounced_efficiency_repair_1h_solusdt_long_short_lb12_e0p03_x-0p005_hold48_cool0_none_4p0x_0p1_44b8ef4f` | SOLUSDT | 1h | long_short | 23.9228% | 13.4658% | 1.5867% | 23.92/55.19/18.03 |
| 564 | `debrepair_debounced_efficiency_repair_1h_solusdt_short_only_lb12_e0p02_x0p005_hold36_cool12_none_4p0x_0p1_ffd0f1b7` | SOLUSDT | 1h | short_only | 22.0625% | 12.1097% | 2.2043% | 23.98/51.31/21.19 |
| 677 | `debrepair_debounced_efficiency_repair_1h_solusdt_short_only_lb12_e0p02_x0p005_hold36_cool6_none_4p0x_0p1_e5d4ce1a` | SOLUSDT | 1h | short_only | 21.0497% | 12.2825% | 2.2043% | 21.39/48.74/21.19 |
| 709 | `debrepair_debounced_efficiency_repair_1h_solusdt_short_only_lb12_e0p02_x0p005_hold36_cool0_none_2p0x_0p15_bfc43af2` | SOLUSDT | 1h | short_only | 19.3343% | 11.2087% | 1.6656% | 25.37/59.31/21.35 |
| 710 | `debrepair_debounced_efficiency_repair_1h_solusdt_short_only_lb12_e0p02_x0p005_hold36_cool0_none_3p0x_0p1_2d4f59e0` | SOLUSDT | 1h | short_only | 19.3343% | 11.2087% | 1.6656% | 25.37/59.31/21.35 |
| 881 | `debrepair_debounced_efficiency_repair_1h_solusdt_short_only_lb6_e0p02_x0p005_hold18_cool0_none_3p0x_0p15_e2b9713c` | SOLUSDT | 1h | short_only | 32.0292% | 11.4117% | 1.8327% | 19.24/27.56/13.58 |
| 946 | `debrepair_debounced_efficiency_repair_1h_solusdt_short_only_lb6_e0p02_x0p005_hold18_cool0_adx20_3p0x_0p15_a9f9a6d7` | SOLUSDT | 1h | short_only | 25.9410% | 11.7165% | 1.8327% | 15.50/28.30/13.58 |
| 1049 | `debrepair_debounced_efficiency_repair_1h_solusdt_long_short_lb12_e0p03_x-0p005_hold48_cool0_none_3p0x_0p1_da1b90f4` | SOLUSDT | 1h | long_short | 18.6111% | 10.1171% | 1.2176% | 24.81/55.28/18.45 |
| 1050 | `debrepair_debounced_efficiency_repair_1h_solusdt_long_short_lb12_e0p03_x-0p005_hold48_cool0_none_2p0x_0p15_ceaf4509` | SOLUSDT | 1h | long_short | 18.6111% | 10.1171% | 1.2176% | 24.81/55.28/18.45 |
| 1094 | `debrepair_debounced_efficiency_repair_1h_solusdt_short_only_lb12_e0p025_x0p0_hold36_cool0_adx20_3p0x_0p15_c2ecca50` | SOLUSDT | 1h | short_only | 32.0204% | 11.2265% | 3.1624% | 33.88/48.92/29.28 |
| 1159 | `debrepair_debounced_efficiency_repair_1h_solusdt_short_only_lb6_e0p02_x0p005_hold18_cool0_none_4p0x_0p1_377fe0d7` | SOLUSDT | 1h | short_only | 28.4910% | 10.1548% | 1.6343% | 19.25/27.59/13.62 |
| 1178 | `debrepair_debounced_efficiency_repair_1h_solusdt_short_only_lb12_e0p02_x0p005_hold36_cool12_none_2p0x_0p15_cc327ac3` | SOLUSDT | 1h | short_only | 16.9081% | 9.0780% | 1.6656% | 24.50/51.29/21.35 |
| 1179 | `debrepair_debounced_efficiency_repair_1h_solusdt_short_only_lb12_e0p02_x0p005_hold36_cool12_none_3p0x_0p1_2bceab04` | SOLUSDT | 1h | short_only | 16.9081% | 9.0780% | 1.6656% | 24.50/51.29/21.35 |
| 1275 | `debrepair_debounced_efficiency_repair_1h_solusdt_short_only_lb6_e0p02_x0p005_hold18_cool0_adx20_4p0x_0p1_def286e6` | SOLUSDT | 1h | short_only | 23.2101% | 10.4227% | 1.6343% | 15.60/28.32/13.62 |
| 1363 | `debrepair_debounced_efficiency_repair_1h_solusdt_short_only_lb12_e0p02_x0p005_hold36_cool6_none_2p0x_0p15_5ee15f39` | SOLUSDT | 1h | short_only | 16.2179% | 9.2182% | 1.6656% | 21.98/48.77/21.35 |
