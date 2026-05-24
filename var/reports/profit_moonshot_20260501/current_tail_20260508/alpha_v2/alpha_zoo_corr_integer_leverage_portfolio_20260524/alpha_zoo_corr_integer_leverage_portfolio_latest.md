# Alpha Zoo Corr Integer-Leverage Portfolio

Generated: `2026-05-24T12:42:48.623687Z`

## Method

- Starts from the latest corr-diversified paper slate, not the full duplicate 136-row book.
- Replays fixed position-state signals and searches integer leverage maps per asset.
- Uses only train+validation return, MDD, liquidation/wipeout, and RPT for leverage-map selection.
- locked-OOS is gate/report-only after the train+validation leverage map is frozen.
- No real-money execution; all outputs remain paper/testnet-only.

## Profile results

| Profile | Tier | Leverage map | Gross | Train | Val | OOS report-only | Val MDD | OOS MDD | Strict promotion | Paper candidate |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| balanced_mdd12_gross5 | `strict_promotion_paper_testnet_candidate` | `{"SOLUSDT": 2, "TRXUSDT": 1}` | 1.00x | 74.6685% | 33.2153% | 5.5300% | 11.6134% | 7.2003% | true | true |
| growth_mdd20_gross8 | `relaxed_paper_testnet_candidate` | `{"ETHUSDT": 8, "SOLUSDT": 4, "TRXUSDT": 12}` | 3.90x | 262.3353% | 71.6291% | 23.3695% | 19.9983% | 9.2371% | false | true |
| aggressive_mdd30_gross10_shadow | `relaxed_paper_testnet_candidate` | `{"ETHUSDT": 8, "SOLUSDT": 4, "TRXUSDT": 12}` | 4.90x | 438.4462% | 117.4976% | 27.5772% | 29.4044% | 12.3630% | false | true |

## Selected recommendation

Use strict-promotion `balanced_mdd12_gross5` for paper/testnet review only: leverage map `{"SOLUSDT": 2, "TRXUSDT": 1}`, validation 33.2153%, locked-OOS report-only 5.5300%.

Also keep relaxed paper/testnet candidates under separate MDD/risk labels:
- `growth_mdd20_gross8` leverage `{"ETHUSDT": 8, "SOLUSDT": 4, "TRXUSDT": 12}`: validation 71.6291%, locked-OOS report-only 23.3695%, validation MDD 19.9983%.
- `aggressive_mdd30_gross10_shadow` leverage `{"ETHUSDT": 8, "SOLUSDT": 4, "TRXUSDT": 12}`: validation 117.4976%, locked-OOS report-only 27.5772%, validation MDD 29.4044%.

## Governance

- ready_for_real=false
- real_money_execution=false
- locked-OOS used for selection=false
