# Alpha Zoo clean new-alpha discovery

- generated: `2026-06-18T13:55:22.219945Z`
- pre-registered search hash: `b4bdb079ba4d9ad5e7202b053858c6d219359126dddaa2a546e38c863ccccca4`
- selection policy: `default_train_validation`
- enabled families: `1`
- integer leverages: `[2]`
- fold workers: `1`
- simulation backend: `python`
- candidate cap sort: `eligible_first_active_train_validation_selection_score`
- candidate rows retained/written: `5`/`5`
- selection input: `train + validation only`
- locked-OOS: `report/gate only after freeze`
- split simulation policy: `continuous_full_period_signal_slice_report_only`
- clean promotion eligible: `false`
- post-OOS selector trusted: `false`
- real-money: `false`
- benchmark gates: shadow `64.42%` or return/MDD `3.49`; clean/paper `34.39%`; max MDD `30.00%`

## Aggregate selected fold result

- OOS comp: `0.00%`
- annualized approx: `0.00%`
- monthly equity MDD: `0.00%`
- max OOS MDD: `0.00%`
- positive folds: `0/0`
- Sharpe approx: `0.00`

## Live realism diagnostics

- live plausibility: `not_supported`
- mean validation return: `0.00%`
- mean locked-OOS return: `0.00%`
- positive locked-OOS fold share: `0.00`
- min validation trade events: `0`
- max validation Sharpe: `0.00`
- blockers: `no_selected_fold_rows`

## Promotion gate summary

- tried families: `1`
- skipped families: `0`
- promotion statuses: `{'rejected_before_full_wf': 5}`
- top rejection reasons: `{'continuous_position_state_across_split_boundaries': 5, 'fresh_forward_required_before_promotion': 5, 'not_train_validation_freeze_eligible': 5, 'validation_cost_turnover_proxy_non_positive': 5}`

## Survivor manifest

- manifest hash: `024446a922c30fb230d9eb407a4948ab041b81ee3f91ba08f47532fb5b00c6ea`
- frozen survivors: `0`
- full-WF retest candidates: `0`
- selection inputs: `train, validation`
- holdout policy: `attach_after_train_validation_freeze_report_gate_only`
- real-money: `false`

## Candidate gate audit rows

| Fold | Model | Family | Selected | Promotion | Advance full WF | Freeze hash | Feature T/V | Rejection reasons |
| --- | --- | --- | ---: | --- | ---: | --- | --- | --- |
| `2026-06` | `az69_squeeze_1h_btcusdt_lb24_q0p15_br24_hold4_lev2_4f71536f` | `volatility_squeeze_breakout` | no | `rejected_before_full_wf` | no | `ac16007e6c5c` | `1.00/1.00` | `continuous_position_state_across_split_boundaries`, `fresh_forward_required_before_promotion`, `not_train_validation_freeze_eligible`, `validation_cost_turnover_proxy_non_positive` |
| `2026-06` | `az69_squeeze_1h_btcusdt_lb72_q0p25_br24_hold4_lev2_0b6f78eb` | `volatility_squeeze_breakout` | no | `rejected_before_full_wf` | no | `63437674486e` | `1.00/1.00` | `continuous_position_state_across_split_boundaries`, `fresh_forward_required_before_promotion`, `not_train_validation_freeze_eligible`, `validation_cost_turnover_proxy_non_positive` |
| `2026-06` | `az69_squeeze_1h_btcusdt_lb24_q0p15_br24_hold8_lev2_fbf70704` | `volatility_squeeze_breakout` | no | `rejected_before_full_wf` | no | `2d61d8f38a69` | `1.00/1.00` | `continuous_position_state_across_split_boundaries`, `fresh_forward_required_before_promotion`, `not_train_validation_freeze_eligible`, `validation_cost_turnover_proxy_non_positive` |
| `2026-06` | `az69_squeeze_1h_btcusdt_lb72_q0p25_br12_hold4_lev2_ce23a56b` | `volatility_squeeze_breakout` | no | `rejected_before_full_wf` | no | `b3481fd1bea7` | `1.00/1.00` | `continuous_position_state_across_split_boundaries`, `fresh_forward_required_before_promotion`, `not_train_validation_freeze_eligible`, `validation_cost_turnover_proxy_non_positive` |
| `2026-06` | `az69_squeeze_1h_btcusdt_lb72_q0p15_br12_hold4_lev2_05340a26` | `volatility_squeeze_breakout` | no | `rejected_before_full_wf` | no | `93f6c1619bf2` | `1.00/1.00` | `continuous_position_state_across_split_boundaries`, `fresh_forward_required_before_promotion`, `not_train_validation_freeze_eligible`, `validation_cost_turnover_proxy_non_positive` |

## Full-WF promotion candidates

- count: `0`
- selection rule: `survivor_manifest_train_validation_freeze_only`
- locked-OOS report gates: `attached_after_freeze_only`
- real-money: `false`

## Fold selections

| Fold | Model | Family | Train | Validation | Locked OOS | OOS MDD |
| --- | --- | --- | ---: | ---: | ---: | ---: |
