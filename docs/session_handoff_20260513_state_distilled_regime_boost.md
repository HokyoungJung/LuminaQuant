# Session handoff — StateDistilledRegimeBoostPortfolio overlay — 2026-05-13

## Scope

Implemented and tested a research-only `StateDistilledRegimeBoostPortfolio` overlay using existing non-calendar state-distilled seeds. This was a thin overlay, not a replacement alpha family and not a live strategy class.

## Strategy used

- Core A: external-risk state-distilled 4x seed `fresh_state_distilled_ext_both_lb168_fast72_z075_ret180_h168_tp600_fl0_xr125`.
- Core B: pure leadership/unwind 4x seed `fresh_state_distilled_both_lb168_fast72_z075_ret180_h168_ls590_ss100_tp600`.
- Booster: conditional sleeve tunable up to 25x, but effective leverage is volatility-targeted per asset and capped by stress/volatility gates.
- Neutral pair overlay: fit/frozen from train/validation lagged features only.

No month/day/hour/calendar entry rules were introduced. The invalid calendar/current-base tuple remains hypothesis-reference-only and was not used for selection or promotion.

## Selection and freeze provenance

- Train/validation-only selection.
- Locked-OOS opened only after frozen config + sidecar hash.
- Grid cap: configured/evaluated/product `64 / 64 / 5832`, hard max `256`.
- Freeze hash: `68db1c473bf43778ccdaba7c2e78ab4a754f71dde2557643fa4267b73d8b3535`.
- Selection ledger: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/state_distilled_regime_boost_20260513/selection_ledger.jsonl`.

## Results

Strict lane selected conservative parameters: core A/B `0.10/0.10`, base leverage `1.0`, stride `24h`, booster allocation `0.10`, neutral pair allocation `0.10`, max effective leverage `4.5x`.

| Split | Return | MDD | Sharpe | Sortino | Calmar | Liq | Min buffer |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | +3.4706% | 3.7181% | 3.4408 | 4.3918 | 0.9334 | 0 | 9837.0960 |
| validation | -1.7179% | 1.7999% | -20.0965 | -10.6949 | -0.9544 | 0 | 9783.9340 |
| locked-OOS | -0.3208% | 0.6141% | -7.9524 | -4.6065 | -0.5224 | 0 | 9936.6557 |

Decision: `deployable_success=false`. Liquidation/buffer/MDD safety was fine, but validation/OOS return and locked-OOS risk-quality metrics were not acceptable. Return/MDD was diagnostic-only.

Diagnostic 5x/6x/10x/15x/25x caps all downshifted to effective `4.5x` under long-term volatility targeting; zero liquidations, but OOS return remained `-0.3208%`. Diagnostic lane remains non-promotable.

## Artifacts

- Runner: `scripts/research/run_state_distilled_regime_boost_portfolio.py`
- Test: `tests/test_profit_moonshot_regime_boost_portfolio.py`
- Artifact assertion guide: `docs/state_distilled_regime_boost_artifact_assertions.md`
- Report dir: `var/reports/profit_moonshot_20260501/current_tail_20260508/alpha_v2/state_distilled_regime_boost_20260513/`
- Summary: `state_distilled_regime_boost_summary_latest.json`
- Markdown: `state_distilled_regime_boost_report_latest.md`
- Freeze: `frozen_config.json` / `frozen_config.sha256.json`
- Locked-OOS gate: `locked_oos_gate.json`

## Memory

Real-data smoke peak RSS: `307077120` bytes (`292.85 MiB`), below the 8 GiB budget. `/usr/bin/time -v` observed max RSS `299880 KiB`.

## Research history/source ledger

No global research inventory/source ledger update was required. The run reused the existing current-tail crypto panel parquet and existing lagged FRED external-state CSV; no new data source family was added.

## Next work

Do not promote this overlay as live. If continuing, either improve the train/validation overlay economics without touching locked-OOS, or return to the stronger Alpha Zoo strict 6x candidate as the current deployable research front-runner under the latest return/MDD-diagnostic policy.
