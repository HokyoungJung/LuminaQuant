# G001 Safety inventory and executable baseline

- generated_at_utc: `2026-07-02T12:45:35Z`
- repo: `/home/hoky/Quants-agent/LuminaQuant`
- plan_reference: `/tmp/.gjc/_session-019f22a1-90f7-7000-ab18-d0fd7010803b/plans/ralplan/019f22a1-90f7-7000-ab18-d0fd7010803b/pending-approval.md`
- plan_reference_sha256: `d0b731a65e81ca27512f92adef975731076abe71fc445adc5b62ebcf291fac88`

## Approval context

- ralplan_gate_result: `User selected Approve execution via ultragoal`
- note: The referenced `pending-approval.md` is the approved plan snapshot for this Ultragoal handoff; its internal `Status: PENDING APPROVAL` remains an audit label from the planning artifact.
- execution_boundary: Approval permits this Ultragoal research execution only; live shadow, paper, and real-money execution remain blocked.

## Git state

```
## private-main...private/main
 M .gjc/state/audit.jsonl
 M .gjc/state/sessions/019edf52-c54c-7000-9168-52eb1573b8aa/ultragoal-state.json
 M apps/dashboard_web/lib/alpha-evidence-server.ts
 M apps/dashboard_web/lib/python-runtime.ts
 M src/lumina_quant/compute/ohlcv_validation.py
 M src/lumina_quant/dashboard/alpha_evidence_service.py
 M tests/test_ohlcv_validation.py
?? .gjc/_session-019f0366-0dc5-7000-83d8-cd839479368e/
?? .gjc/_session-019f046f-794a-7000-91d9-d2274e7e9789/
?? .gjc/_session-019f072d-1c28-7000-88c3-2374500c3f19/
?? .gjc/_session-019f077f-2cb3-7000-852a-82463263ae66/
```

- HEAD: `e0a563b5 (HEAD -> private-main, private/main, private/HEAD) Merge pull request #38 from hoky1227/feat/external-source-indicators`
- upstream ahead/behind: `0	0`

## Pre-existing local changes

These changes predate this ultragoal execution and must be preserved unless a later story explicitly coordinates around them.

- ` M .gjc/state/audit.jsonl`
- ` M .gjc/state/sessions/019edf52-c54c-7000-9168-52eb1573b8aa/ultragoal-state.json`
- ` M apps/dashboard_web/lib/alpha-evidence-server.ts`
- ` M apps/dashboard_web/lib/python-runtime.ts`
- ` M src/lumina_quant/compute/ohlcv_validation.py`
- ` M src/lumina_quant/dashboard/alpha_evidence_service.py`
- ` M tests/test_ohlcv_validation.py`
- `?? .gjc/_session-019f0366-0dc5-7000-83d8-cd839479368e/`
- `?? .gjc/_session-019f046f-794a-7000-91d9-d2274e7e9789/`
- `?? .gjc/_session-019f072d-1c28-7000-88c3-2374500c3f19/`
- `?? .gjc/_session-019f077f-2cb3-7000-852a-82463263ae66/`

## Current incumbent artifacts

- overall_strategy_comparison_latest: exists=True, sha256=`2541612714068b02b291f5eebc369d9cf3312ecc4215ff5500e0845e27b831a5`, path=`/home/hoky/Quants-agent/LuminaQuant/var/reports/current_top_models/overall_strategy_comparison/overall_strategy_comparison_latest.md`
- fresh_forward_shadow_performance_latest: exists=True, sha256=`8908eb3f77fc4d541915dd3676b957790d21966ce3bc056c1a56b763675627e3`, path=`/home/hoky/Quants-agent/LuminaQuant/var/reports/current_top_models/fresh_forward_shadow_eval/fresh_forward_shadow_performance_latest.md`
- h35_shadow_testnet_decision: exists=True, sha256=`ed01a659557646f36f6e227baad025de3371a486f052eb575c5b6af13b2cbe27`, path=`/home/hoky/Quants-agent/LuminaQuant/configs/live/h35_shadow_testnet_decision.json`

## Data coverage snapshot

- market_root: `/home/hoky/Quants-agent/LuminaQuant/data/market_parquet`
- binance_symbol_partition_count: `117`
- symbols_on_disk_count: `117`
- core_static_count: `10`
- tradfi_static_count: `100`
- extended_static_count: `110`
- core_on_disk_count: `10`
- tradfi_on_disk_count: `100`
- extra_on_disk_count: `7`
- tradfi_static_missing_on_disk_count: `0`
- extra_symbols_on_disk_not_static_extended_count_sampled: `7`

## Available execution surfaces

### Scripts
- `scripts/research/refresh_final_portfolio_validation_data.py` exists=True sha256=`807f2050484f254169f8f921126c85f21aee6c93de7283cb18e726512cc5aef6`
- `scripts/research/run_alpha_zoo_69_asset_monthly_refit_walkforward.py` exists=True sha256=`48cd622b0694feb27d623264e890db1ae898e591633d4bfe29bd4cb40ccf48d9`
- `scripts/research/write_tradfi_external_alpha_real_money_path.py` exists=True sha256=`7138189695c06617dc9ff2bb4901dc8887d50a560c967646da131098be8aa3e6`
- `scripts/research/write_tradfi_external_alpha_improvement_followup.py` exists=True sha256=`c1349f527d0f217436ec09e06183c12520dcd1a7f7b7c228d6842341c0258aac`
- `scripts/bench_factor_ic.py` exists=True sha256=`a497132bbc5e49a6e8998ec5a345dba7d8d8cc3563517abe976c8e05ba0b1cf1`
- `scripts/run_research_candidates.py` exists=True sha256=`0d977a8ef7f73303fde6aed4c1bb42d469176b0978d7e2fd5b25b066df6d9fd5`
- `scripts/run_research_pipeline.py` exists=True sha256=`5aa939a106ee5b5c358e2f1ac8c239138c24bdf274fed6c4fff44d17a4d1acb1`

### Modules
- `src/lumina_quant/research_universe.py` exists=True sha256=`3f2f549e2946e0cd81a570cd9e03c26b30f08a546d0d48e994f014f3f00f4e66`
- `src/lumina_quant/research/alpha_search.py` exists=True sha256=`84f6e9f66551212536d86c77fbe72082cabded26ed60ee8721ec538595ad4866`
- `src/lumina_quant/research/tradfi_fetcher.py` exists=True sha256=`fb880e601d63db0f6c255a62cdaeca9e5ad57b06d8a57f566ed8d43e30efa9cd`
- `src/lumina_quant/research/factor_ic.py` exists=True sha256=`18deb3697712253639d913060f7c27e59f2e06a4dac8599a039420e9bff4869e`
- `src/lumina_quant/portfolio/optimizers_extra.py` exists=True sha256=`0110470665985d7bb19ac38a80e8a9ea6abed1616837bf3f7fbefba5529fd431`
- `src/lumina_quant/dashboard/factor_insights_service.py` exists=True sha256=`128eae1199124c7eb2689c043e8c80be29043940c5703c72e3ef597fe5f2e4e5`

## Execution constraints
- No live shadow, paper, or real-money execution in this research run unless later explicit gates pass and user separately approves.
- Tradable research/backtest pool is Binance USD-M core crypto plus Binance USD-M TRADIFI_PERPETUAL instruments.
- Yahoo/Stooq/SEC/web sources are research-only point-in-time priors/features.
- Freeze candidate/portfolio budget manifest before locked-OOS or fresh-forward readout.
- Reject candidates worse than incumbents with reason codes.
- Preserve pre-existing local user changes.
