# Context snapshot — profit moonshot dynamic restart + research history

## Task statement
Restart profit-moonshot research from first principles after rejecting calendar-primary winners. Build/validate pre-defined and newly created **dynamic/state-based** strategies, then create a durable research-history note capturing decisions, strategy properties, metrics, advantages/disadvantages, rejection reasons, and guardrails so future sessions do not repeat the calendar-rule failure.

## Desired outcome
- A plan/PRD/test-spec that permits team/ralph execution without ambiguity.
- A versioned research-history artifact that future sessions must read before profit-moonshot work.
- A renewed dynamic/state-based research run: existing valid candidates plus newly created candidates, screened without locked-OOS leakage, then liquidation-aware/integer-leverage/final-selection validation.
- Git commit + push to `private/main`; CI/private-ci green.

## Known facts / evidence
- Current branch is clean at commit `ebf2263c5004ee8eb2923579be087f2329921557` after strategy-validity audit.
- Latest final selection says `recommendation=no_live_promotion`, `winner=null`.
- Strategy-validity audit: 33 final rows, 31 invalid, 0 deployable; source pool 8,821 rows, 4,392 calendar-primary invalid, 4,429 strategy-valid, but 0 strategy-valid success candidates.
- Prior apparent deployable rows used fixed month/asset calendar rules, mostly TRX/ETH calendar take-profit/rotation; those are rejected as `calendar_primary_alpha_unsupported` and `fixed_asset_calendar_target`.
- Universe: BTC/USDT, ETH/USDT, SOL/USDT, BNB/USDT, TRX/USDT.
- Data/splits used by current artifacts: train `2025-01-01..2025-12-31`, validation `2026-01-01..2026-02-28`, locked-OOS `2026-03-01..2026-05-09`.
- Locked-OOS must remain report-only/gate-only; selection/tuning must use train/validation only.
- Live requires integer leverage and liquidation-aware replay, with Binance USDT perpetual-like conservative margin assumptions, fees/slippage/funding/stress buffer, memory <8 GiB.
- Relevant committed artifacts include:
  - `docs/session_handoff_20260510_profit_moonshot_strategy_validity_audit.md`
  - `var/reports/profit_moonshot_20260501/live_final_selection_20260510/FINAL_HANDOFF_20260510_STRATEGY_VALIDITY_AUDIT.md`
  - `var/reports/profit_moonshot_20260501/live_final_selection_20260510/strategy_validity_audit/strategy_validity_audit_latest.{json,md}`
  - `var/reports/profit_moonshot_20260501/live_final_selection_20260510/final_decision/profit_moonshot_live_final_selection_latest.{json,md}`
- Existing strategy/research touchpoints:
  - `src/lumina_quant/strategies/profit_moonshot.py`
  - `src/lumina_quant/strategies/artifact_portfolio_mode.py`
  - `src/lumina_quant/strategy_factory/candidate_library.py`
  - `scripts/research/replay_profit_moonshot_fresh_start.py`
  - `scripts/research/tune_profit_moonshot_fresh_portfolio.py`
  - `scripts/research/run_profit_moonshot_candidate_hybrid.py`
  - `scripts/research/run_profit_moonshot_liquidation_aware_validation.py`
  - `scripts/research/write_profit_moonshot_live_final_selection.py`
  - `scripts/research/audit_profit_moonshot_strategy_validity.py`
  - `scripts/research/screen_profit_moonshot_external_alpha.py`
- Candidate families already present include trend, breakout, reversion, funding/OI carry/fade, liquidity shock reversion, residual/pair reversion, cross-sectional/topcap momentum, derivatives flow squeeze, taker-flow exhaustion, lead-lag/spillover, adaptive regime momentum, volatility compression/VWAP reversion.

## External source cues for new dynamic hypotheses
- Cross-sectional and time-series crypto momentum are documented in academic/working-paper sources; treat them as hypothesis inspiration only, not live proof.
- Perpetual funding/OI dynamics are an active research area; use them as state variables only when data coverage and live implementability are proven.
- Web search references captured for planning:
  - SSRN: Cryptocurrency Momentum and Reversal — https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3913263
  - SSRN/PDF: Time-Series and Cross-Sectional Momentum in Cryptocurrency Market — https://papers.ssrn.com/sol3/Delivery.cfm/SSRN_ID4698965_code761744.pdf?abstractid=4675565&mirid=1
  - Yale: Common Risk Factors in Cryptocurrency — https://economics.yale.edu/research/common-risk-factors-cryptocurrency
  - MDPI: Two-Tiered Structure of Cryptocurrency Funding Rate Markets — https://www.mdpi.com/2227-7390/14/2/346

## Constraints
- No calendar-primary alpha as a selection signal unless separately pre-registered and proven; do not use fixed month→asset rules.
- No locked-OOS in selection or tuning.
- No non-integer live leverage in final/live candidate or hybrid sleeves.
- Memory across active sessions/commands must stay below 8 GiB; do not run heavy full backtests concurrently.
- Need fresh tests before implementation for new guardrails/history requirements where code changes are needed.
- Commit with Lore protocol and push `private-main:main`; verify GitHub Actions `ci` and `private-ci`.

## Unknowns / open questions
- Whether any existing dynamic source-pool candidate can be rescued by better thresholds/risk management without calendar leakage.
- Whether new dynamic hypotheses can pass train/validation and then locked-OOS gate under conservative liquidation-aware replay.
- Whether current scripts need extension to aggregate a complete research-history artifact from older handoffs automatically or whether a curated note plus machine-readable index is enough for this wave.
- Whether data refresh after `2026-05-10T05:38:38Z` materially changes tail performance.

## Likely codebase touchpoints
- Research-history generator: new script under `scripts/research/` and docs/report artifacts under `docs/` and `var/reports/profit_moonshot_20260501/research_history/`.
- Strategy/candidate definitions: `src/lumina_quant/strategy_factory/candidate_library.py`, `src/lumina_quant/strategies/*`, `src/lumina_quant/strategies/artifact_portfolio_mode.py`.
- Validity/final selection gates: `scripts/research/write_profit_moonshot_live_final_selection.py`, `scripts/research/audit_profit_moonshot_strategy_validity.py`.
- Tests: existing profit-moonshot final-selection/audit/candidate-hybrid/liquidation tests plus new research-history and no-calendar-leak tests.
