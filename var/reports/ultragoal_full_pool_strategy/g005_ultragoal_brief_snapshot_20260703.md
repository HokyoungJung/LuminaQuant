Full-pool LuminaQuant strategy research run recovered from docs/research_note/full_pool_ultragoal_resume_20260702.md. Preserve the approved constraints: live shadow, paper/testnet, and real-money execution remain blocked; imported external performance claims are research-design priors only; G002 source/quarantine flags and G003 point-in-time prior registry are binding inputs; G004 must freeze immutable candidate and portfolio search budgets before any evaluation readout; G005-G007 must obey frozen-budget and no-OOS-selection rules.

@goal: Safety inventory and executable baseline
Inventory the safety gates, current runnable baseline, and verification surface before data refresh or evaluation.

@goal: Refresh Binance USD-M universe and market data coverage
Refresh Binance USD-M core plus TradFi-perp universe coverage, classify missing/quarantined symbols, and preserve support-data refresh evidence without enabling live/paper/real execution.

@goal: Build point-in-time external-prior registry
Build a source-pinned point-in-time external prior registry and cache manifest. Treat all external priors as research-design priors, never LuminaQuant historical evidence.

@goal: Freeze candidate and portfolio search budgets
Build the immutable G004 candidate/portfolio search budget manifest before any evaluation readout. Include repo state, G002 universe/source/feature hashes, G003 prior registry hash, candidate family caps, seeds, operator/window/formula inputs, threshold grids, portfolio grids, cost/turnover/MDD/gross constraints, effective-trials accounting, exclusion/quarantine rules, and no-OOS-selection policy.

@goal: Evaluate candidates with walk-forward and cost stress
Evaluate candidates only under the frozen G004 budget with walk-forward and cost-stress rules. Do not use locked OOS for selection, tuning, tie-breaks, or portfolio construction.

@goal: Construct portfolios and compare incumbents
Construct portfolios under the frozen portfolio budget and compare against incumbents using identical windows, costs, constraints, and source eligibility.

@goal: Produce fail-closed final research decision
Produce the final research decision with fail-closed promotion state, evidence artifacts, and explicit no-real-money/no-paper/no-testnet status unless all gates are proven clean.
