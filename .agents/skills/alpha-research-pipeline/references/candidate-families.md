# Alpha Candidate Families and Skill Routing

Use this reference to select candidate lanes. The `init_alpha_research_run.py` script creates pre-registered seed cards from these families.

## Core Families

### 1. Literature-derived trend/carry/volatility alphas

Skills: `paper-lookup`, `research-lookup`, `literature-review`, `statsmodels`, `statistical-analysis`.

Examples:

- volatility-managed time-series momentum
- crash-state trend suppression
- funding/carry extreme state interaction
- volatility-managed cross-sectional momentum

Validation focus: publication-time provenance, cost stress, sample robustness, no parameter selection on current OOS.

### 2. Time-series state, anomaly, and motif alphas

Skills: `aeon`, `statsmodels`, `exploratory-data-analysis`, optionally `timesfm-forecasting`.

Examples:

- volatility squeeze breakout with participation confirmation
- anomaly reversion conditional on liquidity
- motif/shapelet pre-breakout patterns
- CUSUM/variance-ratio dynamic routing

Validation focus: train-only state labels, embargo, motif lookup boundary, live feature availability.

### 3. Econometric spread/residual alphas

Skills: `statsmodels`, `statistical-analysis`, `experimental-design`.

Examples:

- cointegration residual mean reversion
- VAR lead-lag rotation
- rolling-beta residual momentum
- pair spread regime switching

Validation focus: formation/trading split, hedge-ratio stability, multiple pair testing, funding/spread costs.

### 4. ML/meta-selector alphas

Skills: `scikit-learn`, `shap`, `scientific-critical-thinking`, `experimental-design`.

Examples:

- alpha sleeve quality selector
- regime classifier for enabling/disabling strategies
- leakage sentinel with shuffled/lag-broken controls
- feature interaction discovery

Validation focus: time-series split, purging/embargo, negative controls, SHAP leakage review, calibration.

### 5. Graph and cross-asset propagation alphas

Skills: `networkx`, `statsmodels`, `pymoo`, optionally `torch-geometric`.

Examples:

- lead-lag graph centrality momentum
- correlation-cluster rotation
- leader/peripheral transfer
- community-level risk-on/off propagation

Validation focus: graph built from past only, topology stability, beta neutrality, turnover.

### 6. Macro/external data alphas

Skills: `database-lookup`, `usfiscaldata`, `research-lookup`, `scientific-critical-thinking`.

Examples:

- U.S. liquidity / TGA / rates crypto beta filter
- rates/dollar risk-off guard
- macro event exposure gate
- fiscal auction/liquidity impulse regimes

Validation focus: release timestamp, revision history, data latency, frequency mismatch, source licensing.

### 7. Microstructure and execution alphas

Skills: `polars`, `aeon`, `simpy`, `pymoo`, `exploratory-data-analysis`.

Examples:

- order-book imbalance state
- trade-intensity exhaustion reversal
- liquidity-shock rebound
- cost-aware no-trade filter

Validation focus: feature freshness, fill realism, latency, spread/slippage at 10/15/20bps, live parity.

### 8. Portfolio/risk/ensemble overlays

Skills: `pymoo`, `statistical-analysis`, `pymc`, `scikit-survival`, `scientific-critical-thinking`.

Examples:

- Pareto sleeve blend
- Bayesian uncertainty sizer
- signal decay hazard exit
- entropy/crowding downweight

Validation focus: no locked OOS weight selection, gross/turnover caps, stability, incumbent comparison.

## Conditional Alternative Data Families

Use only when explicitly relevant and source provenance is strong.

- Text/news/filings: `transformers`, `research-lookup`, `database-lookup`.
- Geospatial/weather/supply-chain: `geomaster`, `geopandas`, `database-lookup`.
- Large data/GPU: `dask`, `vaex`, `zarr-python`, `optimize-for-gpu`.

## Default Exclusions

Biomedical, chemistry, lab automation, clinical, quantum, astronomy, and most materials skills are excluded from default LuminaQuant alpha research. Use only for an explicit alternative-data hypothesis or analogy, and record why the scope expansion is justified.
