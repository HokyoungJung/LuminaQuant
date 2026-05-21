# Public Scope

This public repository is limited to a safe, local-only testing pipeline.

Allowed content:

- sample OHLCV CSV data under `sample_data/`,
- one educational moving-average sample strategy,
- local backtesting pipeline,
- local paper-live replay pipeline,
- CI checks, tests, and public usage documentation.

Forbidden content:

- proprietary or production strategies,
- research notes, research reports, experiment artifacts, or optimized parameters,
- production data, private datasets, or data collection code,
- exchange connectors or real order routing,
- credentials, environment files, deployment configuration, or private remotes.

Any future pull request that adds files outside the public scope should update
and pass `tests/public_safety_audit.py` before merge.
