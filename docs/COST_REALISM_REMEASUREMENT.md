# Cost-Realism Edge Re-measurement Guide

**Audience:** whoever runs backtests/walk-forward on the backtest machine.
**Goal:** measure how much of a strategy's headline edge survives *realistic* execution
costs — size/impact-aware slippage, charged funding, enforced risk caps, and protective
stops — instead of the optimistic defaults the headline numbers were produced under.

> **Why this exists.** The audit-hardening pass (merged to `main`, commit `2c8f685`)
> added several cost-realism controls but left them **config-gated OFF by default** so the
> golden regression stays byte-identical and historical numbers remain reproducible.
> The published headline figures (e.g. the lagged-leaf-router `+197%` OOS on the 85-asset
> replay, decaying to `+9.75%` on the expanded universe) were measured with **flat
> size-blind slippage and zero funding**. Those are one-directional optimism sources. This
> guide turns the realism on and re-runs so you know what is actually investable.
>
> All flags below default to the *historical* behavior; enabling them changes backtest
> numerics on purpose. Do this on the backtest PC, not in CI.

---

## 1. The controls

All knobs live in `config.yaml` (no CLI flags needed). Schema:
`src/lumina_quant/configuration/schema.py` (`RiskConfig`, `ExecutionConfig`).

### Execution / fill realism — `execution:`

| Key | Default | Realistic value | Effect |
| :-- | :-- | :-- | :-- |
| `slippage_impact_model` | `"flat"` | `"sqrt_impact"` | `flat` = legacy size-blind slippage (byte-identical golden). `sqrt_impact` adds a square-root market-impact term so large/leveraged orders pay more. |
| `slippage_impact_coefficient` | `0.0` | calibrate (see §5) | Impact strength. Penalty grows with `coefficient * sqrt(participation)`. `0.0` ⇒ no impact even under `sqrt_impact`. |
| `slippage_adv_quote` | `0.0` | per-symbol ADV (quote ccy) | Participation denominator. `0.0` ⇒ fall back to per-bar quote volume. Set to a realistic average daily traded value to model participation. |
| `require_funding_coverage` | `false` | `true` (leveraged) | When `true`, a leveraged backtest **fails loudly** instead of silently charging `0.0` funding when per-bar funding data is absent. Forces you to actually have funding data so funding is charged. |

Funding is charged from `execution.funding_rate_per_8h` (static) and/or per-bar funding
feature data. To charge realistic funding you must **collect funding data**
(`data.kinds` must include `funding`) — `require_funding_coverage: true` is the guard that
proves you did.

### Risk enforcement (changes which trades/sizes happen) — `risk:`

| Key | Default | Realistic value | Effect |
| :-- | :-- | :-- | :-- |
| `allow_metadata_risk_override` | `false` | leave `false` | **Already active by default.** When `false`, a sleeve's signal metadata may only *lower* a risk ceiling — it can no longer raise leverage / exposure / order value / notional above the config caps. Set `true` only to reproduce the old unclamped numbers. |
| `max_leverage` | `0.0` | your hard ceiling | Absolute ceiling for metadata leverage overrides while the clamp is active. `0.0` ⇒ metadata may not raise leverage above the configured run leverage. |
| `attach_default_protective_stop` | `false` | `true` | Give a signal with no `stop_loss` a synthetic stop at `default_stop_loss_pct` so no position runs naked. Changes PnL of unstopped sleeves. |
| `enforce_order_risk_gate_in_backtest` | `false` | `true` | Run the same `RiskManager.check_order` gate in the backtest order path as in live, so one enforcement path governs both. May reject over-cap orders. |
| `hard_drawdown_flatten_pct` | `0.0` | optional, e.g. `0.20` | `> 0` ⇒ flatten ALL positions when intraday drawdown exceeds this fraction, even if `auto_flatten_on_breach` is false. `0.0` ⇒ disabled. |

> **Note:** `live.max_bbo_age_seconds` is a *live-only* safety flag (default `2.0`); it has
> no backtest effect and is not part of re-measurement.

---

## 2. Recommended realism profile

A good "realistic" `config.yaml` overlay to start from:

```yaml
execution:
  slippage_impact_model: "sqrt_impact"
  slippage_impact_coefficient: 0.10   # start here, calibrate per §5
  slippage_adv_quote: 0.0             # 0 = per-bar volume; or set per-symbol ADV
  require_funding_coverage: true      # only if funding data is collected
  # baseline per-side costs (defaults shown; tune for cost stress in §4)
  maker_fee_rate: 0.0002
  taker_fee_rate: 0.0004
  spread_rate: 0.0002
  slippage_rate: 0.0005

risk:
  allow_metadata_risk_override: false   # keep the clamp ON (realistic)
  max_leverage: 0.0                     # or your hard ceiling
  attach_default_protective_stop: true
  enforce_order_risk_gate_in_backtest: true
  hard_drawdown_flatten_pct: 0.0        # optional de-risk tier

data:
  kinds: [ohlcv, funding, feature_points]   # funding MUST be present
```

Keep a separate `config.flat.yaml` (defaults) as the baseline for the A/B in §3.

---

## 3. Re-measurement protocol (A/B)

```bash
# 0) Make sure funding + OHLCV data are collected for the universe/window
uv run lq data collect            # (see docs/EXTERNAL_DATA.md for sources)

# 1) BASELINE — flat costs, no realism (record the numbers)
#    Use the shipped defaults (slippage_impact_model: flat, realism flags off).
uv run lq backtest --run-id baseline_flat
uv run lq optimize --folds 10 --oos-days 30 --validation-days 30 --run-id baseline_flat_wf

# 2) REALISTIC — use the ready-made profile (config.yaml + cost-realism flags ON).
#    LQ_CONFIG_PATH REPLACES config.yaml (no merge), so the profile is a full copy
#    of config.yaml with only the flags flipped — keep it in sync if you edit config.yaml.
LQ_CONFIG_PATH=configs/profiles/backtest_cost_realistic.yaml uv run lq backtest --run-id realistic
LQ_CONFIG_PATH=configs/profiles/backtest_cost_realistic.yaml \
  uv run lq optimize --folds 10 --oos-days 30 --validation-days 30 --run-id realistic_wf
```

> The baseline (step 1) uses the root `config.yaml` (flags OFF) and the realistic
> run uses `configs/profiles/backtest_cost_realistic.yaml` (flags ON) — a clean A/B
> differing only in the cost/risk realism block. To hand-tune instead, apply the §2
> overlay to your own config copy.

- Keep `backtest.random_seed`, universe, window, fold count, and `--oos-days` /
  `--validation-days` **identical** between baseline and realistic runs — only the cost/risk
  flags change.
- Walk-forward (`lq optimize`) is the meaningful comparison for "edge survival"; a single
  `lq backtest` is a quick sanity pass.
- Artifacts land under `var/reports/...` keyed by `--run-id`; compare the two.

---

## 4. Cost-stress grid (10 / 15 / 20 bps round-trip)

The per-side cost is approximately `taker_fee_rate + spread_rate/2 + slippage_rate`
(plus the `sqrt_impact` term). **Round-trip ≈ 2 × per-side.** To stress at a fixed
round-trip cost `X` bps, set the knobs so per-side ≈ `X/2` bps:

| Round-trip target | Suggested `slippage_rate` (with default fees/spread ≈ 5 bps/side) |
| :-- | :-- |
| 10 bps | tune fees+spread+slippage so per-side ≈ 5 bps |
| 15 bps | per-side ≈ 7.5 bps |
| 20 bps | per-side ≈ 10 bps |

Re-run §3 step 2 at each cost level and record the degradation. A genuine edge should stay
positive (and pass the gates in §6) across the grid; an edge that flips negative by 15–20 bps
is cost-fragile. For systematic grids the research runners under
`scripts/research/` (e.g. the `*_cost_stress_*` artifacts referenced in
`docs/research_note/research_note.md`) automate this; the manual config method above is the
reproducible baseline.

---

## 5. Calibrating `sqrt_impact`

Impact penalty ≈ `slippage_impact_coefficient * sqrt(order_notional / denominator)`, where
`denominator = slippage_adv_quote` if set, else the per-bar quote volume (`price * volume`).

- Start with `slippage_impact_coefficient: 0.10` and `slippage_adv_quote: 0.0` (per-bar
  volume) for a conservative first pass.
- For capacity/scale studies, set `slippage_adv_quote` to a realistic per-symbol average
  daily traded value so participation (and thus impact) reflects the size you intend to trade.
- The penalty is clamped to `[0, 0.99]` (no negative fill prices), so a mis-set coefficient
  fails safe rather than crediting PnL.

---

## 6. Interpreting results & go/no-go

Compare baseline vs realistic on the walk-forward OOS aggregate:

| Metric | What to watch |
| :-- | :-- |
| OOS compounded / annualized | The headline. How much survives realistic costs? |
| Sharpe | Risk-adjusted survival |
| Max OOS drawdown | Tail under realism |
| Profit factor | A PF that collapses from implausible (e.g. ~30) toward ~1–3 is a sign the headline was cost-optimism |
| Positive folds (e.g. x/10) | Stability across the walk-forward |
| Turnover / return-per-trade (RPT) | High-turnover edges are the most cost-sensitive |

This re-measurement is **research evidence only** — it does not promote anything to real
money. Real-money go-live still requires the governance gates (see
`src/lumina_quant/live/readiness_policy.py` and `docs/live-readiness/`):
clean train/validation-only selection, locked-OOS report-only walk-forward, fresh-forward
shadow/paper with fill telemetry, the cost-stress grid above, turnover/RPT, BBO/slippage,
partial/reject/cancel/reconciliation evidence, and the `ready_for_real` /
`clean_promotion_eligible` flags flipping — plus human review. Default real-money
allocation remains `0%`.

---

## 7. Reproducibility checklist

- [ ] Same `random_seed`, universe, timeframe, window, folds, `--oos-days`/`--validation-days` across A/B.
- [ ] Funding data collected (`data.kinds` includes `funding`) before enabling `require_funding_coverage`.
- [ ] `validation.golden_rtol` unchanged; the golden suite still passes with the realism flags **off** (defaults).
- [ ] Record both `--run-id`s and the exact `config.yaml` diff used for the realistic run.
- [ ] Note the cost level (10/15/20 bps) on each artifact.

See also: [`CONFIG_SPEC.md`](CONFIG_SPEC.md) · [`MODEL_ASSUMPTIONS.md`](MODEL_ASSUMPTIONS.md) ·
[`METRICS.md`](METRICS.md) · [`FINAL_VALIDATION.md`](FINAL_VALIDATION.md) ·
[`research_note/research_note.md`](research_note/research_note.md).
