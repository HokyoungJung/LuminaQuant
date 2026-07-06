# Real-Money Readiness Audit — 2026-07-06

Follow-up to `codebase_audit_20260703.md`. Scope: what is still **missing / must
improve / can be optimized** before this system trades **real money**, at tip
`c181f31` (branch main, full suite 3186 passed / 21 skipped).

Method: 8-lens fan-out (live-safety-defaults, go-live-promotion, ops-readiness,
live-data-path, risk-capital, perf-optimization, correctness-new-code,
test-verification-gaps), each finding put through 2 adversarial verifiers
(refute + dedup-vs-prior-audit). 56 raw findings → deduplicated below. Every
item was verified against the code at tip; the prior audit's fixed/refuted items
are **not** re-reported. Severity is by real-money impact.

> One correction vs the raw fan-out: `attach_default_protective_stop` is **not**
> a fully dead key — it is consumed in `portfolio_backtest.py:103,1092`. The real
> defect (below) is that enabling it *breaks* real-mode order submission.

---

## Go / No-Go verdict

**No-Go for real money as shipped.** The research/backtest stack is strong, but
the live path ships with its principal loss-defenses either disabled by default,
inert, or unreachable, and with no end-to-end test of the order loop. The gaps
are concentrated and fixable; none require re-architecture. Fix the CRITICAL
block (§1) and the top MAJORs (§2–§4), add the one integration harness (§6),
then re-attempt via testnet → shadow → canary.

---

## 1. CRITICAL — live safety (fix before any real allocation)

### C1. Kill-switch FLATTEN is unreachable at shipped defaults; every drawdown breach is FREEZE-only
`config.yaml:47,60` · `schema.py:55,80,91` · `risk_manager.py:182-264` · `validate.py:150-159`
- Shipped defaults: `auto_flatten_on_breach: false`, `hard_drawdown_flatten_pct: 0.0`, `flatten_escalate_to_market` default `False`. **Verified.**
- With those, the only FLATTEN action `evaluate_portfolio_risk` can emit is `equity<=0`; intraday-drawdown / daily-loss / rolling-1h breaches all resolve to `FREEZE`, which only blocks *new* entries — open positions ride the entire move.
- `validate.py`'s canary/full gate requires only `freeze_new_entries_on_breach`; it does **not** require any flatten tier, so a `stage=full` real run passes validation with a kill-switch that can never de-risk.
- Even when an operator enables flatten: `_flatten_retry_due` latches False after `FLATTEN_MAX_RETRIES=3`; market escalation only fires on the final retry *and* only if `flatten_escalate_to_market=true`; no `FLATTEN_FAILED` alert is emitted when retries exhaust with positions still open; legs lacking a positive limit reference price are silently skipped.
- **Fix:** for `mode=real` (or `stage in {canary,full}`) require `auto_flatten_on_breach=true` or `hard_drawdown_flatten_pct>0` **and** `flatten_escalate_to_market=true` in `validate.py`; emit a `FLATTEN_FAILED` audit + notifier event on exhaustion; let the reduce-only MKT escape cover reference-less legs on every retry, not just the last.

### C2. Real mode has no managed protective stops — and turning on "no naked positions" breaks order submission
`execution_live.py:859-876` (real-mode raise + `_paper_exchange_protection_enabled()==False` when `mode==real`) · `schema.py:72-76` · `portfolio_backtest.py:1092`
- `_validate_protective_order_params` raises `RuntimeError` for any real-mode order carrying `stop_loss`/`take_profit` unless the strategy hand-builds `metadata['exchange_params']` — and no module anywhere produces that mapping. Exchange-side STOP/TAKE_PROFIT auto-placement is paper/testnet-only.
- Consequences: (a) strategies that set stops get **every entry rejected**; 3 raises in 60 s trip the main-loop hard halt while already exposed; (b) strategies that omit stops run **naked** — only soft in-process exits plus the FREEZE-only guards above; (c) enabling `risk.attach_default_protective_stop=true` attaches a stop to every entry (`portfolio_backtest.py:1092`) → makes **every real-mode entry raise**.
- Net: after a process crash / WSL2 sleep / network cut, a 3× isolated-margin perp has **zero** exchange-resident exit orders.
- **Fix:** build a real-mode protective path — translate `event.stop_loss/take_profit` into Binance `STOP_MARKET`/`TAKE_PROFIT_MARKET` reduce-only algo orders (reuse the paper `_build_paper_protective_algo_specs`, drop the paper-only gate). At minimum, refuse real mode when neither exchange-side protection nor a local trigger monitor is available.

### C3. Fat-finger / price-band guard is inert, and a NaN price bypasses **every** risk cap
`execution_live.py:1136-1146,1266-1306` · `schema.py:281,314` · `risk_manager.py:87,91`
- `MAX_BBO_SPREAD_BPS_AT_SUBMIT` / `MAX_ESTIMATED_ONE_WAY_SLIPPAGE_BPS` / `REQUIRE_BBO_FOR_LIMIT_ORDERS` exist in neither `schema.py` nor `_build_live_config_namespace`, so the guard policy is empty for every strategy except alpha_zoo optuna_hybrid. The one knob the 7/3 fix added, `max_bbo_age_seconds=2.0`, is also vacuous because `book_ticker_enabled` defaults **false** → no BBO snapshot ever exists → guard exits at `missing_bbo`.
- `RiskManager.check_order` values notional at **last bar close, not the limit price** (`quantity*current_price`). Micro-repro: BUY LMT 0.04 BTC at 5,000,000 (100× market) → `(True, 'Passed')`.
- **NaN backstop hole (verified):** the price guard is `if current_price <= 0:` — `NaN <= 0` is `False`, so a NaN close passes the guard **and** every subsequent cap comparison (`NaN > max_order_value` etc. all False). One corrupt bar turns the entire pre-trade risk stack into a no-op.
- **Fix:** add the three band fields with fail-closed real-mode defaults + emit them; force `book_ticker_enabled` (or a mark-price source) in real mode; add `if not math.isfinite(current_price) or current_price <= 0: reject` in `check_order`; add a `|limit-mark|/mark` band and price the notional check at `max(limit, close)` for BUYs.

### C4. Live equity/cash is never re-synced after startup — funding drain and exchange-side liquidations are invisible to every equity kill-switch
`trader.py:1386-1441` (sole `get_balance`), `:1837` (sole `_sync_portfolio` call), `:803-843` (accountUpdate → runtime_cache only) · `portfolio_backtest.py:316-328`
- Only the day-start half of 7/3 fix #3c landed. `_sync_portfolio` runs **once** at startup; `execution_live.get_balance` has zero callers; user-stream `accountUpdate` writes wallet balances to `runtime_cache`, never to `portfolio.current_holdings`. Live funding is modeled at 0.
- Result: real funding (~5%+/yr of equity at 50% gross) and any exchange-side isolated-margin liquidation never debit local cash, so `intraday_loss_pct` / `margin_utilization` run on **fictional equity**. A leg liquidated on-exchange can show near-zero local loss while trading continues sizing off capital that no longer exists.
- Startup sync is also wrong-for-perps and fail-open: `get_balance` returns `availableBalance`, then `_sync_portfolio` adds full position notional on top (overstates true `wallet+uPnL`); any exception is swallowed and the run continues on config `initial_capital=10000`.
- **Fix:** periodic equity reconciliation on `RECONCILIATION_INTERVAL_SEC` — fetch marginBalance (`/fapi/v3/account`) + `positionRisk` uPnL, fold funding/commission deltas into `current_holdings` under `portfolio_lock`, alert + freeze on divergence; hard-fail startup sync in real mode.

### C5. Market-data WS treats 1 s of stream idleness as a connection failure — reconnect storm, no backoff, no gap recovery
`binance_market_stream.py:230,245-246` (dead `if raw is None`, outer `except TimeoutError: continue`) vs correct inline handling in `binance_user_stream.py:93-96`
- `websockets` 16.0 sync `recv(timeout=1)` **raises** `TimeoutError` (it never returns None — the `raw is None` branch is dead code proving intent). The raised timeout exits the `with ws_connect(...)` block and hits `except TimeoutError: continue` → immediate full reconnect with **zero sleep**. Micro-repro: 4 reconnects in 4 s on a quiet stream, same trade replayed 4×, 0 `on_error` calls.
- Any 1-second gap with no aggTrade (sparse universe, overnight lull, exchange maintenance; `book_ticker_enabled` defaults false) → ~60 reconnects/min. Binance caps 300 connection attempts / 5 min / IP → 5 min of quiet self-inflicts an IP connect-ban. This path never calls `_on_error`, so `_recover_gap_ticks` never runs and `_ws_consecutive_errors` (fatal at 12) never increments — **invisible to the operator.**
- **Fix:** handle recv timeout *inside* the connection loop like `binance_user_stream.py:93-96` (`except TimeoutError: raw=None; continue` without dropping the socket); reserve the outer handler for real errors, always route through `on_error`, add jittered exponential backoff, and run gap recovery after each successful reconnect.

### C6. No market-data-silence watchdog on the `binance_futures` live path
`trader.py:1896,1908-1911,1973-1982,929-938` · `market_window_rolling.py:94-107`
- The 45 s staleness gate is **event-driven** — it only runs when a MARKET_WINDOW event arrives. When the feed dies quietly (C5, half-open socket, maintenance) no events are emitted, so the trader's `queue.Empty` branch just writes a heartbeat (no data-age field, no notifier) and loops forever, marking equity at the last frozen close. Nothing monitors time-since-last-window (`grep watchdog/data_age/feed_age` → zero hits). The committed `raw_first` source has an equivalent, the real-time source does not.
- **Fix:** in the `queue.Empty`/heartbeat path, track monotonic time since the last MARKET_WINDOW; past the 45 s threshold set `_materialized_stale_block_active`, alert (cadence-limited), and escalate to a risk freeze after N minutes; include last-window age in the heartbeat.

### C7. No end-to-end live order-flow test exists
`tests/test_live_trader_startup_hardening.py:193-220` (every core component stubbed; `_sync_portfolio`, `_evaluate_risk_guards` disabled) · disjoint LiveTrader vs LiveExecutionHandler test sets · zero `vcr/cassette/respx` fixtures
- `trader.run()` is exercised only with `SimpleNamespace` fakes and `lambda: None` risk guards. No test wires real `LiveExecutionHandler + Portfolio + RiskManager` under a real trader loop with only the exchange faked. All three historical live/backtest CRITICALs (retry double-send, flatten latch, SIGNAL-before-FILL ordering) were **seam bugs** found by manual micro-runs, invisible to the current unit tests. No recorded/replayed testnet session anywhere.
- **Fix (highest-leverage test investment):** one scripted-fake-exchange harness (deterministic fills, injectable 5xx/timeouts/partials) + scenario tests: normal round-trip, fill-during-flatten, restart-mid-position, duplicate executionReport, 502-then-success on flatten.

---

## 2. MAJOR — risk & reconciliation

### M1. Restart defeats the kill-switch: consecutive-loss counter not persisted, state file has no identity fingerprint
`trader.py:560-570` · `risk_manager.py:44,110-116` · `persistence.py:21-38`
- The 7/3 audit **confirmed this as MAJOR but it fell out of the fix ledger entirely.** Re-verified open: `_save_state` omits `RiskManager._consecutive_loss_count` (re-arms to 0 on restart → the Phase-5 halt is defeated by exactly the crash it must survive) and `load_state` restores the machine-global `data/state.json` with no `{strategy,symbols,account,mode,saved_at}` fingerprint → strategy B inherits strategy A's positions/entry-prices/day-start-equity. Documented supervisors (`run_bot.bat` unconditional `goto start`, `run_bot.sh`, systemd `Restart=always`) auto-re-arm after a halt-exit. Zero tests cover halt-survives-restart or foreign-state rejection.
- **Fix:** persist `{consecutive_loss_count, hard_halt}`; embed + verify a state fingerprint with max-age; namespace the state path per strategy+account; make `run_bot.bat` check exit codes; add a restart backoff cap + operator-ack after a halt-exit.

### M2. Any freeze self-lifts within one loop iteration — the reconciliation-drift "freeze" policy is inert
`trader.py:1331-1336` (unconditional un-freeze on risk-pass) · `:1037-1088` (drift dedup + freeze) · `schema.py:310`
- `reconciliation_drift_policy` defaults to `alert` (absent from `config.yaml`), so a wrong local book is never corrected. The stronger `freeze` option sets `trading_frozen`, but `_evaluate_risk_guards` clears it with reason `risk_recovered` the next time equity-risk passes — and drift doesn't move equity inputs, so the freeze lasts <10 s while the book stays wrong. The one-shot signature dedup also means identical persistent drift alerts exactly once.
- **Fix:** default `reconciliation_drift_policy='adopt_exchange'` for real mode; track freeze provenance so `risk_recovered` clears only risk-originated freezes; re-apply/re-alert drift on every cycle while it persists.

### M3. Gross exposure & margin utilization measured on NET positions — HEDGE dual legs count as zero
`risk_manager.py:166-176,195-201` · `portfolio_backtest.py:318-326` · `config.yaml:203-204` (HEDGE + isolated)
- `current_holdings[symbol]` is net market value. Configured `position_mode=HEDGE` explicitly anticipates simultaneous LONG+SHORT legs, but the portfolio-wide total-notional and `margin_utilization` accumulations are net-based. Micro-repro: BTC LONG 0.04 + SHORT 0.04 (true gross 4,000 USDT, both isolated-margin funded) contributes **0.0** to the total cap.
- **Fix:** accumulate gross notional from `current_position_legs` in HEDGE mode for both the total cap and margin utilization; add a separate net-direction exposure metric.

### M4. No desk-standard portfolio controls — no order-rate/turnover budget, no max position age, no correlated-exposure cap, no VaR/vol targeting
`risk_manager.py:85-178` (the complete order gate) · `schema.py:41-95,456-489` (no such fields; StrategyQualityConfig is advisory, default off)
- `check_order` evaluates each order in isolation. A mis-looping strategy alternating LONG/EXIT every decision tick passes every check indefinitely, bleeding ~4–9 bps/round-trip until the slow daily-loss guard reacts (~1–3%/day in pure cost). Concentration is per-symbol only: BTC 25% + ETH 25% all-long is one ~0.8-correlated beta bet at the full 50% gross cap, indistinguishable from a hedged book.
- **Fix:** add max new-orders/rolling-minute + daily notional-turnover budget (fail-closed FREEZE), a max-position-age alert/flatten, and a net-direction gross cap as a first-order correlation proxy; longer term, realized-vol-scaled portfolio limit.

### M5. Backtest liquidation simulator runs on the live path and fabricates local LIQUIDATED fills with no exchange order behind them
`live/portfolio.py:14` (reuses backtest `Portfolio`) · `portfolio_backtest.py:790-850` (`_check_liquidations` → `FillEvent(exchange='SIM_LIQUIDATION')`) · `engine.py:193` · `trader.py:1498`
- Every live market event calls `update_timeindex → _check_liquidations`; when the **modeled** margin math (flat 0.005 MMR, last-price highs/lows, local entry prices) declares a breach, it puts a synthetic `LIQUIDATED` fill onto the **live** queue — no real order, no live-mode gate. Binance liquidates on **mark** price with tiered MMR, so near the zone the local sim can fire when the exchange has not (mark/last divergence peaks exactly in fast markets). Local book flips flat, strategy re-enters, real exposure doubles; default `alert`-only drift never converges.
- **Fix:** gate `_check_liquidations` (and `_apply_funding`'s simulated charge) off the live path — real liquidations arrive as exchange fills; at most keep the modeled breach as a WARNING/audit event, never an applied fill.

### M6. Default polling + full research universe = per-1s-row REST poll storm that stalls the decision loop for minutes
`engine.py:448-480` · `execution_live.py:1670,1762-1769` · `binance_futures_client.py:56` · `config.yaml:15-19`
- The 7/3 no-order early-out only applies to handlers exposing `active_orders`; `LiveExecutionHandler` has no such attribute, so live keeps the legacy per-row sweep. In polling mode each `check_open_orders` issues one signed `query_order` per tracked order with no cadence guard. Shipped config omits `symbols` → inherits the ~100+ symbol research universe; one resting order ⇒ ~2,000+ signed calls per 20 s window, floor-throttled at 50 ms ≈ 100+ s of blocking REST/window → the 45 s staleness gate trips and blocks all orders while an order is working. Also a realistic 429→418 ban vector.
- **Fix:** give `LiveExecutionHandler` an `active_orders`-equivalent early-out (or internally rate-limit `check_open_orders` on `event!=None` to `reconciliation_interval_sec`); require an explicit small `trading.symbols` list for real mode in `validate.py`.

---

## 3. MAJOR — live data reliability

### D1. Disconnect gap recovery is structurally lossy — one non-paginated 500-trade fetch at error time only, all failures swallowed
`data_binance_live.py:188-196,260-267` · `binance_futures_exchange.py:314-318` (end_time capped at since+1 h) · `market_window_rolling.py:107-124` + `timeframe_aggregator.py:251-256` (late data rejected)
- Recovery runs only inside `_on_error` (before reconnect), so trades during the outage window are never fetched — and per C5 the timeout reconnect never calls `_on_error` at all. A single `limit=500` agg_trades call recovers only the oldest 500 of a multi-thousand-trade outage and advances the cursor past the rest; outages >1 h can't recover past since+1 h; every fetch exception returns `[]` silently. Since each 1 s window is emitted once and `TimeframeAggregator` discards bars ≤ `_last_seen_ms`, late truth can never repair flat-filled seconds. ATR/breakout/stop logic then runs on understated-volume, clipped-high/low, flat-close bars.
- **Fix:** run recovery after each successful reconnect, paginate from the per-symbol cursor to the first WS trade id, chunk >1 h outages, and alert when recovery truncates or raises.

### D2. Single-symbol feed death is invisible — flat-fill masks a dead symbol as a fresh flat market
`market_window_rolling.py:118,149-157,97` (synthetic prev_close rows, `is_stale=False` hardcoded, global watermark) · `trader.py:1613-1623` (staleness per-event, never per-symbol)
- When one symbol stops trading (suspension, delisting migration, typo'd token silently ignored by the combined stream), `_build_window_snapshot` fabricates a full window at the last close, volume 0, timestamped fresh, every second. The global watermark keeps advancing off surviving symbols so lag stays small and the staleness gate never fires. No per-symbol last-real-trade age, no zero-volume-streak detection, no startup verification that every symbol delivers a tick. A perp in a suspended symbol rides a 10–20% real move while the system sees a dead-flat price.
- **Fix:** track last-real-trade ts per symbol; when `(watermark - last_real_trade[symbol])` exceeds a threshold, mark that symbol's rows stale (extend the contract) or block new orders + alert; cross-check per-symbol liveness against a cheap REST `premiumIndex` before entries.

### D3. `listenKeyExpired` handling is dead code on the user stream; keepalive failures silently discarded
`binance_user_stream.py:101,168-185,202-234` · `trader.py:657-666,765-775`
- The trader has purpose-built `listenKeyExpired` handling (audit log, BBO-cache invalidation, immediate fallback polling), but `BinanceUserStreamClient.parse_message` filters `listenKeyExpired` frames to `None`, so the branch **can never execute**; the client also doesn't reconnect on it. `_run` discards the keepalive result and swallows every keepalive exception, so a REST failure at the 25-min mark is retried only 25 min later — two silent failures expire the key at 60 min. Around every expiry the intended BBO-invalidation + instant fallback never fire → fills land untracked for up to 45 s and the slippage guard can price off a pre-gap BBO.
- **Fix:** pass `listenKeyExpired` through `parse_message` and treat it as immediate break-and-reconnect with a fresh key; check the keepalive return and force reconnect + `on_error` on repeated failure.

### D4. No bar-value sanity gate anywhere in the live decision path
`core/market_window_contract.py:148-192` (type/shape only) · `market_window_rolling.py:120` + `data_materialized.py:285` (`bars_1s_already_normalized=True`) · `compute/ohlcv_loader.py:82-108` (fail-closed validator wired only into the research loader)
- The repo ships a fail-closed OHLCV integrity validator but no live handler or the MARKET_WINDOW contract uses it; both live producers skip normalization, so rows reach strategies exactly as read from parquet/trades (partial/corrupt commits included). Combined with C3's NaN hole this is a direct path to an oversized live order.
- **Fix:** add finite/positive/high≥low checks at the MARKET_WINDOW seam (reuse `ohlcv_validation`) plus the one-line `isfinite` reject in `check_order`.

### D5. (minor) Poll-transport backlog under burst flow; the stale-feed block has no reduce-only exemption
`data_binance_live.py:260` · `trader.py:1930-1945`
- Poll fetches ≤500 aggTrades/~2 s; BTCUSDT bursts exceed 250 trades/s so the cursor falls behind → 45 s stale gate. That gate blocks **all** order events including strategy-initiated reduce-only exits — inconsistent with `RiskManager`'s own reduce-only exemption from freeze/halt. De-risking exits blocked exactly when data lags.
- **Fix:** loop the poll fetch until a page returns <500 rows; exempt `reduce_only=True` from `ORDER_BLOCKED_STALE_FEED`.

---

## 4. MAJOR — go-live promotion pipeline (backtest → real)

### P1. The graduated ramp is not exercisable: canary unreachable, shadow not runnable, readiness chain broken, final gate is self-attestation
`readiness_policy.py:400-408,580-635,345-399` · `shadow_live_runner.py:84-142` · `cli/live.py` · missing `final_portfolio_validation_data_refresh_latest.json`
- **Canary unreachable:** `ready_for_canary` requires `artifact_canary_execution_allowed`, but the strategy-agnostic veto never emits that key (permanently False for non-AlphaZoo), and no tool anywhere writes it. Meanwhile `ready_for_real`/`ready_for_full` require no canary/shadow evidence → the first reachable real stage is **full 100% sizing**; the one 10%-blast-radius stage is the only one you can't enter.
- **Shadow not runnable:** `evaluate_signal_parity` has zero production callers, no CLI flag, no adapter from repo `Strategy` classes, no persisted artifact; `cli/live.py` never injects a ratio so `stage=shadow` always raises. If hand-wired, the gate accepts any float ≥0.99 with no sample-size floor (the "no comparison" sentinel is `1.0`, indistinguishable from perfect).
- **Readiness chain can't start any gated stage at tip:** the required refresh JSON is absent → raw `FileNotFoundError` (stack trace, not structured diagnostics); the only on-disk decision (`live_equivalent_..._ready_for_review`, dated 2026-05-01) is outside the accepted set; the 30-min staleness window is hardcoded.
- **Final gate is self-attestation:** for non-AlphaZoo strategies `ready_for_real/real_execution_allowed/real_money_execution` are satisfied by ANY payload including the decision JSON itself — three booleans hand-typed into the file unlock real money, with no producer tool and no linkage to validation evidence.
- **README overclaims:** README.md:262 promises "readiness artifacts proving fill/slippage/BBO parity" — no fill/slippage/BBO metric is read anywhere in the policy; the only quantitative check is refresh-file age.
- **Fix:** ship an attestation tool that refuses positive flags without embedded+verified references (paper stats, fill-slippage summary, decision lineage hash, operator id+ts); require ≥1 *referenced* artifact (not the payload) to assert positive flags; make `ready_for_full` require recorded canary evidence; build the shadow adapter + `lq live --measure-shadow-parity` + persisted parity artifact; convert `FileNotFoundError` to `LiveReadinessBlockedError`; regenerate the refresh/decision pair + add a per-stage smoke test; correct README.

### P2. No bridge from strategy-factory / G005 winners to a live decision; new families need a source edit to become promotable
`write_portfolio_live_readiness_decision.py:48-155` · `live_selection.py:287-376` (prefix table lacks G005 families) · `readiness_policy.py:186-199`
- G005's top families (`abnormal_return_continuation`, `last_day_liquidity_regime`, `funding_liquidation_crowding_fade`) are `live_opt_in`-registered, but no script converts `candidate_research_latest.json` → a decision, and `_decision_runtime_compatible` ignores an explicit `strategy_name`, resolving via `infer_strategy_class_name(reference)` → `None` (no prefix rule) → all ready flags False. The gate and `cli/live.py` (which *would* instantiate the class) disagree; new families require editing `live_selection.py`.
- **Fix:** make `_decision_runtime_compatible` honor an explicit `strategy_name`/`strategy_class` validated against `get_live_strategy_map(include_opt_in=True)`; add `write_strategy_factory_live_decision.py` that emits a `promote_candidate` decision from a candidate id.

### P3. G005 walk-forward is unfinished and its resume chain is broken by dangling absolute paths
`var/reports/ultragoal_full_pool_strategy/g005_session_stop_handoff_20260703.json` (`repo_path=/home/hoky/Quants-agent/LuminaQuant`, verified nonexistent) · `g005_partial_walkforward_evaluation_attempt_report.md`
- 30m/4h/1d shards completed but their output dirs exist only as `.gz` archives whose recorded source paths are dangling; the 1h timeframe is incomplete (chunks 02/03 partial, 05+ not started; monitor stopped). All handoff paths use a `LuminaQuant/` prefix that no longer exists. The partial "winners" are weak (best val_return 4.2%, val_sharpe 1.09; 4 of top 6 have **negative** report-only OOS), and G006 (cost stress) + G007 (final decision) haven't run. Safety block intact (`execution_enabled=false`).
- **Bottom line: there is currently no G005 winner to promote.** Attempting to trade the partial leaderboard would select on incomplete, un-stress-tested results.
- **Fix (resume steps):** extract the archives to `g005_shard_eval_{30m,4h,1d}/` and verify sha256; remap the `LuminaQuant/` prefix (or add a path-remap key); rerun 1h chunks lacking final artifacts; merge shards → review gate → checkpoint → G006 cost-stress → G007 decision; only then bridge to a live decision (P2).

---

## 5. MAJOR/MINOR — operations (SRE)

- **O1 (major) Telegram alerting cannot work on the documented install:** `requests` is not in base deps or the `live` extra (only reachable via `live-polymarket`), yet `DEPLOYMENT.md:12` documents `uv sync --extra optimize --extra live`. `NotificationManager` degrades to a one-line log per send; the Binance path uses stdlib `urllib` so trading proceeds while **every** FLATTEN/freeze/drift/hard-halt alert silently fails. No preflight checks alerting. → add `requests` to the `live` extra (or port the notifier to `urllib`); make real-mode preflight fail when the notifier can't deliver a startup test message.
- **O2 (major) No dead-man's switch:** heartbeat is written to Postgres and consumed only by pull-based dashboard queries; nothing pages on process death / hang / WSL2 sleep; systemd unit has no `OnFailure`. All portfolio-level guards die with the process. → out-of-process watchdog on heartbeat age → Telegram/webhook; `OnFailure=` on the unit; in-process main-loop-stall detector that force-exits so `Restart=` can act.
- **O3 (minor) Blocking synchronous Telegram I/O in the trading loop** (`notification.py:32`, up to 5 s/message, no queue) sits *between* queueing reduce-only flatten orders and their dispatch, delaying de-risking during a correlated Telegram/network outage. → bounded background send queue, never on the order-managing thread. (one verifier corrected this to **major**.)
- **O4 (minor) Root `Dockerfile`/`docker-compose.yml` are non-functional** (`python:3.10` vs `requires-python>=3.14`, nonexistent `run_live_ws.py`/`dashboard.py` entrypoints, plain-HTTP unchecksummed TA-Lib) while `DEPLOYMENT.md:4` calls Docker out-of-scope. → delete or rewrite for 3.14 + `uv sync` + `lq live`.
- **O5 (minor) Unbounded `logs/crash.log`** duplicates every rotated line; `run_bot.bat` never `mkdir`s `logs\`; disk-full makes `save_state` silently keep a stale crash-recovery file → next restart restores stale positions. → rotate wrapper output; alert + freeze after N `save_state` failures; add the `mkdir`.
- **O6 (minor) Dashboard localhost check trusts client `x-forwarded-for`** (`apps/dashboard_web/middleware.ts:38-43`) — spoofable read-access to positions/PnL if the server is ever bound beyond loopback. → only honor XFF when the direct peer IP is itself loopback.
- **Banner/routing split (in P-lens):** `cli/live.py:92-99` computes "PRODUCTION" from `testnet is False and mode==real`, but routing derives the endpoint solely from `go_live_stage` (`trader.py:157`); `testnet:false + mode:real + stage:testnet` prints "PRODUCTION" while orders route to testnet. → one shared endpoint resolver; wire or delete the dead `live.testnet` key.

---

## 6. MAJOR — verification gaps (beyond C7)

- **V1 Golden numeric pins are write-only except one walk-forward path.** `compare_to_golden` is consumed only by `test_walk_forward_golden.py` (loads only `walk_forward_results_warmup.json`); the event-driven engine goldens (`ma_cross_*`, `buyholdstrategy_*`) are compared by **nothing** in pytest or CI, and `ci.yml` has zero `golden` references — so the "golden 13 green" attestations were manual-only. The engine/execution_sim/portfolio chain (where the funding + reduce_only CRITICALs lived) has no enforced numeric golden. → add a pytest that runs the engine on the shipped fixtures vs `baseline/golden/*` via `compare_to_golden`; extend goldens to sleeves exercising funding/protective orders/liquidation.
- **V2 CI coverage gate excludes `live`, `exchanges`, `core`** (`ci.yml:177-185` covers only research-side packages) — the ~4,600 riskiest lines have no coverage floor and can silently decay. → add `--cov=lumina_quant.{live,exchanges,core}` with a measured floor, ratcheted.
- **V3 Binance API response-shape drift is untested.** `_as_float` coerces every missing/malformed field to `0.0` (`executedQty→0` looks unfilled; `avgPrice→0` poisons PnL + consecutive-loss classifier; `get_balance→0` feeds C4). No recorded fixtures, no negative-shape tests. → capture one testnet session as fixtures; add negative-shape tests asserting loud rejection in real mode; make `_as_float` distinguish absent from 0 for money-critical fields.
- **V4 No fault-injection test of the flatten chain under a failing exchange**, and the shadow parity runner is tested only with toy callables. → chaos scenarios on the C7 harness (502-then-success flatten, repeated unfilled reduce-only, `-2022` rejection mid-flatten) asserting convergence-to-flat or a loud alert.

---

## 7. Optimization (measured at tip; research-throughput, not live loss)

- **X1 (major) `_kama` is still O(n²)-sliced** — the 7/3 audit's *verified byte-identical* slice fix never landed; only the per-bar memo did. Measured: a 20k-bar AdaptiveTrendRider is 192.5 µs/bar; the `_kama` chain is **8.66 s of 13.09 s (66%)** profiled, 1.43M `kaufman_efficiency_ratio` calls. Per-call scaling unchanged from pre-fix (0.35/1.90/17.42 ms at n=250/1000/4000). `max_hold_bars` is tunable to 200,000 → `history_size` scales with it → one optimizer draw can make `_kama` minutes/bar and de-facto starve that hyperparameter region, biasing selection. `line 111`: replace `vals[: idx + 1][-(period_i + 1):]` with `vals[max(0, idx - period_i): idx + 1]` (re-verified bit-identical today over 50 series, 3.25× at n=4000); then a config-gated incremental KAMA for the O(1)/bar win.
- **X2 (major) The 6.2–6.8× columnar memory cut never reaches the windowed (production) path** — `HistoricParquetWindowedDataHandler.__init__` calls `_freeze_rows_as_epoch_ms()` which immediately re-materializes full boxed-tuple history (`data_windowed_parquet.py:42,88-99`); chunked-DB mode supplies prefrozen tuples that were never columnar. `backtest.mode="windowed"` (root + cost-realistic default) sees **no** steady-state saving. The audit-CRITICAL 110-symbol-1s / 8 GiB-cap scenario is unchanged on the path real validation uses. → teach `_freeze_rows_as_epoch_ms` to serve epoch-ms tuples lazily from the columnar store; re-measure with the windowed handler.
- **X3 (major) Next memory ceiling + per-chunk rebuild tax:** `symbol_timestamps_ms` is a Python list of PyLongs — **40.9 B/row (42% of handler RSS)** at 110 sym × 50k rows, built by materializing every row tuple, reconstructing a datetime, and converting back to ms (2.74 µs/row; `chunked_runner` rebuilds per chunk → ~2.6 h of pure timestamp-rebuild for a 1-yr run before any strategy code). → for epoch-encoded symbols set `symbol_timestamps_ms = (rows._timestamps._epoch // unit).astype(np.int64)` (vectorized, identical ints) and switch consumers to `searchsorted`.
- **X4 (minor) Batched factor-IC is argsort-bound** at ~81 ms/factor with no label-rank sharing: a 256-candidate generation pays 20.7 s in the reducer (~17–20 min at a 100k-ts panel). The bb1471d vectorization works (~13× vs pre-fix) but each factor pays 3 full rank passes; cache fwd-rank keyed by the finite-mask row-pattern and reuse across matching factors (bit-identical, coverable by the 57-case parity suite).
- **X5 (minor) `_check_liquidations` recomputes the position-invariant liquidation price twice/bar** (11% of a cheap-strategy backtest) — cache per symbol keyed by (qty, entry_price), invalidate on fill.
- **X6 (minor) Unconditional `print()` in the simulated-fill hot loop** (`execution_sim.py:411,567`) spams stdout in optimizer sweeps and can hide real warnings → `LOGGER.debug`.

---

## 8. New-code correctness (post-audit commits)

- **N1 (major) Columnar bar storage silently coerces null bars to NaN and unverified numeric dtypes to float64** — the "bit-identical tuples" contract is proven only for timestamps (`data.py:44-72`). Micro-repro: a null `open` that legacy materialized as Python `None` (crashes loudly at first `float()`) now becomes `NaN` and flows silently into indicators/signals/fills; Int64 volume → float; int > 2⁵³ loses precision. Tests cover only clean all-float frames. → mirror the timestamp lossless contract for numerics (reject columnar / keep legacy on nulls or non-float64), add null/int/mixed-dtype test cases.
- **N2 (minor) Batched factor-IC diverges from its "bit-identical" oracle on duplicate (symbol,timestamp) rows** — dense scatter is last-write-wins vs the loop oracle keeping every duplicate; micro-repro shows ic_mean 0.95 vs 0.90. `reduce_factor_ic` neither dedups nor rejects duplicates; the 57-case parity suite has no duplicate-coordinate case. → detect duplicate coords and raise/dedup; add the case; state the uniqueness precondition.
- **N3 (minor) Native kernel handshake only catches Cargo.toml version-string mismatches** — the audited scenario (edit `lib.rs`, no version bump, no rebuild) is still silent (`build_info()=0.1.1 == Cargo.toml`). → embed a hash of `native/lumina_compute/src/*.rs` into `build_info()`; or enforce version-bump-per-kernel-change in CI.
- **N4 (minor) Alpha scoreboard `max_liquidations=0` gate is default-open** — `setdefault("liquidation_count", 0.0)` can't distinguish "measured zero" from "never measured"; every returns-only row passes the catastrophic-path gate clean. Derived Sharpe/CAGR hardcode 365 periods/yr regardless of cadence. → fail (or flag) rows missing `liquidation_count`; accept `periods_per_year` for the derive path.
- **N5 (minor) Conditional-fill liquidity cap emits quantity-0.0 FILLED fills on zero-volume trigger bars** and tears down OCO/protective structure with nothing closed — inflates trade_count/frequency (inputs to eligibility gates + the new scoreboard) and dismantles protection on bars where the exit executed nothing. → skip FillEvent + OCO bookkeeping when `executed_qty <= 0`, keep only the remainder chase; add a zero-volume case.

---

## Recommended fix order (real-money critical path)

1. **Live-safety block (C1–C4):** flatten reachability + real-mode validation gate; real-mode protective-stop path; fat-finger band + NaN reject in `check_order`; periodic equity reconciliation. *Numerics: live-only or config-gated; goldens unaffected.*
2. **Data reliability (C5, C6, D1–D4):** WS reconnect/backoff + gap recovery; data-silence watchdog; per-symbol liveness; bar-sanity gate at the MARKET_WINDOW seam.
3. **Reconciliation & restart (M1, M2, M5, M6):** persist risk state + fingerprint state file; freeze provenance; gate `_check_liquidations` off live; live poll early-out + explicit real-mode symbol list.
4. **Test harness (C7, V1–V4):** one fake-exchange integration harness + chaos scenarios; wire engine goldens into CI; add `live/exchanges/core` coverage; API-shape negative tests.
5. **Promotion pipeline (P1–P3):** attestation tool + referenced-artifact requirement; strategy-factory→decision bridge; finish/repair G005 before any promotion.
6. **Ops (O1, O2):** fix Telegram dependency + dead-man's-switch watchdog (both required before *unattended* real trading).
7. **Optimization (X1–X3):** land the `_kama` slice rewrite now (byte-identical); columnar cut into the windowed path; int64 timestamp array. *Research throughput / optimizer integrity, not live loss.*

**Not covered / worth a follow-up pass** (completeness-critic was cut off by a
credit limit): fill accounting/tax-lot export; capital-scaling vs traded-universe
order-book depth (intended sizes vs liquidity); production regime/parameter-drift
& model-decay monitoring and retrain cadence; Binance API-ToS / rate-of-capital
limits; multi-account or multi-exchange failover.
