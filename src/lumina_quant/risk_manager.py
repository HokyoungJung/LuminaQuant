import math
import time
from collections import deque

from lumina_quant.services.portfolio import PortfolioSizingService


class RiskManager:
    """Enforces risk limits before orders are sent to the exchange."""

    def __init__(self, config):
        self.config = config
        self.max_order_value = getattr(config, "MAX_ORDER_VALUE", 5000.0)
        self.max_order_notional_pct = getattr(config, "MAX_ORDER_NOTIONAL_PCT", 0.0)
        self.max_daily_loss = getattr(config, "MAX_DAILY_LOSS_PCT", 0.05)
        self.max_intraday_drawdown_pct = getattr(
            config,
            "MAX_INTRADAY_DRAWDOWN_PCT",
            self.max_daily_loss,
        )
        # Hard de-risk tier above the soft FREEZE threshold: when > 0 and intraday
        # drawdown exceeds it, force FLATTEN even if auto_flatten_on_breach is False.
        # 0.0 disables the tier (legacy behavior).
        self.hard_drawdown_flatten_pct = float(
            getattr(config, "HARD_DRAWDOWN_FLATTEN_PCT", 0.0) or 0.0
        )
        self.max_rolling_loss_pct_1h = getattr(config, "MAX_ROLLING_LOSS_PCT_1H", 0.05)
        self.max_symbol_exposure_pct = getattr(config, "MAX_SYMBOL_EXPOSURE_PCT", 0.25)
        self.max_total_margin_pct = getattr(config, "MAX_TOTAL_MARGIN_PCT", 0.5)
        max_total_notional_pct = getattr(config, "MAX_TOTAL_NOTIONAL_PCT", 0.0)
        self.max_total_notional_pct = (
            float(max_total_notional_pct)
            if float(max_total_notional_pct) > 0.0
            else self.max_total_margin_pct
        )
        self.target_allocation_mode = PortfolioSizingService.normalize_target_allocation_mode(
            getattr(config, "TARGET_ALLOCATION_MODE", "legacy_notional_cap")
        )
        self.leverage = max(1.0, float(getattr(config, "LEVERAGE", 1.0) or 1.0))
        self.freeze_new_entries_on_breach = bool(
            getattr(config, "FREEZE_NEW_ENTRIES_ON_BREACH", True)
        )
        self.auto_flatten_on_breach = bool(getattr(config, "AUTO_FLATTEN_ON_BREACH", False))
        # Phase 5 kill-switch envelope: consecutive-loss halt trigger
        self.consecutive_loss_halt_count = max(
            1, int(getattr(config, "CONSECUTIVE_LOSS_HALT_COUNT", 5))
        )
        self._consecutive_loss_count: int = 0

        # --- Real-money desk controls (real_money_readiness audit 2026-07-06) ---
        # Every new control defaults to the OFF/legacy value so the default config
        # is byte-identical; they only engage when explicitly configured ON (which
        # the real-mode validate gate requires for the live-safety subset).
        # M3: accumulate GROSS per-leg notional in HEDGE mode (else legacy net).
        self.enforce_gross_exposure_in_hedge = bool(
            getattr(config, "ENFORCE_GROSS_EXPOSURE_IN_HEDGE", False)
        )
        self._hedge_mode = (
            str(getattr(config, "POSITION_MODE", "") or "").strip().upper() == "HEDGE"
        )
        # M4: order-rate / daily-turnover / position-age budgets (0 => disabled).
        self.max_orders_per_minute = max(0, int(getattr(config, "MAX_ORDERS_PER_MINUTE", 0) or 0))
        self.max_daily_notional_turnover_pct = float(
            getattr(config, "MAX_DAILY_NOTIONAL_TURNOVER_PCT", 0.0) or 0.0
        )
        self.max_position_age_hours = float(getattr(config, "MAX_POSITION_AGE_HOURS", 0.0) or 0.0)
        # C3: fat-finger price-band guard for limit orders (0 => disabled).
        self.max_limit_price_band_pct = float(
            getattr(config, "MAX_LIMIT_PRICE_BAND_PCT", 0.0) or 0.0
        )
        # M4 runtime state (only mutated when the corresponding budget is enabled,
        # so the default path never touches it).
        self._recent_order_times: deque[float] = deque()
        self._turnover_day: int | None = None
        self._daily_turnover_notional: float = 0.0
        # M1: latched hard-halt flag persisted across restart via get_state/set_state.
        # Default False => byte-identical (nothing trips it unless an operator does).
        self._hard_halt: bool = False

    @staticmethod
    def _portfolio_position_legs(portfolio, symbol) -> dict[str, float] | None:
        if portfolio is None:
            return None
        for attr in ("current_position_legs", "position_legs"):
            payload = getattr(portfolio, attr, None)
            if not isinstance(payload, dict):
                continue
            legs = payload.get(symbol)
            if isinstance(legs, dict):
                return {
                    "LONG": max(0.0, float(legs.get("LONG", 0.0) or 0.0)),
                    "SHORT": max(0.0, float(legs.get("SHORT", 0.0) or 0.0)),
                }
        return None

    @staticmethod
    def _project_hedge_legs(order_event, current_legs: dict[str, float]) -> dict[str, float] | None:
        position_side = str(getattr(order_event, "position_side", "") or "").upper()
        if position_side not in {"LONG", "SHORT"}:
            return None
        direction = str(getattr(order_event, "direction", "") or "").upper()
        quantity = abs(float(getattr(order_event, "quantity", 0.0) or 0.0))
        projected = {
            "LONG": max(0.0, float(current_legs.get("LONG", 0.0) or 0.0)),
            "SHORT": max(0.0, float(current_legs.get("SHORT", 0.0) or 0.0)),
        }
        if position_side == "LONG":
            if direction == "BUY":
                projected["LONG"] += quantity
            elif direction == "SELL":
                projected["LONG"] = max(0.0, projected["LONG"] - quantity)
        elif position_side == "SHORT":
            if direction == "SELL":
                projected["SHORT"] += quantity
            elif direction == "BUY":
                projected["SHORT"] = max(0.0, projected["SHORT"] - quantity)
        return projected

    @staticmethod
    def _order_limit_price(order_event) -> float | None:
        """Return the order's limit price as a positive finite float, else None.

        Market orders (``price is None``) and malformed prices yield ``None`` so
        limit-only logic (C3 band + conservative BUY notional) no-ops for them.
        """
        raw = getattr(order_event, "price", None)
        if raw is None:
            return None
        try:
            price = float(raw)
        except TypeError, ValueError:
            return None
        if math.isfinite(price) and price > 0:
            return price
        return None

    @staticmethod
    def _symbol_mark_price(portfolio, symbol, net_mv: float, net_qty: float) -> float | None:
        """Best-effort per-symbol mark price for GROSS-notional conversion (M3).

        Prefers the portfolio's live bar source (the same price it marks holdings
        with); falls back to deriving price from net market value / net quantity
        when a bar is unavailable. Returns None when no positive finite price can
        be resolved (the caller then falls back to net accounting for that symbol).
        """
        bars = getattr(portfolio, "bars", None)
        getter = getattr(bars, "get_latest_bar_value", None)
        if callable(getter):
            try:
                price = float(getter(symbol, "close"))
            except Exception:
                price = float("nan")
            if math.isfinite(price) and price > 0:
                return price
        if net_qty not in (0, 0.0) and math.isfinite(net_mv):
            price = abs(net_mv) / abs(net_qty)
            if math.isfinite(price) and price > 0:
                return price
        return None

    def _current_total_notional(self, portfolio) -> tuple[float, float]:
        """Return ``(total_notional, net_direction_notional)`` across all symbols.

        Legacy behaviour (``enforce_gross_exposure_in_hedge`` False, or not HEDGE
        mode): ``total_notional`` is the net-abs market value sum — byte-identical to
        the prior inline accumulation. When gross accounting is enabled for a HEDGE
        book, symbols carrying dual legs contribute ``(LONG+SHORT)*price`` (GROSS) so
        a fully-hedged position no longer nets to zero (M3). ``net_direction_notional``
        is the signed net exposure sum (a first-order direction/correlation proxy).
        """
        gross_enabled = self.enforce_gross_exposure_in_hedge and self._hedge_mode
        holdings = getattr(portfolio, "current_holdings", {}) or {}
        positions = getattr(portfolio, "current_positions", {}) or {}
        total = 0.0
        net_direction = 0.0
        for sym in portfolio.symbol_list:
            net_mv = float(holdings.get(sym, 0.0) or 0.0)
            if gross_enabled:
                legs = self._portfolio_position_legs(portfolio, sym)
                if legs is not None and (legs["LONG"] > 0.0 or legs["SHORT"] > 0.0):
                    net_qty = float(positions.get(sym, 0.0) or 0.0)
                    price = self._symbol_mark_price(portfolio, sym, net_mv, net_qty)
                    if price is not None:
                        total += (legs["LONG"] + legs["SHORT"]) * price
                        net_direction += (legs["LONG"] - legs["SHORT"]) * price
                        continue
            total += abs(net_mv)
            net_direction += net_mv
        return total, net_direction

    @staticmethod
    def _order_time_seconds(order_event) -> float:
        """Monotone-ish wall-clock reference for the M4 rolling-window budgets.

        Prefers the order's own ``timestamp_ns`` (deterministic + testable); falls
        back to ``time.time()`` only when the event carries no usable timestamp.
        Only invoked when an M4 budget is enabled, so the default path never calls
        into wall-clock time and stays deterministic/byte-identical.
        """
        ts_ns = getattr(order_event, "timestamp_ns", None)
        if (
            isinstance(ts_ns, (int, float))
            and not isinstance(ts_ns, bool)
            and math.isfinite(ts_ns)
            and ts_ns > 0
        ):
            return float(ts_ns) / 1e9
        return time.time()

    def _purge_order_window(self, now_s: float) -> None:
        """Drop recorded order times older than the trailing 60s window (M4)."""
        cutoff = now_s - 60.0
        window = self._recent_order_times
        while window and window[0] <= cutoff:
            window.popleft()

    def _roll_turnover_day(self, now_s: float) -> None:
        """Reset the daily-turnover accumulator when the UTC day rolls over (M4)."""
        day = int(now_s // 86400.0)
        if self._turnover_day != day:
            self._turnover_day = day
            self._daily_turnover_notional = 0.0

    def check_order(self, order_event, current_price, portfolio=None):
        """Returns True if order is safe, False otherwise."""
        # C3 fail-closed correctness fix: a NaN close silently bypasses EVERY
        # downstream cap comparison (``NaN <= 0`` and ``NaN > cap`` are both False),
        # turning the entire pre-trade risk stack into a no-op on one corrupt bar.
        # Reject any non-finite or non-positive reference price unconditionally.
        if not math.isfinite(current_price) or current_price <= 0:
            return False, "Invalid market price."

        # C3: resolve the order's limit price (if it carries one) for the
        # price-band guard and the conservative notional valuation below.
        # reduce-only exits are exempt from both: they cannot increase exposure
        # (so there is no fat-finger entry to guard) and must never be impeded, in
        # line with the reduce-only de-risk exemptions applied throughout this gate.
        reduce_only = bool(getattr(order_event, "reduce_only", False))
        limit_price = self._order_limit_price(order_event)

        # C3 fat-finger price-band guard: reject a limit order whose price deviates
        # more than ``max_limit_price_band_pct`` from the reference (mark/last close).
        # Gated by the 0 => disabled default so the legacy path is byte-identical.
        if self.max_limit_price_band_pct > 0.0 and limit_price is not None and not reduce_only:
            band_dev = abs(limit_price - current_price) / current_price
            if band_dev > self.max_limit_price_band_pct:
                return (
                    False,
                    f"Limit price {limit_price:.8g} deviates {band_dev:.4%} from "
                    f"reference {current_price:.8g} (> band "
                    f"{self.max_limit_price_band_pct:.4%})",
                )

        # 1. Check Notional Value (absolute per-order cap).
        # C3: for a BUY, an aggressive limit above the last close should be valued
        # at the limit, not the (lower) close, so the per-order + downstream caps
        # cannot be under-counted. SELL / market / reduce-only orders keep the
        # reference price (reduce-only stays byte-identical to the legacy path).
        direction = str(getattr(order_event, "direction", "") or "").upper()
        if limit_price is not None and direction == "BUY" and not reduce_only:
            price_for_notional = max(limit_price, current_price)
        else:
            price_for_notional = current_price
        notional_value = order_event.quantity * price_for_notional
        if float(self.max_order_value) > 0.0 and notional_value > self.max_order_value:
            return (
                False,
                f"Order Value ${notional_value:.2f} exceeds limit ${self.max_order_value}",
            )

        # 2. Check Negative Quantity
        if order_event.quantity <= 0:
            return False, f"Invalid Quantity: {order_event.quantity}"

        # 3. Portfolio-level checks
        if portfolio is not None:
            if getattr(portfolio, "trading_frozen", False) and not bool(
                getattr(order_event, "reduce_only", False)
            ):
                return False, "Trade freeze active: new entries blocked."

            # Phase 5 kill-switch: consecutive-loss auto-halt (reduce-only exempt)
            if self._consecutive_loss_count >= self.consecutive_loss_halt_count and not bool(
                getattr(order_event, "reduce_only", False)
            ):
                return (
                    False,
                    f"Consecutive-loss halt: {self._consecutive_loss_count} losses "
                    f">= limit {self.consecutive_loss_halt_count}. Only reduce-only orders allowed.",
                )

            # M1: latched hard-halt (persisted across restart). Default False, so this
            # branch is inert unless an operator trips it. Reduce-only stays exempt so
            # positions can always be de-risked after a halt.
            if self._hard_halt and not bool(getattr(order_event, "reduce_only", False)):
                return False, "Hard halt active: only reduce-only orders allowed."

            total_equity = float(portfolio.current_holdings.get("total", 0.0))
            if total_equity <= 0:
                return False, "Non-positive equity."

            order_notional_cap = total_equity * float(self.max_order_notional_pct)
            if order_notional_cap > 0 and notional_value > order_notional_cap:
                return (
                    False,
                    f"Order Value ${notional_value:.2f} exceeds equity-scaled cap ${order_notional_cap:.2f}",
                )

            # reduce-only orders bypass circuit-breaker and all downstream exposure caps
            # so the system can always de-risk open positions after a daily-loss trip.
            # (trade-freeze and consecutive-loss halt already exempt reduce-only above.)
            if bool(getattr(order_event, "reduce_only", False)):
                return True, "Passed (reduce-only bypass)."

            if getattr(portfolio, "circuit_breaker_tripped", False):
                return False, "Circuit breaker already tripped."

            # Approximate symbol exposure after order execution.
            current_legs = self._portfolio_position_legs(portfolio, order_event.symbol)
            projected_legs = (
                self._project_hedge_legs(order_event, current_legs) if current_legs else None
            )
            if projected_legs is not None:
                projected_symbol_notional = (
                    float(projected_legs["LONG"]) + float(projected_legs["SHORT"])
                ) * current_price
            else:
                cur_qty = float(portfolio.current_positions.get(order_event.symbol, 0.0))
                signed_order_qty = (
                    order_event.quantity
                    if order_event.direction == "BUY"
                    else -order_event.quantity
                )
                projected_qty = cur_qty + signed_order_qty
                projected_symbol_notional = abs(projected_qty * current_price)
            symbol_cap = total_equity * self.max_symbol_exposure_pct
            if symbol_cap > 0 and projected_symbol_notional > symbol_cap:
                return (
                    False,
                    f"Symbol exposure {projected_symbol_notional:.2f} exceeds cap {symbol_cap:.2f}",
                )

            # Approximate total notional exposure.
            # Uses available holdings valuation from portfolio snapshot. When
            # enforce_gross_exposure_in_hedge is set and the book is in HEDGE mode,
            # this accumulates GROSS per-leg notional so dual LONG+SHORT legs both
            # count (M3); otherwise it is the legacy net-abs market value.
            current_total_notional, _net_direction = self._current_total_notional(portfolio)
            projected_total_notional = current_total_notional + notional_value
            total_cap = total_equity * self.max_total_notional_pct
            if total_cap > 0 and projected_total_notional > total_cap:
                return (
                    False,
                    f"Total exposure {projected_total_notional:.2f} exceeds cap {total_cap:.2f}",
                )

            # M4: order-rate + daily-turnover budgets. Evaluated after every exposure
            # gate so only a fully-admissible NEW entry consumes the budget; reduce-only
            # exits bypassed above are intentionally exempt (de-risking must not throttle).
            # Both gated by their 0 => disabled defaults, so the legacy path is untouched.
            now_s: float | None = None
            if self.max_orders_per_minute > 0 or self.max_daily_notional_turnover_pct > 0.0:
                now_s = self._order_time_seconds(order_event)

            if self.max_orders_per_minute > 0:
                self._purge_order_window(now_s)
                if len(self._recent_order_times) >= self.max_orders_per_minute:
                    return (
                        False,
                        f"Order-rate limit: {len(self._recent_order_times)} orders in "
                        f"trailing 60s >= limit {self.max_orders_per_minute}. "
                        "FREEZE new entries.",
                    )

            projected_turnover: float | None = None
            if self.max_daily_notional_turnover_pct > 0.0:
                self._roll_turnover_day(now_s)
                turnover_cap = total_equity * self.max_daily_notional_turnover_pct
                projected_turnover = self._daily_turnover_notional + notional_value
                if turnover_cap > 0.0 and projected_turnover > turnover_cap:
                    return (
                        False,
                        f"Daily turnover budget: projected {projected_turnover:.2f} "
                        f"exceeds cap {turnover_cap:.2f} "
                        f"({self.max_daily_notional_turnover_pct:.2%} of equity).",
                    )

            # Admitted: record the order against the M4 budgets.
            if self.max_orders_per_minute > 0:
                self._recent_order_times.append(now_s)
            if self.max_daily_notional_turnover_pct > 0.0:
                self._daily_turnover_notional = projected_turnover

        return True, "Passed"

    def evaluate_portfolio_risk(self, portfolio):
        equity = float(portfolio.current_holdings.get("total", 0.0))
        if equity <= 0:
            return False, "Non-positive equity", "FLATTEN", {"equity": equity}

        day_start = float(getattr(portfolio, "day_start_equity", equity) or equity)
        intraday_loss_pct = 0.0
        if day_start > 0:
            intraday_loss_pct = max(0.0, (day_start - equity) / day_start)

        rolling_loss_pct_1h = 0.0
        get_rolling_loss = getattr(portfolio, "get_rolling_loss_pct", None)
        if callable(get_rolling_loss):
            rolling_loss_pct_1h = float(get_rolling_loss(3600))

        # M3: total_notional accumulates GROSS per-leg exposure in HEDGE mode when
        # enforce_gross_exposure_in_hedge is set (dual LONG+SHORT legs both count);
        # otherwise it is the legacy net-abs market value. net_direction_notional is
        # the signed net exposure, exposed as a first-order correlation/direction proxy.
        total_notional, net_direction_notional = self._current_total_notional(portfolio)
        if self.target_allocation_mode == "isolated_margin_fraction":
            margin_utilization = total_notional / (equity * self.leverage) if equity > 0 else 0.0
        else:
            margin_utilization = total_notional / equity if equity > 0 else 0.0

        # Hard de-risk tier (defense-in-depth): a drawdown beyond hard_drawdown_flatten_pct
        # forces FLATTEN regardless of auto_flatten_on_breach. Evaluated before the soft
        # intraday check so the hard tier wins. 0.0 disables the tier (legacy).
        if (
            self.hard_drawdown_flatten_pct > 0.0
            and intraday_loss_pct >= self.hard_drawdown_flatten_pct
        ):
            return (
                False,
                "Hard drawdown flatten breach",
                "FLATTEN",
                {
                    "intraday_loss_pct": intraday_loss_pct,
                    "threshold": float(self.hard_drawdown_flatten_pct),
                },
            )

        if intraday_loss_pct >= float(self.max_intraday_drawdown_pct):
            action = "FLATTEN" if self.auto_flatten_on_breach else "FREEZE"
            return (
                False,
                "Intraday drawdown breach",
                action,
                {
                    "intraday_loss_pct": intraday_loss_pct,
                    "threshold": float(self.max_intraday_drawdown_pct),
                },
            )

        # Daily-loss cap tier: guards when max_daily_loss_pct < max_intraday_drawdown_pct
        # (i.e. the daily cap is tighter than the intraday drawdown cap).  If the two
        # thresholds are equal or the daily cap is looser, this tier is unreachable and
        # enforcement falls through to PortfolioBacktest._check_circuit_breaker(), which
        # measures the same equity-vs-day_start quantity and sets circuit_breaker_tripped.
        # Ordering guarantee: hard_drawdown_flatten_pct > intraday > daily > rolling.
        # When max_daily_loss_pct >= max_intraday_drawdown_pct the tier is intentionally
        # skipped; the intraday check above already provides the enforcement.
        if float(self.max_daily_loss) < float(self.max_intraday_drawdown_pct) and (
            intraday_loss_pct >= float(self.max_daily_loss)
        ):
            action = "FLATTEN" if self.auto_flatten_on_breach else "FREEZE"
            return (
                False,
                "Daily loss breach",
                action,
                {
                    "intraday_loss_pct": intraday_loss_pct,
                    "threshold": float(self.max_daily_loss),
                },
            )

        if rolling_loss_pct_1h >= float(self.max_rolling_loss_pct_1h):
            action = "FLATTEN" if self.auto_flatten_on_breach else "FREEZE"
            return (
                False,
                "Rolling 1h loss breach",
                action,
                {
                    "rolling_loss_pct_1h": rolling_loss_pct_1h,
                    "threshold": float(self.max_rolling_loss_pct_1h),
                },
            )

        if margin_utilization >= float(self.max_total_margin_pct):
            action = "FREEZE" if self.freeze_new_entries_on_breach else "NONE"
            return (
                action == "NONE",
                "Margin utilization breach",
                action,
                {
                    "margin_utilization": margin_utilization,
                    "threshold": float(self.max_total_margin_pct),
                    "target_allocation_mode": self.target_allocation_mode,
                    "leverage": self.leverage,
                    "net_direction_notional": net_direction_notional,
                },
            )

        # M4: max-position-age de-risk tier. Uses the optional portfolio contract
        # ``get_max_position_age_hours()`` (mirrors ``get_rolling_loss_pct``) reporting
        # the age in hours of the oldest open position. Gated by the 0 => disabled
        # default, and inert until the portfolio exposes the method, so the legacy
        # path is byte-identical.
        if self.max_position_age_hours > 0.0:
            age_getter = getattr(portfolio, "get_max_position_age_hours", None)
            if callable(age_getter):
                try:
                    max_age_hours = float(age_getter())
                except Exception:
                    max_age_hours = 0.0
                if math.isfinite(max_age_hours) and max_age_hours >= self.max_position_age_hours:
                    action = "FLATTEN" if self.auto_flatten_on_breach else "FREEZE"
                    return (
                        False,
                        "Position age breach",
                        action,
                        {
                            "max_position_age_hours": max_age_hours,
                            "threshold": float(self.max_position_age_hours),
                            "net_direction_notional": net_direction_notional,
                        },
                    )

        return (
            True,
            "Passed",
            "NONE",
            {
                "intraday_loss_pct": intraday_loss_pct,
                "rolling_loss_pct_1h": rolling_loss_pct_1h,
                "margin_utilization": margin_utilization,
                "target_allocation_mode": self.target_allocation_mode,
                "leverage": self.leverage,
                "net_direction_notional": net_direction_notional,
            },
        )

    def check_portfolio_risk(self, portfolio):
        """Check if daily loss limit is hit."""
        # Already handled in Portfolio circuit breaker, but can add redundancy here.
        return True, "Passed"

    # Phase 5 kill-switch: consecutive-loss tracking
    def record_loss(self, *, realized_pnl: float) -> int:
        """Record a realized loss.  Increments the consecutive-loss counter when
        ``realized_pnl < 0``; resets it on a profitable fill.  Returns the updated
        consecutive-loss count so callers can log or audit it.
        """
        if float(realized_pnl) < 0.0:
            self._consecutive_loss_count += 1
        else:
            self._consecutive_loss_count = 0
        return self._consecutive_loss_count

    def reset_consecutive_losses(self) -> None:
        """Manually reset the consecutive-loss counter (e.g. after operator review)."""
        self._consecutive_loss_count = 0

    # M1: latched hard-halt controls -----------------------------------------
    def trip_hard_halt(self) -> None:
        """Latch a hard halt so only reduce-only orders pass until cleared."""
        self._hard_halt = True

    def clear_hard_halt(self) -> None:
        """Clear the latched hard halt (operator action)."""
        self._hard_halt = False

    # M1: kill-switch state persistence --------------------------------------
    # The consecutive-loss counter and the latched hard-halt must survive a
    # process restart, otherwise the Phase-5 halt re-arms to 0 on exactly the
    # crash it is meant to survive. trader.py wires get_state/set_state into the
    # state.json save/load path.
    def get_state(self) -> dict:
        """Serialize the restart-critical kill-switch state."""
        return {
            "consecutive_loss_count": int(self._consecutive_loss_count),
            "hard_halt": bool(self._hard_halt),
        }

    def set_state(self, state) -> None:
        """Restore kill-switch state from a ``get_state`` payload (tolerant)."""
        if not isinstance(state, dict):
            return
        count = state.get("consecutive_loss_count")
        if (
            isinstance(count, (int, float))
            and not isinstance(count, bool)
            and math.isfinite(count)
            and count >= 0
        ):
            self._consecutive_loss_count = int(count)
        halt = state.get("hard_halt")
        if isinstance(halt, bool):
            self._hard_halt = halt
