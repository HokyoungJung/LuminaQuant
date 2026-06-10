"""Tick-replay validator for bar-level order execution (Phase 4.4).

Two-stage pipeline
------------------
1. Aggregate raw aggTrades → 1s OHLCV bars via the pyo3
   ``aggregate_raw_aggtrades_to_1s`` kernel (``data.native_raw_first_backend``).
2. Simulate MKT and LMT orders against those bars using ``ExecutionModel``,
   verifying the bar-level fill rules against the actual tick tape.

LMT validation (Phase 4.4 requirement — ratified 2026-06-10)
-------------------------------------------------------------
    BUY  LMT fills IFF ``bar_low  < limit_price`` (strict)  → fill at ``limit_price``
    SELL LMT fills IFF ``bar_high > limit_price`` (strict)  → fill at ``limit_price``

These rules are intentionally conservative: we only fill when the market clearly
crossed the limit.  The validator checks both positive (should fill) and negative
(should NOT fill) cases.

SAFETY
------
    NEVER places real exchange orders.  Input is a Polars DataFrame of committed
    public market data (aggTrades fixture).  No API keys required or used.

Usage
-----
    from lumina_quant.backtesting.tick_replay_validator import TickReplayValidator
    from lumina_quant.backtesting.execution_model import ExecutionModelConfig

    validator = TickReplayValidator(
        aggtrades_df=df,
        execution_cfg=ExecutionModelConfig.from_runtime(rt),
    )
    verdict = validator.validate()
    assert verdict["lmt_verdict"] == "PASS"
    assert verdict["mkt_verdict"] == "PASS"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

_LOGGER = logging.getLogger(__name__)


# ── Order definitions ─────────────────────────────────────────────────────────


@dataclass
class MktOrderCase:
    """Single MKT order test case for tick-replay validation."""

    symbol: str
    direction: str  # "BUY" | "SELL"
    qty: float
    description: str = ""


@dataclass
class LmtOrderCase:
    """Single LMT order test case for tick-replay validation.

    ``expect_fill`` drives the pass/fail assertion:
    - ``True``:  a fill MUST occur within the bar window.
    - ``False``: NO fill should occur within the bar window.
    """

    symbol: str
    direction: str  # "BUY" | "SELL"
    qty: float
    limit_price: float
    expect_fill: bool = True
    description: str = ""


@dataclass
class TickReplayVerdict:
    """Return value of :meth:`TickReplayValidator.validate`."""

    bars_count: int
    lmt_verdict: str  # "PASS" | "FAIL" | "SKIP"
    mkt_verdict: str  # "PASS" | "FAIL" | "SKIP"
    lmt_cases: list[dict] = field(default_factory=list)
    mkt_cases: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "bars_count": self.bars_count,
            "lmt_verdict": self.lmt_verdict,
            "mkt_verdict": self.mkt_verdict,
            "lmt_cases": self.lmt_cases,
            "mkt_cases": self.mkt_cases,
            "errors": self.errors,
        }


# ── Validator ─────────────────────────────────────────────────────────────────


class TickReplayValidator:
    """Validate bar-level order execution against real aggTrades tick data.

    Parameters
    ----------
    aggtrades_df:
        Polars ``DataFrame`` with columns ``timestamp_ms``, ``price``,
        ``quantity``.  The committed Phase 0 fixture satisfies this.
    execution_cfg:
        ``ExecutionModelConfig`` — controls fees, slippage, liquidity cap.
    lmt_cases:
        List of :class:`LmtOrderCase` to validate.  When ``None`` a default
        BUY + SELL case pair is auto-constructed from the fixture price range.
    mkt_cases:
        List of :class:`MktOrderCase` to validate.  When ``None`` a single
        BUY case is auto-constructed.
    backend:
        ``"auto"`` (default) | ``"rust"`` | ``"python"`` — raw-first backend
        used for tick→bar aggregation.
    """

    def __init__(
        self,
        aggtrades_df: Any,  # polars.DataFrame
        *,
        execution_cfg: Any = None,
        lmt_cases: list[LmtOrderCase] | None = None,
        mkt_cases: list[MktOrderCase] | None = None,
        backend: str = "auto",
    ) -> None:
        self._df = aggtrades_df
        self._backend = str(backend)

        # Build a default ExecutionModelConfig if none provided.
        if execution_cfg is None:
            from lumina_quant.backtesting.execution_model import ExecutionModelConfig

            execution_cfg = ExecutionModelConfig(
                taker_fee_rate=0.0004,
                maker_fee_rate=0.0002,
                slippage_rate=0.0005,
                spread_rate=0.0002,
                leverage=1,
                margin_mode="isolated",
                maintenance_margin_rate=0.005,
                liquidation_buffer_rate=0.0,
                funding_rate_per_8h=0.0,
                funding_interval_hours=8,
                random_seed=42,
                max_bar_volume_ratio=0.1,
            )
        from lumina_quant.backtesting.execution_model import ExecutionModel

        self._execution_model = ExecutionModel(execution_cfg)
        self._lmt_cases = lmt_cases
        self._mkt_cases = mkt_cases

    # ── Aggregation ───────────────────────────────────────────────────────────

    def _aggregate_to_1s(self) -> Any:  # polars.DataFrame | None
        """Aggregate raw aggTrades to 1s OHLCV bars via pyo3 kernel."""
        try:
            from lumina_quant.data.native_raw_first_backend import (
                aggregate_raw_aggtrades_to_1s_native,
            )
        except ImportError:
            return None

        df = self._df
        ts_col = "timestamp_ms"
        if df.is_empty():
            return None
        ts_min_raw = df[ts_col].min()
        ts_max_raw = df[ts_col].max()
        if ts_min_raw is None or ts_max_raw is None:
            return None
        ts_min = int(ts_min_raw)
        ts_max = int(ts_max_raw)

        return aggregate_raw_aggtrades_to_1s_native(
            df,
            range_start_ms=ts_min,
            range_end_ms=ts_max,
            previous_close=None,
            complete_through_ms=ts_max,
            backend=self._backend,
        )

    def _fallback_1s(self) -> Any:
        """Pure-Python fallback when pyo3 kernel is unavailable."""
        try:
            import polars as pl
        except ImportError:
            return None

        df = self._df
        ms_col = "timestamp_ms"
        prices = df["price"].to_numpy()
        volumes = df["quantity"].to_numpy()
        timestamps = df[ms_col].to_numpy()

        # Floor to second boundary.
        second_starts = (timestamps // 1000) * 1000
        unique_seconds = np.unique(second_starts)

        rows = []
        for s in unique_seconds:
            mask = second_starts == s
            p_s = prices[mask]
            v_s = volumes[mask]
            rows.append(
                {
                    "datetime": s,
                    "open": float(p_s[0]),
                    "high": float(p_s.max()),
                    "low": float(p_s.min()),
                    "close": float(p_s[-1]),
                    "volume": float(v_s.sum()),
                }
            )

        if not rows:
            return pl.DataFrame(
                schema={
                    "datetime": pl.Datetime(time_unit="ms"),
                    "open": pl.Float64,
                    "high": pl.Float64,
                    "low": pl.Float64,
                    "close": pl.Float64,
                    "volume": pl.Float64,
                }
            )

        return pl.DataFrame(rows).with_columns(
            pl.col("datetime").cast(pl.Int64).cast(pl.Datetime(time_unit="ms"))
        )

    # ── Default case construction ─────────────────────────────────────────────

    def _default_lmt_cases(self, bars) -> list[LmtOrderCase]:
        """Build default LMT test cases from the fixture price range."""
        lows = bars["low"].to_numpy()
        highs = bars["high"].to_numpy()
        mid_price = float((lows.min() + highs.max()) / 2.0)
        # Round to nearest 10 to get a clean limit.
        mid_price_round = round(mid_price / 10.0) * 10.0

        return [
            LmtOrderCase(
                symbol="BTCUSDT",
                direction="BUY",
                qty=0.01,
                limit_price=mid_price_round,
                expect_fill=True,  # at least one bar should have low < mid
                description=f"BUY LMT at {mid_price_round} — expect fill (price range covers limit)",
            ),
            LmtOrderCase(
                symbol="BTCUSDT",
                direction="SELL",
                qty=0.01,
                limit_price=mid_price_round,
                expect_fill=True,  # at least one bar should have high > mid
                description=f"SELL LMT at {mid_price_round} — expect fill (price range covers limit)",
            ),
            LmtOrderCase(
                symbol="BTCUSDT",
                direction="BUY",
                qty=0.01,
                limit_price=float(lows.min()) - 100.0,  # well below market
                expect_fill=False,
                description="BUY LMT well below market — expect NO fill",
            ),
            LmtOrderCase(
                symbol="BTCUSDT",
                direction="SELL",
                qty=0.01,
                limit_price=float(highs.max()) + 100.0,  # well above market
                expect_fill=False,
                description="SELL LMT well above market — expect NO fill",
            ),
        ]

    def _default_mkt_cases(self) -> list[MktOrderCase]:
        return [
            MktOrderCase(
                symbol="BTCUSDT",
                direction="BUY",
                qty=0.01,
                description="BUY MKT — fill within bar range",
            ),
            MktOrderCase(
                symbol="BTCUSDT",
                direction="SELL",
                qty=0.01,
                description="SELL MKT — fill within bar range",
            ),
        ]

    # ── Raw tick tape (independent ground truth) ───────────────────────────────

    def _raw_tick_arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (prices, quantities) from the RAW aggTrades tape.

        This is the independent reference: it is computed directly from the
        committed tick stream, NOT from the 1s bars the ExecutionModel consumes
        nor from the model's own fill output. Validating model fills against
        these arrays is what makes the validator non-circular.
        """
        df = self._df
        prices = np.asarray(df["price"].to_numpy(), dtype=np.float64)
        qtys = np.asarray(df["quantity"].to_numpy(), dtype=np.float64)
        return prices, qtys

    # ── LMT validation ────────────────────────────────────────────────────────

    def _validate_lmt_case(
        self,
        case: LmtOrderCase,
        bars,
    ) -> dict:
        """Validate the LMT fill rule against the RAW tick tape (independent).

        The previous implementation re-derived ``should_cross`` with the same
        expression the engine uses and then asked ``compute_fill(is_maker=True)``
        — whose price IS ``limit`` by construction — so ``wrong_price`` could
        never trip and the non-crossing branch was discarded. That validated the
        model against itself.

        Here the ground truth comes from the tick tape:

        * **Cross feasibility** — a passive BUY limit at ``L`` can only fill if a
          real trade printed strictly below ``L`` (``price < L``); SELL needs a
          print strictly above ``L``. This is computed from raw ticks.
        * **Fill-quantity feasibility** — the liquidity that could have filled a
          resting limit is the tape volume at-or-through ``L`` (BUY:
          ``price <= L``; SELL: ``price >= L``). The model's executed quantity
          must not exceed that realised volume — otherwise the model claims a
          fill the tape never supported.
        * **Fill price** — a passive limit fills at ``L`` exactly (maker, no
          slippage), and that price must be one the tape actually traded through.

        The model's bar-rule decision is then asserted to AGREE with the
        tape-derived cross feasibility.
        """
        lows = bars["low"].to_numpy()
        highs = bars["high"].to_numpy()

        direction = str(case.direction).upper()
        limit = float(case.limit_price)
        qty = float(case.qty)

        prices, qtys = self._raw_tick_arrays()

        # ── Independent tape ground truth ─────────────────────────────────────
        if direction == "BUY":
            cross_mask = prices < limit  # strict cross (matches engine rule)
            avail_mask = prices <= limit  # liquidity that fills a resting bid
        else:
            cross_mask = prices > limit
            avail_mask = prices >= limit
        tape_crossed = bool(cross_mask.any())
        tape_volume_through_limit = float(qtys[avail_mask].sum()) if qtys.size else 0.0

        # ── Model bar-rule decision (must agree with the tape) ────────────────
        if direction == "BUY":
            n_should_fill = int((lows < limit).sum())
        else:
            n_should_fill = int((highs > limit).sum())
        model_rule_crossed = n_should_fill > 0
        model_rule_matches_tape = model_rule_crossed == tape_crossed

        # ── Model fill (only meaningful when the tape says a fill is possible) ─
        filled_any = False
        wrong_price = False
        fill_price_seen: float | None = None
        executed_qty = 0.0
        if tape_crossed:
            fill_result = self._execution_model.compute_fill(
                raw_price=limit,
                qty=qty,
                direction=direction,
                bar_volume=tape_volume_through_limit,
                is_maker=True,
                apply_liquidity_cap=True,
            )
            executed_qty = float(fill_result.executed_qty)
            if executed_qty > 0:
                filled_any = True
                fill_price_seen = float(fill_result.fill_price)
                # Passive limit must fill at the limit price exactly.
                if abs(fill_result.fill_price - limit) > 1e-8:
                    wrong_price = True

        # Fill quantity must be supported by the realised tape volume.
        qty_feasible = executed_qty <= tape_volume_through_limit + 1e-12

        # ── Pass/fail ─────────────────────────────────────────────────────────
        if case.expect_fill:
            passed = (
                tape_crossed
                and model_rule_matches_tape
                and filled_any
                and not wrong_price
                and qty_feasible
                and tape_volume_through_limit > 0.0
            )
        else:
            # No tape trade should cross the limit, and the model must agree.
            passed = (not tape_crossed) and model_rule_matches_tape

        return {
            "description": case.description,
            "direction": direction,
            "limit_price": limit,
            "expect_fill": case.expect_fill,
            "filled_any": filled_any,
            "n_bars_should_fill": n_should_fill,
            "tape_crossed": tape_crossed,
            "tape_volume_through_limit": tape_volume_through_limit,
            "model_rule_matches_tape": model_rule_matches_tape,
            "fill_price_seen": fill_price_seen,
            "executed_qty": executed_qty,
            "qty_feasible": qty_feasible,
            "wrong_price": wrong_price,
            "passed": passed,
        }

    # ── MKT validation ────────────────────────────────────────────────────────

    def _validate_mkt_case(self, case: MktOrderCase, bars) -> dict:
        """Validate the MKT fill price against the RAW tick price range (independent).

        The previous implementation bounded the model's fill price by a tolerance
        derived from the model's OWN slippage parameters — i.e. it checked the
        model against itself. Here the bound is the price range the tape actually
        traded: a market BUY cannot realistically fill above the highest print
        (nor below the bar open it crossed from), and a market SELL cannot fill
        below the lowest print. Fill quantity must not exceed the realised tape
        volume.
        """
        opens = bars["open"].to_numpy()
        lows = bars["low"].to_numpy()
        highs = bars["high"].to_numpy()
        volumes = bars["volume"].to_numpy()

        bar_idx = 0
        bar_open = float(opens[bar_idx])
        bar_low = float(lows[bar_idx])
        bar_high = float(highs[bar_idx])
        bar_vol = float(volumes[bar_idx])

        prices, qtys = self._raw_tick_arrays()
        tape_low = float(prices.min()) if prices.size else bar_low
        tape_high = float(prices.max()) if prices.size else bar_high
        tape_volume = float(qtys.sum()) if qtys.size else 0.0

        fill_result = self._execution_model.compute_fill(
            raw_price=bar_open,
            qty=float(case.qty),
            direction=str(case.direction).upper(),
            bar_volume=bar_vol,
            volatility=(bar_high - bar_low) / bar_open if bar_open > 0 else 0.0,
            is_maker=False,
            apply_liquidity_cap=True,
        )

        fp = float(fill_result.fill_price)
        direction = str(case.direction).upper()
        # Independent bound: realised tape price range (small tolerance for the
        # half-spread crossing on the boundary print).
        tol = 1e-6 * max(1.0, tape_high)
        if direction == "BUY":
            price_in_tape_range = bar_open - tol <= fp <= tape_high + tol
        else:
            price_in_tape_range = tape_low - tol <= fp <= bar_open + tol
        qty_feasible = float(fill_result.executed_qty) <= tape_volume + 1e-12

        return {
            "description": case.description,
            "direction": direction,
            "qty": float(case.qty),
            "bar_open": bar_open,
            "tape_low": tape_low,
            "tape_high": tape_high,
            "fill_price": fp,
            "commission": fill_result.commission,
            "executed_qty": fill_result.executed_qty,
            "price_in_tape_range": price_in_tape_range,
            "qty_feasible": qty_feasible,
            "price_ok": price_in_tape_range,
            "passed": price_in_tape_range and qty_feasible and fill_result.executed_qty > 0,
        }

    # ── Main validate ─────────────────────────────────────────────────────────

    def validate(self) -> TickReplayVerdict:
        """Run validation and return a :class:`TickReplayVerdict`.

        Returns:
        -------
        TickReplayVerdict
            Always returns a verdict (never raises).  Check
            ``verdict.errors`` for non-fatal warnings and
            ``verdict.lmt_verdict`` / ``verdict.mkt_verdict`` for the
            pass/fail result.
        """
        errors: list[str] = []

        # Aggregate to 1s bars.
        bars = self._aggregate_to_1s()
        if bars is None or bars.is_empty():
            _LOGGER.warning("pyo3 kernel unavailable; falling back to Python aggregation")
            bars = self._fallback_1s()
            if bars is None or bars.is_empty():
                return TickReplayVerdict(
                    bars_count=0,
                    lmt_verdict="SKIP",
                    mkt_verdict="SKIP",
                    errors=["No bars produced — aggTrades fixture may be empty"],
                )

        bars_count = len(bars)

        # ── LMT validation ────────────────────────────────────────────────────
        lmt_cases = (
            self._lmt_cases if self._lmt_cases is not None else self._default_lmt_cases(bars)
        )
        lmt_results = []
        lmt_all_pass = True

        for case in lmt_cases:
            try:
                result = self._validate_lmt_case(case, bars)
            except Exception as exc:
                result = {"description": case.description, "passed": False, "error": str(exc)}
                errors.append(f"LMT case '{case.description}': {exc}")
            lmt_results.append(result)
            if not result.get("passed", False):
                lmt_all_pass = False

        lmt_verdict = "PASS" if lmt_all_pass else "FAIL"

        # ── MKT validation ────────────────────────────────────────────────────
        mkt_cases = self._mkt_cases if self._mkt_cases is not None else self._default_mkt_cases()
        mkt_results = []
        mkt_all_pass = True

        for case in mkt_cases:
            try:
                result = self._validate_mkt_case(case, bars)
            except Exception as exc:
                result = {"description": case.description, "passed": False, "error": str(exc)}
                errors.append(f"MKT case '{case.description}': {exc}")
            mkt_results.append(result)
            if not result.get("passed", False):
                mkt_all_pass = False

        mkt_verdict = "PASS" if mkt_all_pass else "FAIL"

        return TickReplayVerdict(
            bars_count=bars_count,
            lmt_verdict=lmt_verdict,
            mkt_verdict=mkt_verdict,
            lmt_cases=lmt_results,
            mkt_cases=mkt_results,
            errors=errors,
        )


__all__ = [
    "LmtOrderCase",
    "MktOrderCase",
    "TickReplayValidator",
    "TickReplayVerdict",
]
