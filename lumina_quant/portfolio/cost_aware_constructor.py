"""Cost-aware target-to-order constructor."""

from __future__ import annotations

from dataclasses import dataclass

from lumina_quant.backtesting.cost_models import CostModelParams, estimate_cost_bps


@dataclass(slots=True)
class ConstructorParams:
    no_trade_band_bps: float = 8.0
    turnover_penalty: float = 0.0
    cost_penalty: float = 1.0
    participation_penalty: float = 0.0


class CostAwarePortfolioConstructor:
    """Shrink target trades based on predicted transaction costs and penalties."""

    def __init__(self, params: ConstructorParams | None = None):
        self.params = params or ConstructorParams()

    def construct_orders(
        self,
        *,
        target_weights: dict[str, float],
        current_weights: dict[str, float],
        prices: dict[str, float],
        liquidity: dict[str, dict[str, float]],
        aum: float,
        cost_params: CostModelParams,
        max_participation: float,
    ) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
        orders: dict[str, float] = {}
        diagnostics: dict[str, dict[str, float]] = {}

        safe_aum = max(1.0, float(aum))
        for asset, target_weight in target_weights.items():
            current_weight = float(current_weights.get(asset, 0.0))
            delta_weight = float(target_weight) - current_weight
            if abs(delta_weight) * 10_000.0 < float(self.params.no_trade_band_bps):
                continue

            raw_notional = delta_weight * safe_aum
            market = liquidity.get(asset, {})
            price = float(prices.get(asset, market.get("close", 0.0)) or 0.0)
            if price <= 0.0:
                continue

            sigma = float(market.get("sigma", 0.0))
            adtv = float(market.get("adtv", max(1.0, abs(raw_notional))))
            bar_volume = float(market.get("volume", market.get("adv", 0.0)))

            cost = estimate_cost_bps(
                order_notional=raw_notional,
                sigma=sigma,
                adtv=adtv,
                bar_volume=max(1e-9, bar_volume),
                price=price,
                params=cost_params,
            )

            turnover_cost = float(self.params.turnover_penalty) * abs(delta_weight)
            tx_cost = float(self.params.cost_penalty) * (cost.total_bps / 10_000.0)
            participation_cost = float(self.params.participation_penalty) * cost.participation
            shrink = 1.0 + max(0.0, turnover_cost + tx_cost + participation_cost)
            sized_notional = raw_notional / shrink

            cap_notional = max(0.0, float(max_participation)) * max(0.0, bar_volume) * price
            if cap_notional > 0.0:
                if sized_notional > cap_notional:
                    sized_notional = cap_notional
                elif sized_notional < -cap_notional:
                    sized_notional = -cap_notional

            if abs(sized_notional) < 1e-9:
                continue

            orders[asset] = sized_notional
            diagnostics[asset] = {
                "raw_notional": raw_notional,
                "sized_notional": sized_notional,
                "predicted_total_bps": cost.total_bps,
                "predicted_participation": cost.participation,
            }

        return orders, diagnostics
