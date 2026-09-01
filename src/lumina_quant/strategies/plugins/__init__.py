"""Built-in strategy plugins for cost-aware experiments."""

from lumina_quant.strategies.plugins.trend_momentum import TrendMomentumPlugin
from lumina_quant.strategies.plugins.xs_mean_reversion import CrossSectionalMeanReversionPlugin

__all__ = ["CrossSectionalMeanReversionPlugin", "TrendMomentumPlugin"]
