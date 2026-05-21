from lumina_quant.models import Bar, Signal, TargetPosition


class ThresholdLongStrategy:
    def __init__(self, start_after: int = 1) -> None:
        self.start_after = int(start_after)
        self.count = 0

    def on_bar(self, bar: Bar) -> Signal:
        self.count += 1
        target = TargetPosition.LONG if self.count >= self.start_after else TargetPosition.FLAT
        return Signal(bar.timestamp, target, f"start_after={self.start_after}")
