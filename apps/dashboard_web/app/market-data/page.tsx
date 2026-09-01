import { MarketDataRuntime } from '@/components/market-data-runtime';

export const metadata = { title: 'Market Data · LuminaQuant' };

export default function MarketDataPage() {
  return (
    <div className="page-stack">
      <section className="hero-card">
        <p className="eyebrow">Market data</p>
        <h2>What market is this run trading, and what state is it in?</h2>
        <p>
          Recent price action, indicator readings, and the exact bar window behind them for the
          selected run and symbol.
        </p>
      </section>
      <MarketDataRuntime />
    </div>
  );
}
