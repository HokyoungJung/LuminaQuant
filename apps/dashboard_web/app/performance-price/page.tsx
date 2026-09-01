import { PerformancePriceRuntime } from '@/components/performance-price-runtime';

export const metadata = { title: 'Performance & Price · LuminaQuant' };

export default function PerformancePricePage() {
  return (
    <div className="page-stack">
      <section className="hero-card">
        <p className="eyebrow">Performance &amp; Price</p>
        <h2>How did this run perform, and where were the trades?</h2>
        <p>
          Equity against the BTC benchmark, drawdown and funding, plus every recorded trade for the
          selected run.
        </p>
      </section>
      <PerformancePriceRuntime />
    </div>
  );
}
