import { FactorInsightsRuntime } from '@/components/factor-insights-runtime';

export default function FactorInsightsPage() {
  return (
    <div className="page-stack">
      <section className="hero-card">
        <p className="eyebrow">Factor insights parity</p>
        <h2>Factor IC heatmap &amp; candidate queue</h2>
        <p>
          Read-only view of cross-sectional rank-IC decay per factor and the pending
          research-candidate review queue, sourced from deterministic Python artifacts.
        </p>
      </section>
      <section className="section-card">
        <div className="section-header">
          <div>
            <p className="eyebrow">Python-backed factor feed</p>
            <h3>IC heatmap &amp; candidates</h3>
          </div>
        </div>
        <FactorInsightsRuntime />
      </section>
    </div>
  );
}
