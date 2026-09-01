import { FactorInsightsRuntime } from '@/components/factor-insights-runtime';

export const metadata = { title: 'Factor Insights · LuminaQuant' };

export default function FactorInsightsPage() {
  return (
    <div className="page-stack">
      <section className="hero-card">
        <p className="eyebrow">Factor insights</p>
        <h2>Which factors carry signal, and which candidates deserve review?</h2>
        <p>
          Cross-sectional rank-IC decay per factor, the full factor ranking, and the research-candidate
          review queue — a read-only research view for prioritizing what to promote next.
        </p>
      </section>
      <section className="section-card">
        <div className="section-header">
          <div>
            <p className="eyebrow">Research evidence</p>
            <h3>IC decay heatmap, factor ranking, and candidate queue</h3>
          </div>
        </div>
        <FactorInsightsRuntime />
      </section>
    </div>
  );
}
