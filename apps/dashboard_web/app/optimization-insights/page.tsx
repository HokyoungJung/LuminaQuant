import { OptimizationInsightsRuntime } from '@/components/optimization-insights-runtime';

export const metadata = { title: 'Optimization Insights · LuminaQuant' };

export default function OptimizationInsightsPage() {
  return (
    <div className="page-stack">
      <section className="hero-card">
        <p className="eyebrow">Optimization insights</p>
        <h2>Which candidates are worth promoting — and can you trust them?</h2>
        <p>
          Ranked optimization candidates with out-of-sample versus train evidence, robustness scores, and
          per-stage medians, so a promotion decision rests on more than a single Sharpe number.
        </p>
      </section>
      <section className="section-card">
        <div className="section-header">
          <div>
            <p className="eyebrow">Candidate evidence</p>
            <h3>Best candidate, ranked comparison, and stage medians</h3>
          </div>
        </div>
        <OptimizationInsightsRuntime />
      </section>
    </div>
  );
}
