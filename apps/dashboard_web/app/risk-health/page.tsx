import { RiskHealthRuntime } from '@/components/risk-health-runtime';

export const metadata = { title: 'Risk & Health · LuminaQuant' };

export default function RiskHealthPage() {
  return (
    <div className="page-stack">
      <section className="hero-card">
        <p className="eyebrow">Operations</p>
        <h2>Is the latest run healthy?</h2>
        <p>
          Risk events, heartbeats, and order state transitions from the most recent run, so
          incidents surface before they compound.
        </p>
      </section>
      <section className="section-card">
        <div className="section-header">
          <div>
            <p className="eyebrow">Run telemetry</p>
            <h3>Latest telemetry</h3>
          </div>
        </div>
        <RiskHealthRuntime />
      </section>
    </div>
  );
}
