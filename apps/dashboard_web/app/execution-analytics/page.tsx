import { ExecutionAnalyticsRuntime } from '@/components/execution-analytics-runtime';

export const metadata = { title: 'Execution Analytics · LuminaQuant' };

export default function ExecutionAnalyticsPage() {
  return (
    <div className="page-stack">
      <section className="hero-card">
        <p className="eyebrow">Execution</p>
        <h2>How well are orders filling and trades closing?</h2>
        <p>
          Fill activity, closed-trade outcomes, streaks, and order health for the selected run.
        </p>
      </section>
      <section className="section-card">
        <ExecutionAnalyticsRuntime />
      </section>
    </div>
  );
}
