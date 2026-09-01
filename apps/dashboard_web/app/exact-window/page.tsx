import { ExactWindowRuntime } from '@/components/exact-window-runtime';

export const metadata = { title: 'Exact Window · LuminaQuant' };

export default function ExactWindowPage() {
  return (
    <div className="page-stack">
      <section className="hero-card">
        <p className="eyebrow">Research</p>
        <h2>Which candidates survived the latest exact-window run?</h2>
        <p>
          Summary of the most recent exact-window research bundle: promoted candidates,
          per-timeframe winners, and the fallback portfolio construction.
        </p>
      </section>
      <section className="section-card">
        <ExactWindowRuntime />
      </section>
    </div>
  );
}
