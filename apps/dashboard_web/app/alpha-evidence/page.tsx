import { AlphaEvidenceRuntime } from '@/components/alpha-evidence-runtime';

export const metadata = { title: 'Alpha Evidence · LuminaQuant' };

export default function AlphaEvidencePage() {
  return (
    <div className="page-stack">
      <section className="hero-card">
        <p className="eyebrow">Alpha Evidence</p>
        <h2>Is any alpha ready for real money?</h2>
        <p>
          Classification evidence, reality-gate outcomes, and the live-readiness decision for every
          tracked alpha, in one place.
        </p>
      </section>
      <AlphaEvidenceRuntime />
    </div>
  );
}
