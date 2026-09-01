import { RawDataRuntime } from '@/components/raw-data-runtime';

export const metadata = { title: 'Raw Data · LuminaQuant' };

export default function RawDataPage() {
  return (
    <div className="page-stack">
      <section className="hero-card">
        <p className="eyebrow">Raw data</p>
        <h2>What exactly is stored for this run?</h2>
        <p>
          Frame-by-frame previews of the run&apos;s stored records — runs, equity, fills, orders,
          risk events, and market bars — for spot-checking what the strategy actually saw and did.
        </p>
      </section>
      <RawDataRuntime />
    </div>
  );
}
