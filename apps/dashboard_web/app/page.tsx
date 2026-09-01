import { OverviewRuntime } from '@/components/overview-runtime';

export const metadata = { title: 'Overview · LuminaQuant' };

export default function Home() {
  return (
    <div className="page-stack">
      <section>
        <p className="eyebrow">LuminaQuant workspace</p>
        <h2>Trading overview</h2>
      </section>
      <OverviewRuntime />
    </div>
  );
}
