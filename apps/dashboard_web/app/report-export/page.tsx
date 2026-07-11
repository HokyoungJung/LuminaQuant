import { ReportExportRuntime } from '@/components/report-export-runtime';

export const metadata = { title: 'Report Export · LuminaQuant' };

export default function ReportExportPage() {
  return (
    <div className="page-stack">
      <section className="hero-card">
        <p className="eyebrow">Reporting</p>
        <h2>Export the run snapshot</h2>
        <p>Preview the selected run&apos;s summary report and download it as JSON or Markdown.</p>
      </section>
      <section className="section-card">
        <ReportExportRuntime />
      </section>
    </div>
  );
}
