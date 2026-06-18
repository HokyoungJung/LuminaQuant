'use client';

import type { ReportExportPayload } from '@/lib/dashboard-contracts';
import { useBridgeFetch } from '@/lib/use-bridge-fetch';

function downloadTextFile(filename: string | undefined, content: string, mime: string) {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = filename || 'download.txt';
  anchor.click();
  URL.revokeObjectURL(url);
}

export function ReportExportRuntime() {
  const { payload, error } = useBridgeFetch<ReportExportPayload>(
    '/api/python/dashboard/report-export',
    'report export bridge failed',
  );

  if (error) {
    return <p>{error}</p>;
  }
  if (payload === null) {
    return <p>Loading snapshot export payload…</p>;
  }
  if (payload.status !== 'ok') {
    return <p>No report export payload available yet.</p>;
  }

  return (
    <div className="page-stack">
      <div className="button-row">
        <button
          className="action-button"
          onClick={() =>
            downloadTextFile(
              payload.filenames.json,
              JSON.stringify(payload.json_report, null, 2),
              'application/json',
            )
          }
          type="button"
        >
          Download JSON Snapshot
        </button>
        <button
          className="action-button"
          onClick={() => downloadTextFile(payload.filenames.markdown, payload.markdown_report, 'text/markdown')}
          type="button"
        >
          Download Markdown Snapshot
        </button>
      </div>

      <div className="metric-grid">
        <article>
          <span>Run ID</span>
          <strong>{payload.json_report.run_id ?? 'n/a'}</strong>
        </article>
        <article>
          <span>Strategy</span>
          <strong>{payload.json_report.strategy ?? 'unknown'}</strong>
        </article>
        <article>
          <span>Total Return</span>
          <strong>{String(payload.json_report.total_return ?? 'n/a')}</strong>
        </article>
        <article>
          <span>Realized PnL</span>
          <strong>{String(payload.json_report.realized_pnl ?? 'n/a')}</strong>
        </article>
      </div>

      <section className="section-card">
        <div className="section-header">
          <div>
            <p className="eyebrow">Cutover gate</p>
            <h3>Launcher cutover is complete</h3>
          </div>
          <div className="metric-badge">{payload.cutover_gate.default_launcher}</div>
        </div>
        <ul className="guidance-list">
          {(payload.cutover_gate.evidence ?? []).map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
      </section>

      <div className="card-grid">
        <article className="feature-card">
          <div className="feature-header">
            <h4>JSON Preview</h4>
            <span className="status-pill status-available">live</span>
          </div>
          <pre className="code-block">{JSON.stringify(payload.json_report, null, 2)}</pre>
        </article>
        <article className="feature-card">
          <div className="feature-header">
            <h4>Markdown Preview</h4>
            <span className="status-pill status-available">live</span>
          </div>
          <pre className="code-block">{payload.markdown_report}</pre>
        </article>
      </div>
    </div>
  );
}
