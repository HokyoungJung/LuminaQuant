'use client';

import { useState } from 'react';

import { PageContextBar } from '@/components/page-context-bar';
import { RunSelector } from '@/components/run-selector';
import { SurfaceState } from '@/components/surface-state';
import type { ReportExportPayload } from '@/lib/dashboard-contracts';
import {
  formatCompactTimestamp,
  formatCurrency,
  formatMetricValue,
  pnlClass,
  statusPillClass,
} from '@/lib/format';
import { useBridgeFetch } from '@/lib/use-bridge-fetch';

const ENDPOINT = '/api/python/dashboard/report-export';
const PREVIEW_MAX_HEIGHT = '26rem';

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
  const [selectedRunId, setSelectedRunId] = useState('');
  const url = selectedRunId ? `${ENDPOINT}?run_id=${encodeURIComponent(selectedRunId)}` : ENDPOINT;
  const { payload, error, loading, refetch, lastFetchedAt } = useBridgeFetch<ReportExportPayload>(
    url,
    'report export request failed',
  );

  const runs = payload?.runs ?? [];
  const report = payload?.json_report;
  const reportStatusPill = report?.status ? (
    <span className={statusPillClass(report.status)}>{report.status}</span>
  ) : null;
  const totalReturn = report?.total_return;
  const latestEquity = report?.latest_equity;

  return (
    <div className="page-stack">
      <PageContextBar
        asOf={payload?.as_of}
        runId={payload?.run_id}
        status={payload?.status}
        lastFetchedAt={lastFetchedAt}
        onRefresh={refetch}
        loading={loading}
      />
      {runs.length > 0 ? (
        <RunSelector
          runs={runs}
          value={selectedRunId || payload?.run_id || ''}
          onChange={setSelectedRunId}
        />
      ) : null}
      <SurfaceState
        status={payload?.status}
        error={error || null}
        surface="report export"
        onRetry={refetch}
      />
      {payload === null && !error ? <p>Loading report snapshot…</p> : null}

      {payload !== null && payload.status === 'ok' ? (
        <>
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
              Download JSON snapshot
            </button>
            <button
              className="action-button"
              onClick={() =>
                downloadTextFile(payload.filenames.markdown, payload.markdown_report, 'text/markdown')
              }
              type="button"
            >
              Download Markdown snapshot
            </button>
          </div>

          <div className="metric-grid">
            <article>
              <span>Run</span>
              <strong>{report?.run_id ?? 'n/a'}</strong>
            </article>
            <article>
              <span>Strategy</span>
              <strong>{report?.strategy ?? 'unknown'}</strong>
            </article>
            <article>
              <span>Mode</span>
              <strong>{report?.mode ?? 'n/a'}</strong>
            </article>
            <article>
              <span>Period</span>
              <strong>
                {formatCompactTimestamp(report?.period_start)} →{' '}
                {formatCompactTimestamp(report?.period_end)}
              </strong>
            </article>
            <article>
              <span>Total return</span>
              <strong className={typeof totalReturn === 'number' ? pnlClass(totalReturn) || undefined : undefined}>
                {formatMetricValue(totalReturn, { key: 'total_return' })}
              </strong>
            </article>
            <article>
              <span>Latest equity</span>
              <strong>{formatMetricValue(latestEquity, { key: 'latest_equity' })}</strong>
            </article>
            <article>
              <span>Realized PnL</span>
              <strong className={pnlClass(report?.realized_pnl) || undefined}>
                {formatCurrency(report?.realized_pnl)}
              </strong>
            </article>
            <article>
              <span>Closed trades</span>
              <strong>
                {formatMetricValue(report?.closed_trade_count, { key: 'closed_trade_count' })}
              </strong>
            </article>
          </div>

          <div className="card-grid">
            <article className="feature-card">
              <div className="feature-header">
                <h4>JSON preview</h4>
                {reportStatusPill}
              </div>
              <pre className="code-block" style={{ maxHeight: PREVIEW_MAX_HEIGHT, overflow: 'auto' }}>
                {JSON.stringify(payload.json_report, null, 2)}
              </pre>
            </article>
            <article className="feature-card">
              <div className="feature-header">
                <h4>Markdown preview</h4>
                {reportStatusPill}
              </div>
              <pre className="code-block" style={{ maxHeight: PREVIEW_MAX_HEIGHT, overflow: 'auto' }}>
                {payload.markdown_report}
              </pre>
            </article>
          </div>
        </>
      ) : null}
    </div>
  );
}
