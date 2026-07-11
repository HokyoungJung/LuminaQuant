'use client';

import { useMemo, useState } from 'react';

import { PageContextBar } from '@/components/page-context-bar';
import { RunSelector } from '@/components/run-selector';
import { SurfaceState } from '@/components/surface-state';
import type { RawDataPayload } from '@/lib/dashboard-contracts';
import { formatCompactTimestamp, formatMetricValue } from '@/lib/format';
import { useBridgeFetch } from '@/lib/use-bridge-fetch';

const TIME_COLUMN_PATTERN = /(time|date|_at$|^datetime$)/;

function renderCell(column: string, value: unknown): string {
  if (value === null || value === undefined || value === '') {
    return 'n/a';
  }
  if (typeof value === 'string' && TIME_COLUMN_PATTERN.test(column.toLowerCase())) {
    return formatCompactTimestamp(value);
  }
  if (typeof value === 'number') {
    return formatMetricValue(value, { key: column });
  }
  if (typeof value === 'object') {
    return JSON.stringify(value);
  }
  return String(value);
}

export function RawDataRuntime() {
  const [runId, setRunId] = useState('');

  const url = useMemo(() => {
    const query = new URLSearchParams();
    if (runId) {
      query.set('run_id', runId);
    }
    const suffix = query.toString();
    return `/api/python/dashboard/raw-data${suffix ? `?${suffix}` : ''}`;
  }, [runId]);

  const { payload, error, loading, refetch, lastFetchedAt } = useBridgeFetch<RawDataPayload>(
    url,
    'raw data request failed',
  );

  const runs = payload?.runs ?? [];
  const activeRunId = runId || payload?.run_id || '';
  const showData = payload !== null && payload.status === 'ok';

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
        <div className="button-row">
          <RunSelector runs={runs} value={activeRunId} onChange={setRunId} />
        </div>
      ) : null}
      <SurfaceState
        status={payload?.status}
        error={error || undefined}
        surface="stored run data"
        onRetry={refetch}
      />
      {payload === null && !error ? <p>Loading stored run data…</p> : null}

      {showData ? (
        <>
          <section className="section-card">
            <div className="section-header">
              <div>
                <p className="eyebrow">Stored frames</p>
                <h3>What the database holds for this run</h3>
              </div>
              <div className="metric-badge">{payload.context.source ?? 'n/a'}</div>
            </div>
            <div className="metric-grid">
              <article>
                <span>Run ID</span>
                <strong>{payload.context.run_id || 'n/a'}</strong>
              </article>
              <article>
                <span>Market</span>
                <strong>{payload.context.market || 'n/a'}</strong>
              </article>
              <article>
                <span>Frames</span>
                <strong>{payload.frame_summaries.length}</strong>
              </article>
            </div>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Frame</th>
                    <th>Rows</th>
                    <th>Columns</th>
                  </tr>
                </thead>
                <tbody>
                  {payload.frame_summaries.map((frame) => (
                    <tr key={frame.label}>
                      <td>{frame.label}</td>
                      <td>{formatMetricValue(frame.rows, { key: 'rows' })}</td>
                      <td>{formatMetricValue(frame.columns, { key: 'columns' })}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          {payload.previews.map((preview) => (
            <section key={preview.label} className="section-card">
              <div className="section-header">
                <div>
                  <p className="eyebrow">Preview</p>
                  <h3>{preview.label}</h3>
                </div>
                <div className="metric-badge">
                  {preview.rows.length} {preview.rows.length === 1 ? 'row' : 'rows'}
                </div>
              </div>
              {preview.rows.length > 0 && preview.columns.length > 0 ? (
                <div className="table-wrap">
                  <table>
                    <thead>
                      <tr>
                        {preview.columns.map((column) => (
                          <th key={column}>{column}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {preview.rows.map((row, rowIndex) => (
                        <tr key={rowIndex}>
                          {preview.columns.map((column) => (
                            <td key={column}>{renderCell(column, row[column])}</td>
                          ))}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              ) : (
                <p>No rows stored in this frame for the selected run.</p>
              )}
            </section>
          ))}
        </>
      ) : null}
    </div>
  );
}
