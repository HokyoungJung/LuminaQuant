'use client';

import { PageContextBar } from '@/components/page-context-bar';
import { SurfaceState } from '@/components/surface-state';
import type { AlphaEvidencePayload } from '@/lib/dashboard-contracts';
import { formatCompactTimestamp, formatMetricValue } from '@/lib/format';
import { useBridgeFetch } from '@/lib/use-bridge-fetch';

type EvidenceRecord = Record<string, unknown>;

const MAX_TABLE_ROWS = 20;
const MAX_TABLE_COLUMNS = 8;
const TIMESTAMP_KEY_PATTERN = /(_at|_time|timestamp|date)$/;
const ISO_TIMESTAMP_PATTERN = /^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}/;

/** Identity-style keys pinned to the front of adaptive tables when present. */
const PRIORITY_COLUMN_KEYS = [
  'alpha_id',
  'alpha',
  'name',
  'strategy',
  'classification',
  'run_id',
  'status',
];

function labelFromKey(key: string): string {
  return key.replace(/_/g, ' ');
}

function isScalar(value: unknown): value is string | number | boolean | null {
  return (
    value === null ||
    typeof value === 'string' ||
    typeof value === 'number' ||
    typeof value === 'boolean'
  );
}

/** Format an unknown payload value using the shared metric formatters. */
function formatUnknownValue(key: string, value: unknown): string {
  if (value === null || value === undefined || value === '') {
    return 'n/a';
  }
  if (typeof value === 'boolean') {
    return value ? 'yes' : 'no';
  }
  if (typeof value === 'number') {
    return formatMetricValue(value, { key });
  }
  if (typeof value === 'string') {
    if (TIMESTAMP_KEY_PATTERN.test(key.toLowerCase()) || ISO_TIMESTAMP_PATTERN.test(value)) {
      return formatCompactTimestamp(value);
    }
    return value;
  }
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

/**
 * Pick up to MAX_TABLE_COLUMNS keys for an adaptive record table: identity
 * keys first, then most common scalar-valued keys; keys whose values are
 * never scalar (nested objects/arrays) stay in the per-row details list.
 */
function selectColumns(rows: EvidenceRecord[]): string[] {
  const scalarCounts = new Map<string, number>();
  const firstSeen: string[] = [];
  for (const row of rows) {
    for (const [key, value] of Object.entries(row)) {
      if (!scalarCounts.has(key)) {
        scalarCounts.set(key, 0);
        firstSeen.push(key);
      }
      if (isScalar(value)) {
        scalarCounts.set(key, (scalarCounts.get(key) ?? 0) + 1);
      }
    }
  }
  const candidates = firstSeen.filter((key) => (scalarCounts.get(key) ?? 0) > 0);
  candidates.sort((a, b) => {
    const priorityA = PRIORITY_COLUMN_KEYS.indexOf(a);
    const priorityB = PRIORITY_COLUMN_KEYS.indexOf(b);
    const rankA = priorityA === -1 ? PRIORITY_COLUMN_KEYS.length : priorityA;
    const rankB = priorityB === -1 ? PRIORITY_COLUMN_KEYS.length : priorityB;
    if (rankA !== rankB) {
      return rankA - rankB;
    }
    const countA = scalarCounts.get(a) ?? 0;
    const countB = scalarCounts.get(b) ?? 0;
    if (countA !== countB) {
      return countB - countA;
    }
    return firstSeen.indexOf(a) - firstSeen.indexOf(b);
  });
  return candidates.slice(0, MAX_TABLE_COLUMNS);
}

function extraEntries(row: EvidenceRecord, columns: string[]): Array<[string, unknown]> {
  return Object.entries(row).filter(([key]) => !columns.includes(key));
}

interface AdaptiveRecordTableProps {
  rows: EvidenceRecord[];
  emptyLabel: string;
}

/**
 * Schema-less record table: union of keys becomes columns (capped), the
 * remaining fields of each row fold into a per-row details disclosure.
 */
function AdaptiveRecordTable({ rows, emptyLabel }: AdaptiveRecordTableProps) {
  const visibleRows = rows.slice(0, MAX_TABLE_ROWS);
  const columns = selectColumns(visibleRows);
  const hasDetails = visibleRows.some((row) => extraEntries(row, columns).length > 0);

  if (rows.length === 0) {
    return <p>{emptyLabel}</p>;
  }

  return (
    <>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              {columns.map((column) => (
                <th key={column}>{labelFromKey(column)}</th>
              ))}
              {hasDetails ? <th>more</th> : null}
            </tr>
          </thead>
          <tbody>
            {visibleRows.map((row, index) => {
              const extras = extraEntries(row, columns);
              return (
                <tr key={index}>
                  {columns.map((column) => (
                    <td key={column}>{formatUnknownValue(column, row[column])}</td>
                  ))}
                  {hasDetails ? (
                    <td>
                      {extras.length > 0 ? (
                        <details>
                          <summary>{extras.length} more</summary>
                          <dl>
                            {extras.map(([key, value]) => (
                              <div key={key}>
                                <dt>{labelFromKey(key)}</dt>
                                <dd>{formatUnknownValue(key, value)}</dd>
                              </div>
                            ))}
                          </dl>
                        </details>
                      ) : null}
                    </td>
                  ) : null}
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
      {rows.length > MAX_TABLE_ROWS ? (
        <p className="context-bar-muted">
          Showing {MAX_TABLE_ROWS} of {rows.length} records.
        </p>
      ) : null}
    </>
  );
}

function AlphaEvidenceBody({ payload }: { payload: AlphaEvidencePayload }) {
  const { summary } = payload;
  const classifications = Object.entries(summary.classifications ?? {});
  const realityGates = Object.entries(summary.reality_gates ?? {});
  const liveReadiness = Object.entries(payload.live_readiness ?? {});
  const action = summary.live_readiness_action;

  return (
    <>
      <section className="section-card">
        <div className="section-header">
          <div>
            <p className="eyebrow">Readiness verdict</p>
            <h3>Real-money status</h3>
          </div>
        </div>
        <div className="button-row">
          {payload.advisory_only ? (
            <span className="status-pill status-running">advisory only</span>
          ) : null}
          <span
            className={`status-pill ${
              payload.real_money_execution_enabled ? 'status-ok' : 'status-neutral'
            }`}
          >
            {payload.real_money_execution_enabled
              ? 'real-money execution ENABLED'
              : 'real-money execution disabled'}
          </span>
        </div>
        {action ? (
          <p className="lede">
            <strong>Recommended action:</strong> {action}
          </p>
        ) : null}
        <div className="metric-grid">
          <article>
            <span>Alphas tracked</span>
            <strong>{formatMetricValue(summary.alpha_count, { key: 'alpha_count' })}</strong>
          </article>
          <article>
            <span>Run cards</span>
            <strong>{formatMetricValue(summary.run_card_count, { key: 'run_card_count' })}</strong>
          </article>
          <article>
            <span>Classifications</span>
            {classifications.length > 0 ? (
              <strong>
                {classifications.map(([name, count]) => (
                  <span key={name} className="code-chip">
                    {labelFromKey(name)} · {formatMetricValue(count, { key: 'count' })}{' '}
                  </span>
                ))}
              </strong>
            ) : (
              <strong>n/a</strong>
            )}
          </article>
        </div>
      </section>

      <section className="section-card">
        <div className="section-header">
          <div>
            <p className="eyebrow">Reality gates</p>
            <h3>Which gates hold across every run card?</h3>
          </div>
        </div>
        {realityGates.length > 0 ? (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Gate</th>
                  <th>Result</th>
                  <th>Observations</th>
                </tr>
              </thead>
              <tbody>
                {realityGates.map(([gate, result]) => (
                  <tr key={gate}>
                    <td>{labelFromKey(gate)}</td>
                    <td>
                      <span
                        className={`status-pill ${result.passed ? 'status-ok' : 'status-failed'}`}
                      >
                        {result.passed ? 'passed' : 'failed'}
                      </span>
                    </td>
                    <td>{formatMetricValue(result.observations, { key: 'count' })}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p>No reality-gate results recorded yet — run cards with gate outcomes populate this table.</p>
        )}
      </section>

      <section className="section-card">
        <div className="section-header">
          <div>
            <p className="eyebrow">Evidence</p>
            <h3>Alpha classification evidence</h3>
          </div>
        </div>
        <AdaptiveRecordTable
          rows={payload.evidence}
          emptyLabel="No alpha evidence recorded yet — classification runs populate this table."
        />
      </section>

      <section className="section-card">
        <div className="section-header">
          <div>
            <p className="eyebrow">Run cards</p>
            <h3>Per-run gate and outcome records</h3>
          </div>
        </div>
        <AdaptiveRecordTable
          rows={payload.run_cards}
          emptyLabel="No run cards recorded yet — completed runs with gate outcomes populate this table."
        />
      </section>

      <section className="section-card">
        <div className="section-header">
          <div>
            <p className="eyebrow">Live readiness</p>
            <h3>Latest live-readiness assessment</h3>
          </div>
        </div>
        {liveReadiness.length > 0 ? (
          <dl>
            {liveReadiness.map(([key, value]) => (
              <div key={key}>
                <dt>
                  <strong>{labelFromKey(key)}</strong>
                </dt>
                <dd>{formatUnknownValue(key, value)}</dd>
              </div>
            ))}
          </dl>
        ) : (
          <p>No live-readiness assessment recorded yet.</p>
        )}
      </section>
    </>
  );
}

export function AlphaEvidenceRuntime() {
  const { payload, error, loading, refetch, lastFetchedAt } = useBridgeFetch<AlphaEvidencePayload>(
    '/api/python/dashboard/alpha-evidence',
    'alpha evidence request failed',
  );

  const sourceStatus = payload?.source?.status ?? null;

  return (
    <div className="page-stack">
      <PageContextBar
        asOf={payload?.as_of}
        status={sourceStatus}
        lastFetchedAt={lastFetchedAt}
        onRefresh={refetch}
        loading={loading}
      />
      <SurfaceState
        status={sourceStatus}
        error={error || null}
        surface="alpha evidence"
        onRetry={refetch}
      />
      {payload === null && !error ? <p>Loading alpha evidence…</p> : null}
      {payload !== null ? <AlphaEvidenceBody payload={payload} /> : null}
    </div>
  );
}
