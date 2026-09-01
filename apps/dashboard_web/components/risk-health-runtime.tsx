'use client';

import { PageContextBar } from '@/components/page-context-bar';
import { SurfaceState } from '@/components/surface-state';
import type { RiskHealthPayload } from '@/lib/dashboard-contracts';
import { formatCompactTimestamp, formatNumber } from '@/lib/format';
import { useBridgeFetch } from '@/lib/use-bridge-fetch';

const RISK_EVENT_CELL_STYLE = {
  borderLeft: '3px solid var(--danger)',
} as const;

const DANGER_TILE_STYLE = {
  borderColor: 'var(--danger)',
  background: 'rgba(255, 122, 118, 0.08)',
} as const;

function statusPillClass(value: string | null | undefined): string {
  const slug = (value ?? '')
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9_-]+/g, '-');
  return `status-pill status-${slug || 'unknown'}`;
}

export function RiskHealthRuntime() {
  const { payload, error, loading, refetch, lastFetchedAt } = useBridgeFetch<RiskHealthPayload>(
    '/api/python/dashboard/risk-health',
    'risk & health telemetry failed',
  );

  const status = payload?.status ?? null;
  const riskEventCount = payload?.summary.risk_event_count ?? 0;

  return (
    <div className="page-stack">
      <PageContextBar
        asOf={payload?.as_of}
        runId={payload?.run_id || null}
        status={status}
        lastFetchedAt={lastFetchedAt}
        onRefresh={refetch}
        loading={loading}
      />
      <SurfaceState status={status} error={error} surface="risk & health telemetry" onRetry={refetch} />
      {payload === null && !error ? <p>Loading risk and health telemetry…</p> : null}
      {payload !== null && status === 'ok' ? (
        <>
          <div className="metric-grid">
            <article style={riskEventCount > 0 ? DANGER_TILE_STYLE : undefined}>
              <span>Risk Events</span>
              <strong>{formatNumber(riskEventCount, { digits: 0 })}</strong>
            </article>
            <article>
              <span>Heartbeats</span>
              <strong>{formatNumber(payload.summary.heartbeat_count, { digits: 0 })}</strong>
            </article>
            <article>
              <span>Order States</span>
              <strong>{formatNumber(payload.summary.order_state_count, { digits: 0 })}</strong>
            </article>
          </div>

          <section>
            <h3>Risk events</h3>
            {payload.risk_events.length === 0 ? (
              <p>No risk events for this run.</p>
            ) : (
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Time</th>
                      <th>Reason</th>
                    </tr>
                  </thead>
                  <tbody>
                    {payload.risk_events.map((event, index) => (
                      <tr key={`${event.event_time}-${index}`}>
                        <td style={RISK_EVENT_CELL_STYLE}>
                          {formatCompactTimestamp(event.event_time)}
                        </td>
                        <td>{event.reason || 'n/a'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>

          <section>
            <h3>Heartbeats</h3>
            {payload.heartbeats.length === 0 ? (
              <p>No heartbeats recorded for this run.</p>
            ) : (
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Time</th>
                      <th>Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    {payload.heartbeats.map((beat, index) => (
                      <tr key={`${beat.heartbeat_time}-${index}`}>
                        <td>{formatCompactTimestamp(beat.heartbeat_time)}</td>
                        <td>
                          <span className={statusPillClass(beat.status)}>
                            {beat.status || 'unknown'}
                          </span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>

          <section>
            <h3>Order states</h3>
            {payload.order_states.length === 0 ? (
              <p>No order state transitions recorded for this run.</p>
            ) : (
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Time</th>
                      <th>Symbol</th>
                      <th>State</th>
                      <th>Message</th>
                    </tr>
                  </thead>
                  <tbody>
                    {payload.order_states.map((order, index) => (
                      <tr key={`${order.event_time}-${order.symbol}-${index}`}>
                        <td>{formatCompactTimestamp(order.event_time)}</td>
                        <td>{order.symbol || 'n/a'}</td>
                        <td>
                          <span className={statusPillClass(order.state)}>
                            {order.state || 'unknown'}
                          </span>
                        </td>
                        <td>{order.message || 'n/a'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>
        </>
      ) : null}
    </div>
  );
}
