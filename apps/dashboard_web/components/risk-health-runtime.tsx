'use client';

import type { RiskHealthPayload } from '@/lib/dashboard-contracts';
import { useBridgeFetch } from '@/lib/use-bridge-fetch';

export function RiskHealthRuntime() {
  const { payload, error } = useBridgeFetch<RiskHealthPayload>(
    '/api/python/dashboard/risk-health',
    'risk health bridge failed',
  );

  if (error) {
    return <p>{error}</p>;
  }
  if (payload === null) {
    return <p>Loading risk and health telemetry…</p>;
  }
  if (payload.status !== 'ok') {
    return <p>No risk/health payload available yet.</p>;
  }

  return (
    <div className="page-stack">
      <div className="metric-grid">
        <article>
          <span>Risk Events</span>
          <strong>{payload.summary.risk_event_count}</strong>
        </article>
        <article>
          <span>Heartbeats</span>
          <strong>{payload.summary.heartbeat_count}</strong>
        </article>
        <article>
          <span>Order States</span>
          <strong>{payload.summary.order_state_count}</strong>
        </article>
      </div>

      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Recent Risk Events</th>
              <th>Reason</th>
            </tr>
          </thead>
          <tbody>
            {payload.risk_events.slice(0, 5).map((event, index) => (
              <tr key={`${event.event_time}-${index}`}>
                <td>{event.event_time ?? 'n/a'}</td>
                <td>{event.reason ?? 'n/a'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
