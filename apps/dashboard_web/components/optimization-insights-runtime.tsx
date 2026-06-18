'use client';

import type { OptimizationInsightsPayload } from '@/lib/dashboard-contracts';
import { formatMetricValue } from '@/lib/format';
import { useBridgeFetch } from '@/lib/use-bridge-fetch';

export function OptimizationInsightsRuntime() {
  const { payload, error } = useBridgeFetch<OptimizationInsightsPayload>(
    '/api/python/dashboard/optimization-insights',
    'optimization insights bridge failed',
  );

  if (error) {
    return <p>{error}</p>;
  }
  if (payload === null) {
    return <p>Loading optimization insights parity payload…</p>;
  }
  if (payload.status !== 'ok' && payload.status !== 'no_optimization_results') {
    return <p>No optimization insights payload available yet.</p>;
  }

  return (
    <div className="page-stack">
      <div className="metric-grid">
        {payload.summary_metrics.map((metric) => (
          <article key={metric.key}>
            <span>{metric.label}</span>
            <strong>{formatMetricValue(metric.value)}</strong>
          </article>
        ))}
      </div>

      <div className="card-grid">
        <article className="feature-card">
          <div className="feature-header">
            <h4>Best Candidate</h4>
            <span className="status-pill status-available">{payload.best_candidate?.stage ?? 'n/a'}</span>
          </div>
          <pre className="code-block">{JSON.stringify(payload.best_candidate ?? {}, null, 2)}</pre>
        </article>
        <article className="feature-card">
          <div className="feature-header">
            <h4>Stage Breakdown</h4>
            <span className="status-pill status-available">{payload.stage_breakdown.length} stages</span>
          </div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Stage</th>
                  <th>Count</th>
                  <th>Median Sharpe</th>
                  <th>Median Robustness</th>
                </tr>
              </thead>
              <tbody>
                {payload.stage_breakdown.map((stage) => (
                  <tr key={stage.stage}>
                    <td>{stage.stage}</td>
                    <td>{stage.count}</td>
                    <td>{formatMetricValue(stage.median_sharpe)}</td>
                    <td>{formatMetricValue(stage.median_robustness)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </article>
      </div>

      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Created</th>
              <th>Run ID</th>
              <th>Stage</th>
              <th>Sharpe</th>
              <th>Train Sharpe</th>
              <th>Robustness</th>
              <th>CAGR</th>
              <th>MDD</th>
            </tr>
          </thead>
          <tbody>
            {payload.top_candidates.length > 0 ? (
              payload.top_candidates.map((candidate) => (
                <tr key={`${candidate.created_at}-${candidate.run_id}-${candidate.stage}`}>
                  <td>{candidate.created_at ?? 'n/a'}</td>
                  <td>{candidate.run_id}</td>
                  <td>{candidate.stage}</td>
                  <td>{formatMetricValue(candidate.sharpe)}</td>
                  <td>{formatMetricValue(candidate.train_sharpe)}</td>
                  <td>{formatMetricValue(candidate.robustness_score)}</td>
                  <td>{formatMetricValue(candidate.cagr)}</td>
                  <td>{formatMetricValue(candidate.mdd)}</td>
                </tr>
              ))
            ) : (
              <tr>
                <td colSpan={8}>No optimization rows were available yet.</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
