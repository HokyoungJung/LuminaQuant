'use client';

import type { FactorInsightsPayload } from '@/lib/dashboard-contracts';
import { useBridgeFetch } from '@/lib/use-bridge-fetch';

function formatCell(value: number | null): string {
  return value === null ? 'n/a' : value.toFixed(3);
}

export function FactorInsightsRuntime() {
  const { payload, error } = useBridgeFetch<FactorInsightsPayload>(
    '/api/python/dashboard/factor-insights',
    'factor insights bridge failed',
  );

  if (error) {
    return <p>{error}</p>;
  }
  if (payload === null) {
    return <p>Loading factor IC heatmap and candidate queue…</p>;
  }
  if (payload.status !== 'ok') {
    return <p>No factor insights payload available yet.</p>;
  }

  const { ic_heatmap: heatmap, candidate_queue: queue } = payload;

  return (
    <div className="page-stack">
      <div className="metric-grid">
        <article>
          <span>Factors</span>
          <strong>{payload.summary.factor_count}</strong>
        </article>
        <article>
          <span>Candidates</span>
          <strong>{payload.summary.candidate_count}</strong>
        </article>
        <article>
          <span>Top Factor</span>
          <strong>{payload.summary.top_factor ?? 'n/a'}</strong>
        </article>
      </div>

      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Factor / rank-decay IC</th>
              {heatmap.lags.map((lag) => (
                <th key={lag}>{lag}</th>
              ))}
              <th>IC-IR</th>
            </tr>
          </thead>
          <tbody>
            {heatmap.factors.map((factor, rowIndex) => (
              <tr key={factor}>
                <td>{factor}</td>
                {heatmap.cells[rowIndex].map((cell, cellIndex) => (
                  <td key={`${factor}-${heatmap.lags[cellIndex] ?? cellIndex}`}>
                    {formatCell(cell)}
                  </td>
                ))}
                <td>{formatCell(heatmap.ic_ir[factor] ?? null)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Candidate</th>
              <th>Strategy</th>
              <th>Status</th>
              <th>Score</th>
            </tr>
          </thead>
          <tbody>
            {queue.slice(0, 10).map((candidate) => (
              <tr key={candidate.candidate_id}>
                <td>{candidate.candidate_id}</td>
                <td>{candidate.strategy}</td>
                <td>{candidate.status}</td>
                <td>{formatCell(candidate.score)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
