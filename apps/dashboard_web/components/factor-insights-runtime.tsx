'use client';

import type { CSSProperties } from 'react';
import { useMemo } from 'react';

import { PageContextBar } from '@/components/page-context-bar';
import { SurfaceState } from '@/components/surface-state';
import type { FactorInsightsPayload } from '@/lib/dashboard-contracts';
import { formatCompactTimestamp, formatMetricValue, formatNumber, formatRatio } from '@/lib/format';
import { useBridgeFetch } from '@/lib/use-bridge-fetch';

const QUEUE_ROW_LIMIT = 20;
const HEATMAP_MAX_ALPHA_PCT = 55;
const HEATMAP_MIN_ALPHA_PCT = 8;

const NEUTRAL_PILL_STYLE: CSSProperties = {
  background: 'rgba(148, 163, 184, 0.16)',
  color: 'inherit',
};

/**
 * Diverging cell shading around 0: positive IC leans on the gain token,
 * negative IC on the loss token, |value| drives the background alpha.
 * Text color is left untouched so numbers stay in the text tokens.
 */
function heatmapCellStyle(value: number | null, maxAbs: number): CSSProperties {
  const base: CSSProperties = { fontVariantNumeric: 'tabular-nums', textAlign: 'right' };
  if (value === null || !Number.isFinite(value) || value === 0 || maxAbs <= 0) {
    return base;
  }
  const intensity = Math.min(1, Math.abs(value) / maxAbs);
  const alphaPct = Math.round(
    HEATMAP_MIN_ALPHA_PCT + intensity * (HEATMAP_MAX_ALPHA_PCT - HEATMAP_MIN_ALPHA_PCT),
  );
  const token = value > 0 ? '--pnl-pos' : '--pnl-neg';
  return {
    ...base,
    background: `color-mix(in srgb, var(${token}) ${alphaPct}%, transparent)`,
  };
}

function candidateStatusPill(status: string) {
  const normalized = status.toLowerCase();
  if (normalized === 'promoted' || normalized === 'approved') {
    return <span className="status-pill status-ok">{status}</span>;
  }
  if (normalized === 'rejected' || normalized === 'failed') {
    return <span className="status-pill status-failed">{status}</span>;
  }
  if (normalized === 'queued' || normalized === 'pending' || normalized === 'running') {
    return <span className="status-pill status-running">{status}</span>;
  }
  return (
    <span className="status-pill" style={NEUTRAL_PILL_STYLE}>
      {status}
    </span>
  );
}

function HeatmapLegend() {
  return (
    <div className="chart-legend" aria-hidden>
      <span className="legend-chip">
        <span
          className="legend-swatch"
          style={{ background: `color-mix(in srgb, var(--pnl-neg) ${HEATMAP_MAX_ALPHA_PCT}%, transparent)` }}
        />
        −
      </span>
      <span className="legend-chip">
        <span className="legend-swatch" style={{ background: 'transparent', border: '1px solid var(--grid)' }} />
        0
      </span>
      <span className="legend-chip">
        <span
          className="legend-swatch"
          style={{ background: `color-mix(in srgb, var(--pnl-pos) ${HEATMAP_MAX_ALPHA_PCT}%, transparent)` }}
        />
        +
      </span>
    </div>
  );
}

export function FactorInsightsRuntime() {
  const { payload, error, loading, refetch, lastFetchedAt } = useBridgeFetch<FactorInsightsPayload>(
    '/api/python/dashboard/factor-insights',
    'factor insights request failed',
  );

  const maxAbsCell = useMemo(() => {
    const cells = payload?.ic_heatmap.cells ?? [];
    let maxAbs = 0;
    for (const row of cells) {
      for (const cell of row) {
        if (cell !== null && Number.isFinite(cell)) {
          maxAbs = Math.max(maxAbs, Math.abs(cell));
        }
      }
    }
    return maxAbs;
  }, [payload]);

  const heatmap = payload?.ic_heatmap;
  const queue = payload?.candidate_queue ?? [];
  const queueRows = queue.slice(0, QUEUE_ROW_LIMIT);

  return (
    <div className="page-stack">
      <PageContextBar
        asOf={payload?.as_of}
        status={payload?.status}
        lastFetchedAt={lastFetchedAt}
        onRefresh={refetch}
        loading={loading}
      />
      {payload !== null ? (
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.5rem' }}>
          {payload.advisory_only ? (
            <span
              className="status-pill status-running"
              title="Research output only — nothing on this page places orders or changes allocations."
            >
              advisory only
            </span>
          ) : null}
          <span
            className="status-pill"
            style={NEUTRAL_PILL_STYLE}
            title={
              payload.real_money_execution_enabled
                ? 'Live execution is enabled for this environment.'
                : 'Live execution is off — this feed is informational.'
            }
          >
            live execution {payload.real_money_execution_enabled ? 'on' : 'off'}
          </span>
        </div>
      ) : null}
      <SurfaceState status={payload?.status} error={error || null} surface="factor insights" onRetry={refetch} />
      {payload === null && !error ? <p>Loading factor rankings and candidate queue…</p> : null}
      {payload !== null && payload.status === 'ok' && heatmap ? (
        <>
          <div className="metric-grid">
            <article>
              <span>Factors</span>
              <strong>{formatMetricValue(payload.summary.factor_count, { key: 'factor_count' })}</strong>
            </article>
            <article>
              <span>Candidates</span>
              <strong>{formatMetricValue(payload.summary.candidate_count, { key: 'candidate_count' })}</strong>
            </article>
            <article>
              <span>Decay lags</span>
              <strong>{formatMetricValue(payload.summary.lag_count, { key: 'lag_count' })}</strong>
            </article>
            <article>
              <span>Top factor</span>
              <strong>
                {payload.summary.top_factor ?? 'n/a'}
                {payload.summary.top_factor_ic_ir !== null ? (
                  <span style={{ fontSize: '0.8rem', opacity: 0.75 }}>
                    {' '}
                    IC-IR {formatRatio(payload.summary.top_factor_ic_ir)}
                  </span>
                ) : null}
              </strong>
            </article>
          </div>

          <div>
            <div className="feature-header">
              <h4>Rank-IC decay by lag</h4>
              <HeatmapLegend />
            </div>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Factor</th>
                    {heatmap.lags.map((lag) => (
                      <th key={lag} style={{ textAlign: 'right' }}>
                        {lag}
                      </th>
                    ))}
                    <th style={{ textAlign: 'right' }}>IC-IR</th>
                  </tr>
                </thead>
                <tbody>
                  {heatmap.factors.map((factor, rowIndex) => (
                    <tr key={factor}>
                      <td>{factor}</td>
                      {(heatmap.cells[rowIndex] ?? []).map((cell, cellIndex) => (
                        <td
                          key={`${factor}-${heatmap.lags[cellIndex] ?? cellIndex}`}
                          style={heatmapCellStyle(cell, maxAbsCell)}
                        >
                          {formatNumber(cell, { digits: 3 })}
                        </td>
                      ))}
                      <td style={{ textAlign: 'right', fontVariantNumeric: 'tabular-nums' }}>
                        {formatRatio(heatmap.ic_ir[factor] ?? null)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div>
            <div className="feature-header">
              <h4>Factor ranking</h4>
              <span className="status-pill" style={NEUTRAL_PILL_STYLE}>
                {payload.factor_ranking.length} factors
              </span>
            </div>
            <div className="table-wrap">
              <table style={{ fontVariantNumeric: 'tabular-nums' }}>
                <thead>
                  <tr>
                    <th>Factor</th>
                    <th>IC mean</th>
                    <th>IC-IR</th>
                    <th>IC positive</th>
                    <th>t-stat</th>
                    <th>Turnover</th>
                    <th>Quantile spread</th>
                    <th>Periods</th>
                  </tr>
                </thead>
                <tbody>
                  {payload.factor_ranking.map((row) => (
                    <tr key={row.factor}>
                      <td>{row.factor}</td>
                      <td>{formatNumber(row.ic_mean, { digits: 3 })}</td>
                      <td>{formatMetricValue(row.ic_ir, { key: 'ic_ir' })}</td>
                      <td>{formatMetricValue(row.ic_positive_ratio, { key: 'ic_positive_ratio' })}</td>
                      <td>{formatMetricValue(row.t_stat, { key: 't_stat' })}</td>
                      <td>{formatNumber(row.turnover_mean, { digits: 3 })}</td>
                      <td>{formatNumber(row.quantile_spread_mean, { digits: 4 })}</td>
                      <td>{formatMetricValue(row.n_periods, { key: 'n_periods' })}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div>
            <div className="feature-header">
              <h4>Candidate queue</h4>
              <span className="status-pill" style={NEUTRAL_PILL_STYLE}>
                {queue.length > queueRows.length
                  ? `top ${queueRows.length} of ${queue.length}`
                  : `${queue.length} candidates`}
              </span>
            </div>
            <div className="table-wrap">
              <table style={{ fontVariantNumeric: 'tabular-nums' }}>
                <thead>
                  <tr>
                    <th>Candidate</th>
                    <th>Strategy</th>
                    <th>Status</th>
                    <th>Score</th>
                    <th>Sharpe</th>
                    <th>Robustness</th>
                    <th>Submitted</th>
                  </tr>
                </thead>
                <tbody>
                  {queueRows.length > 0 ? (
                    queueRows.map((candidate) => (
                      <tr key={candidate.candidate_id}>
                        <td>
                          <code className="run-chip">{candidate.candidate_id}</code>
                        </td>
                        <td>{candidate.strategy}</td>
                        <td>{candidateStatusPill(candidate.status)}</td>
                        <td>{formatMetricValue(candidate.score, { key: 'score' })}</td>
                        <td>{formatMetricValue(candidate.sharpe, { key: 'sharpe' })}</td>
                        <td>{formatMetricValue(candidate.robustness_score, { key: 'robustness_score' })}</td>
                        <td>{formatCompactTimestamp(candidate.submitted_at)}</td>
                      </tr>
                    ))
                  ) : (
                    <tr>
                      <td colSpan={7}>No candidates waiting for review.</td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </>
      ) : null}
    </div>
  );
}
