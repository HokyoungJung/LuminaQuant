'use client';

import { useMemo } from 'react';

import { PageContextBar } from '@/components/page-context-bar';
import { SurfaceState } from '@/components/surface-state';
import type { OptimizationInsightsPayload } from '@/lib/dashboard-contracts';
import { formatCompactTimestamp, formatMetricValue, formatRatio, pnlClass } from '@/lib/format';
import { useBridgeFetch } from '@/lib/use-bridge-fetch';

type Candidate = NonNullable<OptimizationInsightsPayload['best_candidate']>;

const OVERFIT_GAP_THRESHOLD = 0.5;
const STRATEGY_PARAM_KEYS = ['strategy', 'strategy_name', 'strategy_class'];

function trainOosGap(candidate: Candidate): number | null {
  if (candidate.train_sharpe === null || candidate.sharpe === null) {
    return null;
  }
  return candidate.train_sharpe - candidate.sharpe;
}

function hasLargeGap(candidate: Candidate): boolean {
  const gap = trainOosGap(candidate);
  return gap !== null && gap > OVERFIT_GAP_THRESHOLD;
}

function paramText(value: unknown): string {
  if (value === null || value === undefined) {
    return 'n/a';
  }
  if (typeof value === 'boolean') {
    return value ? 'yes' : 'no';
  }
  if (typeof value === 'number' || typeof value === 'string') {
    return formatMetricValue(value);
  }
  return JSON.stringify(value);
}

function strategyName(params: Record<string, unknown>): string | null {
  for (const key of STRATEGY_PARAM_KEYS) {
    const raw = params[key];
    if (typeof raw === 'string' && raw !== '') {
      return raw;
    }
  }
  return null;
}

function paramEntries(params: Record<string, unknown>): Array<[string, unknown]> {
  return Object.entries(params).filter(([key]) => !STRATEGY_PARAM_KEYS.includes(key));
}

function paramsSummary(params: Record<string, unknown>, maxEntries = 3): string {
  const entries = paramEntries(params);
  if (entries.length === 0) {
    return '';
  }
  const shown = entries
    .slice(0, maxEntries)
    .map(([key, value]) => `${key}=${paramText(value)}`)
    .join(' · ');
  return entries.length > maxEntries ? `${shown} · …` : shown;
}

/** Thin inline comparison bar; width scales against the column max. */
function InlineBar({ value, max, color }: { value: number | null; max: number; color: string }) {
  if (value === null || !Number.isFinite(value) || max <= 0) {
    return null;
  }
  const widthPct = Math.max(0, Math.min(1, value / max)) * 100;
  return (
    <div aria-hidden className="inline-bar" style={{ marginTop: '0.3rem' }}>
      <div className="inline-bar-fill" style={{ width: `${widthPct}%`, background: color }} />
    </div>
  );
}

function GapChip({ candidate }: { candidate: Candidate }) {
  const gap = trainOosGap(candidate);
  if (gap === null || gap <= OVERFIT_GAP_THRESHOLD) {
    return null;
  }
  return (
    <span
      className="status-pill status-degraded"
      title={`Train Sharpe exceeds out-of-sample Sharpe by ${formatRatio(gap)} — possible overfit.`}
    >
      large train/OOS gap
    </span>
  );
}

function BestCandidateCard({ candidate }: { candidate: Candidate }) {
  const strategy = strategyName(candidate.params);
  const entries = paramEntries(candidate.params);
  return (
    <article className="feature-card">
      <div className="feature-header">
        <h4>Best candidate</h4>
        <span className="status-pill status-running">{candidate.stage || 'unstaged'}</span>
      </div>
      <p>
        {strategy ? <strong>{strategy}</strong> : 'Top-ranked parameter set'}{' '}
        <code className="run-chip">{candidate.run_id || 'n/a'}</code>
        {candidate.created_at ? <span> · {formatCompactTimestamp(candidate.created_at)}</span> : null}
      </p>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1.5rem', alignItems: 'flex-end' }}>
        <div>
          <span style={{ display: 'block', fontSize: '0.78rem', textTransform: 'uppercase' }}>OOS Sharpe</span>
          <strong style={{ fontSize: '2.1rem', lineHeight: 1.1, fontVariantNumeric: 'tabular-nums' }}>
            {formatRatio(candidate.sharpe)}
          </strong>
        </div>
        <div>
          <span style={{ display: 'block', fontSize: '0.78rem', textTransform: 'uppercase' }}>Train Sharpe</span>
          <strong style={{ fontSize: '1.3rem', fontVariantNumeric: 'tabular-nums' }}>
            {formatRatio(candidate.train_sharpe)}
          </strong>
        </div>
        <GapChip candidate={candidate} />
      </div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '1.5rem' }}>
        <div>
          <span style={{ display: 'block', fontSize: '0.78rem', textTransform: 'uppercase' }}>CAGR</span>
          <strong className={pnlClass(candidate.cagr)} style={{ fontVariantNumeric: 'tabular-nums' }}>
            {formatMetricValue(candidate.cagr, { key: 'cagr' })}
          </strong>
        </div>
        <div>
          <span style={{ display: 'block', fontSize: '0.78rem', textTransform: 'uppercase' }}>Max drawdown</span>
          <strong
            className={pnlClass(candidate.mdd === null ? null : -Math.abs(candidate.mdd))}
            style={{ fontVariantNumeric: 'tabular-nums' }}
          >
            {formatMetricValue(candidate.mdd, { key: 'mdd' })}
          </strong>
        </div>
        <div>
          <span style={{ display: 'block', fontSize: '0.78rem', textTransform: 'uppercase' }}>Robustness</span>
          <strong style={{ fontVariantNumeric: 'tabular-nums' }}>
            {formatMetricValue(candidate.robustness_score, { key: 'robustness_score' })}
          </strong>
        </div>
      </div>
      {entries.length > 0 ? (
        <dl
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fill, minmax(10rem, 1fr))',
            gap: '0.35rem 1rem',
            margin: 0,
          }}
        >
          {entries.map(([key, value]) => (
            <div key={key} style={{ display: 'flex', gap: '0.4rem', fontSize: '0.85rem' }}>
              <dt style={{ opacity: 0.7 }}>{key}</dt>
              <dd style={{ margin: 0, fontVariantNumeric: 'tabular-nums' }}>{paramText(value)}</dd>
            </div>
          ))}
        </dl>
      ) : null}
    </article>
  );
}

export function OptimizationInsightsRuntime() {
  const { payload, error, loading, refetch, lastFetchedAt } = useBridgeFetch<OptimizationInsightsPayload>(
    '/api/python/dashboard/optimization-insights',
    'optimization insights request failed',
  );

  const rankedCandidates = useMemo(() => {
    const rows = payload?.top_candidates ?? [];
    return [...rows].sort((a, b) => {
      const left = a.sharpe ?? Number.NEGATIVE_INFINITY;
      const right = b.sharpe ?? Number.NEGATIVE_INFINITY;
      return right - left;
    });
  }, [payload]);

  const maxSharpe = useMemo(
    () => Math.max(0, ...rankedCandidates.map((row) => row.sharpe ?? 0)),
    [rankedCandidates],
  );
  const maxRobustness = useMemo(
    () => Math.max(0, ...rankedCandidates.map((row) => row.robustness_score ?? 0)),
    [rankedCandidates],
  );

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
      <SurfaceState
        status={payload?.status}
        error={error || null}
        surface="optimization insights"
        onRetry={refetch}
      />
      {payload === null && !error ? <p>Loading optimization results…</p> : null}
      {payload !== null && payload.status === 'ok' ? (
        <>
          <div className="metric-grid">
            {payload.summary_metrics.map((metric) => (
              <article key={metric.key}>
                <span>{metric.label}</span>
                <strong>{formatMetricValue(metric.value, { key: metric.key, label: metric.label })}</strong>
              </article>
            ))}
          </div>

          <div className="card-grid">
            {payload.best_candidate ? <BestCandidateCard candidate={payload.best_candidate} /> : null}
            <article className="feature-card">
              <div className="feature-header">
                <h4>Stage medians</h4>
                <span className="status-pill status-running">{payload.stage_breakdown.length} stages</span>
              </div>
              <div className="table-wrap">
                <table>
                  <thead>
                    <tr>
                      <th>Stage</th>
                      <th>Candidates</th>
                      <th>Median Sharpe</th>
                      <th>Median robustness</th>
                    </tr>
                  </thead>
                  <tbody>
                    {payload.stage_breakdown.map((stage) => (
                      <tr key={stage.stage}>
                        <td>{stage.stage}</td>
                        <td>{formatMetricValue(stage.count, { key: 'count' })}</td>
                        <td>{formatMetricValue(stage.median_sharpe, { key: 'median_sharpe' })}</td>
                        <td>{formatMetricValue(stage.median_robustness, { key: 'median_robustness' })}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </article>
          </div>

          <div className="table-wrap">
            <table style={{ fontVariantNumeric: 'tabular-nums' }}>
              <thead>
                <tr>
                  <th>Candidate</th>
                  <th>Stage</th>
                  <th>OOS Sharpe</th>
                  <th>Train Sharpe</th>
                  <th>Robustness</th>
                  <th>CAGR</th>
                  <th>MDD</th>
                  <th>Created</th>
                </tr>
              </thead>
              <tbody>
                {rankedCandidates.length > 0 ? (
                  rankedCandidates.map((candidate, index) => {
                    const strategy = strategyName(candidate.params);
                    const summary = paramsSummary(candidate.params);
                    const largeGap = hasLargeGap(candidate);
                    const gap = trainOosGap(candidate);
                    return (
                      <tr key={`${candidate.run_id}-${candidate.stage}-${candidate.created_at ?? index}`}>
                        <td>
                          <div>
                            {strategy ? <strong>{strategy}</strong> : null}{' '}
                            <code className="run-chip">{candidate.run_id || 'n/a'}</code>
                          </div>
                          {summary ? <div style={{ fontSize: '0.8rem', opacity: 0.75 }}>{summary}</div> : null}
                        </td>
                        <td>
                          <span className="status-pill status-running">{candidate.stage || 'unstaged'}</span>
                        </td>
                        <td>
                          {formatRatio(candidate.sharpe)}
                          <InlineBar value={candidate.sharpe} max={maxSharpe} color="var(--chart-1)" />
                        </td>
                        <td
                          style={largeGap ? { color: 'var(--warn)', fontWeight: 700 } : undefined}
                          title={
                            largeGap && gap !== null
                              ? `Train Sharpe exceeds out-of-sample Sharpe by ${formatRatio(gap)} — possible overfit.`
                              : undefined
                          }
                        >
                          {formatRatio(candidate.train_sharpe)}
                          {largeGap ? ' ⚠' : ''}
                        </td>
                        <td>
                          {formatMetricValue(candidate.robustness_score, { key: 'robustness_score' })}
                          <InlineBar value={candidate.robustness_score} max={maxRobustness} color="var(--chart-2)" />
                        </td>
                        <td className={pnlClass(candidate.cagr)}>
                          {formatMetricValue(candidate.cagr, { key: 'cagr' })}
                        </td>
                        <td className={pnlClass(candidate.mdd === null ? null : -Math.abs(candidate.mdd))}>
                          {formatMetricValue(candidate.mdd, { key: 'mdd' })}
                        </td>
                        <td>{formatCompactTimestamp(candidate.created_at)}</td>
                      </tr>
                    );
                  })
                ) : (
                  <tr>
                    <td colSpan={8}>No candidate rows recorded for this run yet.</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </>
      ) : null}
    </div>
  );
}
