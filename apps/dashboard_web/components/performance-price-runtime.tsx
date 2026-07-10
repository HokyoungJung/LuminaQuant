'use client';

import { useMemo, useState } from 'react';

import {
  TimeSeriesChart,
  type TimeSeriesChartMarker,
  type TimeSeriesChartSeries,
} from '@/components/charts/time-series-chart';
import { PageContextBar } from '@/components/page-context-bar';
import { RunSelector } from '@/components/run-selector';
import { SurfaceState } from '@/components/surface-state';
import type { PerformancePricePayload } from '@/lib/dashboard-contracts';
import {
  formatCompactTimestamp,
  formatCurrency,
  formatMetricValue,
  formatNumber,
  formatPercent,
  pnlClass,
} from '@/lib/format';
import { useBridgeFetch } from '@/lib/use-bridge-fetch';

const TRADE_ROW_CAP = 30;

const PERFORMANCE_METRIC_LABELS: ReadonlyArray<
  [keyof PerformancePricePayload['performance_metrics'], string]
> = [
  ['cagr', 'CAGR'],
  ['annualized_volatility', 'Ann. Volatility'],
  ['sharpe_ratio', 'Sharpe'],
  ['sortino_ratio', 'Sortino'],
  ['calmar_ratio', 'Calmar'],
  ['max_drawdown', 'Max Drawdown'],
];

function toEpochMs(timestamp: string): number {
  return new Date(timestamp).getTime();
}

/** Fill sides arrive as BUY/SELL (and occasionally long/short from older runs). */
function isLongSide(direction: string): boolean {
  return /^(buy|long)$/i.test(direction.trim());
}

export function PerformancePriceRuntime() {
  const [selectedRunId, setSelectedRunId] = useState('');
  const url = selectedRunId
    ? `/api/python/dashboard/performance-price?run_id=${encodeURIComponent(selectedRunId)}`
    : '/api/python/dashboard/performance-price';
  const { payload, error, loading, refetch, lastFetchedAt } =
    useBridgeFetch<PerformancePricePayload>(url, 'Performance & price request failed');

  const ok = payload !== null && payload.status === 'ok';

  const equitySeries = useMemo<TimeSeriesChartSeries[]>(() => {
    if (!ok || payload === null) {
      return [];
    }
    const series: TimeSeriesChartSeries[] = [
      {
        id: 'equity',
        label: 'Equity',
        points: payload.equity_curve.map((point) => ({
          t: toEpochMs(point.timestamp),
          v: point.equity,
        })),
      },
    ];
    if (payload.benchmark_curve.length > 0) {
      series.push({
        id: 'benchmark',
        label: payload.benchmark_symbol ? `${payload.benchmark_symbol} (rebased)` : 'Benchmark (rebased)',
        points: payload.benchmark_curve.map((point) => ({
          t: toEpochMs(point.timestamp),
          v: point.price,
        })),
      });
    }
    return series;
  }, [ok, payload]);

  const tradeMarkers = useMemo<TimeSeriesChartMarker[]>(() => {
    if (!ok || payload === null) {
      return [];
    }
    return payload.trade_markers
      .map((marker) => ({
        t: toEpochMs(marker.timestamp),
        side: (isLongSide(marker.direction) ? 'buy' : 'sell') as 'buy' | 'sell',
        label:
          `${isLongSide(marker.direction) ? 'Buy' : 'Sell'} ${marker.symbol}` +
          ` @ ${formatCurrency(marker.price)}` +
          ` · qty ${formatNumber(marker.quantity, { digits: 4 })}` +
          ` · PnL ${formatCurrency(marker.realized_pnl)}`,
      }))
      .filter((marker) => Number.isFinite(marker.t));
  }, [ok, payload]);

  const drawdownSeries = useMemo<TimeSeriesChartSeries[]>(() => {
    if (!ok || payload === null) {
      return [];
    }
    return [
      {
        id: 'drawdown',
        label: 'Drawdown',
        kind: 'area',
        points: payload.drawdown_curve.map((point) => ({
          t: toEpochMs(point.timestamp),
          v: point.drawdown,
        })),
      },
    ];
  }, [ok, payload]);

  const fundingSeries = useMemo<TimeSeriesChartSeries[]>(() => {
    if (!ok || payload === null || payload.funding_curve.length === 0) {
      return [];
    }
    return [
      {
        id: 'funding',
        label: 'Funding',
        points: payload.funding_curve.map((point) => ({
          t: toEpochMs(point.timestamp),
          v: point.funding,
        })),
      },
    ];
  }, [ok, payload]);

  const sortedTrades = useMemo(() => {
    if (!ok || payload === null) {
      return [];
    }
    return [...payload.trade_markers].sort((a, b) =>
      a.timestamp < b.timestamp ? 1 : a.timestamp > b.timestamp ? -1 : 0,
    );
  }, [ok, payload]);

  const visibleTrades = sortedTrades.slice(0, TRADE_ROW_CAP);
  const runs = payload?.runs ?? [];
  const performanceEntries = payload
    ? PERFORMANCE_METRIC_LABELS.filter(([key]) => typeof payload.performance_metrics[key] === 'number')
    : [];
  const equityWindow = payload?.equity_window;

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
        surface="performance & price"
        onRetry={refetch}
      />
      {payload === null && !error ? <p>Loading run performance…</p> : null}
      {ok && payload !== null ? (
        <>
          <section className="section-card">
            <div className="section-header">
              <div>
                <p className="eyebrow">Run summary</p>
                <h3>Headline numbers</h3>
              </div>
            </div>
            <div className="metric-grid">
              {payload.summary_metrics.map((metric) => (
                <article key={metric.key}>
                  <span>{metric.label}</span>
                  <strong>
                    {formatMetricValue(metric.value, { key: metric.key, label: metric.label })}
                  </strong>
                </article>
              ))}
            </div>
            {performanceEntries.length > 0 ? (
              <div className="metric-grid">
                {performanceEntries.map(([key, label]) => (
                  <article key={key}>
                    <span>{label}</span>
                    <strong>
                      {formatMetricValue(payload.performance_metrics[key], { key, label })}
                    </strong>
                  </article>
                ))}
              </div>
            ) : null}
          </section>

          <section className="section-card">
            <div className="section-header">
              <div>
                <p className="eyebrow">Equity vs benchmark</p>
                <h3>Where the run made and gave back money</h3>
              </div>
            </div>
            <TimeSeriesChart
              series={equitySeries}
              yKind="currency"
              height={260}
              markers={tradeMarkers}
              rebase
              emptyLabel="No equity points recorded for this run yet."
            />
            {equityWindow && equityWindow.points_total > 0 ? (
              <p>
                Window {formatCompactTimestamp(equityWindow.start)} →{' '}
                {formatCompactTimestamp(equityWindow.end)} ·{' '}
                {equityWindow.points_returned.toLocaleString('en-US')} of{' '}
                {equityWindow.points_total.toLocaleString('en-US')} equity points
                {equityWindow.truncated ? ' (newest rows only)' : ''}
              </p>
            ) : null}
          </section>

          <section className="section-card">
            <div className="section-header">
              <div>
                <p className="eyebrow">Drawdown</p>
                <h3>Depth and duration of losses</h3>
              </div>
            </div>
            <TimeSeriesChart
              series={drawdownSeries}
              yKind="percent"
              height={180}
              emptyLabel="No drawdown points recorded for this run yet."
            />
            {fundingSeries.length > 0 ? (
              <>
                <h4>Cumulative funding (quote ccy)</h4>
                <TimeSeriesChart series={fundingSeries} yKind="currency" height={120} />
              </>
            ) : null}
          </section>

          <section className="section-card">
            <div className="section-header">
              <div>
                <p className="eyebrow">Trades</p>
                <h3>Recorded trades for this run</h3>
              </div>
            </div>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Time</th>
                    <th>Symbol</th>
                    <th>Direction</th>
                    <th>Price</th>
                    <th>Qty</th>
                    <th>Realized PnL</th>
                    <th>Return</th>
                    <th>Position After</th>
                  </tr>
                </thead>
                <tbody>
                  {visibleTrades.length > 0 ? (
                    visibleTrades.map((trade, index) => (
                      <tr
                        key={`${trade.timestamp}-${trade.symbol}-${trade.direction}-${trade.price}-${trade.quantity}-${index}`}
                      >
                        <td>{formatCompactTimestamp(trade.timestamp)}</td>
                        <td>{trade.symbol}</td>
                        <td>
                          <span
                            className={`status-pill ${isLongSide(trade.direction) ? 'pnl-pos' : 'pnl-neg'}`}
                          >
                            {isLongSide(trade.direction) ? 'long' : 'short'}
                          </span>
                        </td>
                        <td>{formatCurrency(trade.price)}</td>
                        <td>{formatNumber(trade.quantity, { digits: 4 })}</td>
                        <td className={pnlClass(trade.realized_pnl)}>
                          {trade.realized_pnl > 0
                            ? `+${formatCurrency(trade.realized_pnl)}`
                            : formatCurrency(trade.realized_pnl)}
                        </td>
                        <td className={pnlClass(trade.realized_return_pct)}>
                          {formatPercent(trade.realized_return_pct, { signed: true })}
                        </td>
                        <td>{formatNumber(trade.position_after, { digits: 4 })}</td>
                      </tr>
                    ))
                  ) : (
                    <tr>
                      <td colSpan={8}>No trades recorded for this run yet.</td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
            {sortedTrades.length > 0 ? (
              <p>
                Showing {visibleTrades.length} of {sortedTrades.length} trades, newest first.
              </p>
            ) : null}
          </section>
        </>
      ) : null}
    </div>
  );
}
