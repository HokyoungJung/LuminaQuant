'use client';

import { useId, useMemo, useState } from 'react';

import { TimeSeriesChart, type TimeSeriesChartSeries } from '@/components/charts/time-series-chart';
import { PageContextBar } from '@/components/page-context-bar';
import { RunSelector } from '@/components/run-selector';
import { SurfaceState } from '@/components/surface-state';
import type { MarketDataPayload } from '@/lib/dashboard-contracts';
import {
  formatCompactTimestamp,
  formatCurrency,
  formatMetricValue,
  formatNumber,
  pnlClass,
} from '@/lib/format';
import { useBridgeFetch } from '@/lib/use-bridge-fetch';

/** bar_window is emitted by the bridge but not yet in the shared contracts. */
interface BarWindow {
  start: string | null;
  end: string | null;
  bar_count: number;
}

type MarketDataRuntimePayload = MarketDataPayload & { bar_window?: BarWindow };

/** Context/indicator entries duplicated elsewhere on the page. */
const NON_TILE_INDICATOR_KEYS = new Set(['strategy', 'timeframe_clamped']);

function indicatorHint(
  key: string,
  value: number | string | null,
): { text: string; className: string } | null {
  if (typeof value !== 'number' || !Number.isFinite(value)) {
    return null;
  }
  if (key === 'rsi_14') {
    if (value >= 70) {
      return { text: 'overbought', className: 'status-degraded' };
    }
    if (value <= 30) {
      return { text: 'oversold', className: 'status-degraded' };
    }
  }
  if (key === 'dist_from_20bar_high' && value >= -0.005) {
    return { text: 'at highs', className: 'status-running' };
  }
  if (key === 'dist_from_20bar_low' && value <= 0.005) {
    return { text: 'at lows', className: 'status-degraded' };
  }
  return null;
}

function barPoints(
  bars: MarketDataPayload['recent_bars'],
  field: 'close' | 'high' | 'low',
): Array<{ t: number; v: number }> {
  return bars.flatMap((bar) => {
    const t = Date.parse(bar.timestamp);
    const v = bar[field];
    if (!Number.isFinite(t) || typeof v !== 'number' || !Number.isFinite(v)) {
      return [];
    }
    return [{ t, v }];
  });
}

export function MarketDataRuntime() {
  const [runId, setRunId] = useState('');
  const [symbol, setSymbol] = useState('');
  const symbolSelectId = useId();

  const url = useMemo(() => {
    const query = new URLSearchParams();
    if (runId) {
      query.set('run_id', runId);
    }
    if (symbol) {
      query.set('symbol', symbol);
    }
    const suffix = query.toString();
    return `/api/python/dashboard/market-data${suffix ? `?${suffix}` : ''}`;
  }, [runId, symbol]);

  const { payload, error, loading, refetch, lastFetchedAt } =
    useBridgeFetch<MarketDataRuntimePayload>(url, 'market data request failed');

  const runs = payload?.runs ?? [];
  const activeRunId = runId || payload?.run_id || '';
  const activeSymbol = symbol || payload?.market_context.symbol || '';
  const symbolOptions = useMemo(() => {
    const options = new Set(payload?.symbols ?? []);
    if (activeSymbol) {
      options.add(activeSymbol);
    }
    return [...options].sort();
  }, [payload?.symbols, activeSymbol]);

  const priceSeries = useMemo<TimeSeriesChartSeries[]>(() => {
    const bars = payload?.recent_bars ?? [];
    const close = barPoints(bars, 'close');
    if (close.length === 0) {
      return [];
    }
    const series: TimeSeriesChartSeries[] = [
      { id: 'close', label: 'Close', points: close, kind: 'area' },
    ];
    const high = barPoints(bars, 'high');
    const low = barPoints(bars, 'low');
    if (high.length > 0) {
      series.push({ id: 'high', label: 'High', points: high });
    }
    if (low.length > 0) {
      series.push({ id: 'low', label: 'Low', points: low });
    }
    return series;
  }, [payload?.recent_bars]);

  const showContext =
    payload !== null && (payload.status === 'ok' || payload.status === 'no_market_data');
  const showData = payload !== null && payload.status === 'ok';
  const barWindow = payload?.bar_window;
  const timeframeClamped = payload?.market_context.timeframe_clamped === true;
  const newestBarsFirst = useMemo(
    () => [...(payload?.recent_bars ?? [])].reverse(),
    [payload?.recent_bars],
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
      {runs.length > 0 || symbolOptions.length > 0 ? (
        <div className="button-row">
          {runs.length > 0 ? (
            <RunSelector runs={runs} value={activeRunId} onChange={setRunId} />
          ) : null}
          {symbolOptions.length > 0 ? (
            <div className="run-selector">
              <label htmlFor={symbolSelectId}>Symbol</label>
              <select
                id={symbolSelectId}
                value={activeSymbol}
                onChange={(event) => setSymbol(event.target.value)}
              >
                {symbolOptions.map((option) => (
                  <option key={option} value={option}>
                    {option}
                  </option>
                ))}
              </select>
            </div>
          ) : null}
        </div>
      ) : null}
      <SurfaceState
        status={payload?.status}
        error={error || undefined}
        surface="market data"
        onRetry={refetch}
      />
      {payload === null && !error ? <p>Loading market data…</p> : null}

      {showContext ? (
        <section className="section-card">
          <div className="section-header">
            <div>
              <p className="eyebrow">Market context</p>
              <h3>Instrument and data source</h3>
            </div>
            {timeframeClamped ? (
              <span className="status-pill status-degraded">timeframe clamped</span>
            ) : null}
          </div>
          <div className="metric-grid">
            <article>
              <span>Symbol</span>
              <strong>{payload.market_context.symbol || 'n/a'}</strong>
            </article>
            <article>
              <span>Timeframe</span>
              <strong>{payload.market_context.timeframe || 'n/a'}</strong>
            </article>
            <article>
              <span>Exchange</span>
              <strong>{payload.market_context.exchange || 'n/a'}</strong>
            </article>
            <article>
              <span>Strategy</span>
              <strong>{payload.market_context.strategy || 'n/a'}</strong>
            </article>
            <article>
              <span>Data source</span>
              <strong>{payload.market_context.source || 'n/a'}</strong>
            </article>
          </div>
          {payload.warnings.length > 0 ? (
            <div className="button-row" role="list" aria-label="Data warnings">
              {payload.warnings.map((warning) => (
                <span key={warning} className="status-pill status-degraded" role="listitem">
                  {warning}
                </span>
              ))}
            </div>
          ) : null}
        </section>
      ) : null}

      {showData ? (
        <>
          <div className="metric-grid">
            {payload.summary_metrics.map((metric) => (
              <article key={metric.key}>
                <span>{metric.label}</span>
                <strong
                  className={
                    metric.key === 'price_change_pct' && typeof metric.value === 'number'
                      ? pnlClass(metric.value)
                      : undefined
                  }
                >
                  {formatMetricValue(metric.value, { key: metric.key, label: metric.label })}
                </strong>
              </article>
            ))}
          </div>

          <section className="section-card">
            <div className="section-header">
              <div>
                <p className="eyebrow">Price</p>
                <h3>Recent closes with the high–low band</h3>
              </div>
              {barWindow ? (
                <span className="context-bar-item context-bar-muted">
                  {`over last ${barWindow.bar_count} bars: `}
                  {`${formatCompactTimestamp(barWindow.start)} – ${formatCompactTimestamp(barWindow.end)}`}
                </span>
              ) : null}
            </div>
            <TimeSeriesChart
              series={priceSeries}
              yKind="currency"
              emptyLabel="No market bars to chart for this run and symbol yet."
            />
          </section>

          <section className="section-card">
            <div className="section-header">
              <div>
                <p className="eyebrow">Indicators</p>
                <h3>Current market state</h3>
              </div>
              {barWindow ? (
                <span className="context-bar-item context-bar-muted">
                  {`over last ${barWindow.bar_count} bars: `}
                  {`${formatCompactTimestamp(barWindow.start)} – ${formatCompactTimestamp(barWindow.end)}`}
                </span>
              ) : null}
            </div>
            <div className="metric-grid">
              {payload.indicator_summary
                .filter((metric) => !NON_TILE_INDICATOR_KEYS.has(metric.key))
                .map((metric) => {
                  const hint = indicatorHint(metric.key, metric.value);
                  return (
                    <article key={metric.key}>
                      <span>{metric.label}</span>
                      <strong>
                        {formatMetricValue(metric.value, { key: metric.key, label: metric.label })}{' '}
                        {hint ? (
                          <span className={`status-pill ${hint.className}`}>{hint.text}</span>
                        ) : null}
                      </strong>
                    </article>
                  );
                })}
            </div>
          </section>

          <details className="section-card">
            <summary>{`Show raw bars (${newestBarsFirst.length})`}</summary>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Time (UTC)</th>
                    <th>Open</th>
                    <th>High</th>
                    <th>Low</th>
                    <th>Close</th>
                    <th>Volume</th>
                  </tr>
                </thead>
                <tbody>
                  {newestBarsFirst.length > 0 ? (
                    newestBarsFirst.map((bar) => (
                      <tr key={bar.timestamp}>
                        <td>{formatCompactTimestamp(bar.timestamp)}</td>
                        <td>{formatCurrency(bar.open)}</td>
                        <td>{formatCurrency(bar.high)}</td>
                        <td>{formatCurrency(bar.low)}</td>
                        <td>{formatCurrency(bar.close)}</td>
                        <td>{formatNumber(bar.volume)}</td>
                      </tr>
                    ))
                  ) : (
                    <tr>
                      <td colSpan={6}>No market bars recorded for this run and symbol.</td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          </details>
        </>
      ) : null}
    </div>
  );
}
