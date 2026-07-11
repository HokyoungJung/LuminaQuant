'use client';

import { useMemo, useRef, useState } from 'react';

import {
  areaPath,
  downsample,
  linePath,
  niceTicks,
  rebaseSeries,
  scaleLinear,
  timeTicks,
  type ChartPoint,
} from '@/lib/chart-geometry';
import { formatCompactTimestamp, formatNumber, formatPercent } from '@/lib/format';

export interface TimeSeriesChartSeries {
  id: string;
  label: string;
  points: ChartPoint[];
  kind?: 'line' | 'area';
}

export interface TimeSeriesChartMarker {
  t: number;
  side: 'buy' | 'sell';
  label: string;
}

export interface TimeSeriesChartProps {
  series: TimeSeriesChartSeries[];
  yKind: 'currency' | 'percent' | 'ratio';
  height?: number;
  markers?: TimeSeriesChartMarker[];
  /** Rebase secondary series to the primary's first value (single y-axis). */
  rebase?: boolean;
  emptyLabel?: string;
}

const VIEW_WIDTH = 720;
const MARGIN = { top: 14, right: 84, bottom: 26, left: 64 };
const MAX_RENDER_POINTS = 480;
const SERIES_COLOR_VARS = ['--chart-1', '--chart-2', '--chart-3', '--chart-4'];

interface HoverState {
  t: number;
  xView: number;
  leftPx: number;
  rows: Array<{ label: string; colorVar: string; text: string }>;
}

function formatValue(
  value: number,
  yKind: TimeSeriesChartProps['yKind'],
  digits = 2,
): string {
  if (yKind === 'percent') {
    return formatPercent(value, { digits });
  }
  if (yKind === 'ratio') {
    return value.toFixed(digits);
  }
  return formatNumber(value, { digits });
}

/**
 * Decimal digits needed so adjacent axis ticks render distinctly: derived
 * from the tick step in display units (percent values display *100).
 * Small ranges (e.g. a 0.0001-fraction step) otherwise collapse every tick
 * label onto the same rounded value.  Capped at 6 to avoid float noise.
 */
function tickDecimalDigits(step: number, yKind: TimeSeriesChartProps['yKind']): number {
  if (!Number.isFinite(step) || step <= 0) {
    return 2;
  }
  const displayStep = yKind === 'percent' ? step * 100 : step;
  return Math.min(6, Math.max(2, Math.ceil(-Math.log10(displayStep))));
}

function formatTimeTick(t: number, spanMs: number): string {
  const iso = new Date(t).toISOString();
  if (spanMs >= 300 * 86_400_000) {
    return iso.slice(0, 10);
  }
  if (spanMs >= 2 * 86_400_000) {
    return iso.slice(5, 10);
  }
  return iso.slice(11, 16);
}

function nearestPoint(points: ChartPoint[], t: number): ChartPoint | null {
  let best: ChartPoint | null = null;
  let bestDistance = Infinity;
  for (const point of points) {
    const distance = Math.abs(point.t - t);
    if (distance < bestDistance) {
      bestDistance = distance;
      best = point;
    }
  }
  return best;
}

export function TimeSeriesChart({
  series,
  yKind,
  height = 220,
  markers = [],
  rebase = false,
  emptyLabel,
}: TimeSeriesChartProps) {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const [hover, setHover] = useState<HoverState | null>(null);

  const model = useMemo(() => {
    const prepared = series.map((entry, index) => {
      const finite = entry.points.filter(
        (point) => Number.isFinite(point.t) && Number.isFinite(point.v),
      );
      return { ...entry, points: downsample(finite, MAX_RENDER_POINTS), index };
    });
    const primary = prepared[0];
    const primaryBase = primary?.points.find((point) => point.v !== 0)?.v;
    const rebased = prepared.map((entry) => {
      if (rebase && entry.index > 0 && primaryBase !== undefined) {
        return { ...entry, points: rebaseSeries(entry.points, primaryBase) };
      }
      return entry;
    });
    const allPoints = rebased.flatMap((entry) => entry.points);
    if (allPoints.length === 0) {
      return null;
    }
    let vMin = Infinity;
    let vMax = -Infinity;
    let tMin = Infinity;
    let tMax = -Infinity;
    for (const point of allPoints) {
      if (point.v < vMin) vMin = point.v;
      if (point.v > vMax) vMax = point.v;
      if (point.t < tMin) tMin = point.t;
      if (point.t > tMax) tMax = point.t;
    }
    const hasArea = rebased.length === 1 && rebased[0].kind === 'area';
    if (hasArea) {
      vMin = Math.min(vMin, 0);
      vMax = Math.max(vMax, 0);
    }
    const yTicksRaw = niceTicks(vMin, vMax, 5);
    // Clamp the domain to the data extremes so a peak/trough outside the
    // tick range is never clipped past the plot edge.
    const yDomain: [number, number] = yTicksRaw.length >= 2
      ? [Math.min(yTicksRaw[0], vMin), Math.max(yTicksRaw[yTicksRaw.length - 1], vMax)]
      : [vMin - 1, vMax + 1];
    const xDomain: [number, number] = tMin === tMax ? [tMin - 1, tMax + 1] : [tMin, tMax];
    const yTickStep = yTicksRaw.length >= 2 ? yTicksRaw[1] - yTicksRaw[0] : yDomain[1] - yDomain[0];
    return {
      series: rebased,
      hasArea,
      yTicks: yTicksRaw.filter((tick) => tick >= yDomain[0] && tick <= yDomain[1]),
      xTicks: timeTicks(xDomain[0], xDomain[1], 5),
      yDomain,
      xDomain,
      spanMs: xDomain[1] - xDomain[0],
      yTickStep,
    };
  }, [series, rebase]);

  if (!model) {
    return <div className="chart-empty">{emptyLabel ?? 'No chart data yet.'}</div>;
  }

  const plotWidth = VIEW_WIDTH - MARGIN.left - MARGIN.right;
  const plotHeight = height - MARGIN.top - MARGIN.bottom;
  const xScale = scaleLinear(model.xDomain, [0, plotWidth]);
  const yScale = scaleLinear(model.yDomain, [plotHeight, 0]);
  const valueDigits = tickDecimalDigits(model.yTickStep, yKind);
  const primary = model.series[0];
  const primaryLast = primary.points.length > 0 ? primary.points[primary.points.length - 1] : null;

  const ariaLabel = model.series
    .map((entry) => {
      if (entry.points.length === 0) {
        return `${entry.label}: no data`;
      }
      const values = entry.points.map((point) => point.v);
      const low = formatValue(Math.min(...values), yKind, valueDigits);
      const high = formatValue(Math.max(...values), yKind, valueDigits);
      const last = formatValue(values[values.length - 1], yKind, valueDigits);
      return `${entry.label}: ${low} to ${high}, last ${last}`;
    })
    .join('; ');

  const seriesColor = (entry: (typeof model.series)[number]): string => {
    if (model.hasArea && entry.points.every((point) => point.v <= 0)) {
      // Below-zero area (drawdown) reads as a "serious" tone, not a series hue.
      return 'var(--chart-drawdown)';
    }
    return `var(${SERIES_COLOR_VARS[entry.index % SERIES_COLOR_VARS.length]})`;
  };

  const handlePointerMove = (event: React.PointerEvent<SVGSVGElement>) => {
    const svg = svgRef.current;
    if (!svg) {
      return;
    }
    const rect = svg.getBoundingClientRect();
    if (rect.width === 0) {
      return;
    }
    const xView = ((event.clientX - rect.left) / rect.width) * VIEW_WIDTH;
    const xPlot = Math.max(0, Math.min(plotWidth, xView - MARGIN.left));
    const tHover = model.xDomain[0]
      + (xPlot / plotWidth) * (model.xDomain[1] - model.xDomain[0]);
    const anchor = nearestPoint(primary.points, tHover);
    if (!anchor) {
      return;
    }
    const rows = model.series.flatMap((entry) => {
      const point = nearestPoint(entry.points, anchor.t);
      if (!point) {
        return [];
      }
      return [{
        label: entry.label,
        colorVar: seriesColor(entry),
        text: formatValue(point.v, yKind, valueDigits),
      }];
    });
    setHover({
      t: anchor.t,
      xView: MARGIN.left + xScale(anchor.t),
      leftPx: event.clientX - rect.left,
      rows,
    });
  };

  return (
    <div className="chart-root">
      {model.series.length >= 2 ? (
        <div className="chart-legend" role="list">
          {model.series.map((entry) => (
            <span key={entry.id} className="legend-chip" role="listitem">
              <span className="legend-swatch" style={{ background: seriesColor(entry) }} aria-hidden />
              {entry.label}
            </span>
          ))}
        </div>
      ) : null}
      <svg
        ref={svgRef}
        viewBox={`0 0 ${VIEW_WIDTH} ${height}`}
        className="chart-svg"
        role="img"
        aria-label={ariaLabel}
        onPointerMove={handlePointerMove}
        onPointerLeave={() => setHover(null)}
      >
        <g transform={`translate(${MARGIN.left} ${MARGIN.top})`}>
          {model.yTicks.map((tick) => (
            <g key={`y-${tick}`}>
              <line
                className="chart-grid-line"
                x1={0}
                x2={plotWidth}
                y1={yScale(tick)}
                y2={yScale(tick)}
              />
              <text className="chart-axis-text" x={-8} y={yScale(tick) + 3} textAnchor="end">
                {formatValue(tick, yKind, valueDigits)}
              </text>
            </g>
          ))}
          {model.xTicks.map((tick) => (
            <text
              key={`x-${tick}`}
              className="chart-axis-text"
              x={xScale(tick)}
              y={plotHeight + 16}
              textAnchor="middle"
            >
              {formatTimeTick(tick, model.spanMs)}
            </text>
          ))}
          {model.series.map((entry) => {
            const color = seriesColor(entry);
            const isArea = model.hasArea && entry.kind === 'area';
            const line = linePath(entry.points, plotWidth, plotHeight, model.xDomain, model.yDomain);
            const area = isArea
              ? areaPath(entry.points, plotWidth, plotHeight, model.xDomain, model.yDomain, 0)
              : '';
            return (
              <g key={entry.id}>
                {area ? <path d={area} fill={color} opacity={0.12} stroke="none" /> : null}
                {line ? (
                  <path
                    d={line}
                    fill="none"
                    stroke={color}
                    strokeWidth={2}
                    strokeLinejoin="round"
                    strokeLinecap="round"
                  />
                ) : null}
              </g>
            );
          })}
          {markers.map((marker, index) => {
            const point = nearestPoint(primary.points, marker.t);
            if (!point) {
              return null;
            }
            const x = xScale(point.t);
            const y = yScale(point.v);
            const glyph = marker.side === 'buy'
              ? `M ${x} ${y - 6} L ${x + 5} ${y + 3} L ${x - 5} ${y + 3} Z`
              : `M ${x} ${y + 6} L ${x + 5} ${y - 3} L ${x - 5} ${y - 3} Z`;
            return (
              <path
                key={`marker-${marker.t}-${index}`}
                d={glyph}
                className={marker.side === 'buy' ? 'chart-marker-buy' : 'chart-marker-sell'}
              >
                <title>{marker.label}</title>
              </path>
            );
          })}
          {primaryLast ? (
            <g>
              <circle
                cx={xScale(primaryLast.t)}
                cy={yScale(primaryLast.v)}
                r={4}
                fill={seriesColor(primary)}
                className="chart-endpoint-dot"
              />
              <text
                className="chart-last-value"
                x={plotWidth + 8}
                y={yScale(primaryLast.v) + 3}
              >
                {formatValue(primaryLast.v, yKind, valueDigits)}
              </text>
            </g>
          ) : null}
          {hover ? (
            <line
              className="chart-crosshair"
              x1={hover.xView - MARGIN.left}
              x2={hover.xView - MARGIN.left}
              y1={0}
              y2={plotHeight}
            />
          ) : null}
        </g>
      </svg>
      {hover ? (
        <div
          className="chart-tooltip"
          style={
            hover.xView > VIEW_WIDTH * 0.6
              ? { right: `calc(100% - ${hover.leftPx}px + 12px)`, top: 8 }
              : { left: hover.leftPx + 12, top: 8 }
          }
        >
          <div className="chart-tooltip-time">{formatCompactTimestamp(new Date(hover.t).toISOString())}</div>
          {hover.rows.map((row) => (
            <div key={row.label} className="chart-tooltip-row">
              <span className="legend-swatch" style={{ background: row.colorVar }} aria-hidden />
              <span className="chart-tooltip-label">{row.label}</span>
              <span className="chart-tooltip-value">{row.text}</span>
            </div>
          ))}
        </div>
      ) : null}
    </div>
  );
}
