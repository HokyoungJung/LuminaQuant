/**
 * Shared pure formatting utilities for dashboard runtime components.
 *
 * Cross-side convention: percent-like metrics travel in payloads as RAW
 * FRACTIONS (0.05 = 5%); the frontend multiplies by 100 at render time.
 */

/**
 * Build an SVG polyline path string from an array of numeric values.
 *
 * Returns '' for an empty array (callers suppress the SVG element when the
 * return value is falsy).  A single value produces a horizontal mid-line so
 * the SVG element is never degenerate.
 */
export function buildSparklinePath(
  values: number[],
  width = 420,
  height = 120,
): string {
  if (values.length === 0) {
    return '';
  }
  if (values.length === 1) {
    return `M 0 ${height / 2} L ${width} ${height / 2}`;
  }
  const min = Math.min(...values);
  const max = Math.max(...values);
  const range = max - min || 1;
  return values
    .map((value, index) => {
      const x = (index / (values.length - 1)) * width;
      const y = height - ((value - min) / range) * height;
      return `${index === 0 ? 'M' : 'L'} ${x.toFixed(2)} ${y.toFixed(2)}`;
    })
    .join(' ');
}

export type MetricFormatKind = 'percent' | 'ratio' | 'currency' | 'count' | 'text';

/** Metric keys whose payload values are raw fractions rendered as percent. */
const PERCENT_KEYS = new Set([
  'total_return',
  'cagr',
  'annualized_volatility',
  'max_drawdown',
  'drawdown',
  'mdd',
  'price_change_pct',
  'atr_pct',
  'realized_vol_30',
  'dist_from_20bar_high',
  'dist_from_20bar_low',
  'win_rate',
  'oos_return',
  'oos_max_drawdown',
  'avg_trade_return_pct',
  'realized_return_pct',
  'ic_positive_ratio',
]);

const RATIO_KEY_PATTERN = /(sharpe|sortino|calmar|robustness)/;
const RATIO_KEYS = new Set(['score', 'ic_ir', 't_stat', 'ic_mean', 'weight']);

const CURRENCY_KEY_PATTERN = /(equity|pnl|notional|price|commission|close)/;

// 'points' must be classified before currency so 'equity_points' (a row
// count) never renders as a monetary amount via the 'equity' substring.
const COUNT_KEY_PATTERN = /(count|_trades|trades_|fills|streak|n_periods|rows|columns|points)/;

/**
 * Classify a metric by key/label so numeric values render with the right
 * unit.  Percent detection wins over the substring-based kinds so keys like
 * 'price_change_pct' never fall into the currency bucket.
 */
export function metricFormatKind(key?: string, label?: string): MetricFormatKind {
  const normalized = (key ?? '').toLowerCase();
  if (label && label.includes('%')) {
    return 'percent';
  }
  if (PERCENT_KEYS.has(normalized) || normalized.endsWith('_win_rate')) {
    return 'percent';
  }
  if (RATIO_KEYS.has(normalized) || RATIO_KEY_PATTERN.test(normalized)) {
    return 'ratio';
  }
  // Count before currency: 'closed_trade_count' must not match 'close'.
  if (COUNT_KEY_PATTERN.test(normalized)) {
    return 'count';
  }
  if (CURRENCY_KEY_PATTERN.test(normalized)) {
    return 'currency';
  }
  return 'text';
}

/** Format a raw fraction (0.05 = 5%) as a percent string. */
export function formatPercent(
  fraction: number | null | undefined,
  opts: { digits?: number; signed?: boolean } = {},
): string {
  if (fraction === null || fraction === undefined || !Number.isFinite(fraction)) {
    return 'n/a';
  }
  const { digits = 2, signed = false } = opts;
  const scaled = fraction * 100;
  const sign = signed && scaled > 0 ? '+' : '';
  return `${sign}${scaled.toFixed(digits)}%`;
}

/** Format a number with thousands separators at a fixed decimal count. */
export function formatNumber(
  value: number | null | undefined,
  opts: { digits?: number } = {},
): string {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return 'n/a';
  }
  const { digits = 2 } = opts;
  return value.toLocaleString('en-US', {
    minimumFractionDigits: digits,
    maximumFractionDigits: digits,
  });
}

/** Format a unitless ratio (Sharpe, robustness, ...) at 2 decimals. */
export function formatRatio(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return 'n/a';
  }
  return value.toFixed(2);
}

/** Format a monetary quantity: thousands separators + 2 decimals. */
export function formatCurrency(value: number | null | undefined): string {
  return formatNumber(value, { digits: 2 });
}

/** Format an integer count with thousands separators. */
function formatCount(value: number): string {
  return Math.round(value).toLocaleString('en-US');
}

/**
 * Humanize a duration in milliseconds: '2h 14m', '45m', '30s'.
 * Negative or non-finite input renders as 'n/a'.
 */
export function formatDuration(ms: number): string {
  if (!Number.isFinite(ms) || ms < 0) {
    return 'n/a';
  }
  const totalSeconds = Math.floor(ms / 1000);
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  if (hours > 0) {
    return `${hours}h ${minutes}m`;
  }
  if (minutes > 0) {
    return `${minutes}m`;
  }
  return `${totalSeconds}s`;
}

/**
 * Compact UTC timestamp: 'YYYY-MM-DD HH:mm' (no seconds/ms noise).
 * Unparseable input is passed through unchanged so raw labels still render.
 */
export function formatCompactTimestamp(iso: string | null | undefined): string {
  if (iso === null || iso === undefined || iso === '') {
    return 'n/a';
  }
  const parsed = new Date(iso);
  if (Number.isNaN(parsed.getTime())) {
    return String(iso);
  }
  return parsed.toISOString().slice(0, 16).replace('T', ' ');
}

/** Tone mapping for status pills: normalized status slug -> palette tone. */
const STATUS_TONE_BY_SLUG: Record<string, 'ok' | 'running' | 'failed' | 'warn'> = {
  ok: 'ok',
  success: 'ok',
  succeeded: 'ok',
  completed: 'ok',
  running: 'running',
  queued: 'running',
  pending: 'running',
  starting: 'running',
  stop_requested: 'running',
  failed: 'failed',
  error: 'failed',
  killed: 'failed',
  cancelled: 'failed',
  canceled: 'failed',
  degraded: 'warn',
};

/**
 * CSS classes for a status pill.  Lowercases the raw status into a slug
 * (uppercase DB statuses like 'RUNNING' otherwise match no CSS rule) and
 * appends a tone class so new statuses (killed, stop_requested, queued, ...)
 * inherit the ok/running/failed/warn palette instead of rendering bare.
 */
export function statusPillClass(status: string | null | undefined): string {
  const slug = (status ?? '')
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9_-]+/g, '-');
  const tone = STATUS_TONE_BY_SLUG[slug] ?? 'neutral';
  return `status-pill status-${slug || 'unknown'} status-tone-${tone}`;
}

/** CSS class for signed PnL-style coloring; '' for zero/absent values. */
export function pnlClass(value: number | null | undefined): 'pnl-pos' | 'pnl-neg' | '' {
  if (value === null || value === undefined || !Number.isFinite(value) || value === 0) {
    return '';
  }
  return value > 0 ? 'pnl-pos' : 'pnl-neg';
}

/** Conservative ISO-8601 shape check: date, 'T' or space, then a time. */
const ISO_TIMESTAMP_PATTERN = /^\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}/;

/** Default numeric rendering when the metric kind is unknown. */
function formatPlainNumber(value: number): string {
  if (Number.isInteger(value)) {
    return value.toLocaleString('en-US');
  }
  // Cap floats at 4 significant digits to avoid 17-digit float noise.
  return String(Number(value.toPrecision(4)));
}

/**
 * Format a metric value for display, returning 'n/a' for absent/empty values.
 *
 * With `opts.key`/`opts.label` the value is dispatched on metricFormatKind
 * (percent values arrive as raw fractions and are multiplied by 100 here).
 * Without opts the call stays backwards compatible: strings pass through and
 * finite numbers get a sensible default rendering.
 */
export function formatMetricValue(
  value: number | string | null | undefined,
  opts?: { key?: string; label?: string },
): string {
  if (value === null || value === undefined || value === '') {
    return 'n/a';
  }
  if (typeof value !== 'number') {
    const text = String(value);
    // Strings that are ISO-8601 timestamps render compactly instead of as a
    // raw '2025-12-25T08:00:00.000Z' tile.
    if (ISO_TIMESTAMP_PATTERN.test(text) && !Number.isNaN(Date.parse(text))) {
      return formatCompactTimestamp(text);
    }
    return text;
  }
  if (!Number.isFinite(value)) {
    return 'n/a';
  }
  switch (metricFormatKind(opts?.key, opts?.label)) {
    case 'percent':
      return formatPercent(value);
    case 'ratio':
      return formatRatio(value);
    case 'currency':
      return formatCurrency(value);
    case 'count':
      return formatCount(value);
    default:
      return formatPlainNumber(value);
  }
}
