import { describe, expect, it } from 'vitest';

import {
  formatCompactTimestamp,
  formatCurrency,
  formatDuration,
  formatMetricValue,
  formatNumber,
  formatPercent,
  formatRatio,
  metricFormatKind,
  pnlClass,
} from '../lib/format';

describe('metricFormatKind', () => {
  it('classifies percent-kind keys (payload values are raw fractions)', () => {
    for (const key of [
      'total_return',
      'cagr',
      'annualized_volatility',
      'max_drawdown',
      'drawdown',
      'price_change_pct',
      'atr_pct',
      'realized_vol_30',
      'dist_from_20bar_high',
      'dist_from_20bar_low',
      'win_rate',
      'long_win_rate',
      'oos_return',
      'oos_max_drawdown',
      'avg_trade_return_pct',
      'realized_return_pct',
      'ic_positive_ratio',
    ]) {
      expect(metricFormatKind(key), key).toBe('percent');
    }
  });

  it('treats any label containing % as percent', () => {
    expect(metricFormatKind('some_unknown_key', 'Price Change %')).toBe('percent');
  });

  it('classifies ratio-kind keys', () => {
    for (const key of [
      'sharpe_ratio',
      'oos_sharpe',
      'train_sharpe',
      'median_sharpe',
      'sortino_ratio',
      'calmar_ratio',
      'robustness_score',
      'median_robustness',
      'score',
      'ic_ir',
      't_stat',
    ]) {
      expect(metricFormatKind(key), key).toBe('ratio');
    }
  });

  it('classifies currency-kind and count-kind keys', () => {
    for (const key of ['latest_equity', 'realized_pnl', 'avg_notional', 'total_commission', 'close']) {
      expect(metricFormatKind(key), key).toBe('currency');
    }
    for (const key of ['closed_trade_count', 'order_count', 'buy_fills', 'win_streak_max']) {
      expect(metricFormatKind(key), key).toBe('count');
    }
  });

  it('percent wins over currency substrings', () => {
    // 'price_change_pct' contains 'price' but must render as percent.
    expect(metricFormatKind('price_change_pct')).toBe('percent');
  });

  it('falls back to text for unknown keys', () => {
    expect(metricFormatKind('mystery_key')).toBe('text');
    expect(metricFormatKind(undefined)).toBe('text');
  });
});

describe('formatMetricValue', () => {
  it('returns n/a for absent/empty values', () => {
    expect(formatMetricValue(null)).toBe('n/a');
    expect(formatMetricValue(undefined)).toBe('n/a');
    expect(formatMetricValue('')).toBe('n/a');
    expect(formatMetricValue(NaN, { key: 'cagr' })).toBe('n/a');
  });

  it('multiplies raw fractions by 100 for percent-kind metrics', () => {
    expect(formatMetricValue(0.05, { key: 'total_return' })).toBe('5.00%');
    expect(formatMetricValue(-0.234, { key: 'max_drawdown' })).toBe('-23.40%');
    expect(formatMetricValue(0.012, { label: 'Price Change %' })).toBe('1.20%');
  });

  it('renders ratios at 2dp and currency with separators', () => {
    expect(formatMetricValue(1.23456, { key: 'sharpe_ratio' })).toBe('1.23');
    expect(formatMetricValue(12345.6789, { key: 'latest_equity' })).toBe('12,345.68');
    expect(formatMetricValue(1234567, { key: 'closed_trade_count' })).toBe('1,234,567');
  });

  it('is backwards compatible when opts are omitted', () => {
    expect(formatMetricValue('BTC/USDT')).toBe('BTC/USDT');
    expect(formatMetricValue(1234)).toBe('1,234');
    // Never emits 17-digit float noise: caps at 4 significant digits.
    expect(formatMetricValue(0.1 + 0.2)).toBe('0.3');
    expect(formatMetricValue(1.23456789)).toBe('1.235');
  });
});

describe('formatPercent / formatNumber / formatRatio / formatCurrency', () => {
  it('formats fractions with optional sign and digits', () => {
    expect(formatPercent(0.0512)).toBe('5.12%');
    expect(formatPercent(0.0512, { signed: true })).toBe('+5.12%');
    expect(formatPercent(-0.0512, { signed: true })).toBe('-5.12%');
    expect(formatPercent(0.05123, { digits: 1 })).toBe('5.1%');
    expect(formatPercent(null)).toBe('n/a');
  });

  it('formats plain numbers, ratios and currency', () => {
    expect(formatNumber(1234.5678)).toBe('1,234.57');
    expect(formatNumber(1234.5678, { digits: 0 })).toBe('1,235');
    expect(formatRatio(1.005)).toBe('1.00');
    expect(formatRatio(null)).toBe('n/a');
    expect(formatCurrency(9876543.211)).toBe('9,876,543.21');
    expect(formatCurrency(undefined)).toBe('n/a');
  });
});

describe('formatDuration', () => {
  it('renders hours with minutes', () => {
    expect(formatDuration((2 * 3600 + 14 * 60) * 1000)).toBe('2h 14m');
    expect(formatDuration((25 * 3600 + 5 * 60 + 59) * 1000)).toBe('25h 5m');
  });

  it('renders whole minutes below an hour', () => {
    expect(formatDuration(45 * 60 * 1000)).toBe('45m');
    expect(formatDuration((4 * 60 + 30) * 1000)).toBe('4m');
  });

  it('renders seconds below a minute', () => {
    expect(formatDuration(30 * 1000)).toBe('30s');
    expect(formatDuration(0)).toBe('0s');
    expect(formatDuration(999)).toBe('0s');
  });

  it('returns n/a for negative or non-finite input', () => {
    expect(formatDuration(-1)).toBe('n/a');
    expect(formatDuration(NaN)).toBe('n/a');
    expect(formatDuration(Infinity)).toBe('n/a');
  });
});

describe('formatCompactTimestamp', () => {
  it('renders YYYY-MM-DD HH:mm in UTC without seconds noise', () => {
    expect(formatCompactTimestamp('2026-07-10T09:30:15.123Z')).toBe('2026-07-10 09:30');
    expect(formatCompactTimestamp('2026-07-10T09:30:15+02:00')).toBe('2026-07-10 07:30');
  });

  it('passes through unparseable strings and returns n/a for absent input', () => {
    expect(formatCompactTimestamp('not-a-date')).toBe('not-a-date');
    expect(formatCompactTimestamp(null)).toBe('n/a');
    expect(formatCompactTimestamp('')).toBe('n/a');
  });
});

describe('pnlClass', () => {
  it('maps sign to pnl classes', () => {
    expect(pnlClass(3.2)).toBe('pnl-pos');
    expect(pnlClass(-0.01)).toBe('pnl-neg');
    expect(pnlClass(0)).toBe('');
    expect(pnlClass(null)).toBe('');
    expect(pnlClass(undefined)).toBe('');
    expect(pnlClass(NaN)).toBe('');
  });
});
