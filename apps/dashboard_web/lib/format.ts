/**
 * Shared pure formatting utilities for dashboard runtime components.
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

/**
 * Format a metric value for display, returning 'n/a' for absent/empty values.
 */
export function formatMetricValue(value: number | string | null | undefined): string {
  if (value === null || value === undefined || value === '') {
    return 'n/a';
  }
  return String(value);
}
