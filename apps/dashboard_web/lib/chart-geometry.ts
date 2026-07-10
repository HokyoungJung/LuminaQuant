/**
 * Pure chart geometry helpers for the SVG time-series chart.
 *
 * No React, no DOM: everything here is unit-testable math.  Points are
 * `{ t, v }` pairs where `t` is epoch milliseconds (UTC) and `v` is the
 * series value in payload units.
 */

export interface ChartPoint {
  t: number;
  v: number;
}

const isFinitePoint = (point: ChartPoint): boolean =>
  Number.isFinite(point.t) && Number.isFinite(point.v);

/** Round a raw step to the nearest 1/2/5 * 10^k "nice" step. */
function niceStep(rawStep: number): number {
  const magnitude = 10 ** Math.floor(Math.log10(rawStep));
  const residual = rawStep / magnitude;
  // Geometric-mean thresholds (d3-style) so the tick count stays near target.
  if (residual < Math.SQRT2) return magnitude;
  if (residual < Math.sqrt(10)) return 2 * magnitude;
  if (residual < Math.sqrt(50)) return 5 * magnitude;
  return 10 * magnitude;
}

/**
 * Compute 4-6ish "nice" axis tick values covering [min, max] with 1/2/5
 * steps.  Degenerate or non-finite inputs yield a widened window so callers
 * never receive an empty axis for a flat series.
 */
export function niceTicks(min: number, max: number, count = 5): number[] {
  if (!Number.isFinite(min) || !Number.isFinite(max)) {
    return [];
  }
  let lo = Math.min(min, max);
  let hi = Math.max(min, max);
  if (lo === hi) {
    const pad = Math.abs(lo) * 0.05 || 1;
    lo -= pad;
    hi += pad;
  }
  const step = niceStep((hi - lo) / Math.max(1, count - 1));
  const start = Math.floor(lo / step) * step;
  const ticks: number[] = [];
  // Snap to the step grid to suppress float accumulation noise.
  const pushSnapped = (tick: number) =>
    ticks.push(Number((Math.round(tick / step) * step).toPrecision(12)));
  for (let tick = start; tick <= hi + step / 2; tick += step) {
    pushSnapped(tick);
  }
  // The loop tolerance (step/2) can leave the last tick below hi, which
  // would clip series peaks when callers derive the axis domain from the
  // ticks.  Extend until the ticks cover [lo, hi] completely.
  while (ticks.length > 0 && ticks[ticks.length - 1] < hi) {
    pushSnapped(ticks[ticks.length - 1] + step);
  }
  return ticks;
}

/** Candidate time-tick steps in milliseconds, minutes up to a year. */
const TIME_STEPS_MS: number[] = [
  60_000,
  5 * 60_000,
  15 * 60_000,
  30 * 60_000,
  3_600_000,
  3 * 3_600_000,
  6 * 3_600_000,
  12 * 3_600_000,
  86_400_000,
  2 * 86_400_000,
  7 * 86_400_000,
  14 * 86_400_000,
  30 * 86_400_000,
  90 * 86_400_000,
  180 * 86_400_000,
  365 * 86_400_000,
];

/**
 * Compute epoch-ms tick positions between tMin and tMax on sensible UTC
 * boundaries (multiples of the chosen step from the epoch, which aligns
 * minute/hour/day boundaries in UTC).
 */
export function timeTicks(tMin: number, tMax: number, count = 5): number[] {
  if (!Number.isFinite(tMin) || !Number.isFinite(tMax) || tMax <= tMin) {
    return Number.isFinite(tMin) ? [tMin] : [];
  }
  const rawStep = (tMax - tMin) / Math.max(1, count);
  const step = TIME_STEPS_MS.find((candidate) => candidate >= rawStep)
    ?? Math.ceil(rawStep / (365 * 86_400_000)) * 365 * 86_400_000;
  const ticks: number[] = [];
  for (let tick = Math.ceil(tMin / step) * step; tick <= tMax; tick += step) {
    ticks.push(tick);
  }
  return ticks;
}

/** Linear scale factory mapping `domain` onto `range` (degenerate-safe). */
export function scaleLinear(
  domain: [number, number],
  range: [number, number],
): (value: number) => number {
  const [d0, d1] = domain;
  const [r0, r1] = range;
  const span = d1 - d0;
  if (span === 0 || !Number.isFinite(span)) {
    const mid = (r0 + r1) / 2;
    return () => mid;
  }
  return (value: number) => r0 + ((value - d0) / span) * (r1 - r0);
}

function project(
  points: ChartPoint[],
  width: number,
  height: number,
  xDomain: [number, number],
  yDomain: [number, number],
): Array<{ x: number; y: number }> {
  const x = scaleLinear(xDomain, [0, width]);
  // SVG y grows downward: domain max maps to 0.
  const y = scaleLinear(yDomain, [height, 0]);
  return points.filter(isFinitePoint).map((point) => ({ x: x(point.t), y: y(point.v) }));
}

/**
 * SVG path for a polyline through the finite points, projected into a
 * width*height box.  Returns '' for zero finite points; a single point
 * produces a short horizontal dash so the stroke is visible.
 */
export function linePath(
  points: ChartPoint[],
  width: number,
  height: number,
  xDomain: [number, number],
  yDomain: [number, number],
): string {
  const projected = project(points, width, height, xDomain, yDomain);
  if (projected.length === 0) {
    return '';
  }
  if (projected.length === 1) {
    const { x, y } = projected[0];
    return `M ${(x - 2).toFixed(2)} ${y.toFixed(2)} L ${(x + 2).toFixed(2)} ${y.toFixed(2)}`;
  }
  return projected
    .map(({ x, y }, index) => `${index === 0 ? 'M' : 'L'} ${x.toFixed(2)} ${y.toFixed(2)}`)
    .join(' ');
}

/**
 * Closed SVG path for an area between the series line and a horizontal
 * baseline value (e.g. 0 for drawdown).  Returns '' when fewer than two
 * finite points exist (a one-point area is invisible anyway).
 */
export function areaPath(
  points: ChartPoint[],
  width: number,
  height: number,
  xDomain: [number, number],
  yDomain: [number, number],
  baselineValue: number,
): string {
  const projected = project(points, width, height, xDomain, yDomain);
  if (projected.length < 2) {
    return '';
  }
  const yScale = scaleLinear(yDomain, [height, 0]);
  const baselineY = yScale(baselineValue).toFixed(2);
  const line = projected
    .map(({ x, y }, index) => `${index === 0 ? 'M' : 'L'} ${x.toFixed(2)} ${y.toFixed(2)}`)
    .join(' ');
  const first = projected[0];
  const last = projected[projected.length - 1];
  return `${line} L ${last.x.toFixed(2)} ${baselineY} L ${first.x.toFixed(2)} ${baselineY} Z`;
}

/**
 * Rebase a series so its first finite value equals `base` (index overlay:
 * benchmark rebased to the equity base, single shared y-axis).  If the
 * series has no finite non-zero anchor the points are returned unchanged.
 */
export function rebaseSeries(points: ChartPoint[], base: number): ChartPoint[] {
  const anchor = points.find((point) => isFinitePoint(point) && point.v !== 0);
  if (!anchor || !Number.isFinite(base)) {
    return points;
  }
  const factor = base / anchor.v;
  return points.map((point) => ({ t: point.t, v: point.v * factor }));
}

/**
 * Min-max bucket downsample: preserves the first and last points and, per
 * time bucket, the local minimum and maximum (in time order) so peaks and
 * troughs survive.  Never returns more than `maxPoints` points.
 */
export function downsample(points: ChartPoint[], maxPoints: number): ChartPoint[] {
  const finite = points.filter(isFinitePoint);
  if (maxPoints < 3 || finite.length <= maxPoints) {
    return finite;
  }
  const first = finite[0];
  const last = finite[finite.length - 1];
  const interior = finite.slice(1, -1);
  // Each bucket contributes up to two points (min + max).
  const bucketCount = Math.max(1, Math.floor((maxPoints - 2) / 2));
  const bucketSize = interior.length / bucketCount;
  const sampled: ChartPoint[] = [first];
  for (let bucket = 0; bucket < bucketCount; bucket += 1) {
    const start = Math.floor(bucket * bucketSize);
    const end = Math.min(interior.length, Math.floor((bucket + 1) * bucketSize));
    if (start >= end) {
      continue;
    }
    let minPoint = interior[start];
    let maxPoint = interior[start];
    for (let i = start + 1; i < end; i += 1) {
      if (interior[i].v < minPoint.v) minPoint = interior[i];
      if (interior[i].v > maxPoint.v) maxPoint = interior[i];
    }
    if (minPoint === maxPoint) {
      sampled.push(minPoint);
    } else {
      sampled.push(minPoint.t <= maxPoint.t ? minPoint : maxPoint);
      sampled.push(minPoint.t <= maxPoint.t ? maxPoint : minPoint);
    }
  }
  sampled.push(last);
  return sampled;
}
