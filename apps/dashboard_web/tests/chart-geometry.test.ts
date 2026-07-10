import { describe, expect, it } from 'vitest';

import {
  areaPath,
  downsample,
  linePath,
  niceTicks,
  rebaseSeries,
  scaleLinear,
  timeTicks,
  type ChartPoint,
} from '../lib/chart-geometry';

const HOUR = 3_600_000;

function points(values: number[], t0 = 0, dt = HOUR): ChartPoint[] {
  return values.map((v, index) => ({ t: t0 + index * dt, v }));
}

describe('niceTicks', () => {
  it('produces 1/2/5-step ticks covering the domain', () => {
    const ticks = niceTicks(0, 97, 5);
    expect(ticks.length).toBeGreaterThanOrEqual(4);
    expect(ticks.length).toBeLessThanOrEqual(8);
    expect(ticks[0]).toBeLessThanOrEqual(0);
    expect(ticks[ticks.length - 1]).toBeGreaterThanOrEqual(97);
    const step = ticks[1] - ticks[0];
    const mantissa = step / 10 ** Math.floor(Math.log10(step));
    expect([1, 2, 5]).toContain(Math.round(mantissa));
    for (let i = 1; i < ticks.length; i += 1) {
      expect(ticks[i] - ticks[i - 1]).toBeCloseTo(step, 8);
    }
  });

  it('widens a flat domain instead of returning a single tick', () => {
    const ticks = niceTicks(10, 10, 5);
    expect(ticks.length).toBeGreaterThanOrEqual(2);
    expect(ticks[0]).toBeLessThanOrEqual(10);
    expect(ticks[ticks.length - 1]).toBeGreaterThanOrEqual(10);
  });

  it('returns [] for non-finite input', () => {
    expect(niceTicks(NaN, 10)).toEqual([]);
    expect(niceTicks(0, Infinity)).toEqual([]);
  });
});

describe('timeTicks', () => {
  it('returns sorted ticks inside the domain on step boundaries', () => {
    const t0 = Date.UTC(2026, 0, 1);
    const t1 = Date.UTC(2026, 0, 2);
    const ticks = timeTicks(t0, t1, 5);
    expect(ticks.length).toBeGreaterThanOrEqual(3);
    for (const tick of ticks) {
      expect(tick).toBeGreaterThanOrEqual(t0);
      expect(tick).toBeLessThanOrEqual(t1);
    }
    expect([...ticks].sort((a, b) => a - b)).toEqual(ticks);
  });

  it('handles a degenerate domain', () => {
    expect(timeTicks(1000, 1000, 5)).toEqual([1000]);
  });
});

describe('scaleLinear', () => {
  it('maps domain onto range and stays finite on degenerate domains', () => {
    const scale = scaleLinear([0, 10], [0, 100]);
    expect(scale(5)).toBe(50);
    const flat = scaleLinear([3, 3], [0, 100]);
    expect(Number.isFinite(flat(3))).toBe(true);
  });
});

describe('rebaseSeries', () => {
  it('indexes the series so the first finite value equals the base', () => {
    const rebased = rebaseSeries(points([50, 55, 45]), 100);
    expect(rebased.map((p) => p.v)).toEqual([100, 110, 90]);
  });

  it('anchors on the first finite non-zero value', () => {
    const rebased = rebaseSeries([{ t: 0, v: NaN }, { t: 1, v: 200 }, { t: 2, v: 100 }], 100);
    expect(rebased[1].v).toBe(100);
    expect(rebased[2].v).toBe(50);
  });

  it('returns points unchanged when no usable anchor exists', () => {
    const flat = points([0, 0]);
    expect(rebaseSeries(flat, 100)).toEqual(flat);
  });
});

describe('downsample', () => {
  it('keeps first/last points and global extremes under the cap', () => {
    const values = Array.from({ length: 5000 }, (_, i) => Math.sin(i / 40) * 100);
    values[1234] = 999; // global max
    values[4321] = -999; // global min
    const source = points(values);
    const sampled = downsample(source, 200);
    expect(sampled.length).toBeLessThanOrEqual(200);
    expect(sampled[0]).toEqual(source[0]);
    expect(sampled[sampled.length - 1]).toEqual(source[source.length - 1]);
    expect(sampled.some((p) => p.v === 999)).toBe(true);
    expect(sampled.some((p) => p.v === -999)).toBe(true);
    // Time order preserved.
    for (let i = 1; i < sampled.length; i += 1) {
      expect(sampled[i].t).toBeGreaterThanOrEqual(sampled[i - 1].t);
    }
  });

  it('passes short series through untouched (minus non-finite points)', () => {
    const source = [...points([1, 2, 3]), { t: 99, v: NaN }];
    expect(downsample(source, 100)).toEqual(points([1, 2, 3]));
  });
});

describe('linePath / areaPath', () => {
  const xDomain: [number, number] = [0, 3 * HOUR];
  const yDomain: [number, number] = [0, 10];

  it('produces finite paths and skips non-finite points', () => {
    const path = linePath(
      [...points([1, 5]), { t: 2 * HOUR, v: NaN }, { t: 3 * HOUR, v: 9 }],
      720,
      200,
      xDomain,
      yDomain,
    );
    expect(path.startsWith('M ')).toBe(true);
    expect(path).not.toContain('NaN');
    expect(path).not.toContain('Infinity');
  });

  it('handles 0- and 1-point series without NaN output', () => {
    expect(linePath([], 720, 200, xDomain, yDomain)).toBe('');
    const single = linePath(points([5]), 720, 200, xDomain, yDomain);
    expect(single).not.toContain('NaN');
    expect(single.length).toBeGreaterThan(0);
  });

  it('closes the area path at the baseline value', () => {
    const path = areaPath(points([2, 8, 4, 6]), 720, 200, xDomain, yDomain, 0);
    expect(path.endsWith('Z')).toBe(true);
    expect(path).not.toContain('NaN');
    // Baseline 0 maps to the bottom edge (y = height).
    expect(path).toContain('200.00');
  });

  it('returns "" for an area with fewer than two points', () => {
    expect(areaPath(points([5]), 720, 200, xDomain, yDomain, 0)).toBe('');
  });
});
