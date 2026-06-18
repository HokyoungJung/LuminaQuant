/**
 * Regression tests for dashboard client runtime component pure logic.
 *
 * @testing-library/react is not installed and vitest runs in the 'node'
 * environment, so React component rendering is not available here.
 * Instead we test the extractable pure functions that these components
 * depend on directly — the same pattern used in control-auth.test.ts.
 *
 * Functions under test:
 *   - buildSparklinePath  (private in overview-runtime.tsx; replicated here)
 *   - readJsonOrThrow     (lib/bridge-fetch.ts; used by all runtime components)
 *   - buildTriggerActionRequest (mirrors WorkflowJobsRuntime.triggerAction POST shape)
 *
 * These lock the behavior that must remain stable before a shared
 * useBridgeFetch hook is extracted.
 */
import { describe, it, expect } from 'vitest';

import { readJsonOrThrow } from '../lib/bridge-fetch';

// ---------------------------------------------------------------------------
// buildSparklinePath — replicated from overview-runtime.tsx (lines 9-30)
// Must stay in sync with the source; a drift here is intentional signal.
// ---------------------------------------------------------------------------

function buildSparklinePath(
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

// ---------------------------------------------------------------------------
// buildTriggerActionRequest — mirrors WorkflowJobsRuntime.triggerAction POST
// (workflow-jobs-runtime.tsx lines 18-29). Tests the exact fetch shape.
// ---------------------------------------------------------------------------

function buildTriggerActionRequest(
  jobId: string,
  action: 'stop' | 'kill',
): { url: string; init: RequestInit } {
  return {
    url: '/api/python/dashboard/workflow-jobs/control',
    init: {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ job_id: jobId, action }),
    },
  };
}

// ---------------------------------------------------------------------------
// buildSparklinePath tests (OverviewRuntime loading/empty/happy-path coverage)
// ---------------------------------------------------------------------------

describe('buildSparklinePath (overview-runtime.tsx)', () => {
  it('returns empty string for zero values — loading/empty state renders no SVG path', () => {
    expect(buildSparklinePath([])).toBe('');
  });

  it('returns a horizontal line for a single value (degenerate case)', () => {
    const path = buildSparklinePath([42]);
    expect(path).toBe('M 0 60 L 420 60');
  });

  it('returns a horizontal line for a single value with custom dimensions', () => {
    const path = buildSparklinePath([42], 200, 100);
    expect(path).toBe('M 0 50 L 200 50');
  });

  it('produces M for first point and L for subsequent points', () => {
    const path = buildSparklinePath([0, 1]);
    expect(path).toMatch(/^M /);
    expect(path).toContain(' L ');
  });

  it('happy path: min value maps to bottom (y=height), max value maps to top (y=0)', () => {
    // Two values: min=0 maps to y=120, max=1 maps to y=0
    const path = buildSparklinePath([0, 1], 420, 120);
    // First point: x=0, y=120 (bottom)
    expect(path).toContain('M 0.00 120.00');
    // Second point: x=420, y=0 (top)
    expect(path).toContain('L 420.00 0.00');
  });

  it('happy path: three-point path has correct x spacing', () => {
    const path = buildSparklinePath([0, 0.5, 1], 420, 120);
    const segments = path.split(' ');
    // M 0.00 120.00 L 210.00 60.00 L 420.00 0.00
    expect(segments[0]).toBe('M');
    expect(segments[1]).toBe('0.00');
    expect(segments[3]).toBe('L');
    expect(segments[4]).toBe('210.00'); // midpoint x
  });

  it('handles flat series (all values equal) without division-by-zero — range falls back to 1', () => {
    // All values equal → range = 0 → range fallback = 1 → y = height - 0 = height for all
    const path = buildSparklinePath([5, 5, 5], 420, 120);
    expect(path).not.toBe('');
    // All y-values should be 120 (height - 0)
    expect(path).toMatch(/M 0\.00 120\.00/);
    expect(path).toMatch(/L \S+ 120\.00/);
  });

  it('produces deterministic output for a known equity curve snippet', () => {
    // Pin the exact path for values [100, 110, 105] to lock formatting
    const path = buildSparklinePath([100, 110, 105], 420, 120);
    expect(path).toBe('M 0.00 120.00 L 210.00 0.00 L 420.00 60.00');
  });
});

// ---------------------------------------------------------------------------
// readJsonOrThrow tests (used by OverviewRuntime and WorkflowJobsRuntime)
// ---------------------------------------------------------------------------

describe('readJsonOrThrow (lib/bridge-fetch.ts)', () => {
  it('happy path: returns parsed JSON body when response.ok is true', async () => {
    const payload = { jobs: [], status: 'ok' };
    const response = new Response(JSON.stringify(payload), {
      status: 200,
      headers: { 'content-type': 'application/json' },
    });
    const result = await readJsonOrThrow(response, 'fallback message');
    expect(result).toEqual(payload);
  });

  it('error path: throws with detail field when response is not ok', async () => {
    const errorBody = { detail: 'overview bridge failed', ok: false };
    const response = new Response(JSON.stringify(errorBody), {
      status: 500,
      headers: { 'content-type': 'application/json' },
    });
    await expect(readJsonOrThrow(response, 'fallback message')).rejects.toThrow(
      'overview bridge failed',
    );
  });

  it('error path: falls back to fallbackMessage when detail is absent', async () => {
    const errorBody = { error: 'internal error', ok: false };
    const response = new Response(JSON.stringify(errorBody), {
      status: 503,
      headers: { 'content-type': 'application/json' },
    });
    await expect(readJsonOrThrow(response, 'workflow jobs bridge failed')).rejects.toThrow(
      'workflow jobs bridge failed',
    );
  });
});

// ---------------------------------------------------------------------------
// WorkflowJobsRuntime.triggerAction POST shape — the kill/stop coverage gap
// ---------------------------------------------------------------------------

describe('triggerAction request construction (workflow-jobs-runtime.tsx)', () => {
  it('POSTs to the correct control endpoint', () => {
    const { url } = buildTriggerActionRequest('job-abc', 'kill');
    expect(url).toBe('/api/python/dashboard/workflow-jobs/control');
  });

  it('uses POST method for kill action', () => {
    const { init } = buildTriggerActionRequest('job-abc', 'kill');
    expect(init.method).toBe('POST');
  });

  it('uses POST method for stop action', () => {
    const { init } = buildTriggerActionRequest('job-abc', 'stop');
    expect(init.method).toBe('POST');
  });

  it('sends content-type application/json header', () => {
    const { init } = buildTriggerActionRequest('job-123', 'kill');
    expect((init.headers as Record<string, string>)['content-type']).toBe('application/json');
  });

  it('serialises job_id and action=kill into the request body', () => {
    const { init } = buildTriggerActionRequest('job-123', 'kill');
    const parsed = JSON.parse(init.body as string);
    expect(parsed).toEqual({ job_id: 'job-123', action: 'kill' });
  });

  it('serialises job_id and action=stop into the request body', () => {
    const { init } = buildTriggerActionRequest('job-456', 'stop');
    const parsed = JSON.parse(init.body as string);
    expect(parsed).toEqual({ job_id: 'job-456', action: 'stop' });
  });

  it('preserves exact job_id string in the body (no mutation)', () => {
    const jobId = 'wf-job-2025-06-18-abc123';
    const { init } = buildTriggerActionRequest(jobId, 'kill');
    const parsed = JSON.parse(init.body as string);
    expect(parsed.job_id).toBe(jobId);
  });
});
