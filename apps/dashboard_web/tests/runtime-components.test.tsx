// @vitest-environment jsdom
/**
 * Runtime component tests for the operator dashboard.
 *
 * Rendering runs under jsdom with @testing-library/react; the bridge fetch
 * layer is stubbed per-test with a fake global fetch so each runtime renders
 * the exact payload under test.
 */
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen } from '@testing-library/react';

import { ConfirmButton } from '../components/confirm-button';
import { MarketDataRuntime } from '../components/market-data-runtime';
import { PerformancePriceRuntime } from '../components/performance-price-runtime';
import { WorkflowJobsRuntime } from '../components/workflow-jobs-runtime';
import { readJsonOrThrow } from '../lib/bridge-fetch';
import { buildSparklinePath } from '../lib/format';
import type {
  MarketDataPayload,
  PerformancePricePayload,
  WorkflowJobsPayload,
} from '../lib/dashboard-contracts';

// React reads this flag to allow act()-wrapped updates outside a renderer.
(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true;

function jsonResponse(payload: unknown, status = 200): Response {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { 'content-type': 'application/json' },
  });
}

function stubFetchWith(payload: unknown): ReturnType<typeof vi.fn> {
  const fetchMock = vi.fn(async () => jsonResponse(payload));
  vi.stubGlobal('fetch', fetchMock);
  return fetchMock;
}

beforeEach(() => {
  vi.useRealTimers();
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

// ---------------------------------------------------------------------------
// buildSparklinePath (lib/format.ts)
// ---------------------------------------------------------------------------

describe('buildSparklinePath (lib/format.ts)', () => {
  it('returns empty string for zero values and a mid-line for one value', () => {
    expect(buildSparklinePath([])).toBe('');
    expect(buildSparklinePath([42])).toBe('M 0 60 L 420 60');
  });

  it('produces deterministic output for a known equity curve snippet', () => {
    expect(buildSparklinePath([100, 110, 105], 420, 120)).toBe(
      'M 0.00 120.00 L 210.00 0.00 L 420.00 60.00',
    );
  });
});

// ---------------------------------------------------------------------------
// readJsonOrThrow (lib/bridge-fetch.ts) — shared by every runtime component
// ---------------------------------------------------------------------------

describe('readJsonOrThrow (lib/bridge-fetch.ts)', () => {
  it('happy path: returns parsed JSON body when response.ok is true', async () => {
    const payload = { jobs: [], status: 'ok' };
    const result = await readJsonOrThrow(jsonResponse(payload), 'fallback message');
    expect(result).toEqual(payload);
  });

  it('error path: throws with detail field when response is not ok', async () => {
    const errorBody = { detail: 'overview bridge failed', ok: false };
    await expect(readJsonOrThrow(jsonResponse(errorBody, 500), 'fallback message')).rejects.toThrow(
      'overview bridge failed',
    );
  });

  it("error path: surfaces the JSON body's error field when detail is absent", async () => {
    const errorBody = { error: 'job_not_found', ok: false };
    await expect(
      readJsonOrThrow(jsonResponse(errorBody, 400), 'workflow jobs bridge failed'),
    ).rejects.toThrow('job_not_found');
  });

  it('error path: falls back to the HTTP status when the body carries no message', async () => {
    const response = new Response('<html>bad gateway</html>', { status: 502 });
    await expect(readJsonOrThrow(response, 'bridge request failed')).rejects.toThrow(
      'bridge request failed: HTTP 502',
    );
  });
});

// ---------------------------------------------------------------------------
// WorkflowJobsRuntime
// ---------------------------------------------------------------------------

const RUNNING_JOB = {
  job_id: 'job-1',
  workflow: 'backtest',
  status: 'RUNNING',
  requested_mode: 'paper',
  strategy: 'RsiStrategy',
  run_id: 'run-1',
  started_at: '2026-07-10T08:00:00+00:00',
  ended_at: null,
};

describe('WorkflowJobsRuntime', () => {
  it('renders actionable DSN guidance when the payload status is missing_dsn', async () => {
    const payload: WorkflowJobsPayload = {
      as_of: '2026-07-10T09:00:00+00:00',
      jobs: [],
      status: 'missing_dsn',
    };
    stubFetchWith(payload);

    render(<WorkflowJobsRuntime />);

    expect(await screen.findByText('LQ_POSTGRES_DSN')).toBeTruthy();
    // The old empty-state copy must not mask the configuration problem.
    expect(screen.queryByText(/No workflow jobs recorded yet/)).toBeNull();
  });

  it('renders payload-driven status pills plus two-step Stop/Kill for active jobs', async () => {
    const payload: WorkflowJobsPayload = {
      as_of: '2026-07-10T09:00:00+00:00',
      jobs: [RUNNING_JOB],
      status: 'ok',
    };
    stubFetchWith(payload);

    render(<WorkflowJobsRuntime />);

    const statusPill = await screen.findByText('RUNNING');
    // The pill class derives from the payload status — never a hardcoded 'live'.
    expect(statusPill.className).toContain('status-running');
    expect(screen.getByText('Stop')).toBeTruthy();
    expect(screen.getByText('Kill')).toBeTruthy();
    // First click only arms; no POST goes out until the confirm click.
    fireEvent.click(screen.getByText('Kill'));
    expect(screen.getByText('Confirm kill?')).toBeTruthy();
    const fetchMock = global.fetch as ReturnType<typeof vi.fn>;
    const postCalls = fetchMock.mock.calls.filter(
      (call) => (call[1] as RequestInit | undefined)?.method === 'POST',
    );
    expect(postCalls).toHaveLength(0);
  });
});

// ---------------------------------------------------------------------------
// ConfirmButton
// ---------------------------------------------------------------------------

describe('ConfirmButton', () => {
  it('fires onConfirm exactly once through the two-step arm/confirm flow', () => {
    const onConfirm = vi.fn();
    render(<ConfirmButton label="Stop" onConfirm={onConfirm} />);

    // Arming click must not fire the action.
    fireEvent.click(screen.getByText('Stop'));
    expect(onConfirm).not.toHaveBeenCalled();

    fireEvent.click(screen.getByText('Confirm stop?'));
    expect(onConfirm).toHaveBeenCalledTimes(1);
    // Confirm disarms: the armed confirm control is gone again.
    expect(screen.queryByText('Confirm stop?')).toBeNull();
  });

  it('cancel disarms without firing the action', () => {
    const onConfirm = vi.fn();
    render(<ConfirmButton label="Kill" variant="danger" onConfirm={onConfirm} />);

    fireEvent.click(screen.getByText('Kill'));
    fireEvent.click(screen.getByText('Cancel'));

    expect(onConfirm).not.toHaveBeenCalled();
    expect(screen.getByText('Kill')).toBeTruthy();
  });
});

// ---------------------------------------------------------------------------
// PerformancePriceRuntime
// ---------------------------------------------------------------------------

function performancePricePayload(): PerformancePricePayload {
  return {
    as_of: '2026-07-10T09:00:00+00:00',
    run_id: 'run-123',
    status: 'ok',
    source: { mode: 'next', backend: 'python', status: 'ok', run_id: 'run-123' },
    summary_metrics: [
      { key: 'total_return', label: 'Total Return', value: 0.345206 },
      { key: 'latest_equity', label: 'Latest Equity', value: 1345.21 },
    ],
    performance_metrics: { cagr: 0.22, sharpe_ratio: 1.31, max_drawdown: 0.08 },
    equity_curve: [
      { timestamp: '2026-07-01T00:00:00+00:00', equity: 1000.0 },
      { timestamp: '2026-07-02T00:00:00+00:00', equity: 1345.21 },
    ],
    drawdown_curve: [
      { timestamp: '2026-07-01T00:00:00+00:00', drawdown: 0.0 },
      { timestamp: '2026-07-02T00:00:00+00:00', drawdown: -0.08 },
    ],
    equity_window: {
      start: '2026-07-01T00:00:00+00:00',
      end: '2026-07-02T00:00:00+00:00',
      points_total: 2,
      points_returned: 2,
      truncated: false,
    },
    benchmark_symbol: 'BTCUSDT',
    benchmark_curve: [
      { timestamp: '2026-07-01T00:00:00+00:00', price: 100.0 },
      { timestamp: '2026-07-02T00:00:00+00:00', price: 108.0 },
    ],
    funding_curve: [],
    trade_markers: [],
    runs: [],
  };
}

describe('PerformancePriceRuntime', () => {
  it('renders percent metrics multiplied by 100 (raw-fraction payload convention)', async () => {
    stubFetchWith(performancePricePayload());

    render(<PerformancePriceRuntime />);

    expect(await screen.findByText('34.52%')).toBeTruthy();
  });

  it('renders a two-series chart legend when a benchmark curve is present', async () => {
    stubFetchWith(performancePricePayload());

    render(<PerformancePriceRuntime />);

    expect(await screen.findByText('Equity')).toBeTruthy();
    // Label derives from payload.benchmark_symbol (no hardcoded 'BTC').
    expect(screen.getByText('BTCUSDT (rebased)')).toBeTruthy();
  });
});

// ---------------------------------------------------------------------------
// MarketDataRuntime
// ---------------------------------------------------------------------------

function marketDataPayload(): MarketDataPayload {
  return {
    as_of: '2026-07-10T09:00:00+00:00',
    run_id: 'run-123',
    status: 'ok',
    market_context: {
      symbol: 'BTC/USDT',
      timeframe: '1m',
      timeframe_clamped: false,
      exchange: 'binance',
      strategy: 'RsiStrategy',
      market_db_path: '/data/market.parquet',
      source: 'parquet',
    },
    summary_metrics: [],
    recent_bars: [
      {
        timestamp: '2026-07-10T08:59:00+00:00',
        open: 100,
        high: 101,
        low: 99,
        close: 100.5,
        volume: 12,
      },
    ],
    indicator_summary: [],
    warnings: [],
    runs: [],
    symbols: ['BTC/USDT', 'ETH/USDT'],
  };
}

describe('MarketDataRuntime', () => {
  it('renders the payload symbols as symbol-selector options', async () => {
    stubFetchWith(marketDataPayload());

    render(<MarketDataRuntime />);

    const select = (await screen.findByLabelText('Symbol')) as HTMLSelectElement;
    const options = [...select.options].map((option) => option.value);
    expect(options).toEqual(['BTC/USDT', 'ETH/USDT']);
    expect(select.value).toBe('BTC/USDT');
  });
});
