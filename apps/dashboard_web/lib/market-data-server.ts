import type { MarketDataPayload } from '@/lib/dashboard-contracts';
import { runUvPythonModuleJson } from '@/lib/python-runtime';
import { sanitizeRunId, sanitizeSymbol } from '@/lib/query-params';

export interface MarketDataQuery {
  runId?: string | null;
  symbol?: string | null;
}

export async function loadMarketDataFromPython(
  query: MarketDataQuery = {},
): Promise<MarketDataPayload> {
  const args = [
    '--fn', 'load_market_data_payload',
    '--point-limit', '240',
    '--fill-limit', '80',
    '--json',
  ];
  const runId = sanitizeRunId(query.runId);
  if (runId) {
    args.push('--run-id', runId);
  }
  const symbol = sanitizeSymbol(query.symbol);
  if (symbol) {
    args.push('--symbol', symbol);
  }
  return runUvPythonModuleJson<MarketDataPayload>(
    'lumina_quant.dashboard.cutover_surfaces_service',
    ...args,
  );
}
