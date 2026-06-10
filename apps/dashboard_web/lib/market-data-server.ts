import type { MarketDataPayload } from '@/lib/dashboard-contracts';
import { runUvPythonModuleJson } from '@/lib/python-runtime';

export async function loadMarketDataFromPython(): Promise<MarketDataPayload> {
  return runUvPythonModuleJson<MarketDataPayload>(
    'lumina_quant.dashboard.cutover_surfaces_service',
    '--fn', 'load_market_data_payload',
    '--point-limit', '240',
    '--fill-limit', '80',
    '--json',
  );
}
