import type { PerformancePricePayload } from '@/lib/dashboard-contracts';
import { runUvPythonModuleJson } from '@/lib/python-runtime';

export async function loadPerformancePriceFromPython(): Promise<PerformancePricePayload> {
  return runUvPythonModuleJson<PerformancePricePayload>(
    'lumina_quant.dashboard.cutover_surfaces_service',
    '--fn', 'load_performance_price_payload',
    '--point-limit', '240',
    '--fill-limit', '80',
    '--json',
  );
}
