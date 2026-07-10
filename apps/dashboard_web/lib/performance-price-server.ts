import type { PerformancePricePayload } from '@/lib/dashboard-contracts';
import { runUvPythonModuleJson } from '@/lib/python-runtime';
import { sanitizeRunId } from '@/lib/query-params';

export interface PerformancePriceQuery {
  runId?: string | null;
}

export async function loadPerformancePriceFromPython(
  query: PerformancePriceQuery = {},
): Promise<PerformancePricePayload> {
  const args = [
    '--fn', 'load_performance_price_payload',
    '--point-limit', '240',
    '--fill-limit', '80',
  ];
  const runId = sanitizeRunId(query.runId);
  if (runId) {
    args.push('--run-id', runId);
  }
  args.push('--json');
  return runUvPythonModuleJson<PerformancePricePayload>(
    'lumina_quant.dashboard.cutover_surfaces_service',
    ...args,
  );
}
