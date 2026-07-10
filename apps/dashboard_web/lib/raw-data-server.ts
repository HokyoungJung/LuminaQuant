import type { RawDataPayload } from '@/lib/dashboard-contracts';
import { runUvPythonModuleJson } from '@/lib/python-runtime';
import { sanitizeRunId } from '@/lib/query-params';

export interface RawDataQuery {
  runId?: string | null;
}

export async function loadRawDataFromPython(query: RawDataQuery = {}): Promise<RawDataPayload> {
  const args = [
    '--fn', 'load_raw_data_payload',
    '--point-limit', '60',
    '--json',
  ];
  const runId = sanitizeRunId(query.runId);
  if (runId) {
    args.push('--run-id', runId);
  }
  return runUvPythonModuleJson<RawDataPayload>(
    'lumina_quant.dashboard.cutover_surfaces_service',
    ...args,
  );
}
