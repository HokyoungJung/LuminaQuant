import type { RawDataPayload } from '@/lib/dashboard-contracts';
import { runUvPythonModuleJson } from '@/lib/python-runtime';

export async function loadRawDataFromPython(): Promise<RawDataPayload> {
  return runUvPythonModuleJson<RawDataPayload>(
    'lumina_quant.dashboard.cutover_surfaces_service',
    '--fn', 'load_raw_data_payload',
    '--point-limit', '60',
    '--json',
  );
}
