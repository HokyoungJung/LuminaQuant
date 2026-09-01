import type { ExactWindowPayload } from '@/lib/dashboard-contracts';
import { runUvPythonModuleJson } from '@/lib/python-runtime';

export async function loadExactWindowFromPython(): Promise<ExactWindowPayload> {
  return runUvPythonModuleJson<ExactWindowPayload>(
    'lumina_quant.dashboard.exact_window_service',
    '--json',
  );
}
