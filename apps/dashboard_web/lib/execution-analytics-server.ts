import type { ExecutionAnalyticsPayload } from '@/lib/dashboard-contracts';
import { runUvPythonModuleJson } from '@/lib/python-runtime';

export async function loadExecutionAnalyticsFromPython(): Promise<ExecutionAnalyticsPayload> {
  return runUvPythonModuleJson<ExecutionAnalyticsPayload>(
    'lumina_quant.dashboard.cutover_surfaces_service',
    '--fn', 'load_execution_analytics_payload',
    '--fill-limit', '200',
    '--order-limit', '200',
    '--json',
  );
}
