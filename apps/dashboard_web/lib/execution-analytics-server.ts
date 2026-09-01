import type { ExecutionAnalyticsPayload } from '@/lib/dashboard-contracts';
import { runUvPythonModuleJson } from '@/lib/python-runtime';
import { sanitizeRunId } from '@/lib/query-params';

export interface ExecutionAnalyticsQuery {
  runId?: string | null;
}

export async function loadExecutionAnalyticsFromPython(
  query: ExecutionAnalyticsQuery = {},
): Promise<ExecutionAnalyticsPayload> {
  const args = [
    '--fn', 'load_execution_analytics_payload',
    '--fill-limit', '200',
    '--order-limit', '200',
    '--json',
  ];
  const runId = sanitizeRunId(query.runId);
  if (runId) {
    args.push('--run-id', runId);
  }
  return runUvPythonModuleJson<ExecutionAnalyticsPayload>(
    'lumina_quant.dashboard.cutover_surfaces_service',
    ...args,
  );
}
