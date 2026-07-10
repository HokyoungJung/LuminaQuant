import type { ReportExportPayload } from '@/lib/dashboard-contracts';
import { runUvPythonModuleJson } from '@/lib/python-runtime';
import { sanitizeRunId } from '@/lib/query-params';

export interface ReportExportQuery {
  runId?: string | null;
}

export async function loadReportExportFromPython(
  query: ReportExportQuery = {},
): Promise<ReportExportPayload> {
  const args = [
    '--fn', 'load_report_export_payload',
    '--point-limit', '240',
    '--fill-limit', '200',
    '--event-limit', '50',
    '--json',
  ];
  const runId = sanitizeRunId(query.runId);
  if (runId) {
    args.push('--run-id', runId);
  }
  return runUvPythonModuleJson<ReportExportPayload>(
    'lumina_quant.dashboard.cutover_surfaces_service',
    ...args,
  );
}
