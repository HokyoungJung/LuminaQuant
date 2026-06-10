import type { ReportExportPayload } from '@/lib/dashboard-contracts';
import { runUvPythonModuleJson } from '@/lib/python-runtime';

export async function loadReportExportFromPython(): Promise<ReportExportPayload> {
  return runUvPythonModuleJson<ReportExportPayload>(
    'lumina_quant.dashboard.cutover_surfaces_service',
    '--fn', 'load_report_export_payload',
    '--point-limit', '240',
    '--fill-limit', '200',
    '--event-limit', '50',
    '--json',
  );
}
