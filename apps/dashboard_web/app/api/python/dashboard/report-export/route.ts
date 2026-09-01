import { NextResponse } from 'next/server';

import { loadReportExportFromPython } from '@/lib/report-export-server';

export const dynamic = 'force-dynamic';

export async function GET(request: Request) {
  try {
    const runId = new URL(request.url).searchParams.get('run_id');
    return NextResponse.json(await loadReportExportFromPython({ runId }));
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    return NextResponse.json(
      { error: 'dashboard_report_export_failed', detail },
      { status: 500 },
    );
  }
}
