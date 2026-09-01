import { NextResponse } from 'next/server';

import { loadExecutionAnalyticsFromPython } from '@/lib/execution-analytics-server';

export const dynamic = 'force-dynamic';

export async function GET(request: Request) {
  try {
    const runId = new URL(request.url).searchParams.get('run_id');
    return NextResponse.json(await loadExecutionAnalyticsFromPython({ runId }));
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    return NextResponse.json(
      { error: 'dashboard_execution_analytics_failed', detail },
      { status: 500 },
    );
  }
}
