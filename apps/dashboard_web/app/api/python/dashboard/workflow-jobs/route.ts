import { NextResponse } from 'next/server';
import { loadWorkflowJobsFromPython } from '@/lib/workflow-jobs-server';

export const dynamic = 'force-dynamic';

export async function GET() {
  try {
    return NextResponse.json(await loadWorkflowJobsFromPython());
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    return NextResponse.json(
      {
        error: 'dashboard_workflow_jobs_failed',
        detail,
      },
      { status: 500 },
    );
  }
}
