import { type NextRequest, NextResponse } from 'next/server';

import { loadRawDataFromPython } from '@/lib/raw-data-server';

export const dynamic = 'force-dynamic';

export async function GET(request: NextRequest) {
  try {
    const params = request.nextUrl.searchParams;
    return NextResponse.json(
      await loadRawDataFromPython({ runId: params.get('run_id') }),
    );
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    return NextResponse.json(
      { error: 'dashboard_raw_data_failed', detail },
      { status: 500 },
    );
  }
}
