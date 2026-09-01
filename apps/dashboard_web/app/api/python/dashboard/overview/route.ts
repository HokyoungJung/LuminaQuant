import { NextResponse } from 'next/server';
import { loadOverviewPayloadFromPython } from '@/lib/python-bridge-server';

export const dynamic = 'force-dynamic';

export async function GET() {
  try {
    return NextResponse.json(await loadOverviewPayloadFromPython());
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    return NextResponse.json(
      {
        error: 'dashboard_python_bridge_failed',
        detail,
      },
      { status: 500 },
    );
  }
}
