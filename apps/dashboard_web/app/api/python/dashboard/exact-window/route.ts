import { NextResponse } from 'next/server';
import { loadExactWindowFromPython } from '@/lib/exact-window-server';

export const dynamic = 'force-dynamic';

export async function GET() {
  try {
    return NextResponse.json(await loadExactWindowFromPython());
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    return NextResponse.json(
      { error: 'dashboard_exact_window_failed', detail },
      { status: 500 },
    );
  }
}
