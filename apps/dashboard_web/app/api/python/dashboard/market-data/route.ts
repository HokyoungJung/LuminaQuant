import { type NextRequest, NextResponse } from 'next/server';

import { loadMarketDataFromPython } from '@/lib/market-data-server';

export const dynamic = 'force-dynamic';

export async function GET(request: NextRequest) {
  try {
    const params = request.nextUrl.searchParams;
    return NextResponse.json(
      await loadMarketDataFromPython({
        runId: params.get('run_id'),
        symbol: params.get('symbol'),
      }),
    );
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    return NextResponse.json(
      { error: 'dashboard_market_data_failed', detail },
      { status: 500 },
    );
  }
}
