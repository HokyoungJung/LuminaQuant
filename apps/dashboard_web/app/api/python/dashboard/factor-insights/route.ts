import { NextResponse } from 'next/server';
import { loadFactorInsightsFromPython } from '@/lib/factor-insights-server';

export const dynamic = 'force-dynamic';

export async function GET() {
  try {
    return NextResponse.json(await loadFactorInsightsFromPython());
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    return NextResponse.json(
      { error: 'dashboard_factor_insights_failed', detail },
      { status: 500 },
    );
  }
}
