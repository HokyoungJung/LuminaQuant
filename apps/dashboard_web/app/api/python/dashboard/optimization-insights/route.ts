import { NextResponse } from 'next/server';

import { loadOptimizationInsightsFromPython } from '@/lib/optimization-insights-server';

export const dynamic = 'force-dynamic';

export async function GET() {
  try {
    return NextResponse.json(await loadOptimizationInsightsFromPython());
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    return NextResponse.json(
      { error: 'dashboard_optimization_insights_failed', detail },
      { status: 500 },
    );
  }
}
