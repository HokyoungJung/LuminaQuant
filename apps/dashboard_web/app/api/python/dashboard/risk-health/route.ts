import { NextResponse } from 'next/server';
import { loadRiskHealthFromPython } from '@/lib/risk-health-server';

export const dynamic = 'force-dynamic';

export async function GET() {
  try {
    return NextResponse.json(await loadRiskHealthFromPython());
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    return NextResponse.json(
      { error: 'dashboard_risk_health_failed', detail },
      { status: 500 },
    );
  }
}
