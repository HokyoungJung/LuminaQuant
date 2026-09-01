import { NextResponse } from 'next/server';
import { loadAlphaEvidenceFromPython } from '@/lib/alpha-evidence-server';

export const dynamic = 'force-dynamic';

export async function GET() {
  try {
    return NextResponse.json(await loadAlphaEvidenceFromPython());
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    return NextResponse.json(
      { error: 'dashboard_alpha_evidence_failed', detail },
      { status: 500 },
    );
  }
}
