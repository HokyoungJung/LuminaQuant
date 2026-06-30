import type { AlphaEvidencePayload } from '@/lib/dashboard-contracts';
import { runUvPythonModuleJson } from '@/lib/python-runtime';

export async function loadAlphaEvidenceFromPython(): Promise<AlphaEvidencePayload> {
  return runUvPythonModuleJson<AlphaEvidencePayload>(
    'lumina_quant.dashboard.alpha_evidence_service',
    '--json',
  );
}
