import type { AlphaEvidencePayload } from '@/lib/dashboard-contracts';
import { runUvPythonModuleJson } from '@/lib/python-runtime';

let pendingAlphaEvidence: Promise<AlphaEvidencePayload> | null = null;

export async function loadAlphaEvidenceFromPython(): Promise<AlphaEvidencePayload> {
  if (pendingAlphaEvidence !== null) {
    return pendingAlphaEvidence;
  }
  pendingAlphaEvidence = runUvPythonModuleJson<AlphaEvidencePayload>(
    'lumina_quant.dashboard.alpha_evidence_service',
    '--json',
  ).finally(() => {
    pendingAlphaEvidence = null;
  });
  return pendingAlphaEvidence;
}
