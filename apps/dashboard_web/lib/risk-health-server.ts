import type { RiskHealthPayload } from '@/lib/dashboard-contracts';
import { runUvPythonModuleJson } from '@/lib/python-runtime';

export async function loadRiskHealthFromPython(): Promise<RiskHealthPayload> {
  return runUvPythonModuleJson<RiskHealthPayload>(
    'lumina_quant.dashboard.risk_health_service',
    '--limit', '25',
    '--json',
  );
}
