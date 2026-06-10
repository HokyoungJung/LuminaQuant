import type { OptimizationInsightsPayload } from '@/lib/dashboard-contracts';
import { runUvPythonModuleJson } from '@/lib/python-runtime';

export async function loadOptimizationInsightsFromPython(): Promise<OptimizationInsightsPayload> {
  return runUvPythonModuleJson<OptimizationInsightsPayload>(
    'lumina_quant.dashboard.cutover_surfaces_service',
    '--fn', 'load_optimization_insights_payload',
    '--point-limit', '200',
    '--json',
  );
}
