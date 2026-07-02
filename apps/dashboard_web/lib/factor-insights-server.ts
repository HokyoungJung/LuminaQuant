import type { FactorInsightsPayload } from '@/lib/dashboard-contracts';
import { runUvPythonModuleJson } from '@/lib/python-runtime';

/**
 * Read-only factor IC-heatmap + candidate-queue feed.
 *
 * Module-mode invocation of the Python `factor_insights_service`, matching the
 * additive dashboard bridge surface.  This route is read-only: it never mutates
 * state and never touches broker/order-gateway code.
 */
export async function loadFactorInsightsFromPython(): Promise<FactorInsightsPayload> {
  return runUvPythonModuleJson<FactorInsightsPayload>(
    'lumina_quant.dashboard.factor_insights_service',
    '--json',
  );
}
