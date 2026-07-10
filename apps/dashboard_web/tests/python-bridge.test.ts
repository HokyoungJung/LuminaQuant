import { describe, expect, it } from 'vitest';

import {
  buildOverviewCards,
  dashboardBridgeContract,
  dashboardCutoverGate,
  navigationItems,
} from '../lib/python-bridge';

describe('dashboard bridge contract', () => {
  it('keeps the 8GB baseline explicit', () => {
    expect(dashboardBridgeContract.memoryBudget.hostRamGb).toBe(8);
    expect(dashboardBridgeContract.memoryBudget.targetPeakRssGb).toBeLessThan(8);
  });

  it('maps both legacy dashboard entry views', () => {
    expect(dashboardBridgeContract.legacyViews.map((view) => view.id)).toEqual([
      'dashboard-stub',
    ]);
  });

  it('marks the migrated routes as available in operator navigation order', () => {
    const availableRoutes = navigationItems.filter((item) => item.status === 'available');

    expect(availableRoutes).toHaveLength(13);
    expect(availableRoutes.map((item) => item.href)).toEqual([
      '/',
      '/performance-price',
      '/market-data',
      '/optimization-insights',
      '/exact-window',
      '/factor-insights',
      '/alpha-evidence',
      '/execution-analytics',
      '/risk-health',
      '/workflows',
      '/raw-data',
      '/report-export',
      '/system',
    ]);
  });

  it('builds overview cards from capability metadata', () => {
    const cards = buildOverviewCards();

    expect(cards).toHaveLength(dashboardBridgeContract.capabilities.length);
    expect(cards.some((card) => card.title === 'Python compatibility contract')).toBe(true);
  });

  it('keeps cutover-gate evidence explicit after flipping the launcher to Next', () => {
    expect(dashboardCutoverGate.defaultLauncher).toBe('next');
    expect(dashboardCutoverGate.readyRoutes).toEqual([
      '/performance-price',
      '/execution-analytics',
      '/market-data',
      '/optimization-insights',
      '/raw-data',
      '/report-export',
    ]);
    expect(dashboardCutoverGate.evidence.at(-1)?.detail).toContain('Next dashboard');
  });
});
