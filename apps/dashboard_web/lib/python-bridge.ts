export type NavigationStatus = 'available' | 'planned';

export interface LegacyView {
  id: string;
  label: string;
  source: string;
}

export interface NavigationItem {
  id: string;
  href: string;
  label: string;
  summary: string;
  status: NavigationStatus;
}

export interface CapabilityItem {
  id: string;
  title: string;
  description: string;
  sourceModule: string;
  nextRoute: string;
  status: NavigationStatus;
}

export interface OverviewCard {
  title: string;
  description: string;
  status: NavigationStatus;
}

export const dashboardCutoverGate = {
  defaultLauncher: 'next',
  launcherStatus: 'available',
  readyRoutes: [
    '/performance-price',
    '/execution-analytics',
    '/market-data',
    '/optimization-insights',
    '/raw-data',
    '/report-export',
  ],
  evidence: [
    {
      label: 'Performance',
      detail: 'Equity, drawdown, benchmark, and trade markers are served at /performance-price.',
      status: 'available',
    },
    {
      label: 'Execution',
      detail: 'Fill quality, closed-trade outcomes, streaks, and order status are served at /execution-analytics.',
      status: 'available',
    },
    {
      label: 'Market',
      detail: 'OHLCV bars, indicator readings, and market summary telemetry are served at /market-data.',
      status: 'available',
    },
    {
      label: 'Candidates',
      detail: 'Optimization candidate quality and stage summaries are served at /optimization-insights.',
      status: 'available',
    },
    {
      label: 'Raw Data',
      detail: 'Row counts and capped table previews for the latest run are served at /raw-data.',
      status: 'available',
    },
    {
      label: 'Reports',
      detail: 'JSON and Markdown snapshot exports are served at /report-export.',
      status: 'available',
    },
    {
      label: 'Default launcher',
      detail: 'uv run lq dashboard --run starts the Next dashboard by default; the retired entrypoint remains only as an explicit compatibility stub.',
      status: 'available',
    },
  ],
  remainingGate:
    'This web app is the only primary dashboard runtime. The retired entrypoint remains only for explicit compatibility guidance.',
} as const;

export const dashboardBridgeContract = {
  defaultEntryPoint: 'uv run lq dashboard --run',
  compatibilityPath: '/api/python/dashboard/overview',
  memoryBudget: {
    hostRamGb: 8,
    targetPeakRssGb: 6.5,
    guidance: [
      'Keep heavy verification sequential.',
      'Prefer one active web build/test lane at a time.',
      'Treat this web app as the single primary dashboard runtime.',
    ],
  },
  legacyViews: [
    {
      id: 'dashboard-stub',
      label: 'Retired Dashboard Stub',
      source: 'apps/dashboard/app.py',
    },
  ] satisfies LegacyView[],
  capabilities: [
    {
      id: 'overview',
      title: 'Overview',
      description: 'Landing view with run performance metrics, equity and drawdown, and recent activity.',
      sourceModule: 'src/lumina_quant/dashboard/overview_service.py',
      nextRoute: '/',
      status: 'available',
    },
    {
      id: 'python-compatibility',
      title: 'Python compatibility contract',
      description: 'Overview data contract served to the web app through the bridge route.',
      sourceModule: 'src/lumina_quant/dashboard/bridge.py',
      nextRoute: '/api/python/dashboard/overview',
      status: 'available',
    },
    {
      id: 'exact-window',
      title: 'Research Window',
      description: 'Latest research-window artifact summary with portfolio fallback from the research bundle.',
      sourceModule: 'src/lumina_quant/dashboard/exact_window_service.py',
      nextRoute: '/exact-window',
      status: 'available',
    },
    {
      id: 'performance-price',
      title: 'Performance',
      description: 'Equity, drawdown, benchmark price, funding, and trade markers for a selected run.',
      sourceModule: 'src/lumina_quant/dashboard/cutover_surfaces_service.py',
      nextRoute: '/performance-price',
      status: 'available',
    },
    {
      id: 'execution-analytics',
      title: 'Execution',
      description: 'Fill quality, closed-trade outcomes, streaks, and order-status distribution for a run.',
      sourceModule: 'src/lumina_quant/dashboard/cutover_surfaces_service.py',
      nextRoute: '/execution-analytics',
      status: 'available',
    },
    {
      id: 'market-data',
      title: 'Market',
      description: 'Recent OHLCV bars, indicator readings, and market context for a run and symbol.',
      sourceModule: 'src/lumina_quant/dashboard/cutover_surfaces_service.py',
      nextRoute: '/market-data',
      status: 'available',
    },
    {
      id: 'optimization-insights',
      title: 'Candidates',
      description: 'Optimization candidate quality, stage medians, and best-parameter previews.',
      sourceModule: 'src/lumina_quant/dashboard/cutover_surfaces_service.py',
      nextRoute: '/optimization-insights',
      status: 'available',
    },
    {
      id: 'workflow-jobs',
      title: 'Jobs',
      description: 'Managed backtest, optimization, and live job status with start/stop controls.',
      sourceModule: 'src/lumina_quant/dashboard/workflow_jobs_service.py',
      nextRoute: '/workflows',
      status: 'available',
    },
    {
      id: 'risk-health',
      title: 'Risk & Health',
      description: 'Recent risk events, heartbeats, and order-state changes for the active run.',
      sourceModule: 'src/lumina_quant/dashboard/cutover_surfaces_service.py',
      nextRoute: '/risk-health',
      status: 'available',
    },
    {
      id: 'report-export',
      title: 'Reports',
      description: 'JSON and Markdown snapshot exports of the current run state.',
      sourceModule: 'src/lumina_quant/dashboard/cutover_surfaces_service.py',
      nextRoute: '/report-export',
      status: 'available',
    },
    {
      id: 'raw-data',
      title: 'Raw Data',
      description: 'Row counts and capped previews for the runs, execution, market, and optimization tables.',
      sourceModule: 'src/lumina_quant/dashboard/cutover_surfaces_service.py',
      nextRoute: '/raw-data',
      status: 'available',
    },
  ] satisfies CapabilityItem[],
} as const;

export const navigationItems: NavigationItem[] = [
  {
    id: 'overview',
    href: '/',
    label: 'Overview',
    summary: 'How the latest run is performing: headline metrics, equity, and recent activity.',
    status: 'available',
  },
  {
    id: 'performance-price',
    href: '/performance-price',
    label: 'Performance',
    summary: 'Equity versus benchmark, drawdown, funding, and trade markers for a selected run.',
    status: 'available',
  },
  {
    id: 'market-data',
    href: '/market-data',
    label: 'Market',
    summary: 'Recent price bars and indicator readings for a selected run and symbol.',
    status: 'available',
  },
  {
    id: 'optimization-insights',
    href: '/optimization-insights',
    label: 'Candidates',
    summary: 'Optimization candidate quality, stage summaries, and best parameters.',
    status: 'available',
  },
  {
    id: 'exact-window',
    href: '/exact-window',
    label: 'Research Window',
    summary: 'Latest research-window artifact summary and portfolio snapshot.',
    status: 'available',
  },
  {
    id: 'factor-insights',
    href: '/factor-insights',
    label: 'Factors',
    summary: 'Factor IC decay heatmap and the pending candidate review queue.',
    status: 'available',
  },
  {
    id: 'alpha-evidence',
    href: '/alpha-evidence',
    label: 'Alpha Evidence',
    summary: 'Evaluation evidence and verdicts behind strategy candidates.',
    status: 'available',
  },
  {
    id: 'execution-analytics',
    href: '/execution-analytics',
    label: 'Execution',
    summary: 'Fill quality, closed-trade outcomes, streaks, and order status for a run.',
    status: 'available',
  },
  {
    id: 'risk-health',
    href: '/risk-health',
    label: 'Risk & Health',
    summary: 'Recent risk events, heartbeats, and order-state changes.',
    status: 'available',
  },
  {
    id: 'workflows',
    href: '/workflows',
    label: 'Jobs',
    summary: 'Managed backtest, optimization, and live job status with controls.',
    status: 'available',
  },
  {
    id: 'raw-data',
    href: '/raw-data',
    label: 'Raw Data',
    summary: 'Row counts and capped previews of the underlying data tables.',
    status: 'available',
  },
  {
    id: 'report-export',
    href: '/report-export',
    label: 'Reports',
    summary: 'Snapshot exports of the current run as JSON or Markdown.',
    status: 'available',
  },
  {
    id: 'system',
    href: '/system',
    label: 'System',
    summary: 'Runtime reference: data-source routes, launcher, and memory budget.',
    status: 'available',
  },
];

export function buildOverviewCards(): OverviewCard[] {
  return dashboardBridgeContract.capabilities.map((capability) => ({
    title: capability.title,
    description: capability.description,
    status: capability.status,
  }));
}
