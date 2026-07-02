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
      label: 'Performance & Price parity slice',
      detail: 'Python-backed route is available at /performance-price for the equity, drawdown, and benchmark view.',
      status: 'available',
    },
    {
      label: 'Execution Analytics parity slice',
      detail: 'Python-backed route is available at /execution-analytics for fills, streaks, and order status breakdowns.',
      status: 'available',
    },
    {
      label: 'Market Data parity surface',
      detail: 'Python-backed route is available at /market-data for latest OHLCV context and market summary telemetry.',
      status: 'available',
    },
    {
      label: 'Optimization Insights parity surface',
      detail: 'Python-backed route is available at /optimization-insights for latest candidate quality and stage summaries.',
      status: 'available',
    },
    {
      label: 'Raw Data parity surface',
      detail: 'Python-backed route is available at /raw-data with frame counts and capped previews for the latest run.',
      status: 'available',
    },
    {
      label: 'Report Export parity slice',
      detail: 'Python-backed route is available at /report-export with JSON and Markdown snapshot exports.',
      status: 'available',
    },
    {
      label: 'Default launcher flipped',
      detail: 'uv run lq dashboard --run now targets the Next dashboard by default, while the retired legacy entrypoint only remains as an explicit compatibility stub.',
      status: 'available',
    },
  ],
  remainingGate:
    'Next is the only primary dashboard runtime. The retired legacy path only remains for explicit compatibility guidance.',
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
      'Treat Next as the single primary dashboard runtime.',
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
      title: 'Overview placeholder',
      description:
        'First migrated route that anchors the Next dashboard while staying data-contract first.',
      sourceModule: 'src/lumina_quant/dashboard/overview_service.py',
      nextRoute: '/',
      status: 'available',
    },
    {
      id: 'python-compatibility',
      title: 'Python compatibility contract',
      description:
        'Python-backed compatibility metadata for the first slice, exposed to Next.js via the bridge route.',
      sourceModule: 'src/lumina_quant/dashboard/bridge.py',
      nextRoute: '/api/python/dashboard/overview',
      status: 'available',
    },
    {
      id: 'exact-window',
      title: 'Exact-window migration',
      description: 'Latest exact-window artifact summary and portfolio fallback parity from the Python bundle.',
      sourceModule: 'src/lumina_quant/dashboard/exact_window_service.py',
      nextRoute: '/exact-window',
      status: 'available',
    },
    {
      id: 'performance-price',
      title: 'Performance & Price',
      description: 'Equity, drawdown, benchmark price, funding, and trade-marker parity from Python state.',
      sourceModule: 'src/lumina_quant/dashboard/cutover_surfaces_service.py',
      nextRoute: '/performance-price',
      status: 'available',
    },
    {
      id: 'execution-analytics',
      title: 'Execution Analytics',
      description: 'Fill quality, closed-trade outcomes, streaks, and order-status distribution from Python telemetry.',
      sourceModule: 'src/lumina_quant/dashboard/cutover_surfaces_service.py',
      nextRoute: '/execution-analytics',
      status: 'available',
    },
    {
      id: 'market-data',
      title: 'Market Data',
      description: 'Latest market OHLCV context, summary metrics, and capped price-bar parity for the active run.',
      sourceModule: 'src/lumina_quant/dashboard/cutover_surfaces_service.py',
      nextRoute: '/market-data',
      status: 'available',
    },
    {
      id: 'optimization-insights',
      title: 'Optimization Insights',
      description: 'Optimization candidate quality, stage medians, and best-parameter previews from Python state.',
      sourceModule: 'src/lumina_quant/dashboard/cutover_surfaces_service.py',
      nextRoute: '/optimization-insights',
      status: 'available',
    },
    {
      id: 'workflow-jobs',
      title: 'Workflow jobs',
      description: 'Managed backtest/optimize/live job status and control parity for the web dashboard.',
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
      title: 'Report Export',
      description: 'JSON + Markdown snapshot export preview for the retired-to-Next cutover state.',
      sourceModule: 'src/lumina_quant/dashboard/cutover_surfaces_service.py',
      nextRoute: '/report-export',
      status: 'available',
    },
    {
      id: 'raw-data',
      title: 'Raw Data',
      description: 'Capped frame-count and preview parity for the latest runs, execution, market, and optimization tables.',
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
    summary: 'First parity slice backed by the Python compatibility contract.',
    status: 'available',
  },
  {
    id: 'performance-price',
    href: '/performance-price',
    label: 'Performance & Price',
    summary: 'Equity, drawdown, benchmark, funding, and trade-marker parity from the latest run.',
    status: 'available',
  },
  {
    id: 'execution-analytics',
    href: '/execution-analytics',
    label: 'Execution Analytics',
    summary: 'Closed-trade outcomes, streaks, and order status telemetry for the active run.',
    status: 'available',
  },
  {
    id: 'market-data',
    href: '/market-data',
    label: 'Market Data',
    summary: 'Latest OHLCV bars, market context, and price-range telemetry for the active run.',
    status: 'available',
  },
  {
    id: 'optimization-insights',
    href: '/optimization-insights',
    label: 'Optimization Insights',
    summary: 'Candidate quality, stage breakdowns, and best-parameter previews from Python optimization results.',
    status: 'available',
  },
  {
    id: 'workflows',
    href: '/workflows',
    label: 'Workflow Jobs',
    summary: 'Managed run status, strategy, and mode parity from the Python workflow store.',
    status: 'available',
  },
  {
    id: 'risk-health',
    href: '/risk-health',
    label: 'Risk & Health',
    summary: 'Recent risk, heartbeat, and order-state telemetry from the latest run.',
    status: 'available',
  },
  {
    id: 'exact-window',
    href: '/exact-window',
    label: 'Exact-window',
    summary: 'Latest exact-window artifact summary from the Python research bundle.',
    status: 'available',
  },
  {
    id: 'report-export',
    href: '/report-export',
    label: 'Report Export',
    summary: 'Snapshot JSON/Markdown export preview plus explicit cutover gate evidence.',
    status: 'available',
  },
  {
    id: 'raw-data',
    href: '/raw-data',
    label: 'Raw Data',
    summary: 'Frame counts and capped previews for runs, execution, market, optimization, and workflow state.',
    status: 'available',
  },
  {
    id: 'factor-insights',
    href: '/factor-insights',
    label: 'Factor Insights',
    summary: 'Read-only factor IC-heatmap (rank-decay per factor) and the pending candidate review queue.',
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
