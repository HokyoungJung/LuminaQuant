import { cache } from 'react';

import type { OverviewPayload } from '@/lib/dashboard-contracts';
import { runUvPythonModuleJson } from '@/lib/python-runtime';

export const loadOverviewPayloadFromPython = cache(async (): Promise<OverviewPayload> => (
  runUvPythonModuleJson<OverviewPayload>('lumina_quant.dashboard.bridge', '--json')
));
