import type { WorkflowJobsPayload } from '@/lib/dashboard-contracts';
import { runUvPythonModuleJson } from '@/lib/python-runtime';

export async function loadWorkflowJobsFromPython(): Promise<WorkflowJobsPayload> {
  return runUvPythonModuleJson<WorkflowJobsPayload>(
    'lumina_quant.dashboard.workflow_jobs_service',
    '--fn', 'load',
    '--limit', '10',
    '--json',
  );
}
