'use client';

import { useState } from 'react';

import { ConfirmButton } from '@/components/confirm-button';
import { PageContextBar } from '@/components/page-context-bar';
import { SurfaceState } from '@/components/surface-state';
import { readJsonOrThrow } from '@/lib/bridge-fetch';
import type { WorkflowJobRecord, WorkflowJobsPayload } from '@/lib/dashboard-contracts';
import { formatCompactTimestamp, formatDuration, statusPillClass } from '@/lib/format';
import { useBridgeFetch } from '@/lib/use-bridge-fetch';

const POLL_MS = 10_000;

const TERMINAL_STATUSES = new Set([
  'succeeded',
  'failed',
  'cancelled',
  'canceled',
  'completed',
  'killed',
]);

interface PendingJobAction {
  jobId: string;
  action: 'stop' | 'kill';
}

function isTerminalStatus(status: string): boolean {
  return TERMINAL_STATUSES.has(status.trim().toLowerCase());
}

function jobDuration(job: WorkflowJobRecord): string {
  if (!job.started_at) {
    return 'n/a';
  }
  const started = Date.parse(job.started_at);
  if (!Number.isFinite(started)) {
    return 'n/a';
  }
  if (job.ended_at) {
    const ended = Date.parse(job.ended_at);
    if (!Number.isFinite(ended)) {
      return 'n/a';
    }
    return formatDuration(ended - started);
  }
  if (isTerminalStatus(job.status)) {
    return 'n/a';
  }
  return `running for ${formatDuration(Date.now() - started)}`;
}

const CONTROL_TOKEN_STORAGE_KEY = 'lq-dashboard-control-token';

function readStoredControlToken(): string {
  if (typeof window === 'undefined') {
    return '';
  }
  return window.sessionStorage.getItem(CONTROL_TOKEN_STORAGE_KEY) ?? '';
}

export function WorkflowJobsRuntime() {
  const { payload, error, loading, refetch, lastFetchedAt } = useBridgeFetch<WorkflowJobsPayload>(
    '/api/python/dashboard/workflow-jobs',
    'workflow jobs feed failed',
    { pollMs: POLL_MS },
  );
  const [pendingAction, setPendingAction] = useState<PendingJobAction | null>(null);
  const [actionError, setActionError] = useState<string>('');
  const [controlToken, setControlToken] = useState<string>(readStoredControlToken);

  function updateControlToken(value: string) {
    setControlToken(value);
    if (typeof window !== 'undefined') {
      if (value) {
        window.sessionStorage.setItem(CONTROL_TOKEN_STORAGE_KEY, value);
      } else {
        window.sessionStorage.removeItem(CONTROL_TOKEN_STORAGE_KEY);
      }
    }
  }

  async function triggerAction(jobId: string, action: 'stop' | 'kill') {
    setPendingAction({ jobId, action });
    setActionError('');
    try {
      const headers: Record<string, string> = { 'content-type': 'application/json' };
      if (controlToken.trim()) {
        headers['x-lq-control-token'] = controlToken.trim();
      }
      const response = await fetch('/api/python/dashboard/workflow-jobs/control', {
        method: 'POST',
        headers,
        body: JSON.stringify({ job_id: jobId, action }),
      });
      // The control route maps ok:false results onto non-2xx statuses, so the
      // shared bridge error reader covers every failure shape.
      await readJsonOrThrow<Record<string, unknown>>(response, `${action} request failed`);
      refetch();
    } catch (actionFailure: unknown) {
      const message =
        actionFailure instanceof Error ? actionFailure.message : String(actionFailure);
      const guidance = message.includes('control_token_not_configured')
        ? ' Set LQ_DASHBOARD_CONTROL_TOKEN in the dashboard environment (job control fails closed without it), then paste the same token into the control-token field above.'
        : message.includes('unauthorized')
          ? ' Paste the LQ_DASHBOARD_CONTROL_TOKEN value into the control-token field above and retry.'
          : '';
      setActionError(
        `${action === 'stop' ? 'Stop' : 'Kill'} failed for job ${jobId}: ${message}.${guidance}`,
      );
    } finally {
      setPendingAction(null);
    }
  }

  const status = payload?.status ?? null;
  const jobs = payload?.jobs ?? [];

  return (
    <div className="page-stack">
      <PageContextBar
        asOf={payload?.as_of}
        status={status}
        lastFetchedAt={lastFetchedAt}
        onRefresh={refetch}
        loading={loading}
      />
      <SurfaceState status={status} error={error} surface="workflow jobs" onRetry={refetch} />
      <div className="run-selector">
        <label htmlFor="lq-control-token">Control token</label>
        <input
          id="lq-control-token"
          type="password"
          autoComplete="off"
          value={controlToken}
          onChange={(event) => updateControlToken(event.target.value)}
          placeholder="LQ_DASHBOARD_CONTROL_TOKEN"
          title="Job control fails closed without the shared control token. Paste the LQ_DASHBOARD_CONTROL_TOKEN value; it is kept only for this browser session."
        />
      </div>
      {actionError ? (
        <div className="surface-state surface-state-error" role="alert">
          <p className="surface-state-title">Job action failed</p>
          <p className="surface-state-detail">{actionError}</p>
        </div>
      ) : null}
      {payload === null && !error ? <p>Loading workflow jobs…</p> : null}
      {status === 'ok' && jobs.length === 0 ? (
        <p>No workflow jobs recorded yet. Launch a backtest, optimize, or live workflow and it will appear here.</p>
      ) : null}
      {jobs.length > 0 ? (
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Workflow</th>
                <th>Status</th>
                <th>Mode</th>
                <th>Strategy</th>
                <th>Run ID</th>
                <th>Started</th>
                <th>Ended</th>
                <th>Duration</th>
                <th>Actions</th>
              </tr>
            </thead>
            <tbody>
              {jobs.map((job) => {
                const terminal = isTerminalStatus(job.status);
                const jobPending = pendingAction?.jobId === job.job_id;
                const terminalReason = `Job is already ${job.status.toLowerCase() || 'finished'}; stop and kill only apply to active jobs.`;
                return (
                  <tr key={job.job_id}>
                    <td>{job.workflow || 'n/a'}</td>
                    <td>
                      <span className={statusPillClass(job.status)}>{job.status || 'unknown'}</span>
                    </td>
                    <td>{job.requested_mode || 'n/a'}</td>
                    <td>{job.strategy || 'n/a'}</td>
                    <td>{job.run_id ? <code className="run-chip">{job.run_id}</code> : 'n/a'}</td>
                    <td>{formatCompactTimestamp(job.started_at)}</td>
                    <td>{formatCompactTimestamp(job.ended_at)}</td>
                    <td>{jobDuration(job)}</td>
                    <td>
                      {terminal ? (
                        <span className="confirm-button-group" title={terminalReason}>
                          <button
                            type="button"
                            className="action-button confirm-button"
                            disabled
                            title={terminalReason}
                          >
                            Stop
                          </button>
                          <button
                            type="button"
                            className="action-button confirm-button confirm-button-danger"
                            disabled
                            title={terminalReason}
                          >
                            Kill
                          </button>
                        </span>
                      ) : (
                        <span className="confirm-button-group" aria-busy={jobPending}>
                          <ConfirmButton
                            label="Stop"
                            onConfirm={() => void triggerAction(job.job_id, 'stop')}
                            pending={jobPending && pendingAction?.action === 'stop'}
                            disabled={jobPending}
                          />
                          <ConfirmButton
                            label="Kill"
                            variant="danger"
                            onConfirm={() => void triggerAction(job.job_id, 'kill')}
                            pending={jobPending && pendingAction?.action === 'kill'}
                            disabled={jobPending}
                          />
                        </span>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      ) : null}
    </div>
  );
}
