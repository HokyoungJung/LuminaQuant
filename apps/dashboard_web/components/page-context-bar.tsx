'use client';

import { formatCompactTimestamp, statusPillClass } from '@/lib/format';

export interface PageContextBarProps {
  asOf?: string | null;
  runId?: string | null;
  status?: string | null;
  lastFetchedAt?: string | null;
  onRefresh?: () => void;
  loading?: boolean;
}

/**
 * Compact provenance bar pages mount under their heading: data as-of time,
 * run id, surface status, client fetch time, and a manual refresh.
 */
export function PageContextBar({
  asOf,
  runId,
  status,
  lastFetchedAt,
  onRefresh,
  loading = false,
}: PageContextBarProps) {
  return (
    <div className="context-bar">
      <span className="context-bar-item">as of {formatCompactTimestamp(asOf)}</span>
      {runId ? <code className="run-chip">{runId}</code> : null}
      {status ? <span className={statusPillClass(status)}>{status}</span> : null}
      {lastFetchedAt ? (
        <span className="context-bar-item context-bar-muted">
          fetched {formatCompactTimestamp(lastFetchedAt).slice(11)}
        </span>
      ) : null}
      {onRefresh ? (
        <button
          type="button"
          className="action-button context-bar-refresh"
          onClick={onRefresh}
          disabled={loading}
          aria-busy={loading}
        >
          {loading ? 'Refreshing…' : 'Refresh'}
        </button>
      ) : null}
    </div>
  );
}
