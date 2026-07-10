'use client';

import { PageContextBar } from '@/components/page-context-bar';
import { SurfaceState } from '@/components/surface-state';
import type { ExactWindowPayload } from '@/lib/dashboard-contracts';
import { buildExactWindowEmptyState } from '@/lib/exact-window-status';
import {
  formatCompactTimestamp,
  formatNumber,
  formatPercent,
  pnlClass,
} from '@/lib/format';
import { useBridgeFetch } from '@/lib/use-bridge-fetch';

/** Drawdown magnitude reads as a loss regardless of sign convention. */
function drawdownClass(value: number | null): 'pnl-neg' | undefined {
  if (typeof value !== 'number' || !Number.isFinite(value) || value === 0) {
    return undefined;
  }
  return 'pnl-neg';
}

/** Promoted rows get an ok pill; rejected rows list warn-toned reason chips. */
function PromotionCell({
  promoted,
  rejectReasons,
}: {
  promoted: boolean;
  rejectReasons: string[];
}) {
  if (promoted) {
    return <span className="status-pill status-ok">promoted</span>;
  }
  if (rejectReasons.length === 0) {
    return <>n/a</>;
  }
  return (
    <span style={{ display: 'inline-flex', flexWrap: 'wrap', gap: '0.3rem' }}>
      {rejectReasons.map((reason) => (
        <span key={reason} className="warn-chip">
          {reason}
        </span>
      ))}
    </span>
  );
}

/** Portfolio weight rendered as percent plus an inline proportion bar. */
function WeightCell({ weight }: { weight: number | null }) {
  const pct =
    typeof weight === 'number' && Number.isFinite(weight)
      ? Math.max(0, Math.min(100, weight * 100))
      : 0;
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
      <span aria-hidden className="inline-bar">
        <span className="inline-bar-fill" style={{ width: `${pct}%` }} />
      </span>
      <span>{formatPercent(weight)}</span>
    </div>
  );
}

export function ExactWindowRuntime() {
  const { payload, error, loading, refetch, lastFetchedAt } = useBridgeFetch<ExactWindowPayload>(
    '/api/python/dashboard/exact-window',
    'exact-window research request failed',
  );

  if (error) {
    return <SurfaceState error={error} surface="exact-window research" onRetry={refetch} />;
  }
  if (payload === null) {
    return <p>Loading exact-window research summary…</p>;
  }

  const contextBar = (
    <PageContextBar
      asOf={payload.as_of}
      status={payload.status}
      lastFetchedAt={lastFetchedAt}
      onRefresh={refetch}
      loading={loading}
    />
  );

  if (payload.status !== 'ok') {
    const emptyState = buildExactWindowEmptyState(payload.status, payload.error);

    return (
      <div className="page-stack">
        {contextBar}
        <section className="section-card">
          <div className="section-header">
            <div>
              <p className="eyebrow">Exact-window research</p>
              <h3>Research bundle unavailable</h3>
            </div>
            <div className="metric-badge">{payload.status}</div>
          </div>
          <p>{emptyState.message}</p>
          {emptyState.detail ? <p>{emptyState.detail}</p> : null}
          {payload.root || payload.run_root ? (
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>Field</th>
                    <th>Value</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>Bundle Root</td>
                    <td>{payload.root || 'n/a'}</td>
                  </tr>
                  <tr>
                    <td>Run Root</td>
                    <td>{payload.run_root || 'n/a'}</td>
                  </tr>
                </tbody>
              </table>
            </div>
          ) : null}
        </section>
      </div>
    );
  }

  const windowEntries = Object.entries(payload.time_window);

  return (
    <div className="page-stack">
      {contextBar}
      <div className="metric-grid">
        <article>
          <span>Candidate Count</span>
          <strong>{payload.summary.candidate_count}</strong>
        </article>
        <article>
          <span>Promoted</span>
          <strong>{payload.summary.promoted_count}</strong>
        </article>
        <article>
          <span>Next Action</span>
          <strong>{payload.decision.next_action || 'n/a'}</strong>
        </article>
        <article>
          <span>Peak RSS</span>
          <strong>{formatNumber(payload.memory.peak_rss_mib)} MiB</strong>
        </article>
      </div>

      <section className="section-card">
        <div className="section-header">
          <div>
            <p className="eyebrow">Execution profile</p>
            <h3>Scope and guardrails</h3>
          </div>
          <div className="metric-badge">{payload.memory.status || 'artifact'}</div>
        </div>
        <div className="metric-grid">
          <article>
            <span>Requested timeframes</span>
            <strong>{payload.summary.requested_timeframes.join(', ') || 'n/a'}</strong>
          </article>
          <article>
            <span>Requested symbols</span>
            <strong>{payload.summary.requested_symbols.join(', ') || 'n/a'}</strong>
          </article>
          <article>
            <span>Low-RAM profile</span>
            <strong>{payload.summary.low_ram_profile ? 'enabled' : 'disabled'}</strong>
          </article>
          <article>
            <span>Construction basis</span>
            <strong>{payload.portfolio.construction_basis || 'n/a'}</strong>
          </article>
        </div>
        {windowEntries.length > 0 ? (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Window</th>
                  <th>Value</th>
                </tr>
              </thead>
              <tbody>
                {windowEntries.map(([label, value]) => (
                  <tr key={label}>
                    <td>{label}</td>
                    <td>{value}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : null}
      </section>

      <section className="section-card">
        <div className="section-header">
          <div>
            <p className="eyebrow">Provenance</p>
            <h3>Latest research bundle</h3>
          </div>
          <div className="metric-badge">
            {payload.generated_at ? formatCompactTimestamp(payload.generated_at) : 'pending'}
          </div>
        </div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Field</th>
                <th>Value</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td>Generated At</td>
                <td>{formatCompactTimestamp(payload.generated_at)}</td>
              </tr>
              <tr>
                <td>Bundle Root</td>
                <td>{payload.root || 'n/a'}</td>
              </tr>
              <tr>
                <td>Run Root</td>
                <td>{payload.run_root || 'n/a'}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </section>

      <section className="section-card">
        <div className="section-header">
          <div>
            <p className="eyebrow">Per-timeframe results</p>
            <h3>Best row per timeframe</h3>
          </div>
          <div className="metric-badge">{payload.timeframes.length} rows</div>
        </div>
        {payload.timeframes.length > 0 ? (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Timeframe</th>
                  <th>Candidate</th>
                  <th>Family</th>
                  <th>OOS Return</th>
                  <th>Sharpe</th>
                  <th>Max DD</th>
                  <th>Trades</th>
                  <th>Outcome</th>
                </tr>
              </thead>
              <tbody>
                {payload.timeframes.map((row) => (
                  <tr key={`${row.timeframe}-${row.candidate_id}`}>
                    <td>{row.timeframe || 'n/a'}</td>
                    <td>{row.name || row.candidate_id || 'n/a'}</td>
                    <td>{row.family || 'n/a'}</td>
                    <td className={pnlClass(row.oos_return) || undefined}>
                      {formatPercent(row.oos_return)}
                    </td>
                    <td>{formatNumber(row.oos_sharpe, { digits: 3 })}</td>
                    <td className={drawdownClass(row.oos_max_drawdown)}>
                      {formatPercent(row.oos_max_drawdown)}
                    </td>
                    <td>{formatNumber(row.trade_count, { digits: 0 })}</td>
                    <td>
                      <PromotionCell promoted={row.promoted} rejectReasons={row.reject_reasons} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p>No timeframe-level summary rows were exported in the latest research bundle.</p>
        )}
      </section>

      <section className="section-card">
        <div className="section-header">
          <div>
            <p className="eyebrow">Top candidates</p>
            <h3>Strongest strategy rows</h3>
          </div>
          <div className="metric-badge">{payload.top_candidates.length} rows</div>
        </div>
        {payload.top_candidates.length > 0 ? (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Candidate</th>
                  <th>Timeframe</th>
                  <th>Family</th>
                  <th>OOS Return</th>
                  <th>Sharpe</th>
                  <th>Max DD</th>
                  <th>Outcome</th>
                </tr>
              </thead>
              <tbody>
                {payload.top_candidates.map((row) => (
                  <tr key={`${row.candidate_id}-${row.timeframe}`}>
                    <td>{row.name || row.candidate_id || 'n/a'}</td>
                    <td>{row.timeframe || 'n/a'}</td>
                    <td>{row.family || 'n/a'}</td>
                    <td className={pnlClass(row.oos_return) || undefined}>
                      {formatPercent(row.oos_return)}
                    </td>
                    <td>{formatNumber(row.oos_sharpe, { digits: 3 })}</td>
                    <td className={drawdownClass(row.oos_max_drawdown)}>
                      {formatPercent(row.oos_max_drawdown)}
                    </td>
                    <td>
                      <PromotionCell promoted={row.promoted} rejectReasons={row.reject_reasons} />
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p>No top-strategy rows are available yet.</p>
        )}
      </section>

      <section className="section-card">
        <div className="section-header">
          <div>
            <p className="eyebrow">Portfolio construction</p>
            <h3>Current fallback construction</h3>
          </div>
          <div className="metric-badge">{payload.portfolio_weights.length} sleeves</div>
        </div>
        <div className="metric-grid">
          <article>
            <span>Portfolio OOS Return</span>
            <strong className={pnlClass(payload.portfolio.oos_return) || undefined}>
              {formatPercent(payload.portfolio.oos_return)}
            </strong>
          </article>
          <article>
            <span>Portfolio OOS Sharpe</span>
            <strong>{formatNumber(payload.portfolio.oos_sharpe, { digits: 3 })}</strong>
          </article>
          <article>
            <span>Portfolio Max DD</span>
            <strong className={drawdownClass(payload.portfolio.oos_max_drawdown)}>
              {formatPercent(payload.portfolio.oos_max_drawdown)}
            </strong>
          </article>
          <article>
            <span>Valid Strategy Found</span>
            <strong>{payload.decision.valid_strategy_found ? 'yes' : 'no'}</strong>
          </article>
        </div>
        {payload.portfolio_weights.length > 0 ? (
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Name</th>
                  <th>Timeframe</th>
                  <th>Family</th>
                  <th>Weight</th>
                  <th>OOS Return</th>
                  <th>OOS Sharpe</th>
                </tr>
              </thead>
              <tbody>
                {payload.portfolio_weights.map((row, index) => (
                  <tr key={`${row.name}-${index}`}>
                    <td>{row.name || 'n/a'}</td>
                    <td>{row.timeframe || 'n/a'}</td>
                    <td>{row.family || 'n/a'}</td>
                    <td>
                      <WeightCell weight={row.weight} />
                    </td>
                    <td className={pnlClass(row.oos_return) || undefined}>
                      {formatPercent(row.oos_return)}
                    </td>
                    <td>{formatNumber(row.oos_sharpe, { digits: 3 })}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <p>No portfolio weights were exported in the latest exact-window bundle.</p>
        )}
      </section>

      {payload.notes.length > 0 ? (
        <section className="section-card">
          <div className="section-header">
            <div>
              <p className="eyebrow">Run notes</p>
              <h3>Context captured with the bundle</h3>
            </div>
          </div>
          <div className="table-wrap">
            <table>
              <thead>
                <tr>
                  <th>Note</th>
                  <th>Value</th>
                </tr>
              </thead>
              <tbody>
                {payload.notes.map((note) => (
                  <tr key={note.label}>
                    <td>{note.label}</td>
                    <td>{note.value}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      ) : null}

      {payload.warnings.length > 0 ? (
        <section className="section-card">
          <div className="section-header">
            <div>
              <p className="eyebrow">Warnings</p>
              <h3>Artifact drift to watch</h3>
            </div>
          </div>
          <ul className="guidance-list">
            {payload.warnings.map((warning) => (
              <li key={warning}>{warning}</li>
            ))}
          </ul>
        </section>
      ) : null}
    </div>
  );
}
