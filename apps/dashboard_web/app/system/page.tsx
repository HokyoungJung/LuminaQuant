import { dashboardBridgeContract, dashboardCutoverGate } from '@/lib/python-bridge';

export const metadata = { title: 'System · LuminaQuant' };

export default function SystemPage() {
  return (
    <div className="page-stack">
      <section>
        <p className="eyebrow">Runtime reference</p>
        <h2>How this dashboard runs</h2>
      </section>

      <section className="section-card">
        <div className="section-header">
          <div>
            <p className="eyebrow">Runtime</p>
            <h3>Launcher and memory budget</h3>
          </div>
          <div className="metric-badge">Target peak RSS: {dashboardBridgeContract.memoryBudget.targetPeakRssGb} GB</div>
        </div>
        <div className="metric-grid">
          <article>
            <span>Default entry point</span>
            <strong>{dashboardBridgeContract.defaultEntryPoint}</strong>
          </article>
          <article>
            <span>Host memory baseline</span>
            <strong>{dashboardBridgeContract.memoryBudget.hostRamGb} GB</strong>
          </article>
          <article>
            <span>Overview data route</span>
            <strong>{dashboardBridgeContract.compatibilityPath}</strong>
          </article>
          <article>
            <span>Launcher status</span>
            <strong>{dashboardCutoverGate.launcherStatus}</strong>
          </article>
        </div>
        <ul className="guidance-list">
          {dashboardBridgeContract.memoryBudget.guidance.map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
      </section>

      <section className="section-card">
        <div className="section-header">
          <div>
            <p className="eyebrow">Data sources</p>
            <h3>Page-to-service map</h3>
          </div>
          <div className="metric-badge">
            {
              dashboardBridgeContract.capabilities.filter(
                (capability) => capability.status === 'available',
              ).length
            }{' '}
            routes available
          </div>
        </div>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Surface</th>
                <th>Service module</th>
                <th>Route</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {dashboardBridgeContract.capabilities.map((capability) => (
                <tr key={capability.id}>
                  <td>{capability.title}</td>
                  <td>{capability.sourceModule}</td>
                  <td>{capability.nextRoute}</td>
                  <td>
                    <span className={`status-pill status-${capability.status}`}>{capability.status}</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="section-card">
        <div className="section-header">
          <div>
            <p className="eyebrow">Launch evidence</p>
            <h3>What this runtime serves</h3>
          </div>
          <div className="metric-badge">{dashboardCutoverGate.readyRoutes.length} data-backed routes</div>
        </div>
        <ul className="guidance-list">
          {dashboardCutoverGate.evidence.map((item) => (
            <li key={item.label}>
              <strong>{item.label}:</strong> {item.detail}
            </li>
          ))}
        </ul>
        <p>{dashboardCutoverGate.remainingGate}</p>
      </section>
    </div>
  );
}
