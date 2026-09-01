import { WorkflowJobsRuntime } from '@/components/workflow-jobs-runtime';

export const metadata = { title: 'Workflow Jobs · LuminaQuant' };

export default function WorkflowJobsPage() {
  return (
    <div className="page-stack">
      <section className="hero-card">
        <p className="eyebrow">Operations</p>
        <h2>What is running right now?</h2>
        <p>
          Backtest, optimize, and live jobs with their current status — stop or kill an active job
          from here.
        </p>
      </section>
      <section className="section-card">
        <div className="section-header">
          <div>
            <p className="eyebrow">Job queue</p>
            <h3>Recent jobs</h3>
          </div>
        </div>
        <WorkflowJobsRuntime />
      </section>
    </div>
  );
}
