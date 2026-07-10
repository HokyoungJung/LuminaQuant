'use client';

import { useId } from 'react';

export interface RunSelectorOption {
  run_id: string;
  mode: string;
  status: string;
  strategy: string;
  started_at: string | null;
}

export interface RunSelectorProps {
  runs: RunSelectorOption[];
  value: string;
  onChange: (runId: string) => void;
  disabled?: boolean;
}

/** Accessible run picker: newest run first, labeled native <select>. */
export function RunSelector({ runs, value, onChange, disabled = false }: RunSelectorProps) {
  const selectId = useId();
  const sorted = [...runs].sort((a, b) => {
    // Newest first; runs without a start time sink to the bottom.
    if (a.started_at === b.started_at) return 0;
    if (a.started_at === null) return 1;
    if (b.started_at === null) return -1;
    return a.started_at < b.started_at ? 1 : -1;
  });
  return (
    <div className="run-selector">
      <label htmlFor={selectId}>Run</label>
      <select
        id={selectId}
        value={value}
        onChange={(event) => onChange(event.target.value)}
        disabled={disabled}
      >
        {sorted.map((run) => (
          <option key={run.run_id} value={run.run_id}>
            {`${run.run_id} · ${run.mode} · ${run.strategy} · ${run.status}`}
          </option>
        ))}
      </select>
    </div>
  );
}
