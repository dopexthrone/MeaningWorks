'use client';

import type { CompilePhase } from '@/lib/hooks/useCompile';
import type { TaskStatusResponse } from '@/lib/api/types';

interface CompilationProgressProps {
  phase: CompilePhase;
  taskId: string | null;
  pollData: TaskStatusResponse | null;
}

const STAGES = [
  { key: 'queued', label: 'Queued' },
  { key: 'intent', label: 'Intent Analysis' },
  { key: 'persona', label: 'Persona Mapping' },
  { key: 'entity', label: 'Entity Extraction' },
  { key: 'process', label: 'Process Modeling' },
  { key: 'synthesis', label: 'Synthesis' },
  { key: 'verify', label: 'Verification' },
  { key: 'materialize', label: 'Materialization' },
];

export function CompilationProgress({ phase, taskId, pollData }: CompilationProgressProps) {
  if (phase !== 'submitting' && phase !== 'queued' && phase !== 'polling') return null;

  const status = pollData?.status || 'pending';
  const activeIndex = status === 'running' ? 3 : status === 'pending' ? 0 : 0;

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-sm font-medium">Compiling...</h3>
        {taskId && (
          <span className="text-xs text-[var(--text-muted)] font-mono">{taskId.slice(0, 8)}</span>
        )}
      </div>

      <div className="space-y-2">
        {STAGES.map((stage, i) => {
          const isDone = i < activeIndex;
          const isActive = i === activeIndex;
          return (
            <div key={stage.key} className="flex items-center gap-3">
              <div className={`w-5 h-5 rounded-full flex items-center justify-center text-xs
                ${isDone ? 'bg-emerald-500/20 text-emerald-400' : isActive ? 'bg-brand-500/20 text-brand-500' : 'bg-[var(--bg-tertiary)] text-[var(--text-muted)]'}`}
              >
                {isDone ? (
                  <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
                  </svg>
                ) : isActive ? (
                  <svg className="w-3 h-3 animate-spin" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                  </svg>
                ) : (
                  <span>{i + 1}</span>
                )}
              </div>
              <span className={`text-sm ${isActive ? 'text-[var(--text-primary)] font-medium' : isDone ? 'text-emerald-400' : 'text-[var(--text-muted)]'}`}>
                {stage.label}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
