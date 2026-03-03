'use client';

import type { CompilePhase } from '@/lib/hooks/useCompile';

interface CompileButtonProps {
  phase: CompilePhase;
  onCompile: () => void;
  onCancel: () => void;
  disabled?: boolean;
}

const LABELS: Record<CompilePhase, string> = {
  idle: 'Compile',
  submitting: 'Submitting...',
  queued: 'Queued...',
  polling: 'Compiling...',
  complete: 'Compile',
  error: 'Retry',
  cancelled: 'Compile',
};

export function CompileButton({ phase, onCompile, onCancel, disabled }: CompileButtonProps) {
  const isActive = phase === 'submitting' || phase === 'queued' || phase === 'polling';

  if (isActive) {
    return (
      <div className="flex gap-2">
        <button disabled className="btn-primary flex items-center gap-2 opacity-80">
          <Spinner />
          {LABELS[phase]}
        </button>
        <button onClick={onCancel} className="btn-secondary text-red-400">
          Cancel
        </button>
      </div>
    );
  }

  return (
    <button
      onClick={onCompile}
      disabled={disabled}
      className="btn-primary flex items-center gap-2"
    >
      <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M5 3l14 9-14 9V3z" />
      </svg>
      {LABELS[phase]}
    </button>
  );
}

function Spinner() {
  return (
    <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24">
      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
    </svg>
  );
}
