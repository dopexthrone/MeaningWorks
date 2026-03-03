'use client';

interface ProvenanceTraceProps {
  depth: number;
  chainLength: number;
}

const STRATA = [
  { level: 1, label: 'Input Provenance', description: 'Direct keyword/intent trace' },
  { level: 2, label: 'Dialogue Provenance', description: 'Agent insights and conflicts' },
  { level: 3, label: 'Compilation Patterns', description: 'Stable components and patterns' },
];

export function ProvenanceTrace({ depth, chainLength }: ProvenanceTraceProps) {
  return (
    <div>
      <h4 className="text-xs font-medium text-[var(--text-muted)] mb-2 uppercase tracking-wider">
        Provenance Depth: {depth}/3
      </h4>
      <div className="space-y-2">
        {STRATA.map((s) => {
          const active = s.level <= depth;
          return (
            <div key={s.level} className="flex items-center gap-3">
              <div className={`w-6 h-6 rounded-full flex items-center justify-center text-xs font-medium
                ${active ? 'bg-brand-500/20 text-brand-500' : 'bg-[var(--bg-tertiary)] text-[var(--text-muted)]'}`}
              >
                {s.level}
              </div>
              <div>
                <p className={`text-xs font-medium ${active ? 'text-[var(--text-primary)]' : 'text-[var(--text-muted)]'}`}>
                  {s.label}
                </p>
                <p className="text-[10px] text-[var(--text-muted)]">{s.description}</p>
              </div>
            </div>
          );
        })}
      </div>
      <p className="text-xs text-[var(--text-muted)] mt-2">
        Avg chain length: <span className="font-mono text-[var(--text-secondary)]">{chainLength.toFixed(1)}</span>
      </p>
    </div>
  );
}
