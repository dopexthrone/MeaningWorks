'use client';

import type { BlueprintConstraint } from '@/lib/api/types';

interface ConstraintListProps {
  constraints: BlueprintConstraint[];
}

export function ConstraintList({ constraints }: ConstraintListProps) {
  if (!constraints.length) return null;

  return (
    <div>
      <h4 className="text-xs font-medium text-[var(--text-muted)] mb-2 uppercase tracking-wider">
        Constraints ({constraints.length})
      </h4>
      <div className="space-y-2">
        {constraints.map((c, i) => (
          <div key={i} className="bg-[var(--bg-tertiary)] rounded p-2">
            <p className="text-xs text-[var(--text-secondary)]">{c.description}</p>
            <div className="flex flex-wrap gap-1 mt-1">
              {c.applies_to.map((a) => (
                <span key={a} className="text-[10px] text-[var(--text-muted)] font-mono bg-[var(--bg-primary)] px-1 rounded">
                  {a}
                </span>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
