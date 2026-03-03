'use client';

import { ProgressBar } from '@/components/shared/ProgressBar';
import { formatPercent } from '@/lib/utils/format';

interface DimensionalCoverageProps {
  coverage: Record<string, number>;
}

export function DimensionalCoverage({ coverage }: DimensionalCoverageProps) {
  const entries = Object.entries(coverage);
  if (!entries.length) return null;

  return (
    <div>
      <h4 className="text-xs font-medium text-[var(--text-muted)] mb-3 uppercase tracking-wider">
        Dimensional Coverage
      </h4>
      <div className="space-y-2.5">
        {entries.map(([dim, value]) => (
          <div key={dim}>
            <div className="flex justify-between text-xs mb-1">
              <span className="text-[var(--text-secondary)] capitalize">{dim.replace(/_/g, ' ')}</span>
              <span className="text-[var(--text-muted)] font-mono">{formatPercent(value)}</span>
            </div>
            <ProgressBar value={value * 100} />
          </div>
        ))}
      </div>
    </div>
  );
}
