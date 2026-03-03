'use client';

interface GapReportProps {
  gaps: string[];
}

export function GapReport({ gaps }: GapReportProps) {
  if (!gaps.length) {
    return (
      <div className="text-xs text-emerald-400 flex items-center gap-1">
        <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
          <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
        </svg>
        No gaps detected
      </div>
    );
  }

  return (
    <div>
      <h4 className="text-xs font-medium text-[var(--text-muted)] mb-2 uppercase tracking-wider">
        Gaps ({gaps.length})
      </h4>
      <ul className="space-y-1.5">
        {gaps.map((gap, i) => (
          <li key={i} className="flex gap-2 text-xs">
            <span className="text-amber-400 flex-shrink-0 mt-0.5">
              <svg className="w-3 h-3" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M8.257 3.099c.765-1.36 2.722-1.36 3.486 0l5.58 9.92c.75 1.334-.213 2.98-1.742 2.98H4.42c-1.53 0-2.493-1.646-1.743-2.98l5.58-9.92zM11 13a1 1 0 11-2 0 1 1 0 012 0zm-1-8a1 1 0 00-1 1v3a1 1 0 002 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
              </svg>
            </span>
            <span className="text-[var(--text-secondary)]">{gap}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}
