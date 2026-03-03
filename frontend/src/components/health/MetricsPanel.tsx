'use client';

import type { MetricsResponse } from '@/lib/api/types';

interface MetricsPanelProps {
  metrics: MetricsResponse;
}

export function MetricsPanel({ metrics }: MetricsPanelProps) {
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <div className="card">
          <span className="text-xs text-[var(--text-muted)] uppercase">Total Compilations</span>
          <p className="text-2xl font-bold mt-1">{metrics.total_compilations}</p>
        </div>
        <div className="card">
          <span className="text-xs text-[var(--text-muted)] uppercase">Total Cost</span>
          <p className="text-2xl font-bold mt-1">${metrics.total_cost_usd.toFixed(2)}</p>
        </div>
      </div>

      {Object.keys(metrics.per_domain).length > 0 && (
        <div className="card">
          <h3 className="text-sm font-medium mb-3">Per-Domain Metrics</h3>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-[var(--border)]">
                  <th className="text-left py-2 text-[var(--text-muted)] font-medium">Domain</th>
                  {Object.keys(Object.values(metrics.per_domain)[0] || {}).map((key) => (
                    <th key={key} className="text-right py-2 text-[var(--text-muted)] font-medium capitalize">
                      {key.replace(/_/g, ' ')}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {Object.entries(metrics.per_domain).map(([domain, stats]) => (
                  <tr key={domain} className="border-b border-[var(--border)]/50">
                    <td className="py-2 text-[var(--text-primary)] font-mono">{domain}</td>
                    {Object.values(stats as Record<string, unknown>).map((val, i) => (
                      <td key={i} className="py-2 text-right text-[var(--text-secondary)] font-mono">
                        {typeof val === 'number' ? (val % 1 !== 0 ? val.toFixed(4) : val) : String(val)}
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
