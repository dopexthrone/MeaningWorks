'use client';

import { useEffect, useState } from 'react';
import { getHealth, getMetrics } from '@/lib/api/health';
import { HealthPanel } from '@/components/health/HealthPanel';
import { MetricsPanel } from '@/components/health/MetricsPanel';
import { LoadingSkeleton } from '@/components/shared/LoadingSkeleton';
import type { HealthResponse, MetricsResponse } from '@/lib/api/types';

export default function HealthPage() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [metrics, setMetrics] = useState<MetricsResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([getHealth(), getMetrics()])
      .then(([h, m]) => { setHealth(h); setMetrics(m); })
      .catch((err) => setError(err.message));

    const interval = setInterval(() => {
      getHealth().then(setHealth).catch(() => {});
    }, 15000);

    return () => clearInterval(interval);
  }, []);

  if (error) return <div className="card text-red-400">{error}</div>;

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <h2 className="text-lg font-semibold">System Health</h2>

      {health ? <HealthPanel health={health} /> : <LoadingSkeleton lines={4} />}

      <h2 className="text-lg font-semibold pt-2">Metrics</h2>

      {metrics ? <MetricsPanel metrics={metrics} /> : <LoadingSkeleton lines={3} />}
    </div>
  );
}
