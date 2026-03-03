'use client';

import type { HealthResponse } from '@/lib/api/types';
import { StatusDot } from '@/components/shared/StatusDot';
import { ProgressBar } from '@/components/shared/ProgressBar';
import { formatDuration, formatBytes } from '@/lib/utils/format';

interface HealthPanelProps {
  health: HealthResponse;
}

export function HealthPanel({ health }: HealthPanelProps) {
  const queueStatus = health.worker_queue_depth < 5 ? 'ok' : health.worker_queue_depth < 15 ? 'warning' : 'error';
  const diskUsedMb = health.disk_total_mb - health.disk_free_mb;
  const diskPercent = health.disk_total_mb > 0 ? (diskUsedMb / health.disk_total_mb) * 100 : 0;

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
      {/* Status */}
      <div className="card">
        <div className="flex items-center gap-2 mb-2">
          <StatusDot status={health.status === 'ok' ? 'ok' : 'error'} pulse />
          <span className="text-xs text-[var(--text-muted)] uppercase">Status</span>
        </div>
        <p className="text-lg font-semibold capitalize">{health.status}</p>
        <p className="text-xs text-[var(--text-muted)]">v{health.version}</p>
      </div>

      {/* Uptime */}
      <div className="card">
        <span className="text-xs text-[var(--text-muted)] uppercase">Uptime</span>
        <p className="text-lg font-semibold mt-1">{formatDuration(health.uptime_seconds)}</p>
        <p className="text-xs text-[var(--text-muted)]">{health.corpus_size} compilations</p>
      </div>

      {/* Queue */}
      <div className="card">
        <div className="flex items-center gap-2 mb-2">
          <StatusDot status={queueStatus} />
          <span className="text-xs text-[var(--text-muted)] uppercase">Queue Depth</span>
        </div>
        <p className="text-lg font-semibold">{health.worker_queue_depth}</p>
        <ProgressBar
          value={Math.min(100, health.worker_queue_depth * 5)}
          color={queueStatus === 'ok' ? '#10B981' : queueStatus === 'warning' ? '#F59E0B' : '#EF4444'}
          className="mt-2"
        />
      </div>

      {/* Disk */}
      <div className="card">
        <span className="text-xs text-[var(--text-muted)] uppercase">Disk</span>
        <p className="text-lg font-semibold mt-1">
          {formatBytes(health.disk_free_mb)} free
        </p>
        <ProgressBar value={diskPercent} className="mt-2" />
        <p className="text-xs text-[var(--text-muted)] mt-1">
          {formatBytes(diskUsedMb)} / {formatBytes(health.disk_total_mb)}
        </p>
      </div>

      {/* Domains */}
      <div className="card sm:col-span-2 lg:col-span-4">
        <span className="text-xs text-[var(--text-muted)] uppercase">Available Domains</span>
        <div className="flex flex-wrap gap-2 mt-2">
          {health.domains_available.map((d) => (
            <span key={d} className="text-xs px-2 py-1 rounded bg-[var(--bg-tertiary)] border border-[var(--border)] text-[var(--text-secondary)]">
              {d}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}
