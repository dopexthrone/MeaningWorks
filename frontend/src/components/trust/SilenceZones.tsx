'use client';

interface SilenceZonesProps {
  zones: string[];
}

export function SilenceZones({ zones }: SilenceZonesProps) {
  if (!zones.length) return null;

  return (
    <div>
      <h4 className="text-xs font-medium text-[var(--text-muted)] mb-2 uppercase tracking-wider">
        Silence Zones ({zones.length})
      </h4>
      <div className="flex flex-wrap gap-1.5">
        {zones.map((zone, i) => (
          <span
            key={i}
            className="text-[11px] px-2 py-1 rounded bg-red-500/10 text-red-400 border border-red-500/20"
          >
            {zone}
          </span>
        ))}
      </div>
    </div>
  );
}
