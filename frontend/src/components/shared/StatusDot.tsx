'use client';

interface StatusDotProps {
  status: 'ok' | 'warning' | 'error' | 'unknown';
  size?: 'sm' | 'md';
  pulse?: boolean;
}

const COLORS = {
  ok: 'bg-emerald-400',
  warning: 'bg-amber-400',
  error: 'bg-red-400',
  unknown: 'bg-gray-400',
};

export function StatusDot({ status, size = 'sm', pulse = false }: StatusDotProps) {
  const s = size === 'sm' ? 'w-2 h-2' : 'w-3 h-3';
  return (
    <span className="relative inline-flex">
      <span className={`${s} rounded-full ${COLORS[status]}`} />
      {pulse && <span className={`absolute ${s} rounded-full ${COLORS[status]} animate-ping opacity-50`} />}
    </span>
  );
}
