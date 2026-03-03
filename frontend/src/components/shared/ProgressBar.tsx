'use client';

interface ProgressBarProps {
  value: number; // 0-100
  color?: string;
  className?: string;
}

export function ProgressBar({ value, color, className = '' }: ProgressBarProps) {
  const bg = color || (value >= 70 ? '#10B981' : value >= 40 ? '#F59E0B' : '#EF4444');
  return (
    <div className={`w-full h-2 bg-[var(--bg-tertiary)] rounded-full overflow-hidden ${className}`}>
      <div
        className="h-full rounded-full transition-all duration-500"
        style={{ width: `${Math.min(100, Math.max(0, value))}%`, backgroundColor: bg }}
      />
    </div>
  );
}
