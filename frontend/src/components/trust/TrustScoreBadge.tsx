'use client';

import { trustColor, badgeClasses } from '@/lib/utils/colors';
import type { VerificationBadge } from '@/lib/api/types';

interface TrustScoreBadgeProps {
  score: number;
  badge: VerificationBadge;
  size?: number;
}

export function TrustScoreBadge({ score, badge, size = 120 }: TrustScoreBadgeProps) {
  const color = trustColor(score);
  const radius = (size - 12) / 2;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (score / 100) * circumference;
  const center = size / 2;

  return (
    <div className="flex flex-col items-center gap-2">
      <svg width={size} height={size} className="transform -rotate-90">
        {/* Background circle */}
        <circle
          cx={center}
          cy={center}
          r={radius}
          fill="none"
          stroke="var(--bg-tertiary)"
          strokeWidth="6"
        />
        {/* Score arc */}
        <circle
          cx={center}
          cy={center}
          r={radius}
          fill="none"
          stroke={color}
          strokeWidth="6"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          strokeLinecap="round"
          className="transition-all duration-1000"
        />
      </svg>
      {/* Score text overlay */}
      <div className="absolute flex flex-col items-center" style={{ marginTop: size * 0.28 }}>
        <span className="text-2xl font-bold" style={{ color }}>{Math.round(score)}</span>
        <span className="text-[10px] text-[var(--text-muted)] uppercase">Trust</span>
      </div>

      {/* Badge */}
      <span className={`text-xs px-2 py-0.5 rounded-full border ${badgeClasses(badge)}`}>
        {badge}
      </span>
    </div>
  );
}
