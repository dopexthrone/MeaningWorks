import type { ComponentType, VerificationBadge } from '@/lib/api/types';

export const NODE_COLORS: Record<ComponentType, string> = {
  entity: '#3B82F6',
  process: '#10B981',
  interface: '#8B5CF6',
  event: '#F59E0B',
  constraint: '#EF4444',
  subsystem: '#6B7280',
};

export function trustColor(score: number): string {
  if (score >= 70) return '#10B981';
  if (score >= 40) return '#F59E0B';
  return '#EF4444';
}

export function trustBg(score: number): string {
  if (score >= 70) return 'bg-emerald-500/10 text-emerald-400';
  if (score >= 40) return 'bg-amber-500/10 text-amber-400';
  return 'bg-red-500/10 text-red-400';
}

export function badgeClasses(badge: VerificationBadge): string {
  switch (badge) {
    case 'verified': return 'bg-emerald-500/20 text-emerald-400 border-emerald-500/30';
    case 'partial': return 'bg-amber-500/20 text-amber-400 border-amber-500/30';
    case 'unverified': return 'bg-red-500/20 text-red-400 border-red-500/30';
  }
}

export function coverageColor(value: number): string {
  if (value >= 0.7) return '#10B981';
  if (value >= 0.4) return '#F59E0B';
  return '#EF4444';
}
