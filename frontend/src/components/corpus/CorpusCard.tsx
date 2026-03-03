'use client';

import type { CompilationRecord } from '@/lib/api/types';
import { Badge } from '@/components/shared/Badge';
import { trustBg, badgeClasses } from '@/lib/utils/colors';
import { formatDate, truncate } from '@/lib/utils/format';

interface CorpusCardProps {
  record: CompilationRecord;
}

export function CorpusCard({ record }: CorpusCardProps) {
  const score = record.trust?.overall_score ?? 0;
  const badge = record.trust?.verification_badge ?? 'unverified';

  return (
    <div className="card hover:border-[var(--text-muted)] transition-colors cursor-pointer">
      <div className="flex items-start justify-between mb-2">
        <p className="text-sm text-[var(--text-primary)] font-medium leading-snug">
          {truncate(record.description, 100)}
        </p>
        <Badge className={badgeClasses(badge)}>
          {badge}
        </Badge>
      </div>

      <div className="flex items-center gap-3 text-xs text-[var(--text-muted)]">
        <Badge className="border-[var(--border)]">{record.domain}</Badge>
        <span>{record.component_count ?? record.blueprint?.components?.length ?? 0} components</span>
        <span className={`font-mono ${trustBg(score)} px-1 rounded`}>
          {Math.round(score)}
        </span>
        <span className="ml-auto">{formatDate(record.timestamp)}</span>
      </div>
    </div>
  );
}
