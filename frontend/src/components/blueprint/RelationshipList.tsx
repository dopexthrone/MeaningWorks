'use client';

import type { BlueprintRelationship } from '@/lib/api/types';

interface RelationshipListProps {
  relationships: BlueprintRelationship[];
}

export function RelationshipList({ relationships }: RelationshipListProps) {
  if (!relationships.length) return null;

  return (
    <div>
      <h4 className="text-xs font-medium text-[var(--text-muted)] mb-2 uppercase tracking-wider">
        Relationships ({relationships.length})
      </h4>
      <div className="space-y-1.5">
        {relationships.map((r, i) => (
          <div key={i} className="flex items-center gap-2 text-xs">
            <span className="text-[var(--text-primary)] font-mono">{r.from_component}</span>
            <span className="text-brand-500">{r.type}</span>
            <span className="text-[var(--text-primary)] font-mono">{r.to_component}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
