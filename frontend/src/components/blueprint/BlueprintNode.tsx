'use client';

import { memo } from 'react';
import { Handle, Position } from '@xyflow/react';
import type { NodeProps } from '@xyflow/react';
import type { BlueprintNodeData } from '@/lib/utils/blueprint';
import { Badge } from '@/components/shared/Badge';
import { truncate } from '@/lib/utils/format';

function BlueprintNodeComponent({ data, selected }: NodeProps) {
  const d = data as unknown as BlueprintNodeData;
  const borderColor = selected ? d.color : 'var(--border)';

  return (
    <div
      className="bg-[var(--bg-secondary)] rounded-card px-4 py-3 min-w-[200px] max-w-[260px] cursor-pointer"
      style={{ borderWidth: 2, borderStyle: 'solid', borderColor }}
    >
      <Handle type="target" position={Position.Top} className="!bg-[var(--border)] !w-2 !h-2" />

      <div className="flex items-center gap-2 mb-1.5">
        <div className="w-2 h-2 rounded-full flex-shrink-0" style={{ backgroundColor: d.color }} />
        <span className="font-semibold text-sm text-[var(--text-primary)] truncate">{d.label}</span>
      </div>

      <div className="flex items-center gap-1.5 mb-1.5">
        <Badge className="text-[10px] px-1.5 py-0 border-[var(--border)] text-[var(--text-muted)]">
          {d.type}
        </Badge>
        {d.methods && d.methods.length > 0 && (
          <span className="text-[10px] text-[var(--text-muted)]">{d.methods.length} methods</span>
        )}
      </div>

      {d.description && (
        <p className="text-xs text-[var(--text-muted)] leading-snug">
          {truncate(d.description, 80)}
        </p>
      )}

      <Handle type="source" position={Position.Bottom} className="!bg-[var(--border)] !w-2 !h-2" />
    </div>
  );
}

export const BlueprintNode = memo(BlueprintNodeComponent);
