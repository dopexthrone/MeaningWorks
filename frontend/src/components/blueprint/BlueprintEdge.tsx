'use client';

import { memo } from 'react';
import { BaseEdge, EdgeLabelRenderer, getBezierPath } from '@xyflow/react';
import type { EdgeProps } from '@xyflow/react';

function BlueprintEdgeComponent({
  id,
  sourceX, sourceY, targetX, targetY,
  sourcePosition, targetPosition,
  label,
  style,
}: EdgeProps) {
  const [edgePath, labelX, labelY] = getBezierPath({
    sourceX, sourceY, targetX, targetY,
    sourcePosition, targetPosition,
  });

  return (
    <>
      <BaseEdge
        id={id}
        path={edgePath}
        style={{
          stroke: 'var(--border)',
          strokeWidth: 1.5,
          ...style,
        }}
      />
      {label && (
        <EdgeLabelRenderer>
          <div
            className="absolute bg-[var(--bg-primary)] px-1.5 py-0.5 rounded text-[10px] text-[var(--text-muted)] border border-[var(--border)] pointer-events-none"
            style={{
              transform: `translate(-50%, -50%) translate(${labelX}px,${labelY}px)`,
            }}
          >
            {label}
          </div>
        </EdgeLabelRenderer>
      )}
    </>
  );
}

export const BlueprintEdge = memo(BlueprintEdgeComponent);
