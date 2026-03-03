'use client';

import { useMemo } from 'react';
import type { Blueprint } from '@/lib/api/types';
import { blueprintToGraph } from '@/lib/utils/blueprint';

export function useBlueprintGraph(
  blueprint: Blueprint | null | undefined,
  dimensionalMetadata?: Record<string, unknown>
) {
  return useMemo(() => {
    if (!blueprint) return { nodes: [], edges: [] };
    return blueprintToGraph(blueprint, dimensionalMetadata);
  }, [blueprint, dimensionalMetadata]);
}
