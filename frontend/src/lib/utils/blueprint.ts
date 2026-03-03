import dagre from 'dagre';
import type { Node, Edge } from '@xyflow/react';
import type { Blueprint, BlueprintComponent, BlueprintRelationship } from '@/lib/api/types';
import { NODE_COLORS } from './colors';

const NODE_WIDTH = 240;
const NODE_HEIGHT = 100;

export interface BlueprintNodeData {
  label: string;
  type: string;
  description: string;
  derived_from: string;
  methods: BlueprintComponent['methods'];
  state_machine: BlueprintComponent['state_machine'];
  attributes: BlueprintComponent['attributes'];
  validation_rules: BlueprintComponent['validation_rules'];
  color: string;
  trustScore?: number;
  [key: string]: unknown;
}

export interface BlueprintEdgeData {
  relationType: string;
  description: string;
  derived_from: string;
  [key: string]: unknown;
}

export function blueprintToGraph(
  blueprint: Blueprint,
  dimensionalMetadata?: Record<string, unknown>
): { nodes: Node<BlueprintNodeData>[]; edges: Edge<BlueprintEdgeData>[] } {
  const g = new dagre.graphlib.Graph();
  g.setDefaultEdgeLabel(() => ({}));
  g.setGraph({ rankdir: 'TB', nodesep: 60, ranksep: 80 });

  const components = blueprint.components || [];
  const relationships = blueprint.relationships || [];

  // Coverage per component from dimensional metadata
  const componentCoverage = (dimensionalMetadata?.component_coverage || {}) as Record<string, number>;

  components.forEach((c) => {
    g.setNode(c.name, { width: NODE_WIDTH, height: NODE_HEIGHT });
  });

  relationships.forEach((r) => {
    g.setEdge(r.from_component, r.to_component);
  });

  dagre.layout(g);

  const nodes: Node<BlueprintNodeData>[] = components.map((c) => {
    const pos = g.node(c.name);
    return {
      id: c.name,
      type: 'blueprint',
      position: { x: (pos?.x ?? 0) - NODE_WIDTH / 2, y: (pos?.y ?? 0) - NODE_HEIGHT / 2 },
      data: {
        label: c.name,
        type: c.type,
        description: c.description,
        derived_from: c.derived_from,
        methods: c.methods,
        state_machine: c.state_machine,
        attributes: c.attributes,
        validation_rules: c.validation_rules,
        color: NODE_COLORS[c.type] || '#6B7280',
        trustScore: componentCoverage[c.name],
      },
    };
  });

  const edges: Edge<BlueprintEdgeData>[] = relationships.map((r, i) => ({
    id: `e-${i}`,
    source: r.from_component,
    target: r.to_component,
    type: 'blueprint',
    label: r.type,
    data: {
      relationType: r.type,
      description: r.description,
      derived_from: r.derived_from,
    },
  }));

  return { nodes, edges };
}
