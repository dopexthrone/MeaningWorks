'use client';

import { useCallback, useState } from 'react';
import {
  ReactFlow,
  Controls,
  Background,
  BackgroundVariant,
  useNodesState,
  useEdgesState,
} from '@xyflow/react';
import '@xyflow/react/dist/style.css';
import type { Node, Edge } from '@xyflow/react';
import { BlueprintNode } from './BlueprintNode';
import { BlueprintEdge } from './BlueprintEdge';
import { ComponentDetail } from './ComponentDetail';
import type { BlueprintNodeData } from '@/lib/utils/blueprint';

const nodeTypes = { blueprint: BlueprintNode };
const edgeTypes = { blueprint: BlueprintEdge };

interface BlueprintGraphProps {
  initialNodes: Node[];
  initialEdges: Edge[];
}

export function BlueprintGraph({ initialNodes, initialEdges }: BlueprintGraphProps) {
  const [nodes, , onNodesChange] = useNodesState(initialNodes);
  const [edges, , onEdgesChange] = useEdgesState(initialEdges);
  const [selectedNode, setSelectedNode] = useState<BlueprintNodeData | null>(null);

  const onNodeClick = useCallback((_: React.MouseEvent, node: Node) => {
    setSelectedNode(node.data as unknown as BlueprintNodeData);
  }, []);

  return (
    <div className="flex h-full">
      <div className="flex-1 relative">
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onNodeClick={onNodeClick}
          nodeTypes={nodeTypes}
          edgeTypes={edgeTypes}
          fitView
          minZoom={0.3}
          maxZoom={2}
          proOptions={{ hideAttribution: true }}
        >
          <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="var(--border)" />
          <Controls showInteractive={false} />
        </ReactFlow>
      </div>
      {selectedNode && (
        <ComponentDetail data={selectedNode} onClose={() => setSelectedNode(null)} />
      )}
    </div>
  );
}
