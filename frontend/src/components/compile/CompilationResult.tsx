'use client';

import { useEffect, useState } from 'react';
import { getTaskStatus } from '@/lib/api/compile';
import { useBlueprintGraph } from '@/lib/hooks/useBlueprintGraph';
import { BlueprintGraph } from '@/components/blueprint/BlueprintGraph';
import { TrustScoreBadge } from '@/components/trust/TrustScoreBadge';
import { FidelityRadar } from '@/components/trust/FidelityRadar';
import { ConfidenceTrajectory } from '@/components/trust/ConfidenceTrajectory';
import { GapReport } from '@/components/trust/GapReport';
import { SilenceZones } from '@/components/trust/SilenceZones';
import { ProvenanceTrace } from '@/components/trust/ProvenanceTrace';
import { DimensionalCoverage } from '@/components/trust/DimensionalCoverage';
import { CodeViewer } from '@/components/output/CodeViewer';
import { FileTree } from '@/components/output/FileTree';
import { ExportPanel } from '@/components/output/ExportPanel';
import { RelationshipList } from '@/components/blueprint/RelationshipList';
import { ConstraintList } from '@/components/blueprint/ConstraintList';
import { LoadingSkeleton } from '@/components/shared/LoadingSkeleton';
import type { CompileResponse } from '@/lib/api/types';

type Tab = 'code' | 'blueprint_json' | 'context' | 'interfaces';

interface CompilationResultProps {
  taskId: string;
}

export function CompilationResult({ taskId }: CompilationResultProps) {
  const [result, setResult] = useState<CompileResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<Tab>('code');
  const [selectedFile, setSelectedFile] = useState<string | null>(null);

  useEffect(() => {
    getTaskStatus(taskId)
      .then((task) => {
        if (task.status === 'complete' && task.result) {
          setResult(task.result);
          const files = Object.keys(task.result.materialized_output || {});
          if (files.length) setSelectedFile(files[0]);
        } else if (task.status === 'error') {
          setError(task.error || 'Compilation failed');
        } else {
          setError(`Task status: ${task.status}`);
        }
      })
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, [taskId]);

  const blueprint = result?.blueprint ?? null;
  const trust = result?.trust;
  const { nodes, edges } = useBlueprintGraph(blueprint, result?.dimensional_metadata as Record<string, unknown>);

  if (loading) return <LoadingSkeleton lines={8} />;
  if (error) return <div className="card text-red-400">{error}</div>;
  if (!result) return null;

  const files = result.materialized_output || {};
  const TABS: { key: Tab; label: string }[] = [
    { key: 'code', label: 'Code Output' },
    { key: 'blueprint_json', label: 'Blueprint JSON' },
    { key: 'context', label: 'Context Graph' },
    { key: 'interfaces', label: 'Interface Map' },
  ];

  return (
    <div className="space-y-4">
      {/* Top: Graph + Trust side by side */}
      <div className="grid grid-cols-1 lg:grid-cols-[1fr_340px] gap-4">
        {/* Blueprint Graph */}
        <div className="card p-0 overflow-hidden" style={{ height: 500 }}>
          {nodes.length > 0 ? (
            <BlueprintGraph initialNodes={nodes} initialEdges={edges} />
          ) : (
            <div className="flex items-center justify-center h-full text-[var(--text-muted)] text-sm">
              No components in blueprint
            </div>
          )}
        </div>

        {/* Trust Panel */}
        {trust && (
          <div className="card space-y-5 overflow-y-auto" style={{ maxHeight: 500 }}>
            <div className="flex justify-center relative">
              <TrustScoreBadge score={trust.overall_score} badge={trust.verification_badge} />
            </div>
            <FidelityRadar scores={trust.fidelity_scores} />
            <ConfidenceTrajectory trajectory={trust.confidence_trajectory} />
            <GapReport gaps={trust.gap_report} />
            <SilenceZones zones={trust.silence_zones} />
            <ProvenanceTrace depth={trust.provenance_depth} chainLength={trust.derivation_chain_length} />
            <DimensionalCoverage coverage={trust.dimensional_coverage} />
          </div>
        )}
      </div>

      {/* Blueprint details */}
      {blueprint && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div className="card">
            <RelationshipList relationships={blueprint.relationships || []} />
          </div>
          <div className="card">
            <ConstraintList constraints={blueprint.constraints || []} />
          </div>
        </div>
      )}

      {/* Bottom: Tabbed output */}
      <div className="card">
        <div className="flex gap-1 mb-4 border-b border-[var(--border)] -mx-6 -mt-6 px-6 pt-3">
          {TABS.map((tab) => (
            <button
              key={tab.key}
              onClick={() => setActiveTab(tab.key)}
              className={`px-3 py-2 text-xs font-medium border-b-2 transition-colors
                ${activeTab === tab.key
                  ? 'text-brand-500 border-brand-500'
                  : 'text-[var(--text-muted)] border-transparent hover:text-[var(--text-secondary)]'
                }`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {activeTab === 'code' && (
          <div className="grid grid-cols-1 md:grid-cols-[200px_1fr] gap-4">
            {Object.keys(files).length > 0 ? (
              <>
                <FileTree files={files} selectedFile={selectedFile} onSelect={setSelectedFile} />
                {selectedFile && files[selectedFile] && (
                  <CodeViewer code={files[selectedFile]} filename={selectedFile} />
                )}
              </>
            ) : (
              <p className="text-sm text-[var(--text-muted)] col-span-2">No materialized output</p>
            )}
          </div>
        )}

        {activeTab === 'blueprint_json' && (
          <CodeViewer
            code={JSON.stringify(blueprint, null, 2)}
            language="json"
            filename="blueprint.json"
          />
        )}

        {activeTab === 'context' && (
          <CodeViewer
            code={JSON.stringify(result.context_graph, null, 2)}
            language="json"
            filename="context_graph.json"
          />
        )}

        {activeTab === 'interfaces' && (
          <CodeViewer
            code={JSON.stringify(result.interface_map, null, 2)}
            language="json"
            filename="interface_map.json"
          />
        )}

        <div className="mt-4">
          <ExportPanel
            files={files}
            blueprintJson={JSON.stringify(blueprint, null, 2)}
          />
        </div>
      </div>
    </div>
  );
}
