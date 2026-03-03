'use client';

import { useEffect, useState } from 'react';
import { useRouter } from 'next/navigation';
import { getCompilation } from '@/lib/api/corpus';
import { useBlueprintGraph } from '@/lib/hooks/useBlueprintGraph';
import { BlueprintGraph } from '@/components/blueprint/BlueprintGraph';
import { TrustScoreBadge } from '@/components/trust/TrustScoreBadge';
import { FidelityRadar } from '@/components/trust/FidelityRadar';
import { ConfidenceTrajectory } from '@/components/trust/ConfidenceTrajectory';
import { GapReport } from '@/components/trust/GapReport';
import { DimensionalCoverage } from '@/components/trust/DimensionalCoverage';
import { CodeViewer } from '@/components/output/CodeViewer';
import { FileTree } from '@/components/output/FileTree';
import { ExportPanel } from '@/components/output/ExportPanel';
import { LoadingSkeleton } from '@/components/shared/LoadingSkeleton';
import type { CompilationRecord } from '@/lib/api/types';

interface CorpusDetailProps {
  id: string;
}

export function CorpusDetail({ id }: CorpusDetailProps) {
  const router = useRouter();
  const [record, setRecord] = useState<CompilationRecord | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [selectedFile, setSelectedFile] = useState<string | null>(null);

  useEffect(() => {
    getCompilation(id)
      .then((r) => {
        setRecord(r);
        const files = Object.keys(r.materialized_output || {});
        if (files.length) setSelectedFile(files[0]);
      })
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, [id]);

  const blueprint = record?.blueprint ?? null;
  const trust = record?.trust;
  const { nodes, edges } = useBlueprintGraph(blueprint);

  if (loading) return <LoadingSkeleton lines={8} />;
  if (error) return <div className="card text-red-400">{error}</div>;
  if (!record) return null;

  const files = record.materialized_output || {};

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div>
          <p className="text-sm text-[var(--text-primary)] font-medium">{record.description}</p>
          <p className="text-xs text-[var(--text-muted)] mt-1">
            {record.domain} &middot; {new Date(record.timestamp).toLocaleString()}
          </p>
        </div>
        <button
          onClick={() => {
            if (typeof window !== 'undefined') {
              sessionStorage.setItem('recompile_intent', record.description);
              sessionStorage.setItem('recompile_domain', record.domain);
            }
            router.push('/');
          }}
          className="btn-secondary text-xs"
        >
          Recompile
        </button>
      </div>

      {/* Graph + Trust */}
      <div className="grid grid-cols-1 lg:grid-cols-[1fr_340px] gap-4">
        <div className="card p-0 overflow-hidden" style={{ height: 450 }}>
          {nodes.length > 0 ? (
            <BlueprintGraph initialNodes={nodes} initialEdges={edges} />
          ) : (
            <div className="flex items-center justify-center h-full text-[var(--text-muted)] text-sm">
              No components
            </div>
          )}
        </div>

        {trust && (
          <div className="card space-y-5 overflow-y-auto" style={{ maxHeight: 450 }}>
            <div className="flex justify-center relative">
              <TrustScoreBadge score={trust.overall_score} badge={trust.verification_badge} />
            </div>
            <FidelityRadar scores={trust.fidelity_scores} />
            <ConfidenceTrajectory trajectory={trust.confidence_trajectory} />
            <GapReport gaps={trust.gap_report} />
            <DimensionalCoverage coverage={trust.dimensional_coverage} />
          </div>
        )}
      </div>

      {/* Code output */}
      {Object.keys(files).length > 0 && (
        <div className="card">
          <h3 className="text-sm font-medium mb-3">Output Files</h3>
          <div className="grid grid-cols-1 md:grid-cols-[200px_1fr] gap-4">
            <FileTree files={files} selectedFile={selectedFile} onSelect={setSelectedFile} />
            {selectedFile && files[selectedFile] && (
              <CodeViewer code={files[selectedFile]} filename={selectedFile} />
            )}
          </div>
          <div className="mt-4">
            <ExportPanel files={files} blueprintJson={JSON.stringify(blueprint, null, 2)} />
          </div>
        </div>
      )}
    </div>
  );
}
