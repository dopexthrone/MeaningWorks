'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { IntentInput } from '@/components/compile/IntentInput';
import { DomainSelector } from '@/components/compile/DomainSelector';
import { CompileButton } from '@/components/compile/CompileButton';
import { AdvancedOptions } from '@/components/compile/AdvancedOptions';
import { CompilationProgress } from '@/components/compile/CompilationProgress';
import { useCompile } from '@/lib/hooks/useCompile';
import type { TrustLevel } from '@/lib/api/types';

export default function CompilePage() {
  const router = useRouter();
  const { phase, taskId, result, error, pollData, submit, cancel, reset } = useCompile();

  const [description, setDescription] = useState('');
  const [domain, setDomain] = useState('software');
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [trustLevel, setTrustLevel] = useState<TrustLevel>('standard');
  const [enrich, setEnrich] = useState(false);
  const [canonicalComponents, setCanonicalComponents] = useState('');

  const handleCompile = async () => {
    if (!description.trim()) return;

    const canonical = canonicalComponents.trim()
      ? canonicalComponents.split(',').map((s) => s.trim()).filter(Boolean)
      : undefined;

    await submit({
      description: description.trim(),
      domain,
      trust_level: trustLevel,
      enrich,
      canonical_components: canonical,
    });
  };

  // Navigate to result when complete
  if (phase === 'complete' && taskId) {
    router.push(`/compile?id=${taskId}`);
  }

  const isActive = phase === 'submitting' || phase === 'queued' || phase === 'polling';

  return (
    <div className="max-w-3xl mx-auto space-y-6">
      {/* Intent Input */}
      <div className="card space-y-4">
        <IntentInput
          value={description}
          onChange={setDescription}
          domain={domain}
          disabled={isActive}
        />

        <div className="flex flex-col sm:flex-row gap-3 items-start sm:items-end justify-between">
          <div className="flex gap-3 items-end">
            <div className="w-48">
              <label className="block text-xs text-[var(--text-muted)] mb-1">Domain</label>
              <DomainSelector value={domain} onChange={setDomain} />
            </div>
            <button
              onClick={() => setShowAdvanced(!showAdvanced)}
              className="text-xs text-[var(--text-muted)] hover:text-[var(--text-secondary)] transition-colors pb-2"
            >
              {showAdvanced ? 'Hide' : 'Show'} advanced
            </button>
          </div>
          <CompileButton
            phase={phase}
            onCompile={handleCompile}
            onCancel={cancel}
            disabled={!description.trim()}
          />
        </div>

        {showAdvanced && (
          <AdvancedOptions
            trustLevel={trustLevel}
            onTrustLevelChange={setTrustLevel}
            enrich={enrich}
            onEnrichChange={setEnrich}
            canonicalComponents={canonicalComponents}
            onCanonicalComponentsChange={setCanonicalComponents}
          />
        )}
      </div>

      {/* Progress */}
      <CompilationProgress phase={phase} taskId={taskId} pollData={pollData} />

      {/* Error */}
      {phase === 'error' && error && (
        <div className="card border-red-500/30">
          <p className="text-red-400 font-medium text-sm mb-1">Compilation failed</p>
          <p className="text-xs text-[var(--text-muted)]">{error}</p>
          <button onClick={reset} className="btn-secondary mt-3 text-sm">
            Try again
          </button>
        </div>
      )}

      {/* Cancelled */}
      {phase === 'cancelled' && (
        <div className="card border-amber-500/30">
          <p className="text-amber-400 font-medium text-sm">Compilation cancelled</p>
          <button onClick={reset} className="btn-secondary mt-3 text-sm">
            Start new
          </button>
        </div>
      )}
    </div>
  );
}
