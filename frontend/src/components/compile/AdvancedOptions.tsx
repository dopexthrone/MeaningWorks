'use client';

import { Toggle } from '@/components/shared/Toggle';
import type { TrustLevel } from '@/lib/api/types';

interface AdvancedOptionsProps {
  trustLevel: TrustLevel;
  onTrustLevelChange: (level: TrustLevel) => void;
  enrich: boolean;
  onEnrichChange: (enrich: boolean) => void;
  canonicalComponents: string;
  onCanonicalComponentsChange: (value: string) => void;
}

export function AdvancedOptions({
  trustLevel,
  onTrustLevelChange,
  enrich,
  onEnrichChange,
  canonicalComponents,
  onCanonicalComponentsChange,
}: AdvancedOptionsProps) {
  return (
    <div className="space-y-4 p-4 bg-[var(--bg-tertiary)] rounded-btn border border-[var(--border)]">
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <div>
          <label className="block text-xs text-[var(--text-muted)] mb-1">Trust Level</label>
          <select
            value={trustLevel}
            onChange={(e) => onTrustLevelChange(e.target.value as TrustLevel)}
            className="input w-full text-sm"
          >
            <option value="fast">Fast (minimal verification)</option>
            <option value="standard">Standard (balanced)</option>
            <option value="thorough">Thorough (deep verification)</option>
          </select>
        </div>
        <div className="flex items-end">
          <Toggle checked={enrich} onChange={onEnrichChange} label="Auto-enrich sparse input" />
        </div>
      </div>
      <div>
        <label className="block text-xs text-[var(--text-muted)] mb-1">
          Canonical Components (comma-separated, optional)
        </label>
        <input
          value={canonicalComponents}
          onChange={(e) => onCanonicalComponentsChange(e.target.value)}
          placeholder="e.g. UserService, AuthManager, Database"
          className="input w-full text-sm"
        />
      </div>
    </div>
  );
}
