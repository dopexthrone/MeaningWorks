'use client';

import type { BlueprintNodeData } from '@/lib/utils/blueprint';
import { Badge } from '@/components/shared/Badge';

interface ComponentDetailProps {
  data: BlueprintNodeData;
  onClose: () => void;
}

export function ComponentDetail({ data, onClose }: ComponentDetailProps) {
  return (
    <div className="bg-[var(--bg-secondary)] border-l border-[var(--border)] w-80 h-full overflow-y-auto">
      <div className="flex items-center justify-between p-4 border-b border-[var(--border)]">
        <div className="flex items-center gap-2">
          <div className="w-3 h-3 rounded-full" style={{ backgroundColor: data.color }} />
          <h3 className="font-semibold text-sm">{data.label}</h3>
        </div>
        <button onClick={onClose} className="text-[var(--text-muted)] hover:text-[var(--text-primary)]">
          <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
      </div>

      <div className="p-4 space-y-4">
        {/* Type & Description */}
        <div>
          <Badge className="border-[var(--border)] text-[var(--text-muted)] mb-2">{data.type}</Badge>
          <p className="text-sm text-[var(--text-secondary)]">{data.description}</p>
        </div>

        {/* Derived From */}
        {data.derived_from && (
          <div>
            <h4 className="text-xs font-medium text-[var(--text-muted)] mb-1 uppercase tracking-wider">Derived From</h4>
            <p className="text-xs text-[var(--text-secondary)] italic bg-[var(--bg-tertiary)] p-2 rounded">
              &ldquo;{data.derived_from}&rdquo;
            </p>
          </div>
        )}

        {/* Methods */}
        {data.methods && data.methods.length > 0 && (
          <div>
            <h4 className="text-xs font-medium text-[var(--text-muted)] mb-2 uppercase tracking-wider">
              Methods ({data.methods.length})
            </h4>
            <div className="space-y-2">
              {data.methods.map((m) => (
                <div key={m.name} className="bg-[var(--bg-tertiary)] rounded p-2">
                  <code className="text-xs text-brand-500 font-mono">
                    {m.name}({m.parameters.map((p) => `${p.name}: ${p.type_hint}`).join(', ')})
                  </code>
                  <span className="text-xs text-[var(--text-muted)]"> -&gt; {m.return_type}</span>
                  {m.description && (
                    <p className="text-xs text-[var(--text-muted)] mt-1">{m.description}</p>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* State Machine */}
        {data.state_machine && (
          <div>
            <h4 className="text-xs font-medium text-[var(--text-muted)] mb-2 uppercase tracking-wider">State Machine</h4>
            <div className="bg-[var(--bg-tertiary)] rounded p-2">
              <div className="flex flex-wrap gap-1 mb-2">
                {data.state_machine.states.map((s) => (
                  <Badge
                    key={s}
                    className={`text-[10px] ${s === data.state_machine!.initial_state ? 'bg-brand-500/20 text-brand-500 border-brand-500/30' : 'border-[var(--border)] text-[var(--text-muted)]'}`}
                  >
                    {s}
                  </Badge>
                ))}
              </div>
              {data.state_machine.transitions.map((t, i) => (
                <div key={i} className="text-xs text-[var(--text-muted)] font-mono">
                  {t.from_state} → {t.to_state} <span className="text-[var(--text-secondary)]">({t.trigger})</span>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Validation Rules */}
        {data.validation_rules && data.validation_rules.length > 0 && (
          <div>
            <h4 className="text-xs font-medium text-[var(--text-muted)] mb-1 uppercase tracking-wider">Validation Rules</h4>
            <ul className="text-xs text-[var(--text-secondary)] space-y-1">
              {data.validation_rules.map((r, i) => (
                <li key={i} className="flex gap-1">
                  <span className="text-amber-400">&#x2022;</span> {r}
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </div>
  );
}
