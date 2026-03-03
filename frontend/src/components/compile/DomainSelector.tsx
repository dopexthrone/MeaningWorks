'use client';

import { useDomains } from '@/lib/hooks/useDomains';

interface DomainSelectorProps {
  value: string;
  onChange: (domain: string) => void;
}

export function DomainSelector({ value, onChange }: DomainSelectorProps) {
  const { domains, loading } = useDomains();

  return (
    <select
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className="input w-full"
      disabled={loading}
    >
      {domains.map((d) => (
        <option key={d.name} value={d.name}>
          {d.name} ({d.output_format})
        </option>
      ))}
    </select>
  );
}
