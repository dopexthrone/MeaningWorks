'use client';

import { useRef, useEffect } from 'react';

const DOMAIN_PLACEHOLDERS: Record<string, string> = {
  software: 'Describe what you want to build...\n\nExample: "I need a booking system for a tattoo studio with appointment scheduling, artist portfolios, and client management"',
  process: 'Describe the process you want to model...\n\nExample: "Map out an employee onboarding workflow from offer acceptance to first day"',
  api: 'Describe the API you want to design...\n\nExample: "I need a REST API for a multi-tenant SaaS platform with user management and billing"',
  agent_system: 'Describe the agent system you want to build...\n\nExample: "I need a multi-agent system for automated code review with specialized reviewer agents"',
};

interface IntentInputProps {
  value: string;
  onChange: (value: string) => void;
  domain: string;
  disabled?: boolean;
}

export function IntentInput({ value, onChange, domain, disabled }: IntentInputProps) {
  const ref = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (ref.current) {
      ref.current.style.height = 'auto';
      ref.current.style.height = Math.max(160, ref.current.scrollHeight) + 'px';
    }
  }, [value]);

  return (
    <textarea
      ref={ref}
      value={value}
      onChange={(e) => onChange(e.target.value)}
      placeholder={DOMAIN_PLACEHOLDERS[domain] || DOMAIN_PLACEHOLDERS.software}
      disabled={disabled}
      className="input w-full min-h-[160px] resize-none font-mono text-sm leading-relaxed"
      autoFocus
    />
  );
}
