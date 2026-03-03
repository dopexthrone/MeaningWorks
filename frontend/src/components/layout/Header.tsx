'use client';

import { StatusDot } from '@/components/shared/StatusDot';

interface HeaderProps {
  title: string;
  healthStatus?: 'ok' | 'warning' | 'error' | 'unknown';
  onMenuToggle?: () => void;
}

export function Header({ title, healthStatus = 'unknown', onMenuToggle }: HeaderProps) {
  return (
    <header className="flex items-center justify-between px-6 py-3 border-b border-[var(--border)] bg-[var(--bg-secondary)]">
      <div className="flex items-center gap-3">
        <button className="md:hidden p-1" onClick={onMenuToggle}>
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.5}>
            <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 6.75h16.5M3.75 12h16.5m-16.5 5.25h16.5" />
          </svg>
        </button>
        <h1 className="font-semibold text-sm">{title}</h1>
      </div>
      <div className="flex items-center gap-2 text-xs text-[var(--text-muted)]">
        <StatusDot status={healthStatus} size="sm" />
        <span>{healthStatus === 'ok' ? 'Connected' : 'Disconnected'}</span>
      </div>
    </header>
  );
}
