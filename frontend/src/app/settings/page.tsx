'use client';

import { useState, useEffect } from 'react';
import { Toggle } from '@/components/shared/Toggle';

export default function SettingsPage() {
  const [apiUrl, setApiUrl] = useState('');
  const [apiKey, setApiKey] = useState('');
  const [defaultDomain, setDefaultDomain] = useState('software');
  const [expertMode, setExpertMode] = useState(false);
  const [saved, setSaved] = useState(false);
  const [showKey, setShowKey] = useState(false);

  useEffect(() => {
    if (typeof window !== 'undefined') {
      setApiUrl(localStorage.getItem('motherlabs_api_url') || '');
      setApiKey(localStorage.getItem('motherlabs_api_key') || '');
      setDefaultDomain(localStorage.getItem('motherlabs_default_domain') || 'software');
      setExpertMode(localStorage.getItem('motherlabs_expert_mode') === 'true');
    }
  }, []);

  const save = () => {
    if (typeof window !== 'undefined') {
      if (apiUrl.trim()) {
        localStorage.setItem('motherlabs_api_url', apiUrl.trim());
      } else {
        localStorage.removeItem('motherlabs_api_url');
      }
      if (apiKey.trim()) {
        localStorage.setItem('motherlabs_api_key', apiKey.trim());
      } else {
        localStorage.removeItem('motherlabs_api_key');
      }
      localStorage.setItem('motherlabs_default_domain', defaultDomain);
      localStorage.setItem('motherlabs_expert_mode', String(expertMode));
    }
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  return (
    <div className="max-w-xl mx-auto space-y-6">
      <h2 className="text-lg font-semibold">Settings</h2>

      <div className="card space-y-4">
        <div>
          <label className="block text-xs text-[var(--text-muted)] mb-1">API Key</label>
          <div className="flex gap-2">
            <input
              type={showKey ? 'text' : 'password'}
              value={apiKey}
              onChange={(e) => setApiKey(e.target.value)}
              placeholder="Enter your API key"
              className="input flex-1 text-sm font-mono"
              autoComplete="off"
            />
            <button
              type="button"
              onClick={() => setShowKey(!showKey)}
              className="btn-secondary text-xs px-3"
            >
              {showKey ? 'Hide' : 'Show'}
            </button>
          </div>
          <p className="text-xs text-[var(--text-muted)] mt-1">
            Required for compilation. Contact your admin to get a key.
          </p>
        </div>

        <div>
          <label className="block text-xs text-[var(--text-muted)] mb-1">API Base URL</label>
          <input
            value={apiUrl}
            onChange={(e) => setApiUrl(e.target.value)}
            placeholder="Leave empty for default (same origin)"
            className="input w-full text-sm"
          />
          <p className="text-xs text-[var(--text-muted)] mt-1">
            Only change this if connecting to a different server.
          </p>
        </div>

        <div>
          <label className="block text-xs text-[var(--text-muted)] mb-1">Default Domain</label>
          <select
            value={defaultDomain}
            onChange={(e) => setDefaultDomain(e.target.value)}
            className="input w-full text-sm"
          >
            <option value="software">software</option>
            <option value="process">process</option>
            <option value="api">api</option>
            <option value="agent_system">agent_system</option>
          </select>
        </div>

        <div>
          <Toggle
            checked={expertMode}
            onChange={setExpertMode}
            label="Expert mode (show advanced options by default)"
          />
        </div>

        <button onClick={save} className="btn-primary text-sm">
          {saved ? 'Saved!' : 'Save Settings'}
        </button>
      </div>
    </div>
  );
}
