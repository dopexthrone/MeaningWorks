'use client';

import { useState, useEffect } from 'react';
import { listDomains } from '@/lib/api/domains';
import type { DomainInfo } from '@/lib/api/types';

export function useDomains() {
  const [domains, setDomains] = useState<DomainInfo[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    listDomains()
      .then((res) => setDomains(res.domains))
      .catch(() => setDomains([
        { name: 'software', version: '1.0', output_format: 'python', file_extension: '.py', vocabulary_types: [], relationship_types: [], actionability_checks: [] },
        { name: 'process', version: '1.0', output_format: 'yaml', file_extension: '.yml', vocabulary_types: [], relationship_types: [], actionability_checks: [] },
        { name: 'api', version: '1.0', output_format: 'openapi', file_extension: '.yaml', vocabulary_types: [], relationship_types: [], actionability_checks: [] },
        { name: 'agent_system', version: '1.0', output_format: 'python', file_extension: '.py', vocabulary_types: [], relationship_types: [], actionability_checks: [] },
      ]))
      .finally(() => setLoading(false));
  }, []);

  return { domains, loading };
}
