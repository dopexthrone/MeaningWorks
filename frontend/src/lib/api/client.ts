const getBaseUrl = (): string => {
  if (typeof window !== 'undefined') {
    const stored = localStorage.getItem('motherlabs_api_url');
    if (stored) return stored;
  }
  // In production static export, use relative URLs (same origin via Caddy)
  // Only fall back to localhost in dev when NEXT_PUBLIC_API_URL is explicitly set
  return process.env.NEXT_PUBLIC_API_URL || '';
};

const getApiKey = (): string | null => {
  if (typeof window !== 'undefined') {
    return localStorage.getItem('motherlabs_api_key');
  }
  return null;
};

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = 'ApiError';
  }
}

export async function apiFetch<T>(
  path: string,
  options: RequestInit = {}
): Promise<T> {
  const base = getBaseUrl();
  const url = `${base}${path}`;

  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
  };

  const apiKey = getApiKey();
  if (apiKey) {
    headers['X-API-Key'] = apiKey;
  }

  const res = await fetch(url, {
    ...options,
    headers: {
      ...headers,
      ...(options.headers as Record<string, string>),
    },
  });

  if (!res.ok) {
    const text = await res.text().catch(() => res.statusText);
    if (res.status === 401) {
      throw new ApiError(res.status, 'Invalid or missing API key. Set your key in Settings.');
    }
    if (res.status === 402) {
      throw new ApiError(res.status, 'Budget exceeded. Contact support to increase your limit.');
    }
    if (res.status === 429) {
      throw new ApiError(res.status, 'Rate limit exceeded. Please wait and try again.');
    }
    throw new ApiError(res.status, text);
  }

  return res.json();
}
