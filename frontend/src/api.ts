import type { HistoryItem, Job, JobResults, JobSettings } from './types';

const API_BASE = (import.meta.env.VITE_API_BASE ?? '/api').replace(/\/$/, '');

function responseError(status: number, raw: string): Error {
  let detail: string | { message?: string; code?: string } | undefined;
  try {
    detail = (
      JSON.parse(raw) as {
        detail?: string | { message?: string; code?: string };
      }
    ).detail;
  } catch { /* non-JSON error body */ }
  if (typeof detail === 'string') return new Error(detail);
  if (detail && typeof detail === 'object') {
    return new Error(detail.message || detail.code || `HTTP ${status}`);
  }
  return new Error(raw || `HTTP ${status}`);
}

async function readJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const raw = await response.text();
    throw responseError(response.status, raw);
  }
  return response.json() as Promise<T>;
}

export type HealthState = {
  reachable: boolean;
  ready: boolean;
  demo: boolean;
  status: 'ok' | 'degraded' | 'offline';
};

export async function health(signal?: AbortSignal): Promise<HealthState> {
  try {
    const response = await fetch(`${API_BASE}/health`, { signal });
    if (!response.ok) return { reachable: true, ready: false, demo: false, status: 'degraded' };
    const payload = await response.json() as {
      status?: string;
      demo_mode?: boolean;
      models?: Record<string, { status?: string }>;
    };
    const modelsReady = Object.values(payload.models ?? {}).every((model) => model.status === 'configured');
    return {
      reachable: true,
      ready: modelsReady || payload.demo_mode === true,
      demo: payload.demo_mode === true,
      status: payload.status === 'ok' ? 'ok' : 'degraded',
    };
  } catch {
    return { reachable: false, ready: false, demo: false, status: 'offline' };
  }
}

export function createJob(
  files: File[],
  settings: JobSettings,
  onUploadProgress?: (percent: number) => void,
): Promise<Job> {
  const body = new FormData();
  files.forEach((file) => body.append('files', file, file.webkitRelativePath || file.name));
  body.append('settings', JSON.stringify(settings));
  return new Promise((resolve, reject) => {
    const request = new XMLHttpRequest();
    request.open('POST', `${API_BASE}/jobs`);
    request.upload.addEventListener('progress', (event) => {
      if (event.lengthComputable) {
        onUploadProgress?.(Math.min(100, Math.max(0, (event.loaded / event.total) * 100)));
      }
    });
    request.addEventListener('error', () => reject(new Error('Потеряно соединение с backend.')));
    request.addEventListener('abort', () => reject(new Error('Загрузка отменена.')));
    request.addEventListener('load', () => {
      if (request.status < 200 || request.status >= 300) {
        reject(responseError(request.status, request.responseText));
        return;
      }
      try {
        resolve(JSON.parse(request.responseText) as Job);
      } catch {
        reject(new Error('Backend вернул некорректный ответ.'));
      }
    });
    onUploadProgress?.(0);
    request.send(body);
  });
}

export async function getJob(id: string, signal?: AbortSignal): Promise<Job> {
  return readJson<Job>(await fetch(`${API_BASE}/jobs/${encodeURIComponent(id)}`, { signal }));
}

export async function getResults(id: string): Promise<JobResults> {
  return readJson<JobResults>(
    await fetch(`${API_BASE}/jobs/${encodeURIComponent(id)}/results`),
  );
}

export async function getHistory(limit = 50): Promise<HistoryItem[]> {
  const payload = await readJson<HistoryItem[] | { items?: HistoryItem[] }>(
    await fetch(`${API_BASE}/history?limit=${Math.max(1, Math.min(50, limit))}`),
  );
  return Array.isArray(payload) ? payload : payload.items ?? [];
}

export async function patchImageSettings(
  jobId: string,
  imageId: string,
  settings: JobSettings,
): Promise<Job> {
  return readJson<Job>(
    await fetch(
      `${API_BASE}/jobs/${encodeURIComponent(jobId)}/images/${encodeURIComponent(imageId)}/settings`,
      {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(settings),
      },
    ),
  );
}

export async function appendJobImages(
  jobId: string,
  files: File[],
  settings: JobSettings,
): Promise<Job> {
  const body = new FormData();
  files.forEach((file) => body.append('files', file, file.webkitRelativePath || file.name));
  body.append('settings', JSON.stringify(settings));
  return readJson<Job>(
    await fetch(`${API_BASE}/jobs/${encodeURIComponent(jobId)}/images`, {
      method: 'POST',
      body,
    }),
  );
}
