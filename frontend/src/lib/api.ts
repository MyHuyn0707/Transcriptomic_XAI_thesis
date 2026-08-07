export interface DatasetInfo {
  id: string;
  name: string;
  platform: string;
  n_samples: number;
  n_features: number;
  n_classes: number;
  class_labels: string[];
  description: string;
  fs_models: Record<string, string[]>;
  is_temp?: boolean;
}

export interface JobStatus {
  job_id: string;
  status: 'running' | 'done' | 'error';
  log: string[];
  result: Record<string, unknown> | null;
  error: string | null;
}

export interface RunRecord {
  run_id: string;
  kind: 'feature_selection' | 'model' | 'upload' | 'split';
  dataset: string | null;
  fs_method?: string;
  model?: string;
  status: 'done' | 'error';
  created_at: number;
  summary: Record<string, unknown>;
}

// Every dataset_id/fs_method/model/run_id below is interpolated into a URL
// path or query string — encode each one so a value containing '/', '?',
// '#' or '&' can't corrupt the request instead of just erroring clearly.
const enc = (s: string) => encodeURIComponent(s);

async function req(path: string, options?: RequestInit) {
  const res = await fetch(path, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.detail || `${res.status} ${res.statusText}`);
  }
  return res.json();
}

export const api = {
  getDatasets: (): Promise<DatasetInfo[]> => req('/api/datasets'),

  getOverview: (datasetId: string) => req(`/api/datasets/${enc(datasetId)}/overview`),

  getFeatureSelection: (datasetId: string, fsMethod: string, runId?: string | null) =>
    req(`/api/datasets/${enc(datasetId)}/feature-selection/${enc(fsMethod)}${runId ? `?run_id=${enc(runId)}` : ''}`),

  trainFeatureSelection: (
    datasetId: string,
    fsMethod: string,
    params: Record<string, unknown>,
    splitParams?: Record<string, unknown> | null,
  ) =>
    req(`/api/datasets/${enc(datasetId)}/feature-selection/${enc(fsMethod)}/train`, {
      method: 'POST',
      body: JSON.stringify({ ...params, split_params: splitParams || null }),
    }),

  getModelStats: (datasetId: string, fsMethod: string, model: string, runId?: string | null) =>
    req(`/api/datasets/${enc(datasetId)}/model/${enc(fsMethod)}/${enc(model)}${runId ? `?run_id=${enc(runId)}` : ''}`),

  trainModel: (
    datasetId: string,
    fsMethod: string,
    model: string,
    hyperparams: Record<string, unknown>,
    fsRunId?: string | null,
    splitParams?: Record<string, unknown> | null,
  ) =>
    req(`/api/datasets/${enc(datasetId)}/model/${enc(fsMethod)}/${enc(model)}/train`, {
      method: 'POST',
      body: JSON.stringify({ hyperparams, fs_run_id: fsRunId || null, split_params: splitParams || null }),
    }),

  getJob: (jobId: string): Promise<JobStatus> => req(`/api/jobs/${enc(jobId)}`),

  getRules: (datasetId: string, fsMethod: string, model: string, runId?: string | null) =>
    req(`/api/datasets/${enc(datasetId)}/rules/${enc(fsMethod)}/${enc(model)}${runId ? `?run_id=${enc(runId)}` : ''}`),

  getGenes: (datasetId: string, fsMethod: string, model: string, runId?: string | null) =>
    req(`/api/datasets/${enc(datasetId)}/genes/${enc(fsMethod)}/${enc(model)}${runId ? `?run_id=${enc(runId)}` : ''}`),

  predict: (
    datasetId: string,
    fsMethod: string,
    model: string,
    sampleId: string,
    runId?: string | null,
    splitParams?: Record<string, unknown> | null,
  ) =>
    req(`/api/datasets/${enc(datasetId)}/predict`, {
      method: 'POST',
      body: JSON.stringify({ fs_method: fsMethod, model, sample_id: sampleId, run_id: runId || null, split_params: splitParams || null }),
    }),

  predictUpload: async (
    datasetId: string,
    fsMethod: string,
    model: string,
    file: File,
    runId?: string | null,
  ) => {
    const form = new FormData();
    form.append('fs_method', fsMethod);
    form.append('model', model);
    if (runId) form.append('run_id', runId);
    form.append('file', file);
    const res = await fetch(`/api/datasets/${enc(datasetId)}/predict-upload`, { method: 'POST', body: form });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || `${res.status} ${res.statusText}`);
    }
    return res.json();
  },

  getRuns: (datasetId: string, kind?: 'feature_selection' | 'model' | 'split'): Promise<RunRecord[]> =>
    req(`/api/datasets/${enc(datasetId)}/runs${kind ? `?kind=${kind}` : ''}`),

  splitPreview: (datasetId: string, splitParams: Record<string, unknown>) =>
    req(`/api/datasets/${enc(datasetId)}/split/preview`, {
      method: 'POST',
      body: JSON.stringify(splitParams || {}),
    }),

  getTestSamplesWithSplit: (datasetId: string, fsMethod: string, splitParams?: Record<string, unknown> | null) => {
    const qs = new URLSearchParams();
    if (splitParams?.min_samples_per_class != null) qs.set('min_samples_per_class', String(splitParams.min_samples_per_class));
    if (splitParams?.test_size != null) qs.set('test_size', String(splitParams.test_size));
    const query = qs.toString();
    return req(`/api/datasets/${enc(datasetId)}/test-samples/${enc(fsMethod)}${query ? `?${query}` : ''}`);
  },

  // Plain URL (not a fetch call) for an <a href download> link — works for
  // both a cached dataset AND a fresh upload / custom "Thực hiện lại" split
  // with no cache yet (backend recomputes live in that case).
  getTestSetDownloadUrl: (datasetId: string, fsMethod: string, splitParams?: Record<string, unknown> | null) => {
    const qs = new URLSearchParams();
    if (splitParams?.min_samples_per_class != null) qs.set('min_samples_per_class', String(splitParams.min_samples_per_class));
    if (splitParams?.test_size != null) qs.set('test_size', String(splitParams.test_size));
    const query = qs.toString();
    return `/api/datasets/${enc(datasetId)}/test-samples/${enc(fsMethod)}/download${query ? `?${query}` : ''}`;
  },

  uploadInspect: async (source: 'geo' | 'cumida', tissue: string, file1: File, file2?: File | null) => {
    const form = new FormData();
    form.append('source', source);
    form.append('tissue', tissue);
    form.append('file1', file1);
    if (file2) form.append('file2', file2);
    const res = await fetch('/api/datasets/upload/inspect', { method: 'POST', body: form });
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.detail || `${res.status} ${res.statusText}`);
    }
    return res.json();
  },

  uploadBuild: (uploadId: string, source: 'geo' | 'cumida', classCharacteristic: string, tissue: string) =>
    req('/api/datasets/upload/build', {
      method: 'POST',
      body: JSON.stringify({ upload_id: uploadId, source, class_characteristic: classCharacteristic, tissue }),
    }),

  getUploadHistory: (): Promise<RunRecord[]> => req('/api/datasets/upload/history'),
};

export async function pollJob(
  jobId: string,
  onUpdate: (job: JobStatus) => void,
  intervalMs = 2000,
): Promise<JobStatus> {
  // eslint-disable-next-line no-constant-condition
  while (true) {
    const job = await api.getJob(jobId);
    onUpdate(job);
    if (job.status === 'done' || job.status === 'error') return job;
    await new Promise((r) => setTimeout(r, intervalMs));
  }
}
