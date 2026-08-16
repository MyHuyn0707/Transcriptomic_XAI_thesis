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

export interface TestSample {
  sample_id: string;
  true_label: string;
  row_index: number;
}

// The train/test split summary — returned directly by splitPreview, and as a
// subset of the richer /overview response (OverviewData below).
export interface SplitStats {
  class_labels: string[];
  train_class_counts: Record<string, number>;
  test_class_counts: Record<string, number>;
  raw_class_counts: Record<string, number>;
  dropped_classes: string[];
  n_samples_total: number;
  n_train: number;
  n_test: number;
  test_size: number;
  min_samples_per_class: number;
}

export interface OriginInfo {
  geo_accession?: string;
  platform?: string;
  title?: string;
  source_cumida_csv?: string;
  class_characteristic?: string;
  classes_raw?: Record<string, number>;
  annotation_rows?: number;
  probes_with_gene_symbol?: number;
  probes_from_annotation_file?: number;
  probes_from_mygene?: number;
  probes_without_gene_symbol?: number;
  affx_control_removed?: number;
  unique_probes?: number;
  unique_genes?: number;
  samples_kept?: number;
  na_dropped?: number;
}

export interface ClassContent {
  display_name_vi?: string;
  description_vi?: string;
  canonical_name_en?: string;
  entity_type?: string;
  excluded_from_model?: boolean;
  label_note_vi?: string;
  source_ids?: string[];
}

export interface Reference {
  title: string;
  publisher: string;
  url: string;
  source_type?: string;
}

export interface DiseaseContextEntity {
  entity_id: string;
  name_vi: string;
  canonical_name_en?: string;
  description_vi: string;
  source_ids?: string[];
  media_asset_ids?: string[];
}

export interface DiseaseContext {
  scope_type?: string;
  primary_display_entity_id?: string | null;
  ui_title_vi?: string;
  primary_entities?: DiseaseContextEntity[];
  dataset_relationship_vi?: string;
  media_asset_ids?: string[];
  media_disclaimer_vi?: string;
}

export interface MediaAsset {
  title_vi?: string;
  caption_vi?: string;
  alt_text_vi?: string;
  credit?: string;
  source_page_url?: string;
  source_publisher?: string;
  usage_note_vi?: string;
  url?: string | null;
}

export interface DatasetProvenance {
  local_folder_group?: string;
  download_provider?: {
    provider_id?: string;
    name?: string;
    canonical_name?: string;
    source_ids?: string[];
  };
  provenance_note_vi?: string;
  source_ids?: string[];
}

export interface DatasetStudy {
  arrayexpress_accession?: string;
  original_study_title_en?: string;
  study_objective_vi?: string;
  original_design_vi?: string;
  local_ml_relation_vi?: string;
  source_ids?: string[];
}

export interface ContentInfo {
  title_vi?: string;
  description_vi?: string;
  disease_context?: DiseaseContext;
  dataset_provenance?: DatasetProvenance;
  dataset_study?: DatasetStudy;
  organism?: string;
  experiment_type_vi?: string;
  dataset_note_vi?: string;
  class_descriptions_vi?: Record<string, string>;
  classes?: Record<string, ClassContent>;
  references?: Record<string, Reference>;
  ui_disclaimer_vi?: string;
}

export interface OverviewData extends SplitStats {
  dataset: string;
  visualizations: Record<string, string>;
  origin: OriginInfo;
  content: ContentInfo;
  media_assets?: Record<string, MediaAsset>;
}

// Boruta and mRMR each populate only their own subset of these fields —
// see FeatureExtractionOverview.tsx for which ones are read per method.
export interface FeatureExtractionStats {
  n_original_features?: number;
  n_selected_features: number;
  runtime_seconds?: number;
  feature_selection?: string;
  dataset_name?: string;
  framework?: string;
  n_samples?: number;
  selection_mode?: string;
  confirmed?: number;
  tentative?: number;
  rejected?: number;
  criterion?: string;
  K?: number;
  n_bins?: number;
  n_estimators?: string;
  rf_n_estimators?: number;
  max_depth?: string;
  max_iter?: number;
  perc?: number;
  alpha?: number;
  class_weight?: string;
  random_state?: number;
  k?: number;
}

export interface ModelStatsResponse {
  best_run_test_metrics?: { accuracy?: number; f1_macro?: number };
  n_rules?: number;
  confusion_matrix?: number[][];
  class_labels?: string[];
  hyperparams?: Record<string, unknown> | null;
  rules_summary?: {
    n_rules_raw?: number;
    n_rules_passed_filter?: number;
    n_rules_kept?: number;
    n_dropped_by_filter?: number;
    n_dropped_by_cap?: number;
    filter_config?: {
      min_confidence?: number | null;
      min_fidelity?: number | null;
      min_support?: number | null;
      min_abs_support?: number | null;
      max_conditions?: number | null;
      merge_same_gene?: boolean | null;
      dedup?: boolean | null;
      dedup_sig_figs?: number | null;
      merge_generalization?: boolean | null;
      max_rules_per_class?: number | null;
      max_rules_total?: number | null;
    } | null;
  } | null;
}

export interface PredictedRuleResponse {
  rule_id: number;
  text: string;
  matched: boolean;
  explanation?: string | null;
  sample_values?: Record<string, number>;
  consequent_label: string;
}

export interface PartialMatchResponse {
  rule_id: number;
  text: string;
  consequent_label: string;
  satisfied: number;
  total: number;
  ratio: number;
  conditions?: Array<{ gene: string; probe: string; op: string; threshold: number; actual: number; ok: boolean }>;
}

export interface PredictResponse {
  matched_count: number;
  rules: PredictedRuleResponse[];
  classification: string;
  class_description?: string;
  class_display_name?: string | null;
  class_display_names?: Record<string, string>;
  rule_prediction?: string | null;
  rule_prediction_description?: string | null;
  rule_prediction_display_name?: string | null;
  class_votes?: Record<string, { count: number; percentage: number }>;
  class_votes_over50?: Record<string, { count: number; percentage: number }>;
  partial_matches?: PartialMatchResponse[];
  n_partial_matches_total?: number;
  n_rules_total?: number | null;
  true_label?: string;
  biomedical_summary?: string;
  biomedical_rationale?: string;
  biomedical_model_vs_rule?: string;
  biomedical_disclaimer?: string;
  llm_used?: boolean;
}

// The job.result of a successful /datasets/upload/build job.
export interface BuiltDatasetResult {
  dataset_id: string;
  name: string;
  platform: string;
  n_samples: number;
  n_features: number;
  n_classes: number;
  class_labels: string[];
  description: string;
}

export interface UploadInspectResponse {
  upload_id: string;
  characteristics?: string[];
}

// Every dataset_id/fs_method/model/run_id below is interpolated into a URL
// path or query string — encode each one so a value containing '/', '?',
// '#' or '&' can't corrupt the request instead of just erroring clearly.
const enc = (s: string) => encodeURIComponent(s);

async function req<T = any>(path: string, options?: RequestInit): Promise<T> {
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

  getOverview: (datasetId: string) => req<OverviewData>(`/api/datasets/${enc(datasetId)}/overview`),

  getFeatureSelection: (datasetId: string, fsMethod: string, runId?: string | null) =>
    req<FeatureExtractionStats>(`/api/datasets/${enc(datasetId)}/feature-selection/${enc(fsMethod)}${runId ? `?run_id=${enc(runId)}` : ''}`),

  trainFeatureSelection: (
    datasetId: string,
    fsMethod: string,
    params: Record<string, unknown>,
    splitParams?: Record<string, unknown> | null,
  ) =>
    req<{ job_id: string }>(`/api/datasets/${enc(datasetId)}/feature-selection/${enc(fsMethod)}/train`, {
      method: 'POST',
      body: JSON.stringify({ ...params, split_params: splitParams || null }),
    }),

  getModelStats: (datasetId: string, fsMethod: string, model: string, runId?: string | null) =>
    req<ModelStatsResponse>(`/api/datasets/${enc(datasetId)}/model/${enc(fsMethod)}/${enc(model)}${runId ? `?run_id=${enc(runId)}` : ''}`),

  trainModel: (
    datasetId: string,
    fsMethod: string,
    model: string,
    hyperparams: Record<string, unknown>,
    fsRunId?: string | null,
    splitParams?: Record<string, unknown> | null,
  ) =>
    req<{ job_id: string }>(`/api/datasets/${enc(datasetId)}/model/${enc(fsMethod)}/${enc(model)}/train`, {
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
    req<PredictResponse>(`/api/datasets/${enc(datasetId)}/predict`, {
      method: 'POST',
      body: JSON.stringify({ fs_method: fsMethod, model, sample_id: sampleId, run_id: runId || null, split_params: splitParams || null }),
    }),

  predictUpload: async (
    datasetId: string,
    fsMethod: string,
    model: string,
    file: File,
    runId?: string | null,
  ): Promise<PredictResponse> => {
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
    req<SplitStats>(`/api/datasets/${enc(datasetId)}/split/preview`, {
      method: 'POST',
      body: JSON.stringify(splitParams || {}),
    }),

  getTestSamplesWithSplit: (datasetId: string, fsMethod: string, splitParams?: Record<string, unknown> | null) => {
    const qs = new URLSearchParams();
    if (splitParams?.min_samples_per_class != null) qs.set('min_samples_per_class', String(splitParams.min_samples_per_class));
    if (splitParams?.test_size != null) qs.set('test_size', String(splitParams.test_size));
    const query = qs.toString();
    return req<TestSample[]>(`/api/datasets/${enc(datasetId)}/test-samples/${enc(fsMethod)}${query ? `?${query}` : ''}`);
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

  uploadInspect: async (source: 'geo' | 'cumida', tissue: string, file1: File, file2?: File | null): Promise<UploadInspectResponse> => {
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
    req<{ job_id: string }>('/api/datasets/upload/build', {
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
