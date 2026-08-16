import { DatasetInfo, FeatureExtractionStats, RunRecord, SplitStats, TestSample } from '../lib/api';

// Each config below is also handed to api.ts's request helpers as a plain
// JSON body (typed there as Record<string, unknown>) — the index signature
// lets them cross that boundary without a cast at every call site.

export interface SplitParams {
  min_samples_per_class: number;
  test_size: number;
  [key: string]: unknown;
}

export interface BorutaConfig {
  n_estimators: string;
  rf_n_estimators: number;
  max_depth: string;
  max_iter: number;
  perc: number;
  alpha: number;
  class_weight: string;
  random_state: number;
  selection_mode: 'confirmed' | 'confirmed_tentative' | 'top_k';
  k: number | string;
  [key: string]: unknown;
}

export interface MrmrConfig {
  criterion: string;
  K: number;
  n_bins: number;
  random_state: number;
  [key: string]: unknown;
}

export interface ModelConfig {
  n_estimators: number;
  max_depth: number;
  min_samples_leaf: number;
  class_weight: string;
  random_state: number;
  min_confidence: number;
  min_fidelity: number;
  min_support: number;
  min_abs_support: number;
  max_conditions: number;
  merge_same_gene: boolean;
  dedup: boolean;
  dedup_sig_figs: number;
  merge_generalization: boolean;
  max_rules_per_class: number;
  max_rules_total: number;
  [key: string]: unknown;
}

export interface ModelStatsUI {
  acc: number;
  f1: number;
  rules: number;
  cm: number[][];
  labels: string[];
  hyperparams?: Record<string, unknown> | null;
  rulesSummary?: {
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

// The UI-shaped prediction result applyPredictionResult() builds from a raw
// PredictResponse — camelCase, with the rule/partial-match fields already
// reshaped for direct rendering.
export interface PredictedRule {
  id: number;
  text: string;
  matched: boolean;
  desc: string;
  sampleValues: Record<string, number>;
  class: string;
}

export interface PartialMatch {
  ruleId: number;
  text: string;
  class: string;
  satisfied: number;
  total: number;
  ratio: number;
  conditions: Array<{ gene: string; probe: string; op: string; threshold: number; actual: number; ok: boolean }>;
}

export interface TestResults {
  matchedCount: number;
  rules: PredictedRule[];
  classification: string;
  explanation?: string;
  classDescription?: string;
  classDisplayName?: string | null;
  classDisplayNames?: Record<string, string>;
  rulePrediction?: string | null;
  rulePredictionDescription?: string | null;
  rulePredictionDisplayName?: string | null;
  classVotes?: Record<string, { count: number; percentage: number }>;
  classVotesOver50?: Record<string, { count: number; percentage: number }>;
  partialMatches?: PartialMatch[];
  nPartialMatchesTotal?: number;
  nRulesTotal?: number | null;
  trueLabel?: string;
  biomedicalSummary?: string;
  biomedicalRationale?: string;
  biomedicalModelVsRule?: string;
  biomedicalDisclaimer?: string;
  llmUsed?: boolean;
}

export const DEFAULT_BORUTA_CONFIG: BorutaConfig = {
  n_estimators: 'auto',
  rf_n_estimators: 500,
  max_depth: '',
  max_iter: 100,
  perc: 100,
  alpha: 0.05,
  class_weight: 'balanced',
  random_state: 42,
  selection_mode: 'confirmed',
  k: '',
};

export const DEFAULT_MRMR_CONFIG: MrmrConfig = {
  criterion: 'MIQ',
  K: 50,
  n_bins: 3,
  random_state: 42,
};

export const DEFAULT_MODEL_CONFIG: ModelConfig = {
  n_estimators: 200,
  max_depth: 5,
  min_samples_leaf: 2,
  class_weight: 'balanced_subsample',
  random_state: 42,
  min_confidence: 0.8,
  min_fidelity: 0.0,
  min_support: 0.05,
  min_abs_support: 3,
  max_conditions: 5,
  merge_same_gene: true,
  dedup: true,
  dedup_sig_figs: 2,
  merge_generalization: true,
  max_rules_per_class: 20,
  max_rules_total: 100,
};

export const DEFAULT_SPLIT_INPUTS: SplitParams = { min_samples_per_class: 5, test_size: 0.2 };

// Everything the 5-step wizard needs EXCEPT the "Tải lên dataset mới" wizard
// inside Step 1 (uploadSource/uploadFile1/uploadCharacteristics/...), which
// has no cross-step reset relationship with the rest of this state and lives
// as local useState inside Step1Dataset.tsx instead.
export interface WorkspaceState {
  activeStep: number;

  datasets: DatasetInfo[];
  datasetId: string;
  datasetsError: string;
  datasetTab: 'existing' | 'upload';

  splitParams: SplitParams | null;
  splitInputs: SplitParams;
  splitStats: SplitStats | null;
  isSplitLoading: boolean;
  splitLog: string;
  splitRuns: RunRecord[];

  fsMethod: 'boruta' | 'mrmr';
  fsLog: string;
  isFsLoading: boolean;
  extractionStats: FeatureExtractionStats | null;
  fsRunId: string | null;
  fsRuns: RunRecord[];
  borutaConfig: BorutaConfig;
  mrmrConfig: MrmrConfig;
  mrmrKMode: '50' | '75' | 'custom';

  modelType: 'dt' | 'rf';
  modelStats: ModelStatsUI | null;
  isModelLoading: boolean;
  modelRunId: string | null;
  modelLog: string;
  modelRuns: RunRecord[];
  modelConfig: ModelConfig;
  isModelOverviewCollapsed: boolean;
  isModelConfigCollapsed: boolean;
  isConfusionMatrixCollapsed: boolean;
  isClassificationReportCollapsed: boolean;

  testMode: 'sample' | 'upload';
  testSamples: TestSample[];
  testSampleId: string;
  testUploadFile: File | null;
  isTesting: boolean;
  testResults: TestResults | null;
  isTestResultsCollapsed: boolean;
  expandedRule: number | null;
  isMatchedRulesCollapsed: boolean;
  isPartialMatchesCollapsed: boolean;
}

// K is part of the artifact identifier — this is why cached K=50 and K=75
// runs are loaded through mrmr_k50/mrmr_k75 rather than the obsolete mrmr_miq.
export function fsMethodKeyOf(state: Pick<WorkspaceState, 'fsMethod' | 'mrmrConfig'>): string {
  return state.fsMethod === 'mrmr' ? `mrmr_k${state.mrmrConfig.K}` : state.fsMethod;
}

export const initialWorkspaceState: WorkspaceState = {
  activeStep: 1,

  datasets: [],
  datasetId: '',
  datasetsError: '',
  datasetTab: 'existing',

  splitParams: null,
  splitInputs: DEFAULT_SPLIT_INPUTS,
  splitStats: null,
  isSplitLoading: false,
  splitLog: '',
  splitRuns: [],

  fsMethod: 'boruta',
  fsLog: '',
  isFsLoading: false,
  extractionStats: null,
  fsRunId: null,
  fsRuns: [],
  borutaConfig: DEFAULT_BORUTA_CONFIG,
  mrmrConfig: DEFAULT_MRMR_CONFIG,
  mrmrKMode: '50',

  modelType: 'rf',
  modelStats: null,
  isModelLoading: false,
  modelRunId: null,
  modelLog: '',
  modelRuns: [],
  modelConfig: DEFAULT_MODEL_CONFIG,
  isModelOverviewCollapsed: false,
  isModelConfigCollapsed: false,
  isConfusionMatrixCollapsed: false,
  isClassificationReportCollapsed: false,

  testMode: 'sample',
  testSamples: [],
  testSampleId: '',
  testUploadFile: null,
  isTesting: false,
  testResults: null,
  isTestResultsCollapsed: false,
  expandedRule: null,
  isMatchedRulesCollapsed: false,
  isPartialMatchesCollapsed: false,
};

export type WorkspaceAction =
  | { type: 'active_step_set'; step: number }
  | { type: 'datasets_loaded'; datasets: DatasetInfo[] }
  | { type: 'datasets_load_failed'; message: string }
  | { type: 'dataset_added'; dataset: DatasetInfo }
  | { type: 'dataset_tab_changed'; tab: 'existing' | 'upload' }
  | { type: 'dataset_selected'; datasetId: string }
  | { type: 'split_inputs_changed'; inputs: SplitParams }
  | { type: 'split_runs_loaded'; runs: RunRecord[] }
  | { type: 'split_started'; log: string }
  | { type: 'split_log_appended'; line: string }
  | { type: 'split_computed'; stats: SplitStats; params: SplitParams | null; inputs: SplitParams }
  | { type: 'split_failed'; line: string }
  | { type: 'split_run_loaded'; stats: SplitStats; params: SplitParams; log: string }
  | { type: 'fs_method_changed'; method: 'boruta' | 'mrmr' }
  | { type: 'boruta_config_changed'; config: BorutaConfig }
  | { type: 'mrmr_config_changed'; config: MrmrConfig }
  | { type: 'mrmr_k_mode_changed'; mode: '50' | '75' | 'custom' }
  | { type: 'fs_runs_loaded'; runs: RunRecord[] }
  | { type: 'fs_started'; log: string }
  | { type: 'fs_log_set'; log: string }
  | { type: 'fs_log_appended'; line: string }
  | { type: 'fs_computed'; stats: FeatureExtractionStats; runId: string | null }
  | { type: 'fs_failed'; line: string }
  | { type: 'model_type_changed'; modelType: 'dt' | 'rf' }
  | { type: 'model_config_changed'; config: ModelConfig }
  | { type: 'model_runs_loaded'; runs: RunRecord[] }
  | { type: 'model_started'; log: string }
  | { type: 'model_log_set'; log: string }
  | { type: 'model_log_appended'; line: string }
  | { type: 'model_computed'; stats: ModelStatsUI; runId: string | null }
  | { type: 'model_failed'; line: string }
  | { type: 'model_overview_toggled' }
  | { type: 'model_overview_expanded' }
  | { type: 'model_config_panel_toggled' }
  | { type: 'confusion_matrix_toggled' }
  | { type: 'classification_report_toggled' }
  | { type: 'test_mode_changed'; mode: 'sample' | 'upload' }
  | { type: 'test_samples_loaded'; samples: TestSample[] }
  | { type: 'test_sample_selected'; sampleId: string }
  | { type: 'test_upload_file_changed'; file: File | null }
  | { type: 'test_started' }
  | { type: 'test_computed'; result: TestResults }
  | { type: 'test_failed'; result: TestResults }
  | { type: 'test_results_toggled' }
  | { type: 'expanded_rule_toggled'; ruleId: number }
  | { type: 'matched_rules_toggled' }
  | { type: 'partial_matches_toggled' };
