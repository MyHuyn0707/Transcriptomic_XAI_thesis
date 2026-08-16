import { WorkspaceAction, WorkspaceState } from './types';

// Every level below the one that just changed carries stale results — e.g.
// re-running the split invalidates feature selection, which invalidates the
// model, which invalidates the last test prediction. Each RESET group below
// is the slice a given level clears in ITSELF when a level above it changes;
// downstreamOf(level) unions a level with everything beneath it, so a single
// action can clear its whole downstream in one spread instead of repeating
// the same 6-8 setters at every call site that used to invalidate it by hand.
const SPLIT_RESET = {
  splitStats: null,
  splitParams: null,
  splitLog: '',
};

const FS_RESET = {
  extractionStats: null,
  fsRunId: null,
  fsLog: '',
};

const MODEL_RESET = {
  modelStats: null,
  modelRunId: null,
  modelLog: '',
};

const TEST_RESET = {
  testSamples: [],
  testSampleId: '',
  testResults: null,
  testUploadFile: null,
};

function downstreamOf(level: 'dataset' | 'split' | 'fs' | 'model') {
  switch (level) {
    case 'dataset': return { ...SPLIT_RESET, ...FS_RESET, ...MODEL_RESET, ...TEST_RESET };
    case 'split': return { ...FS_RESET, ...MODEL_RESET, ...TEST_RESET };
    case 'fs': return { ...MODEL_RESET, ...TEST_RESET };
    case 'model': return { ...TEST_RESET };
  }
}

export function workspaceReducer(state: WorkspaceState, action: WorkspaceAction): WorkspaceState {
  switch (action.type) {
    case 'active_step_set':
      // Right-column panels for Bước 4/5 stay mounted once their data exists —
      // only their collapse state follows the active step (collapseSignal
      // pattern), so switching steps folds a panel down instead of removing it.
      return {
        ...state,
        activeStep: action.step,
        isModelOverviewCollapsed: action.step !== 4,
        isTestResultsCollapsed: action.step !== 5,
      };

    case 'datasets_loaded':
      return { ...state, datasets: action.datasets };
    case 'datasets_load_failed':
      return { ...state, datasetsError: action.message };
    case 'dataset_added':
      return { ...state, datasets: [...state.datasets, action.dataset] };
    case 'dataset_tab_changed':
      return { ...state, datasetTab: action.tab };

    case 'dataset_selected':
      // Bước 1 re-select -> ignore steps 2-5 entirely.
      return { ...state, ...downstreamOf('dataset'), datasetId: action.datasetId };

    case 'split_inputs_changed':
      return { ...state, splitInputs: action.inputs };
    case 'split_runs_loaded':
      return { ...state, splitRuns: action.runs };

    case 'split_started':
      return { ...state, ...downstreamOf('split'), isSplitLoading: true, splitLog: action.log };
    case 'split_log_appended':
      return { ...state, splitLog: `${state.splitLog}\n${action.line}` };
    case 'split_computed':
      return {
        ...state,
        isSplitLoading: false,
        splitStats: action.stats,
        splitParams: action.params,
        splitInputs: action.inputs,
      };
    case 'split_failed':
      return { ...state, isSplitLoading: false, splitLog: `${state.splitLog}\n${action.line}` };
    case 'split_run_loaded':
      // Reading a stored run summary straight off disk, not recomputing —
      // synchronous, so isSplitLoading never enters into it.
      return {
        ...state,
        ...downstreamOf('split'),
        splitStats: action.stats,
        splitParams: action.params,
        splitInputs: action.params,
        splitLog: action.log,
      };

    case 'fs_method_changed':
      // The previously-shown extraction stats belong to the PREVIOUS method,
      // so clear the fs level itself too (not just its downstream) rather
      // than leaving a stale panel that doesn't match the radio now selected.
      return { ...state, ...downstreamOf('split'), fsMethod: action.method };
    case 'boruta_config_changed':
      return { ...state, borutaConfig: action.config };
    case 'mrmr_config_changed':
      return { ...state, mrmrConfig: action.config };
    case 'mrmr_k_mode_changed':
      return { ...state, mrmrKMode: action.mode };
    case 'fs_runs_loaded':
      return { ...state, fsRuns: action.runs };

    case 'fs_started':
      return { ...state, ...downstreamOf('fs'), isFsLoading: true, fsLog: action.log };
    case 'fs_log_set':
      return { ...state, fsLog: action.log };
    case 'fs_log_appended':
      return { ...state, fsLog: `${state.fsLog}\n${action.line}` };
    case 'fs_computed':
      return {
        ...state,
        isFsLoading: false,
        extractionStats: action.stats,
        fsRunId: action.runId,
      };
    case 'fs_failed':
      return { ...state, isFsLoading: false, fsLog: `${state.fsLog}\n${action.line}` };

    case 'model_type_changed':
      // Same reasoning as fs_method_changed: the shown model stats belong to
      // the previous model type, clear them too, plus everything below.
      return { ...state, ...downstreamOf('fs'), modelType: action.modelType };
    case 'model_config_changed':
      return { ...state, modelConfig: action.config };
    case 'model_runs_loaded':
      return { ...state, modelRuns: action.runs };

    case 'model_started':
      return { ...state, ...downstreamOf('model'), isModelLoading: true, modelStats: null, modelLog: action.log };
    case 'model_log_set':
      return { ...state, modelLog: action.log };
    case 'model_log_appended':
      return { ...state, modelLog: `${state.modelLog}\n${action.line}` };
    case 'model_computed':
      return {
        ...state,
        isModelLoading: false,
        modelStats: action.stats,
        modelRunId: action.runId,
      };
    case 'model_failed':
      return { ...state, isModelLoading: false, modelLog: `${state.modelLog}\n${action.line}` };

    case 'model_overview_toggled':
      return { ...state, isModelOverviewCollapsed: !state.isModelOverviewCollapsed };
    case 'model_overview_expanded':
      return { ...state, isModelOverviewCollapsed: false };
    case 'model_config_panel_toggled':
      return { ...state, isModelConfigCollapsed: !state.isModelConfigCollapsed };
    case 'confusion_matrix_toggled':
      return { ...state, isConfusionMatrixCollapsed: !state.isConfusionMatrixCollapsed };
    case 'classification_report_toggled':
      return { ...state, isClassificationReportCollapsed: !state.isClassificationReportCollapsed };

    case 'test_mode_changed':
      return { ...state, testMode: action.mode };
    case 'test_samples_loaded':
      return { ...state, testSamples: action.samples };
    case 'test_sample_selected':
      return { ...state, testSampleId: action.sampleId };
    case 'test_upload_file_changed':
      return { ...state, testUploadFile: action.file };

    case 'test_started':
      return { ...state, isTesting: true, testResults: null };
    case 'test_computed':
      return { ...state, isTesting: false, testResults: action.result };
    case 'test_failed':
      return { ...state, isTesting: false, testResults: action.result };
    case 'test_results_toggled':
      return { ...state, isTestResultsCollapsed: !state.isTestResultsCollapsed };

    case 'expanded_rule_toggled':
      return { ...state, expandedRule: state.expandedRule === action.ruleId ? null : action.ruleId };
    case 'matched_rules_toggled':
      return { ...state, isMatchedRulesCollapsed: !state.isMatchedRulesCollapsed };
    case 'partial_matches_toggled':
      return { ...state, isPartialMatchesCollapsed: !state.isPartialMatchesCollapsed };
  }
}
