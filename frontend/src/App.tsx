import React, { useState, useEffect } from 'react';
import {
  Dna, Microscope, Activity, Database, GitMerge,
  CheckCircle2, Info, FileText, Settings2, PlayCircle, Loader2, Table2,
  UploadCloud, ListChecks, ChevronDown, ChevronUp, AlertTriangle,
  Download,
} from 'lucide-react';
import { cn } from './lib/utils';
import { classColor } from './lib/palette';
import { api, pollJob, DatasetInfo, RunRecord } from './lib/api';
import BioNetworkGraph from './components/BioNetworkGraph';
import ConfusionMatrix from './components/ConfusionMatrix';

import DatasetOverview from './components/DatasetOverview';
import DatasetSplit from './components/DatasetSplit';
import FeatureExtractionOverview from './components/FeatureExtractionOverview';
import RuleExtractionResults from './components/RuleExtractionResults';
import RunHistoryList from './components/RunHistoryList';

interface TestSample {
  sample_id: string;
  true_label: string;
  row_index: number;
}

interface ModelStatsUI {
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

// A class's Vietnamese display name if the dataset description defines one
// (same name shown in Dataset Overview), else just the raw label — avoids
// showing only the raw internal class label in the prediction report.
function displayLabel(raw: string, displayNames?: Record<string, string>) {
  return displayNames?.[raw] || raw;
}

function classificationReport(cm: number[][]) {
  return cm.map((row, i) => {
    const support = row.reduce((a, b) => a + b, 0);
    const colSum = cm.reduce((acc, r) => acc + r[i], 0);
    const tp = cm[i][i];
    const precision = colSum > 0 ? tp / colSum : 0;
    const recall = support > 0 ? tp / support : 0;
    const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
    return { precision, recall, f1, support };
  });
}

// Right column's fallback for whichever step is active but hasn't produced
// anything yet (e.g. Bước 3 opened before feature selection has run) — same
// dashed-box treatment the old single "Không gian Phân tích Trống" state used,
// just scoped per-step instead of one blanket message for the whole app.
function EmptyStepPlaceholder({ text }: { text: string }) {
  return (
    <div className="h-full min-h-[400px] bg-white border border-slate-200 border-dashed rounded-3xl flex flex-col items-center justify-center text-slate-400 p-8 text-center shadow-sm">
      <div className="w-16 h-16 bg-slate-50 rounded-full flex items-center justify-center mb-4">
        <Microscope size={28} className="text-slate-300" />
      </div>
      <p className="text-sm max-w-md leading-relaxed text-slate-500">{text}</p>
    </div>
  );
}

export default function App() {
  // Accordion: exactly one of the 5 left-column steps is expanded at a time,
  // and the right column shows ONLY that step's content — replaces the old
  // "everything stacked, scroll forever" layout. Clicking a step's header
  // (or its number) opens it; a step can only open once its prerequisite
  // step has produced something (see stepReady below).
  const [activeStep, setActiveStep] = useState(1);

  // State: Datasets (loaded from the real pipeline outputs)
  const [datasets, setDatasets] = useState<DatasetInfo[]>([]);
  const [datasetId, setDatasetId] = useState('');
  const [datasetsError, setDatasetsError] = useState('');

  // State: Dataset upload (Bước 1, tab "Tải lên dataset mới") — session-only,
  // built via /api/datasets/upload/{inspect,build}.
  const [datasetTab, setDatasetTab] = useState<'existing' | 'upload'>('existing');
  const [uploadSource, setUploadSource] = useState<'geo' | 'cumida'>('geo');
  const [uploadTissue, setUploadTissue] = useState('Dataset');
  const [uploadFile1, setUploadFile1] = useState<File | null>(null);
  const [uploadFile2, setUploadFile2] = useState<File | null>(null);
  const [uploadId, setUploadId] = useState<string | null>(null);
  const [uploadCharacteristics, setUploadCharacteristics] = useState<string[]>([]);
  const [uploadClassChar, setUploadClassChar] = useState('');
  const [isUploadInspecting, setIsUploadInspecting] = useState(false);
  const [isUploadBuilding, setIsUploadBuilding] = useState(false);
  const [uploadLog, setUploadLog] = useState('');
  const [uploadRuns, setUploadRuns] = useState<RunRecord[]>([]);

  // State: Bước 2 "Xử lý & Chia Dữ liệu" — splitParams=null means "dùng mặc
  // định của holdout.yaml"; non-null (từ "Thực hiện lại") flows down into
  // every FS/model/predict call below so they all read the SAME split.
  const [splitParams, setSplitParams] = useState<{ min_samples_per_class: number, test_size: number } | null>(null);
  const [splitInputs, setSplitInputs] = useState({ min_samples_per_class: 5, test_size: 0.2 });
  const [splitStats, setSplitStats] = useState<any>(null);
  const [isSplitLoading, setIsSplitLoading] = useState(false);
  const [splitLog, setSplitLog] = useState('');
  const [splitRuns, setSplitRuns] = useState<RunRecord[]>([]);

  // State: Feature Selection
  const [fsMethod, setFsMethod] = useState<'boruta' | 'mrmr'>('boruta');
  const [fsLog, setFsLog] = useState('');
  const [isFsLoading, setIsFsLoading] = useState(false);
  const [extractionStats, setExtractionStats] = useState<any>(null);
  const [fsRunId, setFsRunId] = useState<string | null>(null);
  const [fsRuns, setFsRuns] = useState<RunRecord[]>([]);

  const [borutaConfig, setBorutaConfig] = useState({
    n_estimators: 'auto',
    rf_n_estimators: 500,
    max_depth: '',
    max_iter: 100,
    perc: 100,
    alpha: 0.05,
    class_weight: 'balanced',
    random_state: 42,
    selection_mode: 'confirmed' as 'confirmed' | 'confirmed_tentative' | 'top_k',
    k: '' as number | string
  });

  const [mrmrConfig, setMrmrConfig] = useState({
    criterion: 'MIQ',
    K: 50,
    n_bins: 3,
    random_state: 42
  });

  // State: Model Selection
  const [modelType, setModelType] = useState<'dt' | 'rf'>('rf');
  const [modelStats, setModelStats] = useState<ModelStatsUI | null>(null);
  const [isModelLoading, setIsModelLoading] = useState(false);
  const [modelRunId, setModelRunId] = useState<string | null>(null);
  const [modelLog, setModelLog] = useState('');
  const [modelRuns, setModelRuns] = useState<RunRecord[]>([]);
  const [modelConfig, setModelConfig] = useState({
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
    max_rules_total: 100
  });
  const [graphType, setGraphType] = useState('go_bp');
  const [isModelOverviewCollapsed, setIsModelOverviewCollapsed] = useState(false);
  const [isModelConfigCollapsed, setIsModelConfigCollapsed] = useState(false);
  const [isConfusionMatrixCollapsed, setIsConfusionMatrixCollapsed] = useState(false);
  const [isClassificationReportCollapsed, setIsClassificationReportCollapsed] = useState(false);
  const [isTestResultsCollapsed, setIsTestResultsCollapsed] = useState(false);

  // State: Testing
  const [testMode, setTestMode] = useState<'sample' | 'upload'>('sample');
  const [testSamples, setTestSamples] = useState<TestSample[]>([]);
  const [testSampleId, setTestSampleId] = useState('');
  const [uploadFile, setUploadFile] = useState<File | null>(null);
  const [isTesting, setIsTesting] = useState(false);
  const [testResults, setTestResults] = useState<any>(null);
  const [expandedRule, setExpandedRule] = useState<number | null>(null);
  const [isMatchedRulesCollapsed, setIsMatchedRulesCollapsed] = useState(false);
  const [isPartialMatchesCollapsed, setIsPartialMatchesCollapsed] = useState(false);

  const fsMethodKey = fsMethod === 'mrmr' ? 'mrmr_miq' : 'boruta';
  const selectedDataset = datasets.find(d => d.id === datasetId);

  useEffect(() => {
    api.getDatasets()
      .then(setDatasets)
      .catch(e => setDatasetsError(e.message));
  }, []);

  // Reset downstream state whenever the dataset changes (Bước 1 re-select ->
  // ignore steps 2-4).
  useEffect(() => {
    let ignore = false;
    setSplitStats(null); setSplitParams(null); setSplitLog('');
    setFsLog(''); setExtractionStats(null); setFsRunId(null);
    setModelStats(null); setModelRunId(null); setModelLog('');
    setTestSamples([]); setTestSampleId(''); setTestResults(null); setUploadFile(null);
    if (datasetId) {
      api.getRuns(datasetId, 'feature_selection').then(r => { if (!ignore) setFsRuns(r); }).catch(() => { if (!ignore) setFsRuns([]); });
      api.getRuns(datasetId, 'model').then(r => { if (!ignore) setModelRuns(r); }).catch(() => { if (!ignore) setModelRuns([]); });
      api.getRuns(datasetId, 'split').then(r => { if (!ignore) setSplitRuns(r); }).catch(() => { if (!ignore) setSplitRuns([]); });
    } else {
      setFsRuns([]); setModelRuns([]); setSplitRuns([]);
    }
    return () => { ignore = true; };
  }, [datasetId]);

  // Lịch sử tải lên (Bước 1) — every past upload across the whole session,
  // not scoped to whichever dataset happens to be selected right now (same
  // reasoning GET /api/datasets/upload/history isn't dataset-scoped).
  useEffect(() => {
    api.getUploadHistory().then(setUploadRuns).catch(() => {});
  }, []);

  // Re-computing the split (step 2, "Thực hiện lại" with custom params) ->
  // every downstream FS/model/test result was computed against the OLD
  // split, same reasoning as the fsMethod/modelType resets below.
  useEffect(() => {
    setExtractionStats(null); setFsRunId(null); setFsLog('');
    setModelStats(null); setModelRunId(null); setModelLog('');
    setTestSamples([]); setTestSampleId(''); setTestResults(null); setUploadFile(null);
  }, [splitParams]);

  // Re-selecting the fs method (Bước 3) -> the currently-shown extraction
  // stats belong to the PREVIOUS method, so clear them too (not just steps
  // 3-4 downstream) rather than leaving a stale panel that doesn't match the
  // radio now selected; don't wait for a new "Huan luyen"/"Tai log cu" click.
  useEffect(() => {
    setExtractionStats(null); setFsRunId(null); setFsLog('');
    setModelStats(null); setModelRunId(null); setModelLog('');
    setTestSamples([]); setTestSampleId(''); setTestResults(null); setUploadFile(null);
  }, [fsMethod]);

  // Re-selecting the model type (Bước 4) -> same reasoning: the shown model
  // stats belong to the previous model type, clear them too, plus step 4.
  useEffect(() => {
    setModelStats(null); setModelRunId(null); setModelLog('');
    setTestSamples([]); setTestSampleId(''); setTestResults(null); setUploadFile(null);
  }, [modelType]);

  // Right-column panels for Bước 4/5 stay mounted once their data exists —
  // only their COLLAPSE state follows the active step (same collapseSignal
  // pattern DatasetOverview/DatasetSplit/FeatureExtractionOverview already
  // use), so switching steps never deletes another step's result, just
  // folds it down to its header.
  useEffect(() => { setIsModelOverviewCollapsed(activeStep !== 4); }, [activeStep]);
  useEffect(() => { setIsTestResultsCollapsed(activeStep !== 5); }, [activeStep]);

  // Load the real held-out test samples once a model is trained/loaded.
  useEffect(() => {
    if (!datasetId || !modelStats) return;
    let ignore = false;
    api.getTestSamplesWithSplit(datasetId, fsMethodKey, splitParams)
      .then(r => { if (!ignore) setTestSamples(r); })
      .catch(() => { if (!ignore) setTestSamples([]); });
    return () => { ignore = true; };
  }, [datasetId, modelStats, fsMethodKey, splitParams]);

  const refreshRuns = () => {
    if (!datasetId) return;
    api.getRuns(datasetId, 'feature_selection').then(setFsRuns).catch(() => {});
    api.getRuns(datasetId, 'model').then(setModelRuns).catch(() => {});
  };

  const refreshSplitRuns = (id: string) => {
    api.getRuns(id, 'split').then(setSplitRuns).catch(() => {});
  };

  const handleUploadInspect = async () => {
    if (!uploadFile1) return;
    setIsUploadInspecting(true);
    setUploadLog('[HỆ THỐNG] Đang phân tích file đã tải lên...');
    setUploadId(null); setUploadCharacteristics([]); setUploadClassChar('');
    try {
      const res = await api.uploadInspect(uploadSource, uploadTissue, uploadFile1, uploadFile2);
      setUploadId(res.upload_id);
      setUploadCharacteristics(res.characteristics || []);
      if (res.characteristics?.length) setUploadClassChar(res.characteristics[0]);
      setUploadLog(prev => prev + '\n[OK] Đã phân tích file — chọn nhãn lớp rồi nhấn "Xây dựng dataset".');
    } catch (e: any) {
      setUploadLog(prev => prev + `\n[LỖI] ${e.message}`);
    } finally {
      setIsUploadInspecting(false);
    }
  };

  const handleUploadBuild = async () => {
    if (!uploadId || !uploadClassChar) return;
    setIsUploadBuilding(true);
    setUploadLog(prev => prev + '\n[HỆ THỐNG] Đang xây dựng dataset (có thể mất vài phút)...');
    try {
      const { job_id } = await api.uploadBuild(uploadId, uploadSource, uploadClassChar, uploadTissue);
      const job = await pollJob(job_id, j => setUploadLog(j.log.join('\n')));
      if (job.status === 'error') {
        throw new Error(job.error || 'Xây dựng dataset thất bại (xem log server).');
      }
      const built = job.result as any;
      setDatasets(prev => [...prev, {
        id: built.dataset_id, name: built.name, platform: built.platform,
        n_samples: built.n_samples, n_features: built.n_features, n_classes: built.n_classes,
        class_labels: built.class_labels, description: built.description, fs_models: {},
        is_temp: true,
      }]);
      setDatasetId(built.dataset_id);
      setDatasetTab('existing');
      setUploadLog(prev => prev + `\n[THÀNH CÔNG] Dataset '${built.dataset_id}' đã sẵn sàng sử dụng.`);
      api.getUploadHistory().then(setUploadRuns).catch(() => {});
    } catch (e: any) {
      setUploadLog(prev => prev + `\n[LỖI] ${e.message}`);
      api.getUploadHistory().then(setUploadRuns).catch(() => {});
    } finally {
      setIsUploadBuilding(false);
    }
  };

  const handleSplitAction = async (action: 'retrain' | 'load') => {
    if (!datasetId) return;
    setIsSplitLoading(true);
    setSplitLog('');
    setExtractionStats(null); setFsRunId(null); setFsLog('');
    setModelStats(null); setModelRunId(null); setModelLog('');
    setTestSamples([]); setTestSampleId(''); setTestResults(null); setUploadFile(null);
    try {
      if (action === 'load') {
        setSplitLog('[HỆ THỐNG] Đang tải số liệu chia dữ liệu (mặc định)...');
        let stats: any;
        try {
          stats = await api.getOverview(datasetId);
        } catch {
          stats = await api.splitPreview(datasetId, {});
        }
        setSplitStats(stats);
        setSplitParams(null);
        setSplitInputs({ min_samples_per_class: stats.min_samples_per_class, test_size: stats.test_size });
        setSplitLog(prev => prev + '\n[CACHE] Đã nạp thành công.');
      } else {
        setSplitLog('[HỆ THỐNG] Đang tính lại chia dữ liệu (train/test)...');
        const stats = await api.splitPreview(datasetId, splitInputs);
        setSplitStats(stats);
        setSplitParams({ ...splitInputs });
        setSplitLog(prev => prev + '\n[OK] Đã tính lại thành công.');
      }
      refreshSplitRuns(datasetId);
    } catch (e: any) {
      setSplitLog(prev => prev + `\n[LỖI] ${e.message}`);
    } finally {
      setIsSplitLoading(false);
    }
  };

  // Loads a past split run directly from its stored summary — no recompute,
  // no new history entry — same "load" semantics loadPastFsRun/loadPastModelRun
  // already use (those GET a cached file; this reads the full summary the
  // /split/preview call already saved into runs_index.json).
  const loadPastSplitRun = (run: RunRecord) => {
    if (!datasetId) return;
    setSplitLog(`[HỆ THỐNG] Đang tải lại kết quả chạy trước (${run.run_id})...`);
    setExtractionStats(null); setFsRunId(null); setFsLog('');
    setModelStats(null); setModelRunId(null); setModelLog('');
    setTestSamples([]); setTestSampleId(''); setTestResults(null); setUploadFile(null);
    const params = {
      min_samples_per_class: run.summary.min_samples_per_class as number,
      test_size: run.summary.test_size as number,
    };
    setSplitStats(run.summary);
    setSplitParams(params);
    setSplitInputs(params);
    setSplitLog(prev => prev + '\n[OK] Đã nạp lại kết quả chạy trước.');
  };

  const handleFsAction = async (action: 'retrain' | 'load') => {
    if (!datasetId) return;
    setIsFsLoading(true);
    setFsLog('');
    setExtractionStats(null);
    setModelStats(null); setModelRunId(null); setModelLog('');
    setTestSamples([]); setTestSampleId(''); setTestResults(null); setUploadFile(null);
    try {
      if (action === 'load') {
        setFsLog(`[HỆ THỐNG] Đang tải kết quả log trước đó cho thuật toán ${fsMethodKey.toUpperCase()}...`);
        const stats = await api.getFeatureSelection(datasetId, fsMethodKey);
        setExtractionStats(stats);
        setFsRunId(null);
        setFsLog(prev => prev + '\n[CACHE] Đã nạp thành công.');
      } else {
        const params = fsMethod === 'mrmr' ? mrmrConfig : borutaConfig;
        setFsLog(`[HỆ THỐNG] Bắt đầu chạy thuật toán ${fsMethodKey.toUpperCase()} (chạy thật, có thể mất vài phút)...`);
        const { job_id } = await api.trainFeatureSelection(datasetId, fsMethodKey, params, splitParams);
        const job = await pollJob(job_id, j => setFsLog(j.log.join('\n')));
        if (job.status === 'error') {
          throw new Error(job.error || 'Trich xuat dac trung that bai (xem log server).');
        }
        const stats = await api.getFeatureSelection(datasetId, fsMethodKey, job_id);
        setExtractionStats(stats);
        setFsRunId(job_id);
        refreshRuns();
      }
    } catch (e: any) {
      setFsLog(prev => prev + `\n[LỖI] ${e.message}`);
    } finally {
      setIsFsLoading(false);
    }
  };

  const loadPastFsRun = async (run: RunRecord) => {
    if (!datasetId) return;
    setIsFsLoading(true);
    setFsLog(`[HỆ THỐNG] Đang tải lại kết quả chạy trước (${run.run_id})...`);
    // This run's features may not match whatever model/test-results are
    // currently shown (same reset as handleFsAction) — otherwise the model
    // panel keeps displaying stale results as if they belonged to this run.
    setModelStats(null); setModelRunId(null); setModelLog('');
    setTestSamples([]); setTestSampleId(''); setTestResults(null); setUploadFile(null);
    try {
      const stats = await api.getFeatureSelection(datasetId, run.fs_method, run.run_id);
      setExtractionStats(stats);
      setFsRunId(run.run_id);
      setFsLog(prev => prev + '\n[OK] Đã nạp lại kết quả chạy trước.');
    } catch (e: any) {
      setFsLog(prev => prev + `\n[LỖI] ${e.message}`);
    } finally {
      setIsFsLoading(false);
    }
  };

  const handleModelAction = async (action: 'retrain' | 'load') => {
    if (!datasetId) return;
    setIsModelLoading(true);
    setModelStats(null);
    setModelLog(action === 'retrain' ? '[HỆ THỐNG] Bắt đầu huấn luyện mô hình (chạy thật)...' : '');
    setTestSamples([]); setTestSampleId(''); setTestResults(null); setUploadFile(null);
    try {
      let stats: any;
      let runId: string | null = null;
      if (action === 'load') {
        stats = await api.getModelStats(datasetId, fsMethodKey, modelType);
      } else {
        const { job_id } = await api.trainModel(datasetId, fsMethodKey, modelType, modelConfig, fsRunId, splitParams);
        const job = await pollJob(job_id, j => setModelLog(j.log.join('\n')));
        if (job.status === 'error') {
          throw new Error(job.error || 'Huan luyen mo hinh that bai (xem log server).');
        }
        stats = await api.getModelStats(datasetId, fsMethodKey, modelType, job_id);
        runId = job_id;
        refreshRuns();
      }
      setModelRunId(runId);
      setModelStats({
        acc: Math.round((stats.best_run_test_metrics?.accuracy ?? 0) * 1000) / 10,
        f1: Math.round((stats.best_run_test_metrics?.f1_macro ?? 0) * 1000) / 10,
        rules: stats.n_rules ?? 0,
        cm: stats.confusion_matrix ?? [],
        labels: stats.class_labels ?? [],
        hyperparams: stats.hyperparams ?? null,
        rulesSummary: stats.rules_summary ?? null,
      });
    } catch (e: any) {
      setModelLog(prev => prev + `\n[LỖI] ${e.message}`);
    } finally {
      setIsModelLoading(false);
    }
  };

  const loadPastModelRun = async (run: RunRecord) => {
    if (!datasetId || !run.model) return;
    setIsModelLoading(true);
    setModelLog(`[HỆ THỐNG] Đang tải lại kết quả chạy trước (${run.run_id})...`);
    // Same reasoning as handleModelAction: a prediction/test result from the
    // PREVIOUS model run must not keep showing next to this newly loaded one.
    setTestSamples([]); setTestSampleId(''); setTestResults(null); setUploadFile(null);
    try {
      const stats = await api.getModelStats(datasetId, run.fs_method, run.model, run.run_id);
      setModelRunId(run.run_id);
      setModelStats({
        acc: Math.round((stats.best_run_test_metrics?.accuracy ?? 0) * 1000) / 10,
        f1: Math.round((stats.best_run_test_metrics?.f1_macro ?? 0) * 1000) / 10,
        rules: stats.n_rules ?? 0,
        cm: stats.confusion_matrix ?? [],
        labels: stats.class_labels ?? [],
        hyperparams: stats.hyperparams ?? null,
        rulesSummary: stats.rules_summary ?? null,
      });
      setModelLog(prev => prev + '\n[OK] Đã nạp lại kết quả chạy trước.');
    } catch (e: any) {
      setModelLog(prev => prev + `\n[LỖI] ${e.message}`);
    } finally {
      setIsModelLoading(false);
    }
  };

  const applyPredictionResult = (result: any) => {
    setTestResults({
      matchedCount: result.matched_count,
      rules: result.rules.map((r: any) => ({
        id: r.rule_id,
        text: r.text,
        matched: r.matched,
        desc: r.explanation || '',
        sampleValues: r.sample_values || {},
        class: r.consequent_label,
      })),
      classification: result.classification,
      classDescription: result.class_description,
      classDisplayName: result.class_display_name || null,
      classDisplayNames: result.class_display_names || {},
      rulePrediction: result.rule_prediction || null,
      rulePredictionDescription: result.rule_prediction_description || null,
      rulePredictionDisplayName: result.rule_prediction_display_name || null,
      classVotes: result.class_votes || {},
      classVotesOver50: result.class_votes_over50 || {},
      partialMatches: (result.partial_matches || []).map((p: any) => ({
        ruleId: p.rule_id,
        text: p.text,
        class: p.consequent_label,
        satisfied: p.satisfied,
        total: p.total,
        ratio: p.ratio,
        conditions: p.conditions || [],
      })),
      nPartialMatchesTotal: result.n_partial_matches_total ?? (result.partial_matches || []).length,
      nRulesTotal: result.n_rules_total ?? null,
      trueLabel: result.true_label,
      biomedicalSummary: result.biomedical_summary,
      biomedicalRationale: result.biomedical_rationale,
      biomedicalModelVsRule: result.biomedical_model_vs_rule,
      biomedicalDisclaimer: result.biomedical_disclaimer,
      llmUsed: result.llm_used,
    });
  };

  const handleTestSample = async () => {
    if (!datasetId) return;
    if (testMode === 'sample' && !testSampleId) return;
    if (testMode === 'upload' && !uploadFile) return;
    setIsTesting(true);
    setTestResults(null);
    try {
      const result = testMode === 'sample'
        ? await api.predict(datasetId, fsMethodKey, modelType, testSampleId, modelRunId, splitParams)
        : await api.predictUpload(datasetId, fsMethodKey, modelType, uploadFile as File, modelRunId);
      applyPredictionResult(result);
    } catch (e: any) {
      setTestResults({ matchedCount: 0, rules: [], classification: 'Loi', explanation: e.message });
    } finally {
      setIsTesting(false);
    }
  };

  const report = modelStats ? classificationReport(modelStats.cm) : [];

  // UX guard: while a later step is running, freeze every earlier step's
  // controls so the user can't change a selection out from under an in-flight
  // job (which would also fire the reset-cascade effects mid-run).
  const datasetLocked = isSplitLoading || isFsLoading || isModelLoading || isTesting;
  const splitSectionLocked = isFsLoading || isModelLoading || isTesting;
  const fsSectionLocked = isSplitLoading || isFsLoading || isModelLoading || isTesting;
  const modelSectionLocked = isModelLoading || isTesting;
  const testSectionLocked = isTesting;
  // A live "Huan luyen" feature-selection run (fsRunId set) produces a
  // feature set that only exists under outputs_live/<fsRunId> — the cached
  // "Tai mo hinh cu" model was trained on the cached (outputs_holdout) fs
  // run's features, so it doesn't line up. Force retraining in that case.
  const modelLoadDisabled = modelSectionLocked || !extractionStats || !!fsRunId;

  // Accordion gating — a step can only be opened once its prerequisite has
  // produced something, same prerequisites the grayscale/lock treatment
  // already uses per section below.
  const stepReady: Record<number, boolean> = {
    1: true,
    2: !!datasetId,
    3: !!splitStats,
    4: !!extractionStats,
    5: !!modelStats,
  };
  const openStep = (n: number) => { if (stepReady[n]) setActiveStep(n); };

  return (
    <div className="min-h-screen bg-slate-50 text-slate-800 font-sans selection:bg-teal-200 pb-12">
      {/* Header */}
      <header className="bg-white border-b border-emerald-100/80 sticky top-0 z-50 backdrop-blur-sm bg-white/90">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-emerald-50 border border-emerald-200 rounded-xl flex items-center justify-center text-emerald-600 shadow-sm">
              <Dna size={22} />
            </div>
            <div>
              <h1 className="text-xl font-bold text-slate-800 tracking-tight">GE-RuGO Workspace</h1>
              <p className="text-[11px] font-semibold tracking-wider uppercase text-emerald-600">Phân lớp dữ liệu Biểu hiện Gene có giải thích</p>
            </div>
          </div>
          <div className="hidden md:flex items-center gap-4 text-sm text-slate-500 font-medium">
            <span className={cn("flex items-center gap-1.5 transition-colors", datasetId ? "text-emerald-600" : "")}><Database size={16}/> Dữ liệu</span>
            <span className="text-slate-300">/</span>
            <span className={cn("flex items-center gap-1.5 transition-colors", splitStats ? "text-emerald-600" : "")}><GitMerge size={16}/> Chia dữ liệu</span>
            <span className="text-slate-300">/</span>
            <span className={cn("flex items-center gap-1.5 transition-colors", fsLog ? "text-emerald-600" : "")}><Settings2 size={16}/> Trích xuất</span>
            <span className="text-slate-300">/</span>
            <span className={cn("flex items-center gap-1.5 transition-colors", modelStats ? "text-emerald-600" : "")}><GitMerge size={16}/> Mô hình</span>
            <span className="text-slate-300">/</span>
            <span className={cn("flex items-center gap-1.5 transition-colors", testResults ? "text-emerald-600" : "")}><Activity size={16}/> Thực nghiệm</span>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-8 grid grid-cols-1 lg:grid-cols-12 gap-10">

        {/* Cột Trái: Cấu hình */}
        <div className="lg:col-span-4 space-y-8 relative">

          <div className="absolute left-[23px] top-12 bottom-12 w-0.5 bg-slate-100 -z-10 hidden lg:block"></div>

          {/* Bước 1: Dataset */}
          <section className="relative">
            <div className="flex items-start gap-4 mb-4">
              <button
                onClick={() => openStep(1)}
                className="w-12 h-12 shrink-0 bg-white border-2 border-teal-500 rounded-full flex items-center justify-center text-teal-600 shadow-sm z-10 hover:bg-teal-50 transition-colors"
              >
                <span className="font-bold text-lg">1</span>
              </button>
              <div className="pt-2 w-full">
                <div
                  className="flex items-center justify-between cursor-pointer group mb-4"
                  onClick={() => openStep(1)}
                >
                  <h2 className="text-lg font-semibold text-slate-800 group-hover:text-teal-700 transition-colors">Chọn Dữ Liệu</h2>
                  <span className="text-slate-400 group-hover:text-teal-600 transition-colors">
                    {activeStep === 1 ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
                  </span>
                </div>
                <div className={cn("bg-white p-5 rounded-2xl shadow-sm border border-slate-200", activeStep !== 1 && "hidden")}>
                  {datasetsError && (
                    <p className="text-sm text-rose-600 mb-3">Không tải được danh sách dataset: {datasetsError}</p>
                  )}

                  <div className="flex gap-2 p-1 bg-slate-100 rounded-lg text-sm mb-4">
                    <button
                      onClick={() => setDatasetTab('existing')}
                      disabled={datasetLocked}
                      className={cn("flex-1 py-1.5 rounded-md flex items-center justify-center gap-1.5 transition-colors", datasetTab === 'existing' ? "bg-white shadow-sm text-teal-700 font-medium" : "text-slate-500")}
                    >
                      <ListChecks size={14} /> Chọn dataset có sẵn
                    </button>
                    <button
                      onClick={() => setDatasetTab('upload')}
                      disabled={datasetLocked}
                      className={cn("flex-1 py-1.5 rounded-md flex items-center justify-center gap-1.5 transition-colors", datasetTab === 'upload' ? "bg-white shadow-sm text-teal-700 font-medium" : "text-slate-500")}
                    >
                      <UploadCloud size={14} /> Tải lên dataset mới
                    </button>
                  </div>

                  {datasetTab === 'existing' ? (
                    <>
                      <select
                        value={datasetId}
                        onChange={(e) => setDatasetId(e.target.value)}
                        disabled={datasetLocked}
                        className="w-full bg-slate-50 border border-slate-200 text-slate-700 rounded-lg px-4 py-2.5 focus:outline-none focus:ring-2 focus:ring-teal-500/30 focus:border-teal-500 transition-colors shadow-sm disabled:opacity-60 disabled:cursor-not-allowed"
                      >
                        <option value="">-- Chọn Dữ liệu --</option>
                        {datasets.map(d => (
                          <option key={d.id} value={d.id}>{d.name || d.id}</option>
                        ))}
                      </select>

                      {selectedDataset && (
                        <div className="mt-4 p-4 bg-teal-50/50 rounded-xl border border-teal-100/50 space-y-2 text-sm">
                          <div className="flex justify-between items-center">
                            <span className="text-slate-500">Mã dataset</span>
                            <span className="font-mono text-xs text-teal-700 bg-teal-100/60 px-2 py-0.5 rounded" title={selectedDataset.id}>{selectedDataset.id}</span>
                          </div>
                          <div className="flex justify-between"><span className="text-slate-500">Nền tảng vi mảng</span> <span className="font-medium text-slate-700">{selectedDataset.platform}</span></div>
                          <div className="flex justify-between"><span className="text-slate-500">Mẫu bệnh phẩm</span> <span className="font-medium text-slate-700">{selectedDataset.n_samples}</span></div>
                          <div className="flex justify-between"><span className="text-slate-500">Đặc trưng</span> <span className="font-medium text-slate-700">{selectedDataset.n_features?.toLocaleString()}</span></div>
                          <div className="flex justify-between"><span className="text-slate-500">Số lớp</span> <span className="font-medium text-teal-700">{selectedDataset.n_classes}</span></div>
                        </div>
                      )}
                    </>
                  ) : (
                    <fieldset disabled={datasetLocked} className="border-0 p-0 m-0 min-w-0 disabled:opacity-60 space-y-3">
                      <div className="flex gap-4">
                        <label className="flex items-center gap-2 cursor-pointer group">
                          <input type="radio" name="uploadSource" checked={uploadSource === 'geo'} onChange={() => { setUploadSource('geo'); setUploadId(null); setUploadCharacteristics([]); }} className="w-4 h-4 shrink-0 accent-teal-600" />
                          <span className="font-medium text-slate-700 group-hover:text-teal-700 transition-colors">Từ NCBI GEO</span>
                        </label>
                        <label className="flex items-center gap-2 cursor-pointer group">
                          <input type="radio" name="uploadSource" checked={uploadSource === 'cumida'} onChange={() => { setUploadSource('cumida'); setUploadId(null); setUploadCharacteristics([]); }} className="w-4 h-4 shrink-0 accent-teal-600" />
                          <span className="font-medium text-slate-700 group-hover:text-teal-700 transition-colors">Từ CuMiDa</span>
                        </label>
                      </div>

                      <div>
                        <label className="text-xs text-slate-500 font-medium block mb-1">Tên thư mục lưu trữ</label>
                        <input type="text" value={uploadTissue} onChange={e => setUploadTissue(e.target.value)} className="w-full bg-slate-50 border border-slate-200 text-slate-700 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-teal-500/30 focus:border-teal-500" />
                      </div>

                      <div>
                        <label className="text-xs text-slate-500 font-medium block mb-1">
                          {uploadSource === 'geo' ? 'File series matrix (.txt.gz)' : 'File probe CuMiDa (.csv)'}
                        </label>
                        <input
                          type="file"
                          accept={uploadSource === 'geo' ? '.gz,.txt' : '.csv'}
                          onChange={e => setUploadFile1(e.target.files?.[0] || null)}
                          className="w-full text-sm text-slate-600"
                        />
                      </div>
                      <div>
                        <label className="text-xs text-slate-500 font-medium block mb-1">
                          File annotation {uploadSource === 'geo' ? 'GPL (tùy chọn)' : 'CuMiDa (tùy chọn)'}
                        </label>
                        <input
                          type="file"
                          onChange={e => setUploadFile2(e.target.files?.[0] || null)}
                          className="w-full text-sm text-slate-600"
                        />
                      </div>

                      <button
                        onClick={handleUploadInspect}
                        disabled={!uploadFile1 || isUploadInspecting || isUploadBuilding}
                        className="w-full bg-white hover:bg-slate-50 text-slate-700 font-medium py-2 px-3 rounded-lg flex items-center justify-center gap-2 transition-colors border border-slate-200 disabled:opacity-70 shadow-sm text-sm"
                      >
                        {isUploadInspecting ? <Loader2 size={16} className="animate-spin" /> : <FileText size={16} className="text-slate-400" />}
                        Phân tích file
                      </button>

                      {uploadCharacteristics.length > 0 && (
                        <div className="animate-in fade-in slide-in-from-top-2">
                          <label className="text-xs text-slate-500 font-medium block mb-1">Cột đặc trưng dùng làm nhãn lớp</label>
                          <select
                            value={uploadClassChar}
                            onChange={e => setUploadClassChar(e.target.value)}
                            className="w-full bg-slate-50 border border-slate-200 text-slate-700 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-teal-500/30 focus:border-teal-500 mb-3"
                          >
                            {uploadCharacteristics.map(c => <option key={c} value={c}>{c}</option>)}
                          </select>
                          <button
                            onClick={handleUploadBuild}
                            disabled={!uploadClassChar || isUploadBuilding}
                            className="w-full bg-teal-600 hover:bg-teal-700 text-white font-medium py-2 px-3 rounded-lg flex items-center justify-center gap-2 transition-colors disabled:opacity-70 shadow-sm text-sm"
                          >
                            {isUploadBuilding ? <Loader2 size={16} className="animate-spin" /> : <PlayCircle size={16} />}
                            Xây dựng dataset
                          </button>
                        </div>
                      )}

                      {uploadLog && (
                        <div className="bg-slate-900 rounded-xl p-4 text-xs font-mono text-teal-400 whitespace-pre-wrap leading-relaxed shadow-inner border border-slate-800 max-h-64 overflow-y-auto">
                          {uploadLog}
                        </div>
                      )}

                      <RunHistoryList
                        title="Lịch sử tải lên"
                        runs={uploadRuns}
                        disabled={isUploadBuilding}
                        isAvailable={run => !!run.dataset && datasets.some(d => d.id === run.dataset)}
                        isSelected={run => datasetId === run.dataset}
                        unavailableTitle="Dataset này không còn tồn tại (mất khi khởi động lại server)"
                        onSelect={run => {
                          if (!run.dataset) return;
                          setDatasetId(run.dataset);
                          setDatasetTab('existing');
                        }}
                        renderLabel={run => `${String(run.summary?.tissue ?? '')} · ${run.run_id}`}
                        renderStatus={run => `${run.summary?.n_samples ?? '?'} mẫu`}
                      />
                    </fieldset>
                  )}
                </div>
              </div>
            </div>
          </section>

          {/* Bước 2: Xử lý & Chia Dữ liệu */}
          <section className="relative">
            <div className={cn("flex items-start gap-4 mb-4 transition-opacity duration-300", !datasetId && "opacity-50 grayscale")}>
              <button
                onClick={() => openStep(2)}
                disabled={!datasetId}
                className={cn("w-12 h-12 shrink-0 bg-white border-2 rounded-full flex items-center justify-center shadow-sm z-10 transition-colors", datasetId ? "border-teal-500 text-teal-600 hover:bg-teal-50" : "border-slate-200 text-slate-400 cursor-not-allowed")}
              >
                <span className="font-bold text-lg">2</span>
              </button>
              <div className="pt-2 w-full">
                <div
                  className="flex items-center justify-between cursor-pointer group mb-4"
                  onClick={() => openStep(2)}
                >
                  <h2 className="text-lg font-semibold text-slate-800 group-hover:text-teal-700 transition-colors">Xử lý &amp; Chia Dữ liệu</h2>
                  <span className="text-slate-400 group-hover:text-teal-600 transition-colors">
                    {activeStep === 2 ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
                  </span>
                </div>
                <div className={cn("bg-white p-5 rounded-2xl shadow-sm border border-slate-200 relative overflow-hidden", activeStep !== 2 && "hidden")}>
                  {!datasetId && <div className="absolute inset-0 z-20 bg-slate-50/50"></div>}

                  <fieldset disabled={splitSectionLocked} className="border-0 p-0 m-0 min-w-0 disabled:opacity-60">
                    <div className="bg-slate-50 p-4 rounded-xl border border-slate-100 mb-5 text-sm">
                      <div className="flex justify-between items-center mb-3">
                        <span className="font-semibold text-slate-700">Cấu hình chia dữ liệu</span>
                        <span className="text-[10px] bg-teal-100 text-teal-700 px-2 py-0.5 rounded font-bold uppercase tracking-wider">Tùy chỉnh</span>
                      </div>
                      <div className="grid grid-cols-1 gap-y-3">
                        <div className="flex justify-between items-center">
                          <span className="text-slate-500">min_samples_per_class</span>
                          <input type="number" min={2} value={splitInputs.min_samples_per_class} onChange={e => setSplitInputs({ ...splitInputs, min_samples_per_class: Number(e.target.value) })} className="w-24 px-2 py-1 text-right font-mono text-teal-700 font-semibold bg-white border border-slate-200 rounded text-sm focus:outline-none focus:ring-1 focus:ring-teal-500" />
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-slate-500">test_size</span>
                          <input type="number" step="0.01" min={0.05} max={0.5} value={splitInputs.test_size} onChange={e => setSplitInputs({ ...splitInputs, test_size: Number(e.target.value) })} className="w-24 px-2 py-1 text-right font-mono text-teal-700 font-semibold bg-white border border-slate-200 rounded text-sm focus:outline-none focus:ring-1 focus:ring-teal-500" />
                        </div>
                      </div>
                    </div>

                    <div className="flex gap-3">
                      <button
                        onClick={() => handleSplitAction('retrain')}
                        disabled={isSplitLoading || !datasetId}
                        className="flex-1 bg-teal-600 hover:bg-teal-700 text-white font-medium py-2 px-3 rounded-lg flex items-center justify-center gap-2 transition-colors disabled:opacity-70 shadow-sm text-sm"
                      >
                        {isSplitLoading ? <Loader2 size={16} className="animate-spin" /> : <PlayCircle size={16} />}
                        Thực hiện lại
                      </button>
                      <button
                        onClick={() => handleSplitAction('load')}
                        disabled={isSplitLoading || !datasetId || selectedDataset?.is_temp}
                        title={selectedDataset?.is_temp ? 'Dataset tải lên chưa từng qua xử lý offline nên không có số liệu chia dữ liệu cache — dùng "Thực hiện lại".' : undefined}
                        className="flex-1 bg-white hover:bg-slate-50 text-slate-700 font-medium py-2 px-3 rounded-lg flex items-center justify-center gap-2 transition-colors border border-slate-200 disabled:opacity-70 shadow-sm text-sm"
                      >
                        <FileText size={16} className="text-slate-400" />
                        Tải dữ liệu có sẵn
                      </button>
                    </div>
                    {selectedDataset?.is_temp && (
                      <p className="text-xs text-amber-600 mt-2">
                        Dataset tải lên chưa có số liệu chia dữ liệu cache — chỉ dùng được "Thực hiện lại".
                      </p>
                    )}

                    {splitLog && (
                      <div className="mt-4 bg-slate-900 rounded-xl p-4 text-xs font-mono text-teal-400 whitespace-pre-wrap leading-relaxed shadow-inner border border-slate-800 max-h-64 overflow-y-auto">
                        {splitLog}
                      </div>
                    )}

                    <RunHistoryList
                      title="Lịch sử xử lý và phân chia dữ liệu"
                      runs={splitRuns}
                      disabled={isSplitLoading}
                      onSelect={loadPastSplitRun}
                      renderLabel={run => `min=${String(run.summary?.min_samples_per_class)} test=${String(run.summary?.test_size)} · ${run.run_id}`}
                      renderStatus={run => `${run.summary?.n_train ?? '?'}/${run.summary?.n_test ?? '?'}`}
                    />
                  </fieldset>
                </div>
              </div>
            </div>
          </section>

          {/* Bước 3: Feature Selection */}
          <section className="relative">
            <div className={cn("flex items-start gap-4 mb-4 transition-opacity duration-300", !splitStats && "opacity-50 grayscale")}>
              <button
                onClick={() => openStep(3)}
                disabled={!splitStats}
                className={cn("w-12 h-12 shrink-0 bg-white border-2 rounded-full flex items-center justify-center shadow-sm z-10 transition-colors", splitStats ? "border-teal-500 text-teal-600 hover:bg-teal-50" : "border-slate-200 text-slate-400 cursor-not-allowed")}
              >
                <span className="font-bold text-lg">3</span>
              </button>
              <div className="pt-2 w-full">
                <div
                  className="flex items-center justify-between cursor-pointer group mb-4"
                  onClick={() => openStep(3)}
                >
                  <h2 className="text-lg font-semibold text-slate-800 group-hover:text-teal-700 transition-colors">Trích xuất đặc trưng</h2>
                  <span className="text-slate-400 group-hover:text-teal-600 transition-colors">
                    {activeStep === 3 ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
                  </span>
                </div>
                <div className={cn("bg-white p-5 rounded-2xl shadow-sm border border-slate-200 relative overflow-hidden", activeStep !== 3 && "hidden")}>
                  {!splitStats && <div className="absolute inset-0 z-20 bg-slate-50/50"></div>}

                  <fieldset disabled={fsSectionLocked} className="border-0 p-0 m-0 min-w-0 disabled:opacity-60">
                  <div className="flex gap-4 mb-5">
                    <label className="flex items-center gap-2 cursor-pointer group">
                      <input type="radio" name="fsMethod" checked={fsMethod === 'boruta'} onChange={() => setFsMethod('boruta')} className="w-4 h-4 shrink-0 accent-teal-600 focus:ring-teal-500" />
                      <span className="font-medium text-slate-700 group-hover:text-teal-700 transition-colors">Boruta</span>
                    </label>
                    <label className="flex items-center gap-2 cursor-pointer group">
                      <input type="radio" name="fsMethod" checked={fsMethod === 'mrmr'} onChange={() => setFsMethod('mrmr')} className="w-4 h-4 shrink-0 accent-teal-600 focus:ring-teal-500" />
                      <span className="font-medium text-slate-700 group-hover:text-teal-700 transition-colors">mRMR</span>
                    </label>
                  </div>

                  {selectedDataset && !selectedDataset.fs_models[fsMethodKey] && (
                    <p className="text-xs text-amber-600 mb-3">Dataset này chưa có kết quả cache cho {fsMethodKey} — chỉ có thể dùng "Huấn luyện".</p>
                  )}

                  {fsMethod === 'boruta' && (
                    <div className="bg-slate-50 p-4 rounded-xl border border-slate-100 mb-5 text-sm animate-in fade-in slide-in-from-top-2">
                      <div className="flex justify-between items-center mb-3">
                        <span className="font-semibold text-slate-700">Cấu hình Boruta</span>
                        <span className="text-[10px] bg-teal-100 text-teal-700 px-2 py-0.5 rounded font-bold uppercase tracking-wider">Tùy chỉnh</span>
                      </div>
                      <div className="grid grid-cols-1 gap-y-3">
                         <div className="flex justify-between items-center">
                           <span className="text-slate-500">n_estimators</span>
                           <input type="text" value={borutaConfig.n_estimators} onChange={e => setBorutaConfig({...borutaConfig, n_estimators: e.target.value})} className="w-24 px-2 py-1 text-right font-mono text-teal-700 font-semibold bg-white border border-slate-200 rounded text-sm focus:outline-none focus:ring-1 focus:ring-teal-500" />
                         </div>
                         <div className="flex justify-between items-center">
                           <span className="text-slate-500">rf_n_estimators</span>
                           <input type="number" value={borutaConfig.rf_n_estimators} onChange={e => setBorutaConfig({...borutaConfig, rf_n_estimators: Number(e.target.value)})} className="w-24 px-2 py-1 text-right font-mono text-teal-700 font-semibold bg-white border border-slate-200 rounded text-sm focus:outline-none focus:ring-1 focus:ring-teal-500" />
                         </div>
                         <div className="flex justify-between items-center">
                           <span className="text-slate-500">max_depth</span>
                           <input type="text" placeholder="null" value={borutaConfig.max_depth} onChange={e => setBorutaConfig({...borutaConfig, max_depth: e.target.value})} className="w-24 px-2 py-1 text-right font-mono text-teal-700 font-semibold bg-white border border-slate-200 rounded text-sm focus:outline-none focus:ring-1 focus:ring-teal-500" />
                         </div>
                         <div className="flex justify-between items-center">
                           <span className="text-slate-500">max_iter</span>
                           <input type="number" value={borutaConfig.max_iter} onChange={e => setBorutaConfig({...borutaConfig, max_iter: Number(e.target.value)})} className="w-24 px-2 py-1 text-right font-mono text-teal-700 font-semibold bg-white border border-slate-200 rounded text-sm focus:outline-none focus:ring-1 focus:ring-teal-500" />
                         </div>
                         <div className="flex justify-between items-center">
                           <span className="text-slate-500">perc</span>
                           <input type="number" value={borutaConfig.perc} onChange={e => setBorutaConfig({...borutaConfig, perc: Number(e.target.value)})} className="w-24 px-2 py-1 text-right font-mono text-teal-700 font-semibold bg-white border border-slate-200 rounded text-sm focus:outline-none focus:ring-1 focus:ring-teal-500" />
                         </div>
                         <div className="flex justify-between items-center">
                           <span className="text-slate-500">alpha</span>
                           <input type="number" step="0.01" value={borutaConfig.alpha} onChange={e => setBorutaConfig({...borutaConfig, alpha: Number(e.target.value)})} className="w-24 px-2 py-1 text-right font-mono text-teal-700 font-semibold bg-white border border-slate-200 rounded text-sm focus:outline-none focus:ring-1 focus:ring-teal-500" />
                         </div>
                         <div className="flex justify-between items-center">
                           <span className="text-slate-500">class_weight</span>
                           <select value={borutaConfig.class_weight} onChange={e => setBorutaConfig({...borutaConfig, class_weight: e.target.value})} className="w-24 px-2 py-1 text-right font-mono text-teal-700 font-semibold bg-white border border-slate-200 rounded text-sm focus:outline-none focus:ring-1 focus:ring-teal-500">
                             <option value="balanced">balanced</option>
                             <option value="balanced_subsample">balanced_subsample</option>
                             <option value="none">none</option>
                           </select>
                         </div>
                         <div className="flex justify-between items-center">
                           <span className="text-slate-500">random_state</span>
                           <input type="number" value={borutaConfig.random_state} onChange={e => setBorutaConfig({...borutaConfig, random_state: Number(e.target.value)})} className="w-24 px-2 py-1 text-right font-mono text-teal-700 font-semibold bg-white border border-slate-200 rounded text-sm focus:outline-none focus:ring-1 focus:ring-teal-500" />
                         </div>
                         <div className="flex justify-between items-center">
                           <span className="text-slate-500">Chế độ chọn đặc trưng</span>
                           <select
                             value={borutaConfig.selection_mode}
                             onChange={e => setBorutaConfig({...borutaConfig, selection_mode: e.target.value as typeof borutaConfig.selection_mode})}
                             className="w-40 px-2 py-1 text-right font-mono text-teal-700 font-semibold bg-white border border-slate-200 rounded text-sm focus:outline-none focus:ring-1 focus:ring-teal-500"
                           >
                             <option value="confirmed">confirmed</option>
                             <option value="confirmed_tentative">confirmed_tentative</option>
                             <option value="top_k">top_k</option>
                           </select>
                         </div>
                         {borutaConfig.selection_mode === 'top_k' && (
                           <div className="flex justify-between items-center">
                             <span className="text-slate-500">k</span>
                             <input
                               type="number"
                               min={1}
                               placeholder="số đặc trưng"
                               value={borutaConfig.k}
                               onChange={e => setBorutaConfig({...borutaConfig, k: e.target.value === '' ? '' : Number(e.target.value)})}
                               className="w-24 px-2 py-1 text-right font-mono text-teal-700 font-semibold bg-white border border-slate-200 rounded text-sm focus:outline-none focus:ring-1 focus:ring-teal-500"
                             />
                           </div>
                         )}
                      </div>
                    </div>
                  )}

                  {fsMethod === 'mrmr' && (
                    <div className="bg-slate-50 p-4 rounded-xl border border-slate-100 mb-5 text-sm animate-in fade-in slide-in-from-top-2">
                      <div className="flex justify-between items-center mb-3">
                        <span className="font-semibold text-slate-700">Cấu hình mRMR</span>
                        <span className="text-[10px] bg-teal-100 text-teal-700 px-2 py-0.5 rounded font-bold uppercase tracking-wider">Tùy chỉnh</span>
                      </div>
                      <div className="grid grid-cols-1 gap-y-3">
                         <div className="flex justify-between items-center">
                           <span className="text-slate-500">criterion</span>
                           <input type="text" value={mrmrConfig.criterion} readOnly className="w-24 px-2 py-1 text-right font-mono text-slate-500 font-semibold bg-slate-100 border border-slate-200 rounded text-sm focus:outline-none cursor-not-allowed" />
                         </div>
                         <div className="flex justify-between items-center">
                           <span className="text-slate-500">K (features)</span>
                           <input type="number" value={mrmrConfig.K} onChange={e => setMrmrConfig({...mrmrConfig, K: Number(e.target.value)})} className="w-24 px-2 py-1 text-right font-mono text-teal-700 font-semibold bg-white border border-slate-200 rounded text-sm focus:outline-none focus:ring-1 focus:ring-teal-500" />
                         </div>
                         <div className="flex justify-between items-center">
                           <span className="text-slate-500">n_bins</span>
                           <input type="number" value={mrmrConfig.n_bins} onChange={e => setMrmrConfig({...mrmrConfig, n_bins: Number(e.target.value)})} className="w-24 px-2 py-1 text-right font-mono text-teal-700 font-semibold bg-white border border-slate-200 rounded text-sm focus:outline-none focus:ring-1 focus:ring-teal-500" />
                         </div>
                         <div className="flex justify-between items-center">
                           <span className="text-slate-500">random_state</span>
                           <input type="number" value={mrmrConfig.random_state} onChange={e => setMrmrConfig({...mrmrConfig, random_state: Number(e.target.value)})} className="w-24 px-2 py-1 text-right font-mono text-teal-700 font-semibold bg-white border border-slate-200 rounded text-sm focus:outline-none focus:ring-1 focus:ring-teal-500" />
                         </div>
                      </div>
                    </div>
                  )}

                  <div className="flex gap-3">
                    <button
                      onClick={() => handleFsAction('retrain')}
                      disabled={isFsLoading || !splitStats}
                      className="flex-1 bg-teal-600 hover:bg-teal-700 text-white font-medium py-2 px-3 rounded-lg flex items-center justify-center gap-2 transition-colors disabled:opacity-70 shadow-sm text-sm"
                    >
                      {isFsLoading ? <Loader2 size={16} className="animate-spin" /> : <PlayCircle size={16} />}
                      Trích xuất lại
                    </button>
                    <button
                      onClick={() => handleFsAction('load')}
                      disabled={isFsLoading || !splitStats || !selectedDataset?.fs_models[fsMethodKey]}
                      className="flex-1 bg-white hover:bg-slate-50 text-slate-700 font-medium py-2 px-3 rounded-lg flex items-center justify-center gap-2 transition-colors border border-slate-200 disabled:opacity-70 shadow-sm text-sm"
                    >
                      <FileText size={16} className="text-slate-400" />
                      Tải kết quả có sẵn
                    </button>
                  </div>

                  {fsLog && (
                    <div className="mt-4 bg-slate-900 rounded-xl p-4 text-xs font-mono text-teal-400 whitespace-pre-wrap leading-relaxed shadow-inner border border-slate-800 max-h-64 overflow-y-auto">
                      {fsLog}
                    </div>
                  )}

                  {fsMethod === 'boruta' && (extractionStats?.selection_mode ?? borutaConfig.selection_mode) === 'confirmed' &&
                    extractionStats && (extractionStats.confirmed ?? extractionStats.n_selected_features) < 5 && (
                    <div className="mt-4 flex items-start gap-2 bg-amber-50 border border-amber-200 rounded-lg px-3 py-2.5 text-amber-800">
                      <AlertTriangle size={14} className="text-amber-500 shrink-0 mt-0.5" />
                      <p className="text-xs leading-relaxed">
                        Số đặc trưng "confirmed" quá ít ({extractionStats.confirmed ?? extractionStats.n_selected_features}) —
                        có thể không đủ để huấn luyện mô hình rule tốt. Thử "confirmed_tentative" hoặc "top_k".
                      </p>
                    </div>
                  )}

                  <RunHistoryList
                    title="Lịch sử trích xuất"
                    runs={fsRuns}
                    disabled={isFsLoading}
                    isSelected={run => fsRunId === run.run_id}
                    onSelect={loadPastFsRun}
                    renderLabel={run => `${run.fs_method} · ${run.run_id}`}
                    renderStatus={run => `${run.summary?.n_selected_features ?? '?'} đặc trưng`}
                  />
                  </fieldset>
                </div>
              </div>
            </div>
          </section>

          {/* Bước 4: Model */}
          <section className="relative">
            <div className={cn("flex items-start gap-4 mb-4 transition-opacity duration-300", !extractionStats && "opacity-50 grayscale")}>
              <button
                onClick={() => openStep(4)}
                disabled={!extractionStats}
                className={cn("w-12 h-12 shrink-0 bg-white border-2 rounded-full flex items-center justify-center shadow-sm z-10 transition-colors", extractionStats ? "border-teal-500 text-teal-600 hover:bg-teal-50" : "border-slate-200 text-slate-400 cursor-not-allowed")}
              >
                <span className="font-bold text-lg">4</span>
              </button>
              <div className="pt-2 w-full">
                <div
                  className="flex items-center justify-between cursor-pointer group mb-4"
                  onClick={() => openStep(4)}
                >
                  <h2 className="text-lg font-semibold text-slate-800 group-hover:text-teal-700 transition-colors">Mô Hình</h2>
                  <span className="text-slate-400 group-hover:text-teal-600 transition-colors">
                    {activeStep === 4 ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
                  </span>
                </div>
                <div className={cn("bg-white p-5 rounded-2xl shadow-sm border border-slate-200 relative overflow-hidden", activeStep !== 4 && "hidden")}>
                  {!extractionStats && <div className="absolute inset-0 z-20 bg-slate-50/50"></div>}

                  <fieldset disabled={modelSectionLocked} className="border-0 p-0 m-0 min-w-0 disabled:opacity-60">
                  <div className="flex gap-4 mb-5">
                    <label className="flex items-center gap-2 cursor-pointer group">
                      <input type="radio" name="modelType" checked={modelType === 'rf'} onChange={() => setModelType('rf')} className="w-4 h-4 shrink-0 accent-teal-600 focus:ring-teal-500" />
                      <span className="font-medium text-slate-700 group-hover:text-teal-700 transition-colors">Random Forest</span>
                    </label>
                    <label className="flex items-center gap-2 cursor-pointer group">
                      <input type="radio" name="modelType" checked={modelType === 'dt'} onChange={() => setModelType('dt')} className="w-4 h-4 shrink-0 accent-teal-600 focus:ring-teal-500" />
                      <span className="font-medium text-slate-700 group-hover:text-teal-700 transition-colors">Decision Tree</span>
                    </label>
                  </div>

                  {fsRunId && (
                    <p className="text-xs text-amber-600 mb-3">
                      Đặc trưng hiện tại đến từ một lần chạy live (run {fsRunId}) — không thể "Tải mô hình cũ" (được huấn luyện trên đặc trưng cache khác), chỉ có thể "Huấn luyện" lại trên tập đặc trưng này.
                    </p>
                  )}

                  <div className="bg-slate-50 p-4 rounded-xl border border-slate-100 mb-5 text-sm animate-in fade-in slide-in-from-top-2">
                    <div className="flex justify-between items-center mb-3">
                      <span className="font-semibold text-slate-700">Cấu hình huấn luyện</span>
                      <span className="text-[10px] bg-teal-100 text-teal-700 px-2 py-0.5 rounded font-bold uppercase tracking-wider">Tùy chỉnh</span>
                    </div>
                    <div className="grid grid-cols-1 gap-y-3">
                      {modelType === 'rf' && (
                        <div className="flex justify-between items-center">
                          <span className="text-slate-500">n_estimators</span>
                          <input type="number" value={modelConfig.n_estimators} onChange={e => setModelConfig({...modelConfig, n_estimators: Number(e.target.value)})} className="w-24 px-2 py-1 text-right font-mono text-teal-700 font-semibold bg-white border border-slate-200 rounded text-sm focus:outline-none focus:ring-1 focus:ring-teal-500" />
                        </div>
                      )}
                      <div className="flex justify-between items-center">
                        <span className="text-slate-500">max_depth</span>
                        <input type="number" value={modelConfig.max_depth} onChange={e => setModelConfig({...modelConfig, max_depth: Number(e.target.value)})} className="w-24 px-2 py-1 text-right font-mono text-teal-700 font-semibold bg-white border border-slate-200 rounded text-sm focus:outline-none focus:ring-1 focus:ring-teal-500" />
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-slate-500">min_samples_leaf</span>
                        <input type="number" value={modelConfig.min_samples_leaf} onChange={e => setModelConfig({...modelConfig, min_samples_leaf: Number(e.target.value)})} className="w-24 px-2 py-1 text-right font-mono text-teal-700 font-semibold bg-white border border-slate-200 rounded text-sm focus:outline-none focus:ring-1 focus:ring-teal-500" />
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-slate-500">class_weight</span>
                        <select value={modelConfig.class_weight} onChange={e => setModelConfig({...modelConfig, class_weight: e.target.value})} className="w-24 px-2 py-1 text-right font-mono text-teal-700 font-semibold bg-white border border-slate-200 rounded text-sm focus:outline-none focus:ring-1 focus:ring-teal-500">
                          <option value="balanced">balanced</option>
                          <option value="balanced_subsample">balanced_subsample</option>
                          <option value="none">none</option>
                        </select>
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-slate-500">random_state</span>
                        <input type="number" value={modelConfig.random_state} onChange={e => setModelConfig({...modelConfig, random_state: Number(e.target.value)})} className="w-24 px-2 py-1 text-right font-mono text-teal-700 font-semibold bg-white border border-slate-200 rounded text-sm focus:outline-none focus:ring-1 focus:ring-teal-500" />
                      </div>
                    </div>
                  </div>

                  <div className="bg-slate-50 p-4 rounded-xl border border-slate-100 mb-5 text-sm animate-in fade-in slide-in-from-top-2">
                    <div className="flex justify-between items-center mb-3">
                      <span className="font-semibold text-slate-700">Cấu hình Lọc Luật</span>
                      <span className="text-[10px] bg-teal-100 text-teal-700 px-2 py-0.5 rounded font-bold uppercase tracking-wider">Tùy chỉnh</span>
                    </div>
                    <div className="grid grid-cols-1 gap-y-3">
                      <div className="flex justify-between items-center">
                        <span className="text-slate-500">min_confidence</span>
                        <input type="number" step="0.01" min="0" max="1" value={modelConfig.min_confidence} onChange={e => setModelConfig({...modelConfig, min_confidence: Number(e.target.value)})} className="w-24 px-2 py-1 text-right font-mono text-teal-700 font-semibold bg-white border border-slate-200 rounded text-sm focus:outline-none focus:ring-1 focus:ring-teal-500" />
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-slate-500">min_fidelity</span>
                        <input type="number" step="0.01" min="0" max="1" value={modelConfig.min_fidelity} onChange={e => setModelConfig({...modelConfig, min_fidelity: Number(e.target.value)})} className="w-24 px-2 py-1 text-right font-mono text-teal-700 font-semibold bg-white border border-slate-200 rounded text-sm focus:outline-none focus:ring-1 focus:ring-teal-500" />
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-slate-500">min_support</span>
                        <input type="number" step="0.01" min="0" max="1" value={modelConfig.min_support} onChange={e => setModelConfig({...modelConfig, min_support: Number(e.target.value)})} className="w-24 px-2 py-1 text-right font-mono text-teal-700 font-semibold bg-white border border-slate-200 rounded text-sm focus:outline-none focus:ring-1 focus:ring-teal-500" />
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-slate-500">min_abs_support</span>
                        <input type="number" value={modelConfig.min_abs_support} onChange={e => setModelConfig({...modelConfig, min_abs_support: Number(e.target.value)})} className="w-24 px-2 py-1 text-right font-mono text-teal-700 font-semibold bg-white border border-slate-200 rounded text-sm focus:outline-none focus:ring-1 focus:ring-teal-500" />
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-slate-500">max_conditions</span>
                        <input type="number" value={modelConfig.max_conditions} onChange={e => setModelConfig({...modelConfig, max_conditions: Number(e.target.value)})} className="w-24 px-2 py-1 text-right font-mono text-teal-700 font-semibold bg-white border border-slate-200 rounded text-sm focus:outline-none focus:ring-1 focus:ring-teal-500" />
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-slate-500">max_rules_per_class</span>
                        <input type="number" value={modelConfig.max_rules_per_class} onChange={e => setModelConfig({...modelConfig, max_rules_per_class: Number(e.target.value)})} className="w-24 px-2 py-1 text-right font-mono text-teal-700 font-semibold bg-white border border-slate-200 rounded text-sm focus:outline-none focus:ring-1 focus:ring-teal-500" />
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-slate-500">max_rules_total</span>
                        <input type="number" value={modelConfig.max_rules_total} onChange={e => setModelConfig({...modelConfig, max_rules_total: Number(e.target.value)})} className="w-24 px-2 py-1 text-right font-mono text-teal-700 font-semibold bg-white border border-slate-200 rounded text-sm focus:outline-none focus:ring-1 focus:ring-teal-500" />
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-slate-500">merge_same_gene</span>
                        <input type="checkbox" checked={modelConfig.merge_same_gene} onChange={e => setModelConfig({...modelConfig, merge_same_gene: e.target.checked})} className="w-4 h-4 accent-teal-600" />
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-slate-500">dedup</span>
                        <input type="checkbox" checked={modelConfig.dedup} onChange={e => setModelConfig({...modelConfig, dedup: e.target.checked})} className="w-4 h-4 accent-teal-600" />
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-slate-500">dedup_sig_figs</span>
                        <input type="number" min={1} max={6} value={modelConfig.dedup_sig_figs} onChange={e => setModelConfig({...modelConfig, dedup_sig_figs: Number(e.target.value)})} className="w-24 px-2 py-1 text-right font-mono text-teal-700 font-semibold bg-white border border-slate-200 rounded text-sm focus:outline-none focus:ring-1 focus:ring-teal-500" />
                      </div>
                      <div className="flex justify-between items-center">
                        <span className="text-slate-500" title="Loại bỏ luật bị luật khác (cùng bộ gene, cùng lớp) bao trùm hoàn toàn — xem giải thích ở khung chat.">merge_generalization</span>
                        <input type="checkbox" checked={modelConfig.merge_generalization} onChange={e => setModelConfig({...modelConfig, merge_generalization: e.target.checked})} className="w-4 h-4 accent-teal-600" />
                      </div>
                    </div>
                  </div>

                  <div className="flex gap-3">
                    <button
                      onClick={() => handleModelAction('retrain')}
                      disabled={isModelLoading || !extractionStats}
                      className="flex-1 bg-teal-600 hover:bg-teal-700 text-white font-medium py-2 px-3 rounded-lg flex items-center justify-center gap-2 transition-colors disabled:opacity-70 shadow-sm text-sm"
                    >
                      {isModelLoading ? <Loader2 size={16} className="animate-spin" /> : <PlayCircle size={16} />}
                      Huấn luyện lại
                    </button>
                    <button
                      onClick={() => handleModelAction('load')}
                      disabled={modelLoadDisabled}
                      title={fsRunId ? 'Không khả dụng khi đặc trưng hiện tại đến từ một lần chạy live' : undefined}
                      className="flex-1 bg-white hover:bg-slate-50 text-slate-700 font-medium py-2 px-3 rounded-lg flex items-center justify-center gap-2 transition-colors border border-slate-200 disabled:opacity-70 shadow-sm text-sm"
                    >
                      <FileText size={16} className="text-slate-400" />
                      Tải mô hình có sẵn
                    </button>
                  </div>

                  {modelLog && (
                    <div className="mt-4 bg-slate-900 rounded-xl p-4 text-xs font-mono text-teal-400 whitespace-pre-wrap leading-relaxed shadow-inner border border-slate-800 max-h-64 overflow-y-auto">
                      {modelLog}
                    </div>
                  )}

                  <RunHistoryList
                    title="Lịch sử huấn luyện"
                    runs={modelRuns}
                    disabled={isModelLoading}
                    isSelected={run => modelRunId === run.run_id}
                    onSelect={loadPastModelRun}
                    renderLabel={run => `${run.fs_method}/${run.model} · ${run.run_id}`}
                    renderStatus={run => `acc=${((run.summary?.accuracy as number ?? 0) * 100).toFixed(1)}%`}
                  />
                  </fieldset>
                </div>
              </div>
            </div>
          </section>

          {/* Bước 5: Testing */}
          <section className="relative">
            <div className={cn("flex items-start gap-4 transition-opacity duration-300", !modelStats && "opacity-50 grayscale")}>
              <button
                onClick={() => openStep(5)}
                disabled={!modelStats}
                className={cn("w-12 h-12 shrink-0 bg-white border-2 rounded-full flex items-center justify-center shadow-sm z-10 transition-colors", modelStats ? "border-teal-500 text-teal-600 hover:bg-teal-50" : "border-slate-200 text-slate-400 cursor-not-allowed")}
              >
                <span className="font-bold text-lg">5</span>
              </button>
              <div className="pt-2 w-full">
                <div
                  className="flex items-center justify-between cursor-pointer group mb-4"
                  onClick={() => openStep(5)}
                >
                  <h2 className="text-lg font-semibold text-slate-800 group-hover:text-teal-700 transition-colors">Kiểm Thử (Thực Nghiệm)</h2>
                  <span className="text-slate-400 group-hover:text-teal-600 transition-colors">
                    {activeStep === 5 ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
                  </span>
                </div>
                <div className={cn("bg-white p-5 rounded-2xl shadow-sm border border-slate-200 relative overflow-hidden", activeStep !== 5 && "hidden")}>
                  {!modelStats && <div className="absolute inset-0 z-20 bg-slate-50/50"></div>}

                  <fieldset disabled={testSectionLocked} className="border-0 p-0 m-0 min-w-0 disabled:opacity-60">
                  <div className="flex flex-col gap-4">
                     <div className="flex gap-2 p-1 bg-slate-100 rounded-lg text-sm">
                       <button
                         onClick={() => setTestMode('sample')}
                         className={cn("flex-1 py-1.5 rounded-md flex items-center justify-center gap-1.5 transition-colors", testMode === 'sample' ? "bg-white shadow-sm text-teal-700 font-medium" : "text-slate-500")}
                       >
                         <ListChecks size={14} /> Chọn mẫu có sẵn
                       </button>
                       <button
                         onClick={() => setTestMode('upload')}
                         className={cn("flex-1 py-1.5 rounded-md flex items-center justify-center gap-1.5 transition-colors", testMode === 'upload' ? "bg-white shadow-sm text-teal-700 font-medium" : "text-slate-500")}
                       >
                         <UploadCloud size={14} /> Tải lên file
                       </button>
                     </div>

                     {/* Download the exact held-out test set (test_set.csv +
                         manifest.csv + samples/*.json) — for transparency
                         (audit which rows were held out) and for re-testing
                         via "Tải lên file" above. Works for both a cached
                         dataset and a fresh upload / custom "Thực hiện lại"
                         split (backend recomputes live when there's no cache
                         yet) — the URL already carries the current splitParams. */}
                     {datasetId && (
                       <a
                         href={api.getTestSetDownloadUrl(datasetId, fsMethodKey, splitParams)}
                         download
                         className="w-full py-2 border border-slate-200 hover:bg-slate-50 text-slate-600 rounded-lg flex items-center justify-center gap-1.5 transition-colors text-xs font-medium"
                       >
                         <Download size={14} /> Tải xuống test set (ZIP) — minh bạch dữ liệu đánh giá
                       </a>
                     )}

                     {testMode === 'sample' ? (
                       <select
                         value={testSampleId}
                         onChange={(e) => setTestSampleId(e.target.value)}
                         className="w-full bg-slate-50 border border-slate-200 text-slate-700 rounded-lg px-4 py-2.5 focus:outline-none focus:ring-2 focus:ring-teal-500/30 focus:border-teal-500 transition-colors shadow-sm"
                       >
                         <option value="">-- Chọn mẫu bệnh phẩm (test set thật) --</option>
                         {testSamples.map(s => (
                           <option key={s.sample_id} value={s.sample_id}>{s.sample_id} (nhãn thật: {s.true_label})</option>
                         ))}
                       </select>
                     ) : (
                       <label className="w-full border-2 border-dashed border-teal-200 bg-teal-50/30 hover:bg-teal-50/80 text-teal-700 rounded-xl px-4 py-6 flex flex-col items-center justify-center cursor-pointer transition-colors text-center">
                         <UploadCloud size={24} className="mb-2 text-teal-500" />
                         <span className="font-medium text-sm">{uploadFile ? uploadFile.name : 'Tải lên mẫu bệnh phẩm'}</span>
                         <span className="text-xs text-slate-400 mt-1">
                           Định dạng <span className="font-mono">.json</span> (giống mẫu test_set — chỉ cần giữ lại dữ liệu microarray, các trường khác có thể lược bỏ),
                           {' '}<span className="font-mono">.csv</span> (cột "samples,type,{'{probe}'}...") hoặc <span className="font-mono">.txt</span> (2 cột probe,giá trị)
                         </span>
                         <input type="file" accept=".json,.csv,.txt" className="hidden" onChange={(e) => setUploadFile(e.target.files?.[0] || null)} />
                       </label>
                     )}

                      <button
                        onClick={handleTestSample}
                        disabled={(testMode === 'sample' ? !testSampleId : !uploadFile) || isTesting}
                        className="w-full py-2.5 bg-teal-600 hover:bg-teal-700 text-white font-medium rounded-xl flex items-center justify-center gap-2 transition-colors disabled:opacity-50 disabled:cursor-not-allowed shadow-sm text-sm"
                      >
                        {isTesting ? <Loader2 size={18} className="animate-spin" /> : <Activity size={18} />}
                        Dự Đoán Kết Quả
                      </button>
                  </div>
                  </fieldset>
                </div>
              </div>
            </div>
          </section>
        </div>

        {/* Cột Phải: Visualizations & Testing — every step's panel that HAS data
            stays mounted (never deleted just because another step is active),
            since the results are related/shareable across steps; only its
            collapse state follows the active step (collapseSignal below), so
            switching steps folds panels down instead of removing them. Sticky +
            its own scroll region so browsing doesn't move the header/left column. */}
        <div className="lg:col-span-8 lg:sticky lg:top-24 lg:max-h-[calc(100vh-7rem)] overflow-y-auto pr-1 -mr-1">
          <div className="flex flex-col gap-8 pb-4">

          {datasetId ? (
            <DatasetOverview datasetId={datasetId} collapseSignal={activeStep !== 1} isTemp={selectedDataset?.is_temp} />
          ) : (
            activeStep === 1 && <EmptyStepPlaceholder text='Chọn hoặc tải lên một dataset ở Bước 1 để xem tổng quan dữ liệu.' />
          )}

          {datasetId && splitStats ? (
            <DatasetSplit datasetId={datasetId} stats={splitStats} collapseSignal={activeStep !== 2} />
          ) : (
            activeStep === 2 && <EmptyStepPlaceholder text='Nhấn "Tải dữ liệu có sẵn" hoặc "Thực hiện lại" ở Bước 2 để xem số liệu chia dữ liệu.' />
          )}

          {extractionStats ? (
            <FeatureExtractionOverview stats={extractionStats} collapseSignal={activeStep !== 3} />
          ) : (
            activeStep === 3 && <EmptyStepPlaceholder text='Chạy trích xuất đặc trưng ở Bước 3 để xem kết quả.' />
          )}

          {activeStep === 4 && !modelStats && (
            <EmptyStepPlaceholder text='Huấn luyện hoặc tải mô hình ở Bước 4 để xem tổng quan mô hình.' />
          )}

          {activeStep === 5 && !modelStats && (
            <EmptyStepPlaceholder text='Cần có mô hình (Bước 4) trước khi kiểm thử.' />
          )}

          {modelStats && (
            <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">

              {/* Tổng Quan Mô Hình — parent card; Confusion Matrix, Chi tiết Phân Loại,
                  Rule Extraction and Bio Graph are all sub-sections of this one card,
                  collapsible together so the testing result below isn't buried under them. */}
              <div className="bg-white rounded-2xl shadow-sm border border-slate-200 p-6">
                <div
                  className="flex items-center justify-between gap-4 cursor-pointer group"
                  onClick={() => setIsModelOverviewCollapsed(c => !c)}
                >
                  <div className="flex items-center gap-3 min-w-0">
                    <div className="w-10 h-10 shrink-0 rounded-full bg-teal-50 flex items-center justify-center text-teal-600 border border-teal-100">
                      <GitMerge size={20} />
                    </div>
                    <h2 className="text-2xl font-bold text-slate-800 tracking-tight group-hover:text-teal-700 transition-colors truncate">Tổng Quan Mô Hình</h2>
                  </div>
                  <button className="p-2 rounded-full hover:bg-teal-50 text-slate-500 hover:text-teal-600 transition-colors shrink-0">
                    {isModelOverviewCollapsed ? <ChevronDown size={24} /> : <ChevronUp size={24} />}
                  </button>
                </div>

                {/* Stat badges on their own row below the title — keeping them beside
                    the title squeezed the heading on narrower widths. */}
                <div className="flex flex-wrap gap-3 mt-4">
                  <div className="bg-white px-5 py-2 rounded-xl border border-slate-200 shadow-sm flex items-center gap-3">
                    <span className="text-[10px] font-bold uppercase tracking-widest text-slate-400">Accuracy</span>
                    <span className="text-xl font-bold text-teal-600">{modelStats.acc}%</span>
                  </div>
                  <div className="bg-white px-5 py-2 rounded-xl border border-slate-200 shadow-sm flex items-center gap-3">
                    <span className="text-[10px] font-bold uppercase tracking-widest text-slate-400">F1</span>
                    <span className="text-xl font-bold text-teal-600">{modelStats.f1}%</span>
                  </div>
                  <div className="bg-white px-5 py-2 rounded-xl border border-slate-200 shadow-sm flex items-center gap-3">
                    <span className="text-[10px] font-bold uppercase tracking-widest text-slate-400">Rules</span>
                    <span className="text-xl font-bold text-teal-600">{modelStats.rules}</span>
                  </div>
                </div>

                {!isModelOverviewCollapsed && (
                  <div className="mt-6 space-y-8 animate-in fade-in slide-in-from-top-4 duration-300">
                    <div className="space-y-6">
                      {/* Tham số cấu hình — the actual hyperparams/filter thresholds this
                          run used, mirroring "Tổng quan Trích xuất Đặc trưng"'s config
                          card so the model results side isn't missing the config context
                          the feature-extraction results side already has. Filter config
                          falls back to the current Bước 4 sidebar values (modelConfig) when
                          the backend hasn't recorded them (older cached runs, before
                          filter_config was added to rules_summary.json) — always shows both
                          columns instead of the card silently shrinking to one when data
                          is missing. */}
                      {(() => {
                        const fc = modelStats.rulesSummary?.filter_config;
                        const filterDisplay = fc ?? {
                          min_confidence: modelConfig.min_confidence,
                          min_fidelity: modelConfig.min_fidelity,
                          min_support: modelConfig.min_support,
                          min_abs_support: modelConfig.min_abs_support,
                          max_conditions: modelConfig.max_conditions,
                          merge_same_gene: modelConfig.merge_same_gene,
                          dedup: modelConfig.dedup,
                          dedup_sig_figs: modelConfig.dedup_sig_figs,
                          merge_generalization: modelConfig.merge_generalization,
                          max_rules_per_class: modelConfig.max_rules_per_class,
                          max_rules_total: modelConfig.max_rules_total,
                        };
                        return (
                          <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
                            <div
                              className="flex items-center justify-between cursor-pointer group/cfg"
                              onClick={() => setIsModelConfigCollapsed(c => !c)}
                            >
                              <div className="flex items-center gap-2">
                                <Settings2 className="text-teal-600" size={20} />
                                <h3 className="text-base font-semibold text-slate-800 group-hover/cfg:text-teal-700 transition-colors">Tham số cấu hình</h3>
                              </div>
                              <button className="p-1.5 rounded-full hover:bg-teal-50 text-slate-400 transition-colors shrink-0">
                                {isModelConfigCollapsed ? <ChevronDown size={20} /> : <ChevronUp size={20} />}
                              </button>
                            </div>
                            {!isModelConfigCollapsed && (
                              <div className="mt-6 grid grid-cols-1 lg:grid-cols-2 gap-6 animate-in fade-in slide-in-from-top-2 duration-200">
                                <div className="bg-slate-50 rounded-xl p-5 border border-slate-200">
                                  <h4 className="text-sm font-bold text-slate-800 mb-4 flex items-center gap-2">
                                    <span className="w-1.5 h-4 bg-teal-500 rounded-full inline-block"></span>
                                    Siêu tham số mô hình ({modelType === 'rf' ? 'Random Forest' : 'Decision Tree'})
                                  </h4>
                                  <div className="grid grid-cols-1 gap-y-3">
                                    {modelStats.hyperparams ? Object.entries(modelStats.hyperparams).map(([k, v]) => (
                                      <div key={k} className="flex justify-between py-2 border-b border-slate-200 border-dashed">
                                        <span className="text-slate-500 text-sm font-mono">{k}</span>
                                        <span className="text-slate-800 font-medium text-sm font-mono">{v === null ? 'null' : String(v)}</span>
                                      </div>
                                    )) : (
                                      <p className="text-sm text-slate-400 italic">Không có dữ liệu.</p>
                                    )}
                                  </div>
                                </div>
                                <div className="bg-slate-50 rounded-xl p-5 border border-slate-200">
                                  <h4 className="text-sm font-bold text-slate-800 mb-4 flex items-center gap-2">
                                    <span className="w-1.5 h-4 bg-blue-500 rounded-full inline-block"></span>
                                    Cấu hình Lọc Luật
                                  </h4>
                                  <div className="grid grid-cols-1 gap-y-3">
                                    {Object.entries(filterDisplay).map(([k, v]) => (
                                      <div key={k} className="flex justify-between py-2 border-b border-slate-200 border-dashed">
                                        <span className="text-slate-500 text-sm font-mono">{k}</span>
                                        <span className="text-slate-800 font-medium text-sm font-mono">{v == null ? '—' : String(v)}</span>
                                      </div>
                                    ))}
                                  </div>
                                </div>
                              </div>
                            )}
                          </div>
                        );
                      })()}

                      {/* Confusion Matrix — own card + collapse, stacked (not side-by-side with
                          the report) so neither gets clipped/overlapped on narrower screens. */}
                      <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
                        <div
                          className="flex items-center justify-between cursor-pointer group/cm"
                          onClick={() => setIsConfusionMatrixCollapsed(c => !c)}
                        >
                          <div className="flex items-center gap-2">
                            <Table2 className="text-teal-600" size={20} />
                            <h3 className="text-base font-semibold text-slate-800 group-hover/cm:text-teal-700 transition-colors">Ma trận nhầm lẫn</h3>
                          </div>
                          <button className="p-1.5 rounded-full hover:bg-teal-50 text-slate-400 transition-colors shrink-0">
                            {isConfusionMatrixCollapsed ? <ChevronDown size={20} /> : <ChevronUp size={20} />}
                          </button>
                        </div>
                        {!isConfusionMatrixCollapsed && (
                          <div className="mt-6 flex items-center justify-center animate-in fade-in slide-in-from-top-2 duration-200">
                            <ConfusionMatrix data={modelStats.cm} labels={modelStats.labels} />
                          </div>
                        )}
                      </div>

                      {/* Detailed Classification Report */}
                      <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
                        <div
                          className="flex items-center justify-between cursor-pointer group/cr"
                          onClick={() => setIsClassificationReportCollapsed(c => !c)}
                        >
                          <div className="flex items-center gap-2">
                            <FileText className="text-teal-600" size={20} />
                            <h3 className="text-base font-semibold text-slate-800 group-hover/cr:text-teal-700 transition-colors">Chi tiết Phân Loại</h3>
                          </div>
                          <button className="p-1.5 rounded-full hover:bg-teal-50 text-slate-400 transition-colors shrink-0">
                            {isClassificationReportCollapsed ? <ChevronDown size={20} /> : <ChevronUp size={20} />}
                          </button>
                        </div>
                        {!isClassificationReportCollapsed && (
                          <div className="mt-6 animate-in fade-in slide-in-from-top-2 duration-200">
                            <table className="w-full text-sm text-left table-fixed">
                              <thead className="text-xs text-slate-400 uppercase bg-slate-50/80 rounded-t-lg">
                                <tr>
                                  <th className="px-3 py-3 font-semibold rounded-tl-lg w-2/5">Lớp (Class)</th>
                                  <th className="px-3 py-3 font-semibold">Precision</th>
                                  <th className="px-3 py-3 font-semibold">Recall</th>
                                  <th className="px-3 py-3 font-semibold">F1</th>
                                  <th className="px-3 py-3 font-semibold rounded-tr-lg">Support</th>
                                </tr>
                              </thead>
                              <tbody className="divide-y divide-slate-100">
                                {modelStats.labels.map((label, idx) => (
                                  <tr key={label} className="hover:bg-slate-50/50 transition-colors">
                                    <td className="px-3 py-4 font-medium text-slate-700 truncate" title={label}>{label}</td>
                                    <td className="px-3 py-4 text-slate-600 font-mono">{report[idx]?.precision.toFixed(2)}</td>
                                    <td className="px-3 py-4 text-slate-600 font-mono">{report[idx]?.recall.toFixed(2)}</td>
                                    <td className="px-3 py-4 text-slate-600 font-mono">{report[idx]?.f1.toFixed(2)}</td>
                                    <td className="px-3 py-4 text-slate-600 font-mono">{report[idx]?.support}</td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        )}
                      </div>
                    </div>

                    {/* Rule Extraction Results */}
                    <RuleExtractionResults
                      datasetId={datasetId}
                      fsMethod={fsMethodKey}
                      model={modelType}
                      runId={modelRunId}
                      rulesSummary={modelStats.rulesSummary}
                      labels={modelStats.labels}
                    />

                    {/* Bio Graph */}
                    <div className="bg-white p-6 rounded-2xl shadow-sm border border-slate-200">
                      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6">
                        <div className="flex items-center gap-2">
                          <Microscope className="text-teal-600" size={20} />
                          <h3 className="text-base font-semibold text-slate-800">Mạng Lưới Tương Tác Gen (Biology Graph)</h3>
                        </div>
                        <select
                          value={graphType}
                          onChange={(e) => setGraphType(e.target.value)}
                          className="bg-slate-50 border border-slate-200 text-slate-700 text-sm rounded-lg px-3 py-2 focus:outline-none focus:ring-2 focus:ring-teal-500/30 focus:border-teal-500 transition-colors shadow-sm"
                        >
                          <option value="go_bp">GO Biological Process</option>
                          <option value="go_cc">GO Cellular Component</option>
                          <option value="go_mf">GO Molecular Function</option>
                          <option value="kegg">KEGG Pathways</option>
                        </select>
                      </div>
                      <div className="bg-slate-50/50 rounded-xl overflow-hidden border border-slate-100">
                        <BioNetworkGraph graphType={graphType} />
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {modelStats && (
            <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
              <div>
                <div
                  className="flex items-center justify-between mb-6 cursor-pointer group"
                  onClick={() => setIsTestResultsCollapsed(c => !c)}
                >
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 shrink-0 rounded-full bg-emerald-50 flex items-center justify-center text-emerald-600 border border-emerald-100">
                      <Activity size={20} />
                    </div>
                    <h2 className="text-2xl font-bold text-slate-800 tracking-tight group-hover:text-emerald-700 transition-colors">Kết quả Thực nghiệm</h2>
                  </div>
                  <button className="p-2 rounded-full hover:bg-emerald-50 text-slate-500 hover:text-emerald-600 transition-colors shrink-0">
                    {isTestResultsCollapsed ? <ChevronDown size={24} /> : <ChevronUp size={24} />}
                  </button>
                </div>

                {!isTestResultsCollapsed && (
                <div className="w-full">
                  {testResults ? (
                    testResults.classification === 'Loi' ? (
                    <div className="bg-slate-900 text-slate-50 p-6 rounded-2xl shadow-xl animate-in fade-in slide-in-from-right-4 duration-500 border border-red-800/40 h-full">
                      <div className="flex items-center gap-3 mb-4 pb-4 border-b border-slate-800">
                        <AlertTriangle className="text-red-400" size={24} />
                        <h3 className="text-lg font-bold">Lỗi khi thực nghiệm</h3>
                      </div>
                      <p className="text-sm text-red-300 bg-red-950/30 border border-red-800/40 rounded-lg px-3 py-2">
                        {testResults.explanation || 'Đã có lỗi xảy ra, vui lòng thử lại.'}
                      </p>
                    </div>
                    ) : (
                    <div className="bg-slate-900 text-slate-50 p-6 rounded-2xl shadow-xl animate-in fade-in slide-in-from-right-4 duration-500 border border-slate-800 h-full">
                      <div className="flex items-center gap-3 mb-6 pb-4 border-b border-slate-800">
                        <CheckCircle2 className="text-teal-400" size={24} />
                        <h3 className="text-lg font-bold">Báo cáo Phân loại</h3>
                        <span className="ml-auto bg-teal-500/20 text-teal-300 border border-teal-500/30 text-xs py-1 px-3 rounded-full font-medium">
                          Khớp {testResults.matchedCount} rules
                        </span>
                      </div>

                      {testResults.trueLabel ? (
                        <p className="text-sm text-slate-400 mb-4">
                          Nhãn thật: <span className="font-semibold text-slate-200">
                            {displayLabel(testResults.trueLabel, testResults.classDisplayNames)}
                          </span>
                          {testResults.classDisplayNames?.[testResults.trueLabel] && (
                            <span className="text-slate-500 font-mono text-xs ml-1.5">({testResults.trueLabel})</span>
                          )}
                        </p>
                      ) : (
                        <p className="text-sm text-slate-500 mb-4 italic">Dữ liệu upload không có nhãn</p>
                      )}

                      {/* Two independent predictions shown side by side: the model's
                          own predict() call vs. the majority label among matched rules —
                          these can disagree (see class_votes note above), so both are
                          surfaced explicitly instead of only the model's result. */}
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mb-6">
                        <div className="rounded-xl p-4 border" style={{
                          backgroundColor: classColor(testResults.classification, modelStats?.labels || []) + '14',
                          borderColor: classColor(testResults.classification, modelStats?.labels || []) + '55',
                        }}>
                          <span className="text-slate-400 text-xs uppercase tracking-wider font-semibold block mb-1.5">Dự đoán theo Mô hình (RF/DT)</span>
                          <span className="font-bold text-xl" style={{ color: classColor(testResults.classification, modelStats?.labels || []) }}>
                            {testResults.classDisplayName || testResults.classification}
                          </span>
                          {testResults.classDisplayName && (
                            <span className="text-slate-500 font-mono text-xs block mt-0.5">{testResults.classification}</span>
                          )}
                        </div>
                        <div className="rounded-xl p-4 border" style={{
                          backgroundColor: testResults.rulePrediction ? classColor(testResults.rulePrediction, modelStats?.labels || []) + '14' : 'rgba(100,116,139,0.1)',
                          borderColor: testResults.rulePrediction ? classColor(testResults.rulePrediction, modelStats?.labels || []) + '55' : 'rgba(100,116,139,0.3)',
                        }}>
                          <span className="text-slate-400 text-xs uppercase tracking-wider font-semibold block mb-1.5">Dự đoán theo Tập luật (rule-based)</span>
                          {testResults.rulePrediction ? (
                            <>
                              <span className="font-bold text-xl" style={{ color: classColor(testResults.rulePrediction, modelStats?.labels || []) }}>
                                {testResults.rulePredictionDisplayName || testResults.rulePrediction}
                              </span>
                              {testResults.rulePredictionDisplayName && (
                                <span className="text-slate-500 font-mono text-xs block mt-0.5">{testResults.rulePrediction}</span>
                              )}
                            </>
                          ) : (
                            <span className="font-bold text-xl text-slate-500">Không có luật khớp</span>
                          )}
                        </div>
                      </div>
                      {testResults.rulePrediction && testResults.rulePrediction !== testResults.classification && (
                        <p className="text-xs text-amber-400/90 bg-amber-950/30 border border-amber-800/40 rounded-lg px-3 py-2 mb-6">
                          Hai dự đoán không trùng nhau — mô hình quyết định dựa trên toàn bộ đặc trưng, còn tập luật chỉ phản ánh các luật đơn giản tình cờ khớp với mẫu này (xem "Tỷ lệ Luật Khớp theo Lớp" bên dưới).
                        </p>
                      )}

                      {/* Matched-rules list — same bordered-box level as the vote breakdown
                          and biomedical assessment below it (previously this list floated
                          without its own box, unlike its siblings). */}
                      <div className="bg-slate-800/50 rounded-xl p-5 border border-slate-700/50 mb-6">
                        <div
                          className="flex items-center justify-between mb-4 cursor-pointer group/rules"
                          onClick={() => setIsMatchedRulesCollapsed(c => !c)}
                        >
                          <h4 className="font-semibold tracking-wide text-xs uppercase text-teal-400 group-hover/rules:text-teal-300 transition-colors">
                            Danh sách Luật Khớp ({testResults.rules.length})
                          </h4>
                          <button className="text-slate-500 hover:text-teal-300 transition-colors">
                            {isMatchedRulesCollapsed ? <ChevronDown size={18} /> : <ChevronUp size={18} />}
                          </button>
                        </div>
                        {!isMatchedRulesCollapsed && (
                        <div className="space-y-3">
                          {testResults.rules.map((rule: { id: number, text: string, matched: boolean, desc: string, sampleValues: Record<string, number>, class: string }) => {
                            const ruleColor = classColor(rule.class, modelStats?.labels || []);
                            return (
                            <div key={rule.id} className={cn("rounded-xl overflow-hidden transition-all", rule.matched ? "bg-teal-900/30 border border-teal-700/50" : "bg-slate-800/30 border border-slate-700/50 opacity-60")}>
                              <div className="px-4 py-3 flex items-center justify-between">
                                <div className="flex items-center gap-3">
                                  <span
                                    className="px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider border"
                                    style={{ backgroundColor: ruleColor + '2a', color: ruleColor, borderColor: ruleColor + '55' }}
                                  >
                                    {displayLabel(rule.class, testResults.classDisplayNames)}
                                  </span>
                                  <code className="font-mono text-xs md:text-sm text-teal-100/90">{rule.text}</code>
                                </div>
                                <button
                                  onClick={() => setExpandedRule(expandedRule === rule.id ? null : rule.id)}
                                  className="text-teal-300 hover:text-white transition-colors shrink-0 ml-4"
                                  title="Giải thích Y sinh"
                                >
                                  <Info size={16} />
                                </button>
                              </div>
                              {expandedRule === rule.id && (
                                <div className="px-4 py-3 bg-slate-950/50 text-sm text-slate-300 border-t border-slate-800/80 leading-relaxed">
                                  <p className="mb-3">{rule.desc || 'Chưa có mô tả sinh học cho luật này (chạy scripts/generate_bio_descriptions.py để sinh).'}</p>
                                  <div className="flex flex-wrap gap-2">
                                    {Object.entries(rule.sampleValues).map(([gene, val]) => (
                                      <span key={gene} className="bg-slate-800 px-2 py-1 rounded-md text-xs font-mono border border-slate-700">
                                        <span className="text-teal-300">{gene}</span> = {val}
                                      </span>
                                    ))}
                                  </div>
                                </div>
                              )}
                            </div>
                            );
                          })}
                        </div>
                        )}
                      </div>

                      {/* Genes appearing in the matched rules — just the list (no
                          redefinition, that already lives in "Gene xuất hiện trong Danh
                          sách Luật" above); click jumps straight to that gene's own card. */}
                      {(() => {
                        const matchedGenes = Array.from(new Set(
                          testResults.rules.flatMap((rule: { sampleValues: Record<string, number> }) => Object.keys(rule.sampleValues)),
                        )) as string[];
                        if (matchedGenes.length === 0) return null;
                        return (
                          <div className="bg-slate-800/50 rounded-xl p-5 border border-slate-700/50 mb-6">
                            <h4 className="font-semibold tracking-wide text-xs uppercase text-teal-400 mb-4">Danh sách Gene trong Luật Khớp</h4>
                            <div className="flex flex-wrap gap-2">
                              {matchedGenes.map(gene => (
                                <button
                                  key={gene}
                                  onClick={() => {
                                    // "Tổng Quan Mô Hình" (which contains the gene list target)
                                    // auto-collapses — and unmounts — once a prediction result
                                    // shows, so it must be re-expanded and given a tick to
                                    // remount before the jump-to-gene listener can catch this.
                                    setIsModelOverviewCollapsed(false);
                                    setTimeout(() => {
                                      window.dispatchEvent(new CustomEvent('jump-to-gene', { detail: { gene } }));
                                    }, 60);
                                  }}
                                  className="px-2.5 py-1 rounded-md text-xs font-mono bg-slate-900/60 border border-slate-700 text-teal-300 hover:bg-teal-900/40 hover:border-teal-700 hover:text-teal-200 transition-colors"
                                  title="Xem chi tiết gene này ở danh sách Gene phía trên"
                                >
                                  {gene}
                                </button>
                              ))}
                            </div>
                          </div>
                        );
                      })()}

                      {/* Rules that satisfy >=50% of their conditions but didn't fully match —
                          transparency into "near misses", sorted by match ratio, top 10.
                          Monochrome styling matches "Danh sách Luật Khớp" above (colored class
                          badge only — rule text and condition chips stay a single neutral tone,
                          the ✓/✗ mark is just appended after the value instead of color-coding
                          the whole chip); the ratio badge sits on its own header row so it never
                          crowds out the rule text like it did when squeezed inline before. */}
                      {testResults.partialMatches && testResults.partialMatches.length > 0 && (
                        <div className="bg-slate-800/50 rounded-xl p-5 border border-slate-700/50 mb-6">
                          <div
                            className="flex items-center justify-between mb-4 cursor-pointer group/partial"
                            onClick={() => setIsPartialMatchesCollapsed(c => !c)}
                          >
                            <h4 className="font-semibold tracking-wide text-xs uppercase text-teal-400 group-hover/partial:text-teal-300 transition-colors">
                              Luật Khớp Một Phần ({testResults.partialMatches.length})
                            </h4>
                            <button className="text-slate-500 hover:text-teal-300 transition-colors">
                              {isPartialMatchesCollapsed ? <ChevronDown size={18} /> : <ChevronUp size={18} />}
                            </button>
                          </div>
                          {!isPartialMatchesCollapsed && (
                          <div className="space-y-3">
                            {testResults.partialMatches.map((p: { ruleId: number, text: string, class: string, satisfied: number, total: number, ratio: number, conditions: Array<{ gene: string, probe: string, op: string, threshold: number, actual: number, ok: boolean }> }) => {
                              const ruleColor = classColor(p.class, modelStats?.labels || []);
                              return (
                                <div key={p.ruleId} className="rounded-xl overflow-hidden bg-slate-800/30 border border-slate-700/50 p-4">
                                  <div className="flex items-center justify-between gap-3 mb-2">
                                    <span
                                      className="px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider border shrink-0"
                                      style={{ backgroundColor: ruleColor + '2a', color: ruleColor, borderColor: ruleColor + '55' }}
                                    >
                                      {displayLabel(p.class, testResults.classDisplayNames)}
                                    </span>
                                    <span className="shrink-0 bg-amber-500/10 text-amber-300 border border-amber-500/30 text-[11px] py-1 px-2.5 rounded-full font-semibold">
                                      {p.satisfied}/{p.total} điều kiện — {Math.round(p.ratio * 100)}%
                                    </span>
                                  </div>
                                  <code className="font-mono text-xs md:text-sm text-teal-100/90 block break-words">{p.text}</code>
                                  <div className="mt-3 flex flex-wrap gap-2">
                                    {p.conditions.map((c, ci) => (
                                      <span
                                        key={ci}
                                        className="px-2 py-1 rounded-md text-xs font-mono border border-slate-700 bg-slate-900/40 text-slate-300"
                                      >
                                        {c.gene}={c.actual} {c.ok ? '✓' : '✗'}
                                      </span>
                                    ))}
                                  </div>
                                </div>
                              );
                            })}
                          </div>
                          )}
                        </div>
                      )}

                      {/* Genes that actually SATISFIED a condition in a partial-match rule —
                          not every gene referenced (a failed condition's gene isn't "in" the
                          rule the way a matched one is), mirroring "Danh sách Gene trong Luật
                          Khớp" above but scoped to partialMatches (capped at 10, same as the
                          list itself — can't list genes from rules that aren't shown). */}
                      {(() => {
                        const partialMatchedGenes = Array.from(new Set(
                          (testResults.partialMatches || []).flatMap(
                            (p: { conditions: Array<{ gene: string, ok: boolean }> }) =>
                              p.conditions.filter(c => c.ok).map(c => c.gene),
                          ),
                        )) as string[];
                        if (partialMatchedGenes.length === 0) return null;
                        return (
                          <div className="bg-slate-800/50 rounded-xl p-5 border border-slate-700/50 mb-6">
                            <h4 className="font-semibold tracking-wide text-xs uppercase text-teal-400 mb-4">Danh sách Gene trong Luật Khớp Một Phần</h4>
                            <div className="flex flex-wrap gap-2">
                              {partialMatchedGenes.map(gene => (
                                <button
                                  key={gene}
                                  onClick={() => {
                                    setIsModelOverviewCollapsed(false);
                                    setTimeout(() => {
                                      window.dispatchEvent(new CustomEvent('jump-to-gene', { detail: { gene } }));
                                    }, 60);
                                  }}
                                  className="px-2.5 py-1 rounded-md text-xs font-mono bg-slate-900/60 border border-slate-700 text-teal-300 hover:bg-teal-900/40 hover:border-teal-700 hover:text-teal-200 transition-colors"
                                  title="Xem chi tiết gene này ở danh sách Gene phía trên"
                                >
                                  {gene}
                                </button>
                              ))}
                            </div>
                          </div>
                        );
                      })()}

                      {/* Per-class matched-rule vote breakdown — how many/what % of the
                          matched rules point to each class, independent of the model's
                          own predict() call (a transparency/sanity-check signal). */}
                      {Object.keys(testResults.classVotes || {}).length > 0 && (
                        <div className="bg-slate-800/50 rounded-xl p-5 border border-slate-700/50 mb-6">
                          <h4 className="font-semibold tracking-wide text-xs uppercase text-teal-400 mb-4">Tỷ lệ Luật Khớp hoàn toàn theo Lớp</h4>
                          <div className="space-y-3">
                            {Object.entries(testResults.classVotes as Record<string, { count: number, percentage: number }>).map(([label, v]) => {
                              const color = classColor(label, modelStats?.labels || []);
                              return (
                                <div key={label} className="space-y-1">
                                  <div className="flex items-center justify-between gap-3">
                                    <span className="text-sm font-semibold leading-snug" style={{ color }}>
                                      {displayLabel(label, testResults.classDisplayNames)}
                                    </span>
                                    <span className="shrink-0 text-right text-slate-400 font-mono text-xs">{v.count} luật ({v.percentage}%)</span>
                                  </div>
                                  <div className="h-2.5 rounded-full bg-slate-700/50 overflow-hidden">
                                    <div className="h-full rounded-full transition-all" style={{ width: `${v.percentage}%`, backgroundColor: color }} />
                                  </div>
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      )}

                      {/* Per-class breakdown of PARTIAL matches only (>=50% but <100% of
                          conditions, unbounded — not just the capped top-10 partialMatches
                          list) — the per-class counterpart of "Danh sách Luật Khớp một Phần",
                          complementing "Tỷ lệ Luật Khớp theo Lớp" above (full matches only)
                          instead of duplicating it with matched rules mixed back in. */}
                      {Object.keys(testResults.classVotesOver50 || {}).length > 0 && (
                        <div className="bg-slate-800/50 rounded-xl p-5 border border-slate-700/50 mb-6">
                          <h4 className="font-semibold tracking-wide text-xs uppercase text-teal-400 mb-4">Tỷ lệ Luật Khớp một phần theo Lớp</h4>
                          <div className="space-y-3">
                            {Object.entries(testResults.classVotesOver50 as Record<string, { count: number, percentage: number }>).map(([label, v]) => {
                              const color = classColor(label, modelStats?.labels || []);
                              return (
                                <div key={label} className="space-y-1">
                                  <div className="flex items-center justify-between gap-3">
                                    <span className="text-sm font-semibold leading-snug" style={{ color }}>
                                      {displayLabel(label, testResults.classDisplayNames)}
                                    </span>
                                    <span className="shrink-0 text-right text-slate-400 font-mono text-xs">{v.count} luật ({v.percentage}%)</span>
                                  </div>
                                  <div className="h-2.5 rounded-full bg-slate-700/50 overflow-hidden">
                                    <div className="h-full rounded-full transition-all" style={{ width: `${v.percentage}%`, backgroundColor: color }} />
                                  </div>
                                </div>
                              );
                            })}
                          </div>
                        </div>
                      )}
                      
                      {/* Comparison of match TIERS (đủ 100% vs một phần >=50%) out of every
                          rule this model has — same row-per-category bar layout as "Tỷ lệ
                          Luật Khớp theo Lớp" below, just comparing match tiers instead of
                          classes. Uses nPartialMatchesTotal (unbounded), not
                          partialMatches.length (capped at 10 for display), so the % doesn't
                          undercount. */}
                      {!!testResults.nRulesTotal && (
                        <div className="bg-slate-800/50 rounded-xl p-5 border border-slate-700/50 mb-6">
                          <h4 className="font-semibold tracking-wide text-xs uppercase text-teal-400 mb-4">Tỷ lệ Luật Khớp Trên 50%</h4>
                          {(() => {
                            const total = testResults.nRulesTotal;
                            const tiers = [
                              { label: 'Khớp đủ (100%)', count: testResults.matchedCount, color: '#2dd4bf' },
                              { label: 'Khớp một phần (≥50%)', count: testResults.nPartialMatchesTotal || 0, color: '#fbbf24' },
                            ];
                            return (
                              <div className="space-y-3">
                                {tiers.map(t => {
                                  const pct = Math.round((t.count / total) * 1000) / 10;
                                  return (
                                    <div key={t.label} className="space-y-1">
                                      <div className="flex items-center justify-between gap-3">
                                        <span className="text-sm font-semibold leading-snug" style={{ color: t.color }}>
                                          {t.label}
                                        </span>
                                        <span className="shrink-0 text-right text-slate-400 font-mono text-xs">{t.count}/{total} ({pct}%)</span>
                                      </div>
                                      <div className="h-2.5 rounded-full bg-slate-700/50 overflow-hidden">
                                        <div className="h-full rounded-full transition-all" style={{ width: `${pct}%`, backgroundColor: t.color }} />
                                      </div>
                                    </div>
                                  );
                                })}
                              </div>
                            );
                          })()}
                        </div>
                      )}

                      <div className="bg-slate-800/50 rounded-xl p-5 border border-slate-700/50">
                        <div className="flex items-center gap-2 mb-4 text-teal-400">
                          <Brain size={18} />
                          <h4 className="font-semibold tracking-wide text-xs uppercase">
                            Đánh Giá Y Sinh {testResults.llmUsed === false && <span className="text-slate-500 normal-case">(template — Gemini không khả dụng)</span>}
                          </h4>
                        </div>
                        <div className="space-y-5">
                          {testResults.biomedicalSummary && (
                            <div>
                              <span className="text-slate-400 text-xs uppercase tracking-wider font-semibold block mb-1.5">Tóm tắt</span>
                              <p className="text-slate-100 leading-relaxed text-base font-medium">{testResults.biomedicalSummary}</p>
                            </div>
                          )}
                          {testResults.biomedicalRationale && (
                            <div>
                              <span className="text-slate-400 text-xs uppercase tracking-wider font-semibold block mb-1.5">Cơ sở Sinh học</span>
                              <p className="text-slate-300 leading-relaxed text-base">{testResults.biomedicalRationale}</p>
                            </div>
                          )}
                          {testResults.biomedicalModelVsRule && (
                            <div className={cn(
                              "rounded-lg p-4 border",
                              testResults.rulePrediction && testResults.rulePrediction !== testResults.classification
                                ? "bg-amber-950/30 border-amber-800/40"
                                : "bg-teal-950/30 border-teal-800/40"
                            )}>
                              <span className={cn(
                                "text-xs uppercase tracking-wider font-semibold block mb-1.5",
                                testResults.rulePrediction && testResults.rulePrediction !== testResults.classification ? "text-amber-400" : "text-teal-400"
                              )}>
                                So sánh Mô hình &harr; Tập luật
                              </span>
                              <p className="text-slate-200 leading-relaxed text-base">{testResults.biomedicalModelVsRule}</p>
                            </div>
                          )}
                          {testResults.biomedicalDisclaimer && (
                            <div className="flex items-start gap-2 bg-amber-950/30 border border-amber-800/40 rounded-lg px-3 py-2.5 mt-1">
                              <AlertTriangle size={14} className="text-amber-500 shrink-0 mt-0.5" />
                              <p className="text-amber-400/90 text-xs leading-relaxed">{testResults.biomedicalDisclaimer}</p>
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                    )
                  ) : (
                    <div className="h-full bg-slate-100/50 border border-slate-200 border-dashed rounded-2xl flex flex-col items-center justify-center text-slate-400 p-8 text-center min-h-[300px]">
                       <div className="w-16 h-16 bg-white rounded-full flex items-center justify-center mb-4 border border-slate-200 shadow-sm">
                         <Activity size={24} className="text-slate-300" />
                       </div>
                       <p className="text-sm max-w-sm font-medium text-slate-500">
                         Chọn một mẫu bệnh phẩm và nhấn "Dự Đoán Kết Quả" ở Bước 5 để bắt đầu.
                       </p>
                    </div>
                  )}
                </div>
                )}
              </div>
            </div>
          )}

          </div>
        </div>
      </main>
    </div>
  );
}

// Icon helper
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function Brain(props: any) {
  return (
    <svg {...props} xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 5a3 3 0 1 0-5.997.125 4 4 0 0 0-2.526 5.77 4 4 0 0 0 .556 6.588A4 4 0 1 0 12 18Z"/>
      <path d="M12 5a3 3 0 1 1 5.997.125 4 4 0 0 1 2.526 5.77 4 4 0 0 1-.556 6.588A4 4 0 1 1 12 18Z"/>
      <path d="M15 13a4.5 4.5 0 0 1-3-4 4.5 4.5 0 0 1-3 4"/>
      <path d="M17.599 6.5a3 3 0 0 0 .399-1.375"/>
      <path d="M6.003 5.125A3 3 0 0 0 6.401 6.5"/>
      <path d="M3.477 10.896a4 4 0 0 1 .585-.396"/>
      <path d="M19.938 10.5a4 4 0 0 1 .585.396"/>
      <path d="M6 18a4 4 0 0 1-1.967-.516"/>
      <path d="M19.967 17.484A4 4 0 0 1 18 18"/>
    </svg>
  )
}
