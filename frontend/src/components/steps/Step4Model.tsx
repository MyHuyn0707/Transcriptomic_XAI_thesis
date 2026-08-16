import React, { useEffect } from 'react';
import { ChevronDown, ChevronUp, FileText, GitMerge, Loader2, PlayCircle, Settings2, Table2 } from 'lucide-react';
import { api, ModelStatsResponse, pollJob, RunRecord } from '../../lib/api';
import { classificationReport } from '../../lib/metrics';
import { errorMessage } from '../../lib/utils';
import { canonicalClassLabels, fsMethodKeyOf, ModelStatsUI, WorkspaceAction, WorkspaceState } from '../../state/types';
import Button from '../ui/Button';
import Panel from '../ui/Panel';
import Badge from '../ui/Badge';
import StatusLine from '../ui/StatusLine';
import RunHistoryList from '../RunHistoryList';
import ConfusionMatrix from '../ConfusionMatrix';
import RuleExtractionResults from '../RuleExtractionResults';

interface Props {
  state: WorkspaceState;
  dispatch: React.Dispatch<WorkspaceAction>;
}

function buildModelStatsUI(stats: ModelStatsResponse): ModelStatsUI {
  return {
    acc: Math.round((stats.best_run_test_metrics?.accuracy ?? 0) * 1000) / 10,
    f1: Math.round((stats.best_run_test_metrics?.f1_macro ?? 0) * 1000) / 10,
    rules: stats.n_rules ?? 0,
    cm: stats.confusion_matrix ?? [],
    labels: stats.class_labels ?? [],
    hyperparams: stats.hyperparams ?? null,
    rulesSummary: stats.rules_summary ?? null,
  };
}

/** Bước 4 "Mô Hình" — train (or load a cached) Random Forest / Decision Tree
 * on top of Bước 3's selected features, plus the rule-filtering thresholds
 * applied to the rules it extracts. */
export default function Step4Model({ state, dispatch }: Props) {
  const {
    datasetId, splitParams, fsRunId, extractionStats,
    modelType, modelConfig, modelLog, modelRuns, modelRunId,
    isSplitLoading, isFsLoading, isModelLoading, isTesting,
  } = state;
  const fsMethodKey = fsMethodKeyOf(state);
  const modelSectionLocked = isModelLoading || isTesting;
  // A live "Huan luyen" feature-selection run (fsRunId set) produces a
  // feature set that only exists under outputs_live/<fsRunId> — the cached
  // "Tai mo hinh cu" model was trained on the cached (outputs_holdout) fs
  // run's features, so it doesn't line up. Force retraining in that case.
  const modelLoadDisabled = modelSectionLocked || !extractionStats || !!fsRunId;

  useEffect(() => {
    let ignore = false;
    if (datasetId) {
      api.getRuns(datasetId, 'model').then(runs => { if (!ignore) dispatch({ type: 'model_runs_loaded', runs }); }).catch(() => { if (!ignore) dispatch({ type: 'model_runs_loaded', runs: [] }); });
    } else {
      dispatch({ type: 'model_runs_loaded', runs: [] });
    }
    return () => { ignore = true; };
  }, [datasetId, dispatch]);

  const refreshModelRuns = () => {
    if (!datasetId) return;
    api.getRuns(datasetId, 'model').then(runs => dispatch({ type: 'model_runs_loaded', runs })).catch(() => {});
  };

  const handleModelAction = async (action: 'retrain' | 'load') => {
    if (!datasetId) return;
    dispatch({ type: 'model_started', log: action === 'retrain' ? '[HỆ THỐNG] Bắt đầu huấn luyện mô hình (chạy thật)...' : '' });
    try {
      let stats: ModelStatsResponse;
      let runId: string | null = null;
      if (action === 'load') {
        stats = await api.getModelStats(datasetId, fsMethodKey, modelType);
      } else {
        const { job_id } = await api.trainModel(datasetId, fsMethodKey, modelType, modelConfig, fsRunId, splitParams);
        const job = await pollJob(job_id, j => dispatch({ type: 'model_log_set', log: j.log.join('\n') }));
        if (job.status === 'error') {
          throw new Error(job.error || 'Huan luyen mo hinh that bai (xem log server).');
        }
        stats = await api.getModelStats(datasetId, fsMethodKey, modelType, job_id);
        runId = job_id;
        refreshModelRuns();
      }
      dispatch({ type: 'model_computed', stats: buildModelStatsUI(stats), runId });
    } catch (e) {
      dispatch({ type: 'model_failed', line: `[LỖI] ${errorMessage(e)}` });
    }
  };

  const loadPastModelRun = async (run: RunRecord) => {
    if (!datasetId || !run.model || !run.fs_method) return;
    dispatch({ type: 'model_started', log: `[HỆ THỐNG] Đang tải lại kết quả chạy trước (${run.run_id})...` });
    try {
      const stats = await api.getModelStats(datasetId, run.fs_method, run.model, run.run_id);
      dispatch({ type: 'model_computed', stats: buildModelStatsUI(stats), runId: run.run_id });
      dispatch({ type: 'model_log_appended', line: '[OK] Đã nạp lại kết quả chạy trước.' });
    } catch (e) {
      dispatch({ type: 'model_failed', line: `[LỖI] ${errorMessage(e)}` });
    }
  };

  return (
    <fieldset disabled={modelSectionLocked} className="border-0 p-0 m-0 min-w-0 disabled:opacity-60">
      <div className="flex gap-4 mb-5">
        <label className="flex items-center gap-2 cursor-pointer group">
          <input type="radio" name="modelType" checked={modelType === 'rf'} onChange={() => dispatch({ type: 'model_type_changed', modelType: 'rf' })} className="w-4 h-4 shrink-0 accent-brand-600 focus:ring-brand-500" />
          <span className="font-medium text-neutral-800 group-hover:text-brand-700 transition-colors">Random Forest</span>
        </label>
        <label className="flex items-center gap-2 cursor-pointer group">
          <input type="radio" name="modelType" checked={modelType === 'dt'} onChange={() => dispatch({ type: 'model_type_changed', modelType: 'dt' })} className="w-4 h-4 shrink-0 accent-brand-600 focus:ring-brand-500" />
          <span className="font-medium text-neutral-800 group-hover:text-brand-700 transition-colors">Decision Tree</span>
        </label>
      </div>

      {fsRunId && (
        <p className="text-xs text-warning-600 mb-3">
          Đặc trưng hiện tại đến từ một lần chạy live (run {fsRunId}) — không thể "Tải mô hình cũ" (được huấn luyện trên đặc trưng cache khác), chỉ có thể "Huấn luyện" lại trên tập đặc trưng này.
        </p>
      )}

      <Panel className="mb-5 text-sm animate-in fade-in slide-in-from-top-2">
        <div className="flex justify-between items-center mb-3">
          <span className="font-semibold text-neutral-800">Cấu hình huấn luyện</span>
          <Badge tone="brand">Tùy chỉnh</Badge>
        </div>
        <div className="grid grid-cols-1 gap-y-3">
          {modelType === 'rf' && (
            <div className="flex justify-between items-center">
              <span className="text-neutral-600">n_estimators</span>
              <input type="number" value={modelConfig.n_estimators} onChange={e => dispatch({ type: 'model_config_changed', config: {...modelConfig, n_estimators: Number(e.target.value)} })} className="w-24 px-2 py-1 text-right font-mono text-brand-700 font-semibold bg-white border border-neutral-200 rounded text-sm focus:outline-none focus:ring-1 focus:ring-brand-500" />
            </div>
          )}
          <div className="flex justify-between items-center">
            <span className="text-neutral-600">max_depth</span>
            <input type="number" value={modelConfig.max_depth} onChange={e => dispatch({ type: 'model_config_changed', config: {...modelConfig, max_depth: Number(e.target.value)} })} className="w-24 px-2 py-1 text-right font-mono text-brand-700 font-semibold bg-white border border-neutral-200 rounded text-sm focus:outline-none focus:ring-1 focus:ring-brand-500" />
          </div>
          <div className="flex justify-between items-center">
            <span className="text-neutral-600">min_samples_leaf</span>
            <input type="number" value={modelConfig.min_samples_leaf} onChange={e => dispatch({ type: 'model_config_changed', config: {...modelConfig, min_samples_leaf: Number(e.target.value)} })} className="w-24 px-2 py-1 text-right font-mono text-brand-700 font-semibold bg-white border border-neutral-200 rounded text-sm focus:outline-none focus:ring-1 focus:ring-brand-500" />
          </div>
          <div className="flex justify-between items-center">
            <span className="text-neutral-600">class_weight</span>
            <select value={modelConfig.class_weight} onChange={e => dispatch({ type: 'model_config_changed', config: {...modelConfig, class_weight: e.target.value} })} className="w-24 px-2 py-1 text-right font-mono text-brand-700 font-semibold bg-white border border-neutral-200 rounded text-sm focus:outline-none focus:ring-1 focus:ring-brand-500">
              <option value="balanced">balanced</option>
              <option value="balanced_subsample">balanced_subsample</option>
              <option value="none">none</option>
            </select>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-neutral-600">random_state</span>
            <input type="number" value={modelConfig.random_state} onChange={e => dispatch({ type: 'model_config_changed', config: {...modelConfig, random_state: Number(e.target.value)} })} className="w-24 px-2 py-1 text-right font-mono text-brand-700 font-semibold bg-white border border-neutral-200 rounded text-sm focus:outline-none focus:ring-1 focus:ring-brand-500" />
          </div>
        </div>
      </Panel>

      <Panel className="mb-5 text-sm animate-in fade-in slide-in-from-top-2">
        <div className="flex justify-between items-center mb-3">
          <span className="font-semibold text-neutral-800">Cấu hình Lọc Luật</span>
          <Badge tone="brand">Tùy chỉnh</Badge>
        </div>
        <div className="grid grid-cols-1 gap-y-3">
          <div className="flex justify-between items-center">
            <span className="text-neutral-600">min_confidence</span>
            <input type="number" step="0.01" min="0" max="1" value={modelConfig.min_confidence} onChange={e => dispatch({ type: 'model_config_changed', config: {...modelConfig, min_confidence: Number(e.target.value)} })} className="w-24 px-2 py-1 text-right font-mono text-brand-700 font-semibold bg-white border border-neutral-200 rounded text-sm focus:outline-none focus:ring-1 focus:ring-brand-500" />
          </div>
          <div className="flex justify-between items-center">
            <span className="text-neutral-600">min_fidelity</span>
            <input type="number" step="0.01" min="0" max="1" value={modelConfig.min_fidelity} onChange={e => dispatch({ type: 'model_config_changed', config: {...modelConfig, min_fidelity: Number(e.target.value)} })} className="w-24 px-2 py-1 text-right font-mono text-brand-700 font-semibold bg-white border border-neutral-200 rounded text-sm focus:outline-none focus:ring-1 focus:ring-brand-500" />
          </div>
          <div className="flex justify-between items-center">
            <span className="text-neutral-600">min_support</span>
            <input type="number" step="0.01" min="0" max="1" value={modelConfig.min_support} onChange={e => dispatch({ type: 'model_config_changed', config: {...modelConfig, min_support: Number(e.target.value)} })} className="w-24 px-2 py-1 text-right font-mono text-brand-700 font-semibold bg-white border border-neutral-200 rounded text-sm focus:outline-none focus:ring-1 focus:ring-brand-500" />
          </div>
          <div className="flex justify-between items-center">
            <span className="text-neutral-600">min_abs_support</span>
            <input type="number" value={modelConfig.min_abs_support} onChange={e => dispatch({ type: 'model_config_changed', config: {...modelConfig, min_abs_support: Number(e.target.value)} })} className="w-24 px-2 py-1 text-right font-mono text-brand-700 font-semibold bg-white border border-neutral-200 rounded text-sm focus:outline-none focus:ring-1 focus:ring-brand-500" />
          </div>
          <div className="flex justify-between items-center">
            <span className="text-neutral-600">max_conditions</span>
            <input type="number" value={modelConfig.max_conditions} onChange={e => dispatch({ type: 'model_config_changed', config: {...modelConfig, max_conditions: Number(e.target.value)} })} className="w-24 px-2 py-1 text-right font-mono text-brand-700 font-semibold bg-white border border-neutral-200 rounded text-sm focus:outline-none focus:ring-1 focus:ring-brand-500" />
          </div>
          <div className="flex justify-between items-center">
            <span className="text-neutral-600">max_rules_per_class</span>
            <input type="number" value={modelConfig.max_rules_per_class} onChange={e => dispatch({ type: 'model_config_changed', config: {...modelConfig, max_rules_per_class: Number(e.target.value)} })} className="w-24 px-2 py-1 text-right font-mono text-brand-700 font-semibold bg-white border border-neutral-200 rounded text-sm focus:outline-none focus:ring-1 focus:ring-brand-500" />
          </div>
          <div className="flex justify-between items-center">
            <span className="text-neutral-600">max_rules_total</span>
            <input type="number" value={modelConfig.max_rules_total} onChange={e => dispatch({ type: 'model_config_changed', config: {...modelConfig, max_rules_total: Number(e.target.value)} })} className="w-24 px-2 py-1 text-right font-mono text-brand-700 font-semibold bg-white border border-neutral-200 rounded text-sm focus:outline-none focus:ring-1 focus:ring-brand-500" />
          </div>
          <div className="flex justify-between items-center">
            <span className="text-neutral-600">merge_same_gene</span>
            <input type="checkbox" checked={modelConfig.merge_same_gene} onChange={e => dispatch({ type: 'model_config_changed', config: {...modelConfig, merge_same_gene: e.target.checked} })} className="w-4 h-4 accent-brand-600" />
          </div>
          <div className="flex justify-between items-center">
            <span className="text-neutral-600">dedup</span>
            <input type="checkbox" checked={modelConfig.dedup} onChange={e => dispatch({ type: 'model_config_changed', config: {...modelConfig, dedup: e.target.checked} })} className="w-4 h-4 accent-brand-600" />
          </div>
          <div className="flex justify-between items-center">
            <span className="text-neutral-600">dedup_sig_figs</span>
            <input type="number" min={1} max={6} value={modelConfig.dedup_sig_figs} onChange={e => dispatch({ type: 'model_config_changed', config: {...modelConfig, dedup_sig_figs: Number(e.target.value)} })} className="w-24 px-2 py-1 text-right font-mono text-brand-700 font-semibold bg-white border border-neutral-200 rounded text-sm focus:outline-none focus:ring-1 focus:ring-brand-500" />
          </div>
          <div className="flex justify-between items-center">
            <span className="text-neutral-600" title="Loại bỏ luật bị luật khác (cùng bộ gene, cùng lớp) bao trùm hoàn toàn — xem giải thích ở khung chat.">merge_generalization</span>
            <input type="checkbox" checked={modelConfig.merge_generalization} onChange={e => dispatch({ type: 'model_config_changed', config: {...modelConfig, merge_generalization: e.target.checked} })} className="w-4 h-4 accent-brand-600" />
          </div>
        </div>
      </Panel>

      <div className="flex gap-3">
        <Button
          onClick={() => handleModelAction('retrain')}
          disabled={isModelLoading || !extractionStats}
          className="flex-1"
        >
          {isModelLoading ? <Loader2 size={16} className="animate-spin" /> : <PlayCircle size={16} />}
          Huấn luyện lại
        </Button>
        <Button
          variant="secondary"
          onClick={() => handleModelAction('load')}
          disabled={modelLoadDisabled}
          title={fsRunId ? 'Không khả dụng khi đặc trưng hiện tại đến từ một lần chạy live' : undefined}
          className="flex-1"
        >
          <FileText size={16} className="text-neutral-500" />
          Tải mô hình có sẵn
        </Button>
      </div>

      {modelLog && <StatusLine log={modelLog} loading={isModelLoading} />}

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
  );
}

/** Right-column "Tổng Quan Mô Hình" card — stats, hyperparameters/filter
 * config, confusion matrix, classification report and the extracted rules,
 * all as sub-sections of one collapsible card so the test report below isn't
 * buried under them. */
export function ModelOverviewPanel({ state, dispatch }: Props) {
  const {
    datasetId, modelStats, modelType, modelConfig, modelRunId,
    isModelOverviewCollapsed, isModelConfigCollapsed, isConfusionMatrixCollapsed, isClassificationReportCollapsed,
  } = state;
  const fsMethodKey = fsMethodKeyOf(state);
  if (!modelStats) return null;
  const report = classificationReport(modelStats.cm);

  return (
    <div className="bg-white rounded-2xl shadow-sm border border-neutral-200 p-6">
      <div
        className="flex items-center justify-between gap-4 cursor-pointer group"
        onClick={() => dispatch({ type: 'model_overview_toggled' })}
      >
        <div className="flex items-center gap-3 min-w-0">
          <div className="w-10 h-10 shrink-0 rounded-full bg-brand-50 flex items-center justify-center text-brand-600 border border-brand-100">
            <GitMerge size={20} />
          </div>
          <h2 className="text-2xl font-bold text-neutral-900 tracking-tight group-hover:text-brand-700 transition-colors truncate">Tổng Quan Mô Hình</h2>
        </div>
        <button className="p-2 rounded-full hover:bg-brand-50 text-neutral-600 hover:text-brand-600 transition-colors shrink-0">
          {isModelOverviewCollapsed ? <ChevronDown size={24} /> : <ChevronUp size={24} />}
        </button>
      </div>

      {/* Stat badges on their own row below the title — keeping them beside
          the title squeezed the heading on narrower widths. */}
      <div className="flex flex-wrap gap-3 mt-4">
        <div className="bg-white px-5 py-2 rounded-xl border border-neutral-200 shadow-sm flex items-center gap-3">
          <span className="text-[10px] font-bold uppercase tracking-widest text-neutral-500">Accuracy</span>
          <span className="text-xl font-bold text-brand-600">{modelStats.acc}%</span>
        </div>
        <div className="bg-white px-5 py-2 rounded-xl border border-neutral-200 shadow-sm flex items-center gap-3">
          <span className="text-[10px] font-bold uppercase tracking-widest text-neutral-500">F1</span>
          <span className="text-xl font-bold text-brand-600">{modelStats.f1}%</span>
        </div>
        <div className="bg-white px-5 py-2 rounded-xl border border-neutral-200 shadow-sm flex items-center gap-3">
          <span className="text-[10px] font-bold uppercase tracking-widest text-neutral-500">Rules</span>
          <span className="text-xl font-bold text-brand-600">{modelStats.rules}</span>
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
                <div className="bg-white p-6 rounded-2xl shadow-sm border border-neutral-200">
                  <div
                    className="flex items-center justify-between cursor-pointer group/cfg"
                    onClick={() => dispatch({ type: 'model_config_panel_toggled' })}
                  >
                    <div className="flex items-center gap-2">
                      <Settings2 className="text-brand-600" size={20} />
                      <h3 className="text-base font-semibold text-neutral-900 group-hover/cfg:text-brand-700 transition-colors">Tham số cấu hình</h3>
                    </div>
                    <button className="p-1.5 rounded-full hover:bg-brand-50 text-neutral-500 transition-colors shrink-0">
                      {isModelConfigCollapsed ? <ChevronDown size={20} /> : <ChevronUp size={20} />}
                    </button>
                  </div>
                  {!isModelConfigCollapsed && (
                    <div className="mt-6 grid grid-cols-1 lg:grid-cols-2 gap-6 animate-in fade-in slide-in-from-top-2 duration-200">
                      <div className="bg-neutral-50 rounded-xl p-5 border border-neutral-200 h-full flex flex-col">
                        <h4 className="text-sm font-bold text-neutral-900 mb-4 flex items-center gap-2 shrink-0">
                          <span className="w-1.5 h-4 bg-brand-500 rounded-full inline-block"></span>
                          Siêu tham số mô hình ({modelType === 'rf' ? 'Random Forest' : 'Decision Tree'})
                        </h4>
                        <div className="grid grid-cols-1 gap-y-3 flex-1 content-center">
                          {modelStats.hyperparams ? Object.entries(modelStats.hyperparams).map(([k, v]) => (
                            <div key={k} className="flex justify-between gap-3 py-2 border-b border-neutral-200 border-dashed">
                              <span className="text-neutral-600 text-sm font-mono">{k}</span>
                              <span className="text-neutral-900 font-medium text-sm font-mono">{v === null ? 'null' : String(v)}</span>
                            </div>
                          )) : (
                            <p className="text-sm text-neutral-500 italic">Không có dữ liệu.</p>
                          )}
                        </div>
                      </div>
                      <div className="bg-neutral-50 rounded-xl p-5 border border-neutral-200 h-full flex flex-col">
                        <h4 className="text-sm font-bold text-neutral-900 mb-4 flex items-center gap-2 shrink-0">
                          <span className="w-1.5 h-4 bg-info-500 rounded-full inline-block"></span>
                          Cấu hình Lọc Luật
                        </h4>
                        <div className="grid grid-cols-1 gap-y-3 flex-1 content-center">
                          {Object.entries(filterDisplay).map(([k, v]) => (
                            <div key={k} className="flex justify-between gap-3 py-2 border-b border-neutral-200 border-dashed">
                              <span className="text-neutral-600 text-sm font-mono">{k}</span>
                              <span className="text-neutral-900 font-medium text-sm font-mono">{v == null ? '—' : String(v)}</span>
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
            <div className="bg-white p-6 rounded-2xl shadow-sm border border-neutral-200">
              <div
                className="flex items-center justify-between cursor-pointer group/cm"
                onClick={() => dispatch({ type: 'confusion_matrix_toggled' })}
              >
                <div className="flex items-center gap-2">
                  <Table2 className="text-brand-600" size={20} />
                  <h3 className="text-base font-semibold text-neutral-900 group-hover/cm:text-brand-700 transition-colors">Ma trận nhầm lẫn</h3>
                </div>
                <button className="p-1.5 rounded-full hover:bg-brand-50 text-neutral-500 transition-colors shrink-0">
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
            <div className="bg-white p-6 rounded-2xl shadow-sm border border-neutral-200">
              <div
                className="flex items-center justify-between cursor-pointer group/cr"
                onClick={() => dispatch({ type: 'classification_report_toggled' })}
              >
                <div className="flex items-center gap-2">
                  <FileText className="text-brand-600" size={20} />
                  <h3 className="text-base font-semibold text-neutral-900 group-hover/cr:text-brand-700 transition-colors">Chi tiết Phân Loại</h3>
                </div>
                <button className="p-1.5 rounded-full hover:bg-brand-50 text-neutral-500 transition-colors shrink-0">
                  {isClassificationReportCollapsed ? <ChevronDown size={20} /> : <ChevronUp size={20} />}
                </button>
              </div>
              {!isClassificationReportCollapsed && (
                <div className="mt-6 animate-in fade-in slide-in-from-top-2 duration-200">
                  <table className="w-full text-sm text-left table-fixed">
                    <thead className="text-xs text-neutral-500 uppercase bg-neutral-50/80 rounded-t-lg">
                      <tr>
                        <th className="px-3 py-3 font-semibold rounded-tl-lg w-2/5">Lớp (Class)</th>
                        <th className="px-3 py-3 font-semibold">Precision</th>
                        <th className="px-3 py-3 font-semibold">Recall</th>
                        <th className="px-3 py-3 font-semibold">F1</th>
                        <th className="px-3 py-3 font-semibold rounded-tr-lg">Support</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-neutral-100">
                      {modelStats.labels.map((label, idx) => (
                        <tr key={label} className="hover:bg-neutral-50/50 transition-colors">
                          <td className="px-3 py-4 font-medium text-neutral-800 truncate" title={label}>{label}</td>
                          <td className="px-3 py-4 text-neutral-700 font-mono">{report[idx]?.precision.toFixed(2)}</td>
                          <td className="px-3 py-4 text-neutral-700 font-mono">{report[idx]?.recall.toFixed(2)}</td>
                          <td className="px-3 py-4 text-neutral-700 font-mono">{report[idx]?.f1.toFixed(2)}</td>
                          <td className="px-3 py-4 text-neutral-700 font-mono">{report[idx]?.support}</td>
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
            labels={canonicalClassLabels(state)}
          />
        </div>
      )}
    </div>
  );
}
