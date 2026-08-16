import React, { useEffect } from 'react';
import { AlertTriangle } from 'lucide-react';
import { FileText, Reload, Play } from '@tailgrids/icons';
import { api, pollJob, RunRecord } from '../../lib/api';
import { fsMethodKeyOf, WorkspaceAction, WorkspaceState } from '../../state/types';
import { errorMessage } from '../../lib/utils';
import { Button } from '../tailgrids/core/button';
import Panel from '../ui/Panel';
import { Badge } from '../tailgrids/core/badge';
import StatusLine from '../ui/StatusLine';
import RunHistoryList from '../RunHistoryList';

interface Props {
  state: WorkspaceState;
  dispatch: React.Dispatch<WorkspaceAction>;
}

/** Bước 3 "Trích xuất đặc trưng" — the API artifact key for mRMR is derived
 * from its K (mrmr_k50, mrmr_k75, ...); Boruta and mRMR share this one step
 * since they're mutually exclusive feature-selection methods. */
export default function Step3FeatureExtraction({ state, dispatch }: Props) {
  const {
    datasets, datasetId, splitParams, splitStats,
    fsMethod, borutaConfig, mrmrConfig, mrmrKMode, fsLog, fsRuns, fsRunId, extractionStats,
    isSplitLoading, isFsLoading, isModelLoading, isTesting,
  } = state;
  const isMrmr = fsMethod === 'mrmr';
  const fsMethodKey = fsMethodKeyOf(state);
  const fsSectionLocked = isSplitLoading || isFsLoading || isModelLoading || isTesting;
  const selectedDataset = datasets.find(d => d.id === datasetId);

  useEffect(() => {
    let ignore = false;
    if (datasetId) {
      api.getRuns(datasetId, 'feature_selection').then(runs => { if (!ignore) dispatch({ type: 'fs_runs_loaded', runs }); }).catch(() => { if (!ignore) dispatch({ type: 'fs_runs_loaded', runs: [] }); });
    } else {
      dispatch({ type: 'fs_runs_loaded', runs: [] });
    }
    return () => { ignore = true; };
  }, [datasetId, dispatch]);

  const refreshFsRuns = () => {
    if (!datasetId) return;
    api.getRuns(datasetId, 'feature_selection').then(runs => dispatch({ type: 'fs_runs_loaded', runs })).catch(() => {});
  };

  const handleFsAction = async (action: 'retrain' | 'load') => {
    if (!datasetId) return;
    try {
      if (action === 'load') {
        dispatch({ type: 'fs_started', log: `[HỆ THỐNG] Đang tải kết quả log trước đó cho thuật toán ${fsMethodKey.toUpperCase()}...` });
        const stats = await api.getFeatureSelection(datasetId, fsMethodKey);
        dispatch({ type: 'fs_computed', stats, runId: null });
        dispatch({ type: 'fs_log_appended', line: '[CACHE] Đã nạp thành công.' });
      } else {
        // Keep the requested K and the artifact key (mrmr_k{K}) aligned.
        const params = isMrmr ? mrmrConfig : borutaConfig;
        dispatch({ type: 'fs_started', log: `[HỆ THỐNG] Bắt đầu chạy thuật toán ${fsMethodKey.toUpperCase()} (chạy thật, có thể mất vài phút)...` });
        const { job_id } = await api.trainFeatureSelection(datasetId, fsMethodKey, params, splitParams);
        const job = await pollJob(job_id, j => dispatch({ type: 'fs_log_set', log: j.log.join('\n') }));
        if (job.status === 'error') {
          throw new Error(job.error || 'Trich xuat dac trung that bai (xem log server).');
        }
        const stats = await api.getFeatureSelection(datasetId, fsMethodKey, job_id);
        dispatch({ type: 'fs_computed', stats, runId: job_id });
        refreshFsRuns();
      }
    } catch (e) {
      dispatch({ type: 'fs_failed', line: `[LỖI] ${errorMessage(e)}` });
    }
  };

  const loadPastFsRun = async (run: RunRecord) => {
    if (!datasetId || !run.fs_method) return;
    dispatch({ type: 'fs_started', log: `[HỆ THỐNG] Đang tải lại kết quả chạy trước (${run.run_id})...` });
    try {
      const stats = await api.getFeatureSelection(datasetId, run.fs_method, run.run_id);
      dispatch({ type: 'fs_computed', stats, runId: run.run_id });
      dispatch({ type: 'fs_log_appended', line: '[OK] Đã nạp lại kết quả chạy trước.' });
    } catch (e) {
      dispatch({ type: 'fs_failed', line: `[LỖI] ${errorMessage(e)}` });
    }
  };

  return (
    <fieldset disabled={fsSectionLocked} className="border-0 p-0 m-0 min-w-0 disabled:opacity-60">
      <div className="flex gap-4 mb-5">
        <label className="flex items-center gap-2 cursor-pointer group">
          <input type="radio" name="fsMethod" checked={fsMethod === 'boruta'} onChange={() => dispatch({ type: 'fs_method_changed', method: 'boruta' })} className="w-4 h-4 shrink-0 accent-brand-600 focus:ring-brand-500" />
          <span className="font-medium text-neutral-800 group-hover:text-brand-700 transition-colors">Boruta</span>
        </label>
        <label className="flex items-center gap-2 cursor-pointer group">
          <input type="radio" name="fsMethod" checked={fsMethod === 'mrmr'} onChange={() => dispatch({ type: 'fs_method_changed', method: 'mrmr' })} className="w-4 h-4 shrink-0 accent-brand-600 focus:ring-brand-500" />
          <span className="font-medium text-neutral-800 group-hover:text-brand-700 transition-colors">mRMR</span>
        </label>
      </div>

      {selectedDataset && !selectedDataset.fs_models[fsMethodKey] && (
        <p className="text-xs text-warning-600 mb-3">Dataset này chưa có kết quả cache cho {fsMethodKey} — chỉ có thể dùng "Huấn luyện".</p>
      )}

      {fsMethod === 'boruta' && (
        <Panel className="mb-5 text-sm animate-in fade-in slide-in-from-top-2">
          <div className="flex justify-between items-center mb-3">
            <span className="font-semibold text-neutral-800">Cấu hình Boruta</span>
            <Badge color="primary">Tùy chỉnh</Badge>
          </div>
          <div className="grid grid-cols-1 gap-y-3">
             <div className="flex justify-between items-center">
               <span className="text-neutral-600">n_estimators</span>
               <input type="text" value={borutaConfig.n_estimators} onChange={e => dispatch({ type: 'boruta_config_changed', config: {...borutaConfig, n_estimators: e.target.value} })} className="w-24 px-2 py-1 text-right font-mono text-brand-700 font-semibold bg-white border border-neutral-200 rounded text-sm focus:outline-none focus:ring-1 focus:ring-brand-500" />
             </div>
             <div className="flex justify-between items-center">
               <span className="text-neutral-600">rf_n_estimators</span>
               <input type="number" value={borutaConfig.rf_n_estimators} onChange={e => dispatch({ type: 'boruta_config_changed', config: {...borutaConfig, rf_n_estimators: Number(e.target.value)} })} className="w-24 px-2 py-1 text-right font-mono text-brand-700 font-semibold bg-white border border-neutral-200 rounded text-sm focus:outline-none focus:ring-1 focus:ring-brand-500" />
             </div>
             <div className="flex justify-between items-center">
               <span className="text-neutral-600">max_depth</span>
               <input type="text" placeholder="null" value={borutaConfig.max_depth} onChange={e => dispatch({ type: 'boruta_config_changed', config: {...borutaConfig, max_depth: e.target.value} })} className="w-24 px-2 py-1 text-right font-mono text-brand-700 font-semibold bg-white border border-neutral-200 rounded text-sm focus:outline-none focus:ring-1 focus:ring-brand-500" />
             </div>
             <div className="flex justify-between items-center">
               <span className="text-neutral-600">max_iter</span>
               <input type="number" value={borutaConfig.max_iter} onChange={e => dispatch({ type: 'boruta_config_changed', config: {...borutaConfig, max_iter: Number(e.target.value)} })} className="w-24 px-2 py-1 text-right font-mono text-brand-700 font-semibold bg-white border border-neutral-200 rounded text-sm focus:outline-none focus:ring-1 focus:ring-brand-500" />
             </div>
             <div className="flex justify-between items-center">
               <span className="text-neutral-600">perc</span>
               <input type="number" value={borutaConfig.perc} onChange={e => dispatch({ type: 'boruta_config_changed', config: {...borutaConfig, perc: Number(e.target.value)} })} className="w-24 px-2 py-1 text-right font-mono text-brand-700 font-semibold bg-white border border-neutral-200 rounded text-sm focus:outline-none focus:ring-1 focus:ring-brand-500" />
             </div>
             <div className="flex justify-between items-center">
               <span className="text-neutral-600">alpha</span>
               <input type="number" step="0.01" value={borutaConfig.alpha} onChange={e => dispatch({ type: 'boruta_config_changed', config: {...borutaConfig, alpha: Number(e.target.value)} })} className="w-24 px-2 py-1 text-right font-mono text-brand-700 font-semibold bg-white border border-neutral-200 rounded text-sm focus:outline-none focus:ring-1 focus:ring-brand-500" />
             </div>
             <div className="flex justify-between items-center">
               <span className="text-neutral-600">class_weight</span>
               <select value={borutaConfig.class_weight} onChange={e => dispatch({ type: 'boruta_config_changed', config: {...borutaConfig, class_weight: e.target.value} })} className="w-24 px-2 py-1 text-right font-mono text-brand-700 font-semibold bg-white border border-neutral-200 rounded text-sm focus:outline-none focus:ring-1 focus:ring-brand-500">
                 <option value="balanced">balanced</option>
                 <option value="balanced_subsample">balanced_subsample</option>
                 <option value="none">none</option>
               </select>
             </div>
             <div className="flex justify-between items-center">
               <span className="text-neutral-600">random_state</span>
               <input type="number" value={borutaConfig.random_state} onChange={e => dispatch({ type: 'boruta_config_changed', config: {...borutaConfig, random_state: Number(e.target.value)} })} className="w-24 px-2 py-1 text-right font-mono text-brand-700 font-semibold bg-white border border-neutral-200 rounded text-sm focus:outline-none focus:ring-1 focus:ring-brand-500" />
             </div>
             <div className="flex justify-between items-center">
               <span className="text-neutral-600">Chế độ chọn đặc trưng</span>
               <select
                 value={borutaConfig.selection_mode}
                 onChange={e => dispatch({ type: 'boruta_config_changed', config: {...borutaConfig, selection_mode: e.target.value as typeof borutaConfig.selection_mode} })}
                 className="w-40 px-2 py-1 text-right font-mono text-brand-700 font-semibold bg-white border border-neutral-200 rounded text-sm focus:outline-none focus:ring-1 focus:ring-brand-500"
               >
                 <option value="confirmed">confirmed</option>
                 <option value="confirmed_tentative">confirmed_tentative</option>
                 <option value="top_k">top_k</option>
               </select>
             </div>
             {borutaConfig.selection_mode === 'top_k' && (
               <div className="flex justify-between items-center">
                 <span className="text-neutral-600">k</span>
                 <input
                   type="number"
                   min={1}
                   placeholder="số đặc trưng"
                   value={borutaConfig.k}
                   onChange={e => dispatch({ type: 'boruta_config_changed', config: {...borutaConfig, k: e.target.value === '' ? '' : Number(e.target.value)} })}
                   className="w-24 px-2 py-1 text-right font-mono text-brand-700 font-semibold bg-white border border-neutral-200 rounded text-sm focus:outline-none focus:ring-1 focus:ring-brand-500"
                 />
               </div>
             )}
          </div>
        </Panel>
      )}

      {isMrmr && (
        <Panel className="mb-5 text-sm animate-in fade-in slide-in-from-top-2">
          <div className="flex justify-between items-center mb-3">
            <span className="font-semibold text-neutral-800">Cấu hình mRMR</span>
            <Badge color="primary">Tùy chỉnh</Badge>
          </div>
          <div className="grid grid-cols-1 gap-y-3">
             <div className="flex justify-between items-center">
               <span className="text-neutral-600">criterion</span>
               <input type="text" value={mrmrConfig.criterion} readOnly className="w-24 px-2 py-1 text-right font-mono text-neutral-600 font-semibold bg-neutral-100 border border-neutral-200 rounded text-sm focus:outline-none cursor-not-allowed" />
             </div>
             <div className="flex justify-between items-center">
               <span className="text-neutral-600">K (features)</span>
               <select
                 value={mrmrKMode}
                 onChange={e => {
                   const mode = e.target.value as typeof mrmrKMode;
                   dispatch({ type: 'mrmr_k_mode_changed', mode });
                   if (mode !== 'custom') {
                     dispatch({ type: 'mrmr_config_changed', config: { ...mrmrConfig, K: Number(mode) } });
                   }
                 }}
                 className="w-24 px-2 py-1 font-mono text-brand-700 font-semibold bg-white border border-neutral-200 rounded text-sm focus:outline-none focus:ring-1 focus:ring-brand-500"
               >
                 <option value="50">50</option>
                 <option value="75">75</option>
                 <option value="custom">K mới</option>
               </select>
             </div>
             {mrmrKMode === 'custom' && (
               <div className="flex justify-between items-center">
                 <span className="text-neutral-600">Giá trị K</span>
                 <input
                   type="number"
                   min={1}
                   value={mrmrConfig.K}
                   onChange={e => dispatch({ type: 'mrmr_config_changed', config: { ...mrmrConfig, K: Math.max(1, Number(e.target.value) || 1) } })}
                   className="w-24 px-2 py-1 text-right font-mono text-brand-700 font-semibold bg-white border border-neutral-200 rounded text-sm focus:outline-none focus:ring-1 focus:ring-brand-500"
                   aria-label="Giá trị K mới"
                 />
               </div>
             )}
             <div className="flex justify-between items-center">
               <span className="text-neutral-600">n_bins</span>
               <input type="number" value={mrmrConfig.n_bins} onChange={e => dispatch({ type: 'mrmr_config_changed', config: {...mrmrConfig, n_bins: Number(e.target.value)} })} className="w-24 px-2 py-1 text-right font-mono text-brand-700 font-semibold bg-white border border-neutral-200 rounded text-sm focus:outline-none focus:ring-1 focus:ring-brand-500" />
             </div>
             <div className="flex justify-between items-center">
               <span className="text-neutral-600">random_state</span>
               <input type="number" value={mrmrConfig.random_state} onChange={e => dispatch({ type: 'mrmr_config_changed', config: {...mrmrConfig, random_state: Number(e.target.value)} })} className="w-24 px-2 py-1 text-right font-mono text-brand-700 font-semibold bg-white border border-neutral-200 rounded text-sm focus:outline-none focus:ring-1 focus:ring-brand-500" />
             </div>
          </div>
        </Panel>
      )}

      <div className="flex gap-3">
        <Button
          onClick={() => handleFsAction('retrain')}
          disabled={isFsLoading || !splitStats}
          className="flex-1"
        >
          {isFsLoading ? <Reload size={16} className="animate-spin" /> : <Play size={16} />}
          Trích xuất lại
        </Button>
        <Button
          variant="primary" appearance="outline"
          onClick={() => handleFsAction('load')}
          disabled={isFsLoading || !splitStats || !selectedDataset?.fs_models[fsMethodKey]}
          className="flex-1"
        >
          <FileText size={16} className="text-neutral-500" />
          Tải kết quả có sẵn
        </Button>
      </div>

      {fsLog && <StatusLine log={fsLog} loading={isFsLoading} />}

      {fsMethod === 'boruta' && (extractionStats?.selection_mode ?? borutaConfig.selection_mode) === 'confirmed' &&
        extractionStats && (extractionStats.confirmed ?? extractionStats.n_selected_features) < 5 && (
        <div className="mt-4 flex items-start gap-2 bg-warning-50 border border-warning-200 rounded-lg px-3 py-2.5 text-warning-800">
          <AlertTriangle size={14} className="text-warning-500 shrink-0 mt-0.5" />
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
  );
}
