import React, { useEffect } from 'react';
import { AlertTriangle } from 'lucide-react';
import { FileText, Reload, Play } from '@tailgrids/icons';
import { api, pollJob, RunRecord } from '../../lib/api';
import { fsMethodKeyOf, WorkspaceAction, WorkspaceState } from '../../state/types';
import { errorMessage } from '../../lib/utils';
import { Button } from '../tailgrids/core/button';
import { Input } from '../tailgrids/core/input';
import { RadioInput } from '../tailgrids/core/radio-input';
import { Select, SelectContent, SelectIndicator, SelectItem, SelectTrigger, SelectValue } from '../tailgrids/core/select';
import Panel from '../ui/Panel';
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
          <RadioInput name="fsMethod" checked={fsMethod === 'boruta'} onChange={() => dispatch({ type: 'fs_method_changed', method: 'boruta' })} />
          <span className="font-medium text-neutral-800 group-hover:text-brand-700 transition-colors">Boruta</span>
        </label>
        <label className="flex items-center gap-2 cursor-pointer group">
          <RadioInput name="fsMethod" checked={fsMethod === 'mrmr'} onChange={() => dispatch({ type: 'fs_method_changed', method: 'mrmr' })} />
          <span className="font-medium text-neutral-800 group-hover:text-brand-700 transition-colors">mRMR</span>
        </label>
      </div>

      {selectedDataset && !selectedDataset.fs_models[fsMethodKey] && (
        <Panel surface="warning" padding="xs" className="mb-4 flex gap-2.5">
          <AlertTriangle size={14} className="text-warning-800 shrink-0 mt-0.5" />
          <p className="text-xs text-warning-800 leading-relaxed">
            Dataset này chưa có kết quả cache cho {fsMethodKey} — chỉ có thể dùng "Huấn luyện".
          </p>
        </Panel>
      )}

      {fsMethod === 'boruta' && (
        <div className="space-y-3 mb-5 animate-in fade-in slide-in-from-top-2">
          <div>
            <label className="text-sm text-neutral-600 font-medium block mb-1">n_estimators</label>
            <Input value={borutaConfig.n_estimators} onChange={e => dispatch({ type: 'boruta_config_changed', config: {...borutaConfig, n_estimators: e.target.value} })} className="w-full text-sm" />
          </div>
          <div>
            <label className="text-sm text-neutral-600 font-medium block mb-1">rf_n_estimators</label>
            <Input type="number" value={borutaConfig.rf_n_estimators} onChange={e => dispatch({ type: 'boruta_config_changed', config: {...borutaConfig, rf_n_estimators: Number(e.target.value)} })} className="w-full text-sm" />
          </div>
          <div>
            <label className="text-sm text-neutral-600 font-medium block mb-1">max_depth</label>
            <Input placeholder="null" value={borutaConfig.max_depth} onChange={e => dispatch({ type: 'boruta_config_changed', config: {...borutaConfig, max_depth: e.target.value} })} className="w-full text-sm" />
          </div>
          <div>
            <label className="text-sm text-neutral-600 font-medium block mb-1">max_iter</label>
            <Input type="number" value={borutaConfig.max_iter} onChange={e => dispatch({ type: 'boruta_config_changed', config: {...borutaConfig, max_iter: Number(e.target.value)} })} className="w-full text-sm" />
          </div>
          <div>
            <label className="text-sm text-neutral-600 font-medium block mb-1">perc</label>
            <Input type="number" value={borutaConfig.perc} onChange={e => dispatch({ type: 'boruta_config_changed', config: {...borutaConfig, perc: Number(e.target.value)} })} className="w-full text-sm" />
          </div>
          <div>
            <label className="text-sm text-neutral-600 font-medium block mb-1">alpha</label>
            <Input type="number" step="0.01" value={borutaConfig.alpha} onChange={e => dispatch({ type: 'boruta_config_changed', config: {...borutaConfig, alpha: Number(e.target.value)} })} className="w-full text-sm" />
          </div>
          <div>
            <label className="text-sm text-neutral-600 font-medium block mb-1">class_weight</label>
            <Select
              aria-label="class_weight"
              value={borutaConfig.class_weight}
              onChange={key => dispatch({ type: 'boruta_config_changed', config: {...borutaConfig, class_weight: String(key)} })}
              className="w-full"
            >
              <SelectTrigger className="w-full">
                <SelectValue />
                <SelectIndicator />
              </SelectTrigger>
              <SelectContent>
                <SelectItem id="balanced" textValue="balanced">balanced</SelectItem>
                <SelectItem id="balanced_subsample" textValue="balanced_subsample">balanced_subsample</SelectItem>
                <SelectItem id="none" textValue="none">none</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div>
            <label className="text-sm text-neutral-600 font-medium block mb-1">random_state</label>
            <Input type="number" value={borutaConfig.random_state} onChange={e => dispatch({ type: 'boruta_config_changed', config: {...borutaConfig, random_state: Number(e.target.value)} })} className="w-full text-sm" />
          </div>
          <div>
            <label className="text-sm text-neutral-600 font-medium block mb-1">Chế độ chọn đặc trưng</label>
            <Select
              aria-label="Chế độ chọn đặc trưng"
              value={borutaConfig.selection_mode}
              onChange={key => dispatch({ type: 'boruta_config_changed', config: {...borutaConfig, selection_mode: String(key) as typeof borutaConfig.selection_mode} })}
              className="w-full"
            >
              <SelectTrigger className="w-full">
                <SelectValue />
                <SelectIndicator />
              </SelectTrigger>
              <SelectContent>
                <SelectItem id="confirmed" textValue="confirmed">confirmed</SelectItem>
                <SelectItem id="confirmed_tentative" textValue="confirmed_tentative">confirmed_tentative</SelectItem>
                <SelectItem id="top_k" textValue="top_k">top_k</SelectItem>
              </SelectContent>
            </Select>
          </div>
          {borutaConfig.selection_mode === 'top_k' && (
            <div>
              <label className="text-sm text-neutral-600 font-medium block mb-1">k</label>
              <Input
                type="number"
                min={1}
                placeholder="số đặc trưng"
                value={borutaConfig.k}
                onChange={e => dispatch({ type: 'boruta_config_changed', config: {...borutaConfig, k: e.target.value === '' ? '' : Number(e.target.value)} })}
                className="w-full text-sm"
                aria-label="k"
              />
            </div>
          )}
        </div>
      )}

      {isMrmr && (
        <div className="space-y-3 mb-5 animate-in fade-in slide-in-from-top-2">
          <div>
            <label className="text-sm text-neutral-600 font-medium block mb-1">criterion</label>
            <Input value={mrmrConfig.criterion} readOnly disabled className="w-full text-sm" />
          </div>
          <div>
            <label className="text-sm text-neutral-600 font-medium block mb-1">K (features)</label>
            <Select
              aria-label="K (features)"
              value={mrmrKMode}
              onChange={key => {
                const mode = String(key) as typeof mrmrKMode;
                dispatch({ type: 'mrmr_k_mode_changed', mode });
                if (mode !== 'custom') {
                  dispatch({ type: 'mrmr_config_changed', config: { ...mrmrConfig, K: Number(mode) } });
                }
              }}
              className="w-full"
            >
              <SelectTrigger className="w-full">
                <SelectValue />
                <SelectIndicator />
              </SelectTrigger>
              <SelectContent>
                <SelectItem id="50" textValue="50">50</SelectItem>
                <SelectItem id="75" textValue="75">75</SelectItem>
                <SelectItem id="custom" textValue="K mới">K mới</SelectItem>
              </SelectContent>
            </Select>
          </div>
          {mrmrKMode === 'custom' && (
            <div>
              <label className="text-sm text-neutral-600 font-medium block mb-1">Giá trị K</label>
              <Input
                type="number"
                min={1}
                value={mrmrConfig.K}
                onChange={e => dispatch({ type: 'mrmr_config_changed', config: { ...mrmrConfig, K: Math.max(1, Number(e.target.value) || 1) } })}
                className="w-full text-sm"
                aria-label="Giá trị K mới"
              />
            </div>
          )}
          <div>
            <label className="text-sm text-neutral-600 font-medium block mb-1">n_bins</label>
            <Input type="number" value={mrmrConfig.n_bins} onChange={e => dispatch({ type: 'mrmr_config_changed', config: {...mrmrConfig, n_bins: Number(e.target.value)} })} className="w-full text-sm" />
          </div>
          <div>
            <label className="text-sm text-neutral-600 font-medium block mb-1">random_state</label>
            <Input type="number" value={mrmrConfig.random_state} onChange={e => dispatch({ type: 'mrmr_config_changed', config: {...mrmrConfig, random_state: Number(e.target.value)} })} className="w-full text-sm" />
          </div>
        </div>
      )}

      <div className="flex flex-col gap-3">
        <Button
          onClick={() => handleFsAction('retrain')}
          disabled={isFsLoading || !splitStats}
          className="w-full"
        >
          {isFsLoading ? <Reload size={16} className="animate-spin" /> : <Play size={16} />}
          Trích xuất lại
        </Button>
        <Button
          variant="primary" appearance="outline"
          onClick={() => handleFsAction('load')}
          disabled={isFsLoading || !splitStats || !selectedDataset?.fs_models[fsMethodKey]}
          className="w-full"
        >
          <FileText size={16} className="text-neutral-600" />
          Tải kết quả có sẵn
        </Button>
      </div>

      {fsLog && <StatusLine log={fsLog} loading={isFsLoading} />}

      {fsMethod === 'boruta' && (extractionStats?.selection_mode ?? borutaConfig.selection_mode) === 'confirmed' &&
        extractionStats && (extractionStats.confirmed ?? extractionStats.n_selected_features) < 5 && (
        <Panel surface="warning" padding="xs" className="mt-4 flex gap-2.5">
          <AlertTriangle size={14} className="text-warning-800 shrink-0 mt-0.5" />
          <p className="text-xs text-warning-800 leading-relaxed">
            Số đặc trưng "confirmed" quá ít ({extractionStats.confirmed ?? extractionStats.n_selected_features}) —
            có thể không đủ để huấn luyện mô hình rule tốt. Thử "confirmed_tentative" hoặc "top_k".
          </p>
        </Panel>
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
