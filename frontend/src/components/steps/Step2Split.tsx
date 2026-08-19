import React, { useEffect } from 'react';
import { AlertTriangle } from 'lucide-react';
import { FileText, Reload, Play } from '@tailgrids/icons';
import { api, RunRecord, SplitStats } from '../../lib/api';
import { WorkspaceAction, WorkspaceState } from '../../state/types';
import { errorMessage } from '../../lib/utils';
import { Button } from '../tailgrids/core/button';
import { Input } from '../tailgrids/core/input';
import Panel from '../ui/Panel';
import StatusLine from '../ui/StatusLine';
import RunHistoryList from '../RunHistoryList';

interface Props {
  state: WorkspaceState;
  dispatch: React.Dispatch<WorkspaceAction>;
}

/** Bước 2 "Xử lý & Chia Dữ liệu" — splitParams=null means "dùng mặc định của
 * holdout.yaml"; non-null (từ "Thực hiện lại") flows down into every
 * FS/model/predict call so they all read the SAME split. */
export default function Step2Split({ state, dispatch }: Props) {
  const {
    datasets, datasetId, splitInputs, splitLog, splitRuns,
    isSplitLoading, isFsLoading, isModelLoading, isTesting,
  } = state;
  const splitSectionLocked = isFsLoading || isModelLoading || isTesting;
  const selectedDataset = datasets.find(d => d.id === datasetId);

  useEffect(() => {
    let ignore = false;
    if (datasetId) {
      api.getRuns(datasetId, 'split').then(runs => { if (!ignore) dispatch({ type: 'split_runs_loaded', runs }); }).catch(() => { if (!ignore) dispatch({ type: 'split_runs_loaded', runs: [] }); });
    } else {
      dispatch({ type: 'split_runs_loaded', runs: [] });
    }
    return () => { ignore = true; };
  }, [datasetId, dispatch]);

  const refreshSplitRuns = (id: string) => {
    api.getRuns(id, 'split').then(runs => dispatch({ type: 'split_runs_loaded', runs })).catch(() => {});
  };

  const handleSplitAction = async (action: 'retrain' | 'load') => {
    if (!datasetId) return;
    try {
      if (action === 'load') {
        dispatch({ type: 'split_started', log: '[HỆ THỐNG] Đang tải số liệu chia dữ liệu (mặc định)...' });
        let stats: SplitStats;
        try {
          stats = await api.getOverview(datasetId);
        } catch {
          stats = await api.splitPreview(datasetId, {});
        }
        dispatch({
          type: 'split_computed',
          stats,
          params: null,
          inputs: { min_samples_per_class: stats.min_samples_per_class, test_size: stats.test_size },
        });
        dispatch({ type: 'split_log_appended', line: '[CACHE] Đã nạp thành công.' });
      } else {
        dispatch({ type: 'split_started', log: '[HỆ THỐNG] Đang tính lại chia dữ liệu (train/test)...' });
        const stats = await api.splitPreview(datasetId, splitInputs);
        dispatch({ type: 'split_computed', stats, params: { ...splitInputs }, inputs: splitInputs });
        dispatch({ type: 'split_log_appended', line: '[OK] Đã tính lại thành công.' });
      }
      refreshSplitRuns(datasetId);
    } catch (e) {
      dispatch({ type: 'split_failed', line: `[LỖI] ${errorMessage(e)}` });
    }
  };

  // Loads a past split run directly from its stored summary — no recompute,
  // no new history entry — same "load" semantics loadPastFsRun/loadPastModelRun
  // already use (those GET a cached file; this reads the full summary the
  // /split/preview call already saved into runs_index.json).
  const loadPastSplitRun = (run: RunRecord) => {
    if (!datasetId) return;
    const params = {
      min_samples_per_class: run.summary.min_samples_per_class as number,
      test_size: run.summary.test_size as number,
    };
    dispatch({
      type: 'split_run_loaded',
      stats: run.summary as unknown as SplitStats,
      params,
      log: `[HỆ THỐNG] Đang tải lại kết quả chạy trước (${run.run_id})...\n[OK] Đã nạp lại kết quả chạy trước.`,
    });
  };

  return (
    <fieldset disabled={splitSectionLocked} className="border-0 p-0 m-0 min-w-0 disabled:opacity-60">
      <div className="space-y-3 mb-5">
        <div>
          <label className="text-sm text-neutral-600 font-medium block mb-1">min_samples_per_class</label>
          <Input type="number" min={2} value={splitInputs.min_samples_per_class} onChange={e => dispatch({ type: 'split_inputs_changed', inputs: { ...splitInputs, min_samples_per_class: Number(e.target.value) } })} className="w-full text-sm" />
        </div>
        <div>
          <label className="text-sm text-neutral-600 font-medium block mb-1">test_size</label>
          <Input type="number" step="0.01" min={0.05} max={0.5} value={splitInputs.test_size} onChange={e => dispatch({ type: 'split_inputs_changed', inputs: { ...splitInputs, test_size: Number(e.target.value) } })} className="w-full text-sm" />
        </div>
      </div>

      <div className="flex flex-col gap-3">
        <Button
          onClick={() => handleSplitAction('retrain')}
          disabled={isSplitLoading || !datasetId}
          className="w-full"
        >
          {isSplitLoading ? <Reload size={16} className="animate-spin" /> : <Play size={16} />}
          Thực hiện lại
        </Button>
        <Button
          variant="primary" appearance="outline"
          onClick={() => handleSplitAction('load')}
          disabled={isSplitLoading || !datasetId || selectedDataset?.is_temp}
          title={selectedDataset?.is_temp ? 'Dataset tải lên chưa từng qua xử lý offline nên không có số liệu chia dữ liệu cache — dùng "Thực hiện lại".' : undefined}
          className="w-full"
        >
          <FileText size={16} className="text-neutral-600" />
          Tải dữ liệu có sẵn
        </Button>
      </div>
      {selectedDataset?.is_temp && (
        <Panel surface="warning" padding="xs" className="mt-3 flex gap-2.5">
          <AlertTriangle size={14} className="text-warning-800 shrink-0 mt-0.5" />
          <p className="text-xs text-warning-800 leading-relaxed">
            Dataset tải lên chưa có số liệu chia dữ liệu cache — chỉ dùng được "Thực hiện lại".
          </p>
        </Panel>
      )}

      {splitLog && <StatusLine log={splitLog} loading={isSplitLoading} />}

      <RunHistoryList
        title="Lịch sử xử lý và phân chia dữ liệu"
        runs={splitRuns}
        disabled={isSplitLoading}
        onSelect={loadPastSplitRun}
        renderLabel={run => `min=${String(run.summary?.min_samples_per_class)} test=${String(run.summary?.test_size)} · ${run.run_id}`}
        renderStatus={run => `${run.summary?.n_train ?? '?'}/${run.summary?.n_test ?? '?'}`}
      />
    </fieldset>
  );
}
