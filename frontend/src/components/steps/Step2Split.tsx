import React, { useEffect } from 'react';
import { FileText, Loader2, PlayCircle } from 'lucide-react';
import { api, RunRecord, SplitStats } from '../../lib/api';
import { WorkspaceAction, WorkspaceState } from '../../state/types';
import { errorMessage } from '../../lib/utils';
import Button from '../ui/Button';
import Panel from '../ui/Panel';
import Badge from '../ui/Badge';
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
      <Panel className="mb-5 text-sm">
        <div className="flex justify-between items-center mb-3">
          <span className="font-semibold text-neutral-800">Cấu hình chia dữ liệu</span>
          <Badge tone="brand">Tùy chỉnh</Badge>
        </div>
        <div className="grid grid-cols-1 gap-y-3">
          <div className="flex justify-between items-center">
            <span className="text-neutral-600">min_samples_per_class</span>
            <input type="number" min={2} value={splitInputs.min_samples_per_class} onChange={e => dispatch({ type: 'split_inputs_changed', inputs: { ...splitInputs, min_samples_per_class: Number(e.target.value) } })} className="w-24 px-2 py-1 text-right font-mono text-brand-700 font-semibold bg-white border border-neutral-200 rounded text-sm focus:outline-none focus:ring-1 focus:ring-brand-500" />
          </div>
          <div className="flex justify-between items-center">
            <span className="text-neutral-600">test_size</span>
            <input type="number" step="0.01" min={0.05} max={0.5} value={splitInputs.test_size} onChange={e => dispatch({ type: 'split_inputs_changed', inputs: { ...splitInputs, test_size: Number(e.target.value) } })} className="w-24 px-2 py-1 text-right font-mono text-brand-700 font-semibold bg-white border border-neutral-200 rounded text-sm focus:outline-none focus:ring-1 focus:ring-brand-500" />
          </div>
        </div>
      </Panel>

      <div className="flex gap-3">
        <Button
          onClick={() => handleSplitAction('retrain')}
          disabled={isSplitLoading || !datasetId}
          className="flex-1"
        >
          {isSplitLoading ? <Loader2 size={16} className="animate-spin" /> : <PlayCircle size={16} />}
          Thực hiện lại
        </Button>
        <Button
          variant="secondary"
          onClick={() => handleSplitAction('load')}
          disabled={isSplitLoading || !datasetId || selectedDataset?.is_temp}
          title={selectedDataset?.is_temp ? 'Dataset tải lên chưa từng qua xử lý offline nên không có số liệu chia dữ liệu cache — dùng "Thực hiện lại".' : undefined}
          className="flex-1"
        >
          <FileText size={16} className="text-neutral-500" />
          Tải dữ liệu có sẵn
        </Button>
      </div>
      {selectedDataset?.is_temp && (
        <p className="text-xs text-warning-600 mt-2">
          Dataset tải lên chưa có số liệu chia dữ liệu cache — chỉ dùng được "Thực hiện lại".
        </p>
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
