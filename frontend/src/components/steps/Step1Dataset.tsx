import React, { useState, useEffect } from 'react';
import { FileText, Loader2, ListChecks, PlayCircle, UploadCloud } from 'lucide-react';
import { cn } from '../../lib/utils';
import { api, BuiltDatasetResult, pollJob, RunRecord } from '../../lib/api';
import { WorkspaceAction, WorkspaceState } from '../../state/types';
import { errorMessage } from '../../lib/utils';
import Button from '../ui/Button';
import StatusLine from '../ui/StatusLine';
import RunHistoryList from '../RunHistoryList';

interface Props {
  state: WorkspaceState;
  dispatch: React.Dispatch<WorkspaceAction>;
}

/** Bước 1 "Chọn Dữ Liệu" — pick an already-processed dataset, or build a new
 * one from a fresh GEO/CuMiDa upload. The upload wizard's own fields
 * (uploadSource..uploadLog below) are session-only and never read outside
 * this step, so they stay local instead of joining the shared reducer. */
export default function Step1Dataset({ state, dispatch }: Props) {
  const { datasets, datasetId, datasetsError, datasetTab, isSplitLoading, isFsLoading, isModelLoading, isTesting } = state;
  const datasetLocked = isSplitLoading || isFsLoading || isModelLoading || isTesting;
  const selectedDataset = datasets.find(d => d.id === datasetId);

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

  // Lịch sử tải lên — every past upload across the whole session, not scoped
  // to whichever dataset happens to be selected right now (same reasoning
  // GET /api/datasets/upload/history isn't dataset-scoped).
  useEffect(() => {
    api.getUploadHistory().then(setUploadRuns).catch(() => {});
  }, []);

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
    } catch (e) {
      setUploadLog(prev => prev + `\n[LỖI] ${errorMessage(e)}`);
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
      const built = job.result as unknown as BuiltDatasetResult;
      dispatch({
        type: 'dataset_added',
        dataset: {
          id: built.dataset_id, name: built.name, platform: built.platform,
          n_samples: built.n_samples, n_features: built.n_features, n_classes: built.n_classes,
          class_labels: built.class_labels, description: built.description, fs_models: {},
          is_temp: true,
        },
      });
      dispatch({ type: 'dataset_selected', datasetId: built.dataset_id });
      dispatch({ type: 'dataset_tab_changed', tab: 'existing' });
      setUploadLog(prev => prev + `\n[THÀNH CÔNG] Dataset '${built.dataset_id}' đã sẵn sàng sử dụng.`);
      api.getUploadHistory().then(setUploadRuns).catch(() => {});
    } catch (e) {
      setUploadLog(prev => prev + `\n[LỖI] ${errorMessage(e)}`);
      api.getUploadHistory().then(setUploadRuns).catch(() => {});
    } finally {
      setIsUploadBuilding(false);
    }
  };

  return (
    <>
      {datasetsError && (
        <p className="text-sm text-danger-600 mb-3">Không tải được danh sách dataset: {datasetsError}</p>
      )}

      <div className="flex gap-2 p-1 bg-neutral-100 rounded-lg text-sm mb-4">
        <button
          onClick={() => dispatch({ type: 'dataset_tab_changed', tab: 'existing' })}
          disabled={datasetLocked}
          className={cn("flex-1 py-1.5 rounded-md flex items-center justify-center gap-1.5 transition-colors", datasetTab === 'existing' ? "bg-white shadow-sm text-brand-700 font-medium" : "text-neutral-600")}
        >
          <ListChecks size={14} /> Chọn dataset có sẵn
        </button>
        <button
          onClick={() => dispatch({ type: 'dataset_tab_changed', tab: 'upload' })}
          disabled={datasetLocked}
          className={cn("flex-1 py-1.5 rounded-md flex items-center justify-center gap-1.5 transition-colors", datasetTab === 'upload' ? "bg-white shadow-sm text-brand-700 font-medium" : "text-neutral-600")}
        >
          <UploadCloud size={14} /> Tải lên dataset mới
        </button>
      </div>

      {datasetTab === 'existing' ? (
        <>
          <select
            value={datasetId}
            onChange={(e) => dispatch({ type: 'dataset_selected', datasetId: e.target.value })}
            disabled={datasetLocked}
            className="w-full bg-neutral-50 border border-neutral-200 text-neutral-800 rounded-lg px-4 py-2.5 focus:outline-none focus:ring-2 focus:ring-brand-500/30 focus:border-brand-500 transition-colors shadow-sm disabled:opacity-60 disabled:cursor-not-allowed"
          >
            <option value="">-- Chọn Dữ liệu --</option>
            {datasets.map(d => (
              <option key={d.id} value={d.id}>{d.name || d.id}</option>
            ))}
          </select>

          {selectedDataset && (
            <div className="mt-4 bg-neutral-50 p-5 rounded-xl border border-neutral-100 space-y-1 shadow-sm text-sm">
              <div className="flex justify-between gap-3 py-2 border-b border-neutral-200/60">
                <span className="text-neutral-600 font-medium">Mã</span>
                <span className="text-neutral-900 font-bold font-mono truncate" title={selectedDataset.id}>{selectedDataset.id}</span>
              </div>
              <div className="flex justify-between gap-3 py-2 border-b border-neutral-200/60">
                <span className="text-neutral-600 font-medium">Nền tảng vi mảng</span>
                <span className="text-neutral-900 font-medium">{selectedDataset.platform}</span>
              </div>
              <div className="flex justify-between gap-3 py-2 border-b border-neutral-200/60">
                <span className="text-neutral-600 font-medium">Mẫu bệnh phẩm</span>
                <span className="text-neutral-900 font-medium">{selectedDataset.n_samples}</span>
              </div>
              <div className="flex justify-between gap-3 py-2 border-b border-neutral-200/60">
                <span className="text-neutral-600 font-medium">Đặc trưng</span>
                <span className="text-neutral-900 font-medium">{selectedDataset.n_features?.toLocaleString()}</span>
              </div>
              <div className="flex justify-between gap-3 py-2">
                <span className="text-neutral-600 font-medium">Số lớp</span>
                <span className="text-brand-700 font-bold">{selectedDataset.n_classes}</span>
              </div>
            </div>
          )}
        </>
      ) : (
        <fieldset disabled={datasetLocked} className="border-0 p-0 m-0 min-w-0 disabled:opacity-60 space-y-3">
          <div className="flex gap-4">
            <label className="flex items-center gap-2 cursor-pointer group">
              <input type="radio" name="uploadSource" checked={uploadSource === 'geo'} onChange={() => { setUploadSource('geo'); setUploadId(null); setUploadCharacteristics([]); }} className="w-4 h-4 shrink-0 accent-brand-600" />
              <span className="font-medium text-neutral-800 group-hover:text-brand-700 transition-colors">Từ NCBI GEO</span>
            </label>
            <label className="flex items-center gap-2 cursor-pointer group">
              <input type="radio" name="uploadSource" checked={uploadSource === 'cumida'} onChange={() => { setUploadSource('cumida'); setUploadId(null); setUploadCharacteristics([]); }} className="w-4 h-4 shrink-0 accent-brand-600" />
              <span className="font-medium text-neutral-800 group-hover:text-brand-700 transition-colors">Từ CuMiDa</span>
            </label>
          </div>

          <div>
            <label className="text-xs text-neutral-600 font-medium block mb-1">Tên thư mục lưu trữ</label>
            <input type="text" value={uploadTissue} onChange={e => setUploadTissue(e.target.value)} className="w-full bg-neutral-50 border border-neutral-200 text-neutral-800 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-brand-500/30 focus:border-brand-500" />
          </div>

          <div>
            <label className="text-xs text-neutral-600 font-medium block mb-1">
              {uploadSource === 'geo' ? 'File series matrix (.txt.gz)' : 'File probe CuMiDa (.csv)'}
            </label>
            <input
              type="file"
              accept={uploadSource === 'geo' ? '.gz,.txt' : '.csv'}
              onChange={e => setUploadFile1(e.target.files?.[0] || null)}
              className="w-full text-sm text-neutral-700"
            />
          </div>
          <div>
            <label className="text-xs text-neutral-600 font-medium block mb-1">
              File annotation {uploadSource === 'geo' ? 'GPL (tùy chọn)' : 'CuMiDa (tùy chọn)'}
            </label>
            <input
              type="file"
              onChange={e => setUploadFile2(e.target.files?.[0] || null)}
              className="w-full text-sm text-neutral-700"
            />
          </div>

          <Button
            variant="secondary"
            onClick={handleUploadInspect}
            disabled={!uploadFile1 || isUploadInspecting || isUploadBuilding}
            className="w-full"
          >
            {isUploadInspecting ? <Loader2 size={16} className="animate-spin" /> : <FileText size={16} className="text-neutral-500" />}
            Phân tích file
          </Button>

          {uploadCharacteristics.length > 0 && (
            <div className="animate-in fade-in slide-in-from-top-2">
              <label className="text-xs text-neutral-600 font-medium block mb-1">Cột đặc trưng dùng làm nhãn lớp</label>
              <select
                value={uploadClassChar}
                onChange={e => setUploadClassChar(e.target.value)}
                className="w-full bg-neutral-50 border border-neutral-200 text-neutral-800 rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-brand-500/30 focus:border-brand-500 mb-3"
              >
                {uploadCharacteristics.map(c => <option key={c} value={c}>{c}</option>)}
              </select>
              <Button
                onClick={handleUploadBuild}
                disabled={!uploadClassChar || isUploadBuilding}
                className="w-full"
              >
                {isUploadBuilding ? <Loader2 size={16} className="animate-spin" /> : <PlayCircle size={16} />}
                Xây dựng dataset
              </Button>
            </div>
          )}

          {uploadLog && <StatusLine log={uploadLog} loading={isUploadInspecting || isUploadBuilding} />}

          <RunHistoryList
            title="Lịch sử tải lên"
            runs={uploadRuns}
            disabled={isUploadBuilding}
            isAvailable={run => !!run.dataset && datasets.some(d => d.id === run.dataset)}
            isSelected={run => datasetId === run.dataset}
            unavailableTitle="Dataset này không còn tồn tại (mất khi khởi động lại server)"
            onSelect={run => {
              if (!run.dataset) return;
              dispatch({ type: 'dataset_selected', datasetId: run.dataset });
              dispatch({ type: 'dataset_tab_changed', tab: 'existing' });
            }}
            renderLabel={run => `${String(run.summary?.tissue ?? '')} · ${run.run_id}`}
            renderStatus={run => `${run.summary?.n_samples ?? '?'} mẫu`}
          />
        </fieldset>
      )}
    </>
  );
}
