import React, { useState, useEffect } from 'react';
import { TabContent, TabList, TabRoot, TabTrigger } from '../tailgrids/core/tabs';
import { Archive, ListChecks, ScanSearch } from 'lucide-react';
import { FileText, UploadCloud, Reload, Play } from '@tailgrids/icons';
import { api, BuiltDatasetResult, pollJob, RunRecord } from '../../lib/api';
import { WorkspaceAction, WorkspaceState } from '../../state/types';
import { errorMessage } from '../../lib/utils';
import { Button } from '../tailgrids/core/button';
import { Select, SelectContent, SelectIndicator, SelectItem, SelectTrigger, SelectValue } from '../tailgrids/core/select';
import { RadioInput } from '../tailgrids/core/radio-input';
import { Input } from '../tailgrids/core/input';
import StatusLine from '../ui/StatusLine';
import FilePicker from '../ui/FilePicker';
import FieldLabel from '../ui/FieldLabel';
import StatTile from '../ui/StatTile';
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

      <TabRoot
        value={datasetTab}
        onValueChange={(tab) => dispatch({ type: 'dataset_tab_changed', tab: tab as 'existing' | 'upload' })}
        className="border-0 rounded-none max-w-full [&>div:first-child]:border-none! [&>div:first-child]:p-0!"
      >
        {/* overflow-visible + flex-wrap replace Tabs' default horizontal-scroll
            behavior (built for many tabs in a row) — with only 2 triggers we
            want them sharing the row width evenly and wrapping their own
            text onto a 2nd line on narrow screens, never scrolling/clipping. */}
        <TabList className="!p-0 !border-0 mb-4 overflow-visible flex-wrap">
          <TabTrigger value="existing" disabled={datasetLocked} icon={<Archive size={14} className="shrink-0" />} className="flex-1 justify-center whitespace-normal text-center">
            Kho lưu trữ
          </TabTrigger>
          <TabTrigger value="upload" disabled={datasetLocked} icon={<UploadCloud size={14} className="shrink-0" />} className="flex-1 justify-center whitespace-normal text-center">
            Tải lên
          </TabTrigger>
        </TabList>

          <TabContent value="existing" className="!p-0">
            <Select
              aria-label="Chọn dataset"
              value={datasetId}
              onChange={(key) => dispatch({ type: 'dataset_selected', datasetId: String(key) })}
              isDisabled={datasetLocked}
              className="w-full"
            >
              <SelectTrigger className="w-full">
                <SelectValue />
                <SelectIndicator />
              </SelectTrigger>
              <SelectContent>
                <SelectItem id="" textValue="-- Chọn dữ liệu --">-- Chọn dữ liệu --</SelectItem>
                {datasets.map(d => (
                  <SelectItem key={d.id} id={d.id} textValue={d.name || d.id}>{d.name || d.id}</SelectItem>
                ))}
              </SelectContent>
            </Select>

            {selectedDataset && (
              <div className="mt-4 space-y-3">
                <div className="flex items-baseline justify-between gap-2">
                  <FieldLabel>Mã dataset</FieldLabel>
                  <span className="font-mono text-sm text-neutral-700 truncate" title={selectedDataset.id}>{selectedDataset.id}</span>
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <StatTile label="Nền tảng vi mảng" value={selectedDataset.platform} truncate mono />
                  <StatTile label="Mẫu bệnh phẩm" value={selectedDataset.n_samples} mono />
                  <StatTile label="Đặc trưng" value={selectedDataset.n_features?.toLocaleString()} mono />
                  <StatTile label="Số lớp" value={selectedDataset.n_classes} mono />
                </div>
              </div>
            )}
          </TabContent>

          <TabContent value="upload" className="!p-0">
            <fieldset disabled={datasetLocked} className="border-0 p-0 m-0 min-w-0 disabled:opacity-60 space-y-3">
              <div className="flex gap-4">
                <label className="flex items-center gap-2 cursor-pointer group">
                  <RadioInput
                    name="uploadSource"
                    checked={uploadSource === 'geo'}
                    onChange={() => { setUploadSource('geo'); setUploadId(null); setUploadCharacteristics([]); }}
                  />
                  <span className="font-medium text-neutral-800 group-hover:text-brand-700 transition-colors">Từ NCBI GEO</span>
                </label>
                <label className="flex items-center gap-2 cursor-pointer group">
                  <RadioInput
                    name="uploadSource"
                    checked={uploadSource === 'cumida'}
                    onChange={() => { setUploadSource('cumida'); setUploadId(null); setUploadCharacteristics([]); }}
                  />
                  <span className="font-medium text-neutral-800 group-hover:text-brand-700 transition-colors">Từ CuMiDa</span>
                </label>
              </div>

              <div>
                <label className="text-sm text-neutral-600 font-medium block mb-1">Tên thư mục lưu trữ</label>
                <Input value={uploadTissue} onChange={e => setUploadTissue(e.target.value)} className="w-full text-sm" />
              </div>

              <div>
                <label className="text-sm text-neutral-600 font-medium block mb-1">
                  {uploadSource === 'geo' ? 'File series matrix' : 'File probe CuMiDa'}
                </label>
                <FilePicker
                  value={uploadFile1}
                  onChange={setUploadFile1}
                  accept={uploadSource === 'geo' ? '.gz,.txt' : '.csv'}
                />
              </div>
              <div>
                <label className="text-sm text-neutral-600 font-medium block mb-1">
                  File annotation {uploadSource === 'geo' ? 'GPL' : 'CuMiDa'} (không bắt buộc)
                </label>
                <FilePicker
                  value={uploadFile2}
                  onChange={setUploadFile2}
                />
              </div>

              <Button
                variant="primary" appearance="fill"
                onClick={handleUploadInspect}
                disabled={!uploadFile1 || isUploadInspecting || isUploadBuilding}
                className="w-full text-base"
              >
                {isUploadInspecting ? <Reload size={16} className="animate-spin" /> : <ScanSearch size={16} className="text-neutral-600" />}
                Phân tích
              </Button>

              {uploadCharacteristics.length > 0 && (
                <div className="animate-in fade-in slide-in-from-top-2">
                  <label className="text-xs text-neutral-600 font-medium block mb-1">Cột đặc trưng dùng làm nhãn lớp</label>
                  <Select
                    aria-label="Cột đặc trưng dùng làm nhãn lớp"
                    value={uploadClassChar}
                    onChange={(key) => setUploadClassChar(String(key))}
                    className="w-full mb-3"
                  >
                    <SelectTrigger className="w-full">
                      <SelectValue />
                      <SelectIndicator />
                    </SelectTrigger>
                    <SelectContent>
                      {uploadCharacteristics.map(c => (
                        <SelectItem key={c} id={c} textValue={c}>{c}</SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <Button
                    onClick={handleUploadBuild}
                    disabled={!uploadClassChar || isUploadBuilding}
                    className="w-full"
                  >
                    {isUploadBuilding ? <Reload size={16} className="animate-spin" /> : <Play size={16} />}
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
          </TabContent>
      </TabRoot>
    </>
  );
}
