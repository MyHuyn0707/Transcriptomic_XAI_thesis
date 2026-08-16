import React, { useEffect, useReducer } from 'react';
import { Activity, Database, Dna, GitMerge, Microscope, Settings2 } from 'lucide-react';
import { cn } from './lib/utils';
import { api } from './lib/api';
import { workspaceReducer } from './state/workspaceReducer';
import { initialWorkspaceState } from './state/types';
import WizardStep from './components/WizardStep';
import Step1Dataset from './components/steps/Step1Dataset';
import Step2Split from './components/steps/Step2Split';
import Step3FeatureExtraction from './components/steps/Step3FeatureExtraction';
import Step4Model, { ModelOverviewPanel } from './components/steps/Step4Model';
import Step5Testing, { TestResultsPanel } from './components/steps/Step5Testing';
import DatasetOverview from './components/DatasetOverview';
import DatasetSplit from './components/DatasetSplit';
import FeatureExtractionOverview from './components/FeatureExtractionOverview';

// Right column's fallback for whichever step is active but hasn't produced
// anything yet (e.g. Bước 3 opened before feature selection has run) — same
// dashed-box treatment the old single "Không gian Phân tích Trống" state used,
// just scoped per-step instead of one blanket message for the whole app.
function EmptyStepPlaceholder({ text }: { text: string }) {
  return (
    <div className="h-full min-h-[400px] bg-white border border-neutral-200 border-dashed rounded-3xl flex flex-col items-center justify-center text-neutral-500 p-8 text-center shadow-sm">
      <div className="w-16 h-16 bg-neutral-50 rounded-full flex items-center justify-center mb-4">
        <Microscope size={28} className="text-neutral-400" />
      </div>
      <p className="text-sm max-w-md leading-relaxed text-neutral-600">{text}</p>
    </div>
  );
}

export default function App() {
  // Every dataset/split/fs/model/test field (and the cascade that clears a
  // step's downstream results once an upstream choice changes) lives in one
  // reducer instead of ~40 separate useState hooks — see
  // state/workspaceReducer.ts for the reset rules.
  const [state, dispatch] = useReducer(workspaceReducer, initialWorkspaceState);
  const { activeStep, datasets, datasetId, splitStats, fsLog, extractionStats, modelStats, testResults } = state;
  const selectedDataset = datasets.find(d => d.id === datasetId);

  useEffect(() => {
    api.getDatasets()
      .then(loaded => dispatch({ type: 'datasets_loaded', datasets: loaded }))
      .catch(e => dispatch({ type: 'datasets_load_failed', message: e.message }));
  }, []);

  // "Tổng Quan Mô Hình" (which contains the gene list a chip jumps to) auto-
  // collapses — and unmounts — once a prediction result shows, so it must be
  // re-expanded and given a tick to remount before the jump-to-gene listener
  // can catch this.
  const jumpToGene = (gene: string) => {
    dispatch({ type: 'model_overview_expanded' });
    setTimeout(() => {
      window.dispatchEvent(new CustomEvent('jump-to-gene', { detail: { gene } }));
    }, 60);
  };

  // Accordion: exactly one of the 5 left-column steps is expanded at a time,
  // and the right column shows ONLY that step's content — clicking a step's
  // header (or its number) opens it; a step can only open once its
  // prerequisite step has produced something.
  const stepReady: Record<number, boolean> = {
    1: true,
    2: !!datasetId,
    3: !!splitStats,
    4: !!extractionStats,
    5: !!modelStats,
  };
  const openStep = (n: number) => { if (stepReady[n]) dispatch({ type: 'active_step_set', step: n }); };

  return (
    <div className="min-h-screen bg-neutral-50 text-neutral-900 font-sans selection:bg-brand-200 pb-12">
      {/* Header */}
      <header className="bg-white border-b border-success-100/80 sticky top-0 z-50 backdrop-blur-sm bg-white/90">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-success-50 border border-success-200 rounded-xl flex items-center justify-center text-success-600 shadow-sm">
              <Dna size={22} />
            </div>
            <div>
              <h1 className="text-xl font-bold text-neutral-900 tracking-tight">GE-RuGO Workspace</h1>
              <p className="text-[11px] font-semibold tracking-wider uppercase text-success-600">Phân lớp dữ liệu Biểu hiện Gene có giải thích</p>
            </div>
          </div>
          <div className="hidden md:flex items-center gap-4 text-sm text-neutral-600 font-medium">
            <span className={cn("flex items-center gap-1.5 transition-colors", datasetId ? "text-success-600" : "")}><Database size={16}/> Dữ liệu</span>
            <span className="text-neutral-400">/</span>
            <span className={cn("flex items-center gap-1.5 transition-colors", splitStats ? "text-success-600" : "")}><GitMerge size={16}/> Chia dữ liệu</span>
            <span className="text-neutral-400">/</span>
            <span className={cn("flex items-center gap-1.5 transition-colors", fsLog ? "text-success-600" : "")}><Settings2 size={16}/> Trích xuất</span>
            <span className="text-neutral-400">/</span>
            <span className={cn("flex items-center gap-1.5 transition-colors", modelStats ? "text-success-600" : "")}><GitMerge size={16}/> Mô hình</span>
            <span className="text-neutral-400">/</span>
            <span className={cn("flex items-center gap-1.5 transition-colors", testResults ? "text-success-600" : "")}><Activity size={16}/> Thực nghiệm</span>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-8 grid grid-cols-1 lg:grid-cols-12 gap-10">

        {/* Cột Trái: Cấu hình */}
        <div className="lg:col-span-4 space-y-8 relative">
          <div className="absolute left-[23px] top-12 bottom-12 w-0.5 bg-neutral-100 -z-10 hidden lg:block"></div>

          <WizardStep number={1} title="Chọn Dữ Liệu" active={activeStep === 1} ready={stepReady[1]} onOpen={() => openStep(1)}>
            <Step1Dataset state={state} dispatch={dispatch} />
          </WizardStep>

          <WizardStep number={2} title="Xử lý & Chia Dữ liệu" active={activeStep === 2} ready={stepReady[2]} onOpen={() => openStep(2)}>
            <Step2Split state={state} dispatch={dispatch} />
          </WizardStep>

          <WizardStep number={3} title="Trích xuất đặc trưng" active={activeStep === 3} ready={stepReady[3]} onOpen={() => openStep(3)}>
            <Step3FeatureExtraction state={state} dispatch={dispatch} />
          </WizardStep>

          <WizardStep number={4} title="Mô Hình" active={activeStep === 4} ready={stepReady[4]} onOpen={() => openStep(4)}>
            <Step4Model state={state} dispatch={dispatch} />
          </WizardStep>

          <WizardStep number={5} title="Kiểm Thử (Thực Nghiệm)" active={activeStep === 5} ready={stepReady[5]} onOpen={() => openStep(5)}>
            <Step5Testing state={state} dispatch={dispatch} />
          </WizardStep>
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
              <ModelOverviewPanel state={state} dispatch={dispatch} />
            </div>
          )}

          {modelStats && (
            <div className="animate-in fade-in slide-in-from-bottom-4 duration-500">
              <TestResultsPanel state={state} dispatch={dispatch} onJumpToGene={jumpToGene} />
            </div>
          )}

          </div>
        </div>
      </main>
    </div>
  );
}
