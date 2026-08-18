import React, { useState, useEffect } from 'react';
import { Filter, ChevronDown, ChevronUp, ClockThree, FileText, CheckCircle1 } from '@tailgrids/icons';
import { FeatureExtractionStats } from '../lib/api';
import ConfigRow from './ui/ConfigRow';

interface Props {
  stats: FeatureExtractionStats | null;
  collapseSignal?: boolean;
}

export default function FeatureExtractionOverview({ stats, collapseSignal }: Props) {
  const [isCollapsed, setIsCollapsed] = useState(false);

  useEffect(() => {
    if (collapseSignal) {
      setIsCollapsed(true);
    } else {
      setIsCollapsed(false);
    }
  }, [collapseSignal]);

  if (!stats) return null;

  return (
    <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div className="bg-white p-6 rounded-2xl shadow-xs border border-neutral-200 transition-all duration-300">
        <div
          className="flex items-center justify-between cursor-pointer group"
          onClick={() => setIsCollapsed(!isCollapsed)}
        >
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-brand-50 flex items-center justify-center text-brand-600 border border-brand-100">
              <Filter size={20} />
            </div>
            <h2 className="text-lg font-semibold text-neutral-800 group-hover:text-brand-600 transition-colors">
              Tổng quan Trích xuất Đặc trưng
            </h2>
          </div>
          <button className="p-2 rounded-full hover:bg-neutral-100 text-neutral-600 transition-colors">
            {isCollapsed ? <ChevronDown size={24} /> : <ChevronUp size={24} />}
          </button>
        </div>

        {!isCollapsed && (
          <div className="mt-6 animate-in fade-in slide-in-from-top-4 duration-300">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
              <div className="rounded-xl p-4 border border-neutral-200 flex items-center gap-4">
                <div className="w-10 h-10 rounded-full bg-info-100 text-info-600 flex items-center justify-center">
                  <FileText size={20} />
                </div>
                <div>
                  <p className="text-[10px] text-neutral-600 font-bold uppercase tracking-wider">Đặc trưng ban đầu</p>
                  <p className="text-xl font-bold text-neutral-800 font-mono">{(stats.n_original_features ?? 0).toLocaleString()}</p>
                </div>
              </div>
              <div className="rounded-xl p-4 border border-neutral-200 flex items-center gap-4">
                <div className="w-10 h-10 rounded-full bg-success-100 text-success-600 flex items-center justify-center">
                  <CheckCircle1 size={20} />
                </div>
                <div>
                  <p className="text-[10px] text-neutral-600 font-bold uppercase tracking-wider">Đã chọn</p>
                  <p className="text-xl font-bold text-neutral-800 font-mono">{stats.n_selected_features}</p>
                </div>
              </div>
              <div className="rounded-xl p-4 border border-neutral-200 flex items-center gap-4">
                <div className="w-10 h-10 rounded-full bg-warning-100 text-warning-600 flex items-center justify-center">
                  <ClockThree size={20} />
                </div>
                <div>
                  <p className="text-[10px] text-neutral-600 font-bold uppercase tracking-wider">Thời gian chạy</p>
                  <p className="text-xl font-bold text-neutral-800 font-mono">{(stats.runtime_seconds ?? 0).toFixed(0)} giây</p>
                </div>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Cấu hình */}
              <div className="rounded-xl border border-neutral-200 p-5">
                <h3 className="text-sm font-bold text-neutral-800 mb-4 flex items-center gap-2">
                  <span className="w-1.5 h-4 bg-brand-500 rounded-full inline-block"></span>
                  Tham số cấu hình
                </h3>

                <div className="grid grid-cols-1 gap-y-1">
                   {(stats.feature_selection ?? '').includes('mrmr') && (
                     <>
                        <ConfigRow label="Criterion" value={stats.criterion} />
                        <ConfigRow label="K (Target Features)" value={stats.K} />
                        <ConfigRow label="n_bins" value={stats.n_bins || 3} />
                        <ConfigRow label="Random State" value={stats.random_state || 42} />
                     </>
                   )}

                   {stats.feature_selection === 'boruta' && (
                     <>
                        <ConfigRow label="n_estimators" value={stats.n_estimators || 'auto'} />
                        <ConfigRow label="rf_n_estimators" value={stats.rf_n_estimators || 500} />
                        <ConfigRow label="max_depth" value={stats.max_depth || 'null'} />
                        <ConfigRow label="max_iter" value={stats.max_iter} />
                        <ConfigRow label="perc" value={stats.perc || 100} />
                        <ConfigRow label="Alpha" value={stats.alpha} />
                        <ConfigRow label="Class Weight" value={stats.class_weight || 'balanced'} />
                        <ConfigRow label="Random State" value={stats.random_state || 42} />
                        <ConfigRow label="Chế độ chọn đặc trưng" value={stats.selection_mode || 'confirmed'} />
                        {stats.selection_mode === 'top_k' && (
                          <ConfigRow label="k" value={stats.k} />
                        )}
                     </>
                   )}
                </div>
              </div>

              {/* Kết quả */}
              <div className="rounded-xl border border-neutral-200 p-5">
                <h3 className="text-sm font-bold text-neutral-800 mb-4 flex items-center gap-2">
                  <span className="w-1.5 h-4 bg-info-500 rounded-full inline-block"></span>
                  Kết quả phân tích
                </h3>
                <div className="grid grid-cols-1 gap-y-1">
                  <ConfigRow label="Dataset" value={stats.dataset_name} />
                  <ConfigRow label="Framework" value={stats.framework} mono={false} />
                  <ConfigRow label="Số lượng mẫu" value={stats.n_samples} />

                  {stats.feature_selection === 'boruta' && (
                    <>
                      <ConfigRow label="Confirmed Features" value={stats.confirmed} valueClassName="!text-success-600 font-bold" />
                      <ConfigRow label="Tentative Features" value={stats.tentative} valueClassName="!text-warning-500 font-bold" />
                      <ConfigRow label="Rejected Features" value={stats.rejected?.toLocaleString()} valueClassName="!text-danger-500 font-bold" />
                    </>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
