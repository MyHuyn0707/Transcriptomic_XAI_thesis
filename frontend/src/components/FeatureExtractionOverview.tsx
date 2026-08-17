import React, { useState, useEffect } from 'react';
import { Filter, ChevronDown, ChevronUp, ClockThree, FileText, CheckCircle1 } from '@tailgrids/icons';
import { FeatureExtractionStats } from '../lib/api';

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
      <div className="bg-white p-6 rounded-2xl shadow-sm border border-neutral-200 transition-all duration-300">
        <div 
          className="flex items-center justify-between cursor-pointer group"
          onClick={() => setIsCollapsed(!isCollapsed)}
        >
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-brand-50 flex items-center justify-center text-brand-600 border border-brand-100">
              <Filter size={20} />
            </div>
            <h2 className="text-xl font-bold text-neutral-900 tracking-tight group-hover:text-brand-600 transition-colors">
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
              <div className="bg-neutral-50 rounded-xl p-4 border border-neutral-100 flex items-center gap-4">
                <div className="w-10 h-10 rounded-full bg-info-100 text-info-600 flex items-center justify-center">
                  <FileText size={20} />
                </div>
                <div>
                  <p className="text-[10px] text-neutral-600 font-bold uppercase tracking-wider">Đặc trưng ban đầu</p>
                  <p className="text-xl font-bold text-neutral-900 font-mono">{(stats.n_original_features ?? 0).toLocaleString()}</p>
                </div>
              </div>
              <div className="bg-neutral-50 rounded-xl p-4 border border-neutral-100 flex items-center gap-4">
                <div className="w-10 h-10 rounded-full bg-success-100 text-success-600 flex items-center justify-center">
                  <CheckCircle1 size={20} />
                </div>
                <div>
                  <p className="text-[10px] text-neutral-600 font-bold uppercase tracking-wider">Đã chọn</p>
                  <p className="text-xl font-bold text-neutral-900 font-mono">{stats.n_selected_features}</p>
                </div>
              </div>
              <div className="bg-neutral-50 rounded-xl p-4 border border-neutral-100 flex items-center gap-4">
                <div className="w-10 h-10 rounded-full bg-warning-100 text-warning-600 flex items-center justify-center">
                  <ClockThree size={20} />
                </div>
                <div>
                  <p className="text-[10px] text-neutral-600 font-bold uppercase tracking-wider">Thời gian chạy</p>
                  <p className="text-xl font-bold text-neutral-900 font-mono">{(stats.runtime_seconds ?? 0).toFixed(0)} giây</p>
                </div>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Cấu hình */}
              <div className="bg-neutral-50 rounded-xl p-5 border border-neutral-200">
                <h3 className="text-sm font-bold text-neutral-900 mb-4 flex items-center gap-2">
                  <span className="w-1.5 h-4 bg-brand-500 rounded-full inline-block"></span>
                  Tham số cấu hình
                </h3>
                
                <div className="grid grid-cols-1 gap-y-3">
                   {(stats.feature_selection ?? '').includes('mrmr') && (
                     <>
                        <div className="flex justify-between gap-3 py-2 border-b border-neutral-200 border-dashed">
                          <span className="text-neutral-600 text-sm">Criterion</span>
                          <span className="text-neutral-900 font-medium text-sm font-mono">{stats.criterion}</span>
                        </div>
                        <div className="flex justify-between gap-3 py-2 border-b border-neutral-200 border-dashed">
                          <span className="text-neutral-600 text-sm">K (Target Features)</span>
                          <span className="text-neutral-900 font-medium text-sm font-mono">{stats.K}</span>
                        </div>
                        <div className="flex justify-between gap-3 py-2 border-b border-neutral-200 border-dashed">
                          <span className="text-neutral-600 text-sm">n_bins</span>
                          <span className="text-neutral-900 font-medium text-sm font-mono">{stats.n_bins || 3}</span>
                        </div>
                        <div className="flex justify-between gap-3 py-2 border-b border-neutral-200 border-dashed">
                          <span className="text-neutral-600 text-sm">Random State</span>
                          <span className="text-neutral-900 font-medium text-sm font-mono">{stats.random_state || 42}</span>
                        </div>
                     </>
                   )}

                   {stats.feature_selection === 'boruta' && (
                     <>
                        <div className="flex justify-between gap-3 py-2 border-b border-neutral-200 border-dashed">
                          <span className="text-neutral-600 text-sm">n_estimators</span>
                          <span className="text-neutral-900 font-medium text-sm font-mono">{stats.n_estimators || 'auto'}</span>
                        </div>
                        <div className="flex justify-between gap-3 py-2 border-b border-neutral-200 border-dashed">
                          <span className="text-neutral-600 text-sm">rf_n_estimators</span>
                          <span className="text-neutral-900 font-medium text-sm font-mono">{stats.rf_n_estimators || 500}</span>
                        </div>
                        <div className="flex justify-between gap-3 py-2 border-b border-neutral-200 border-dashed">
                          <span className="text-neutral-600 text-sm">max_depth</span>
                          <span className="text-neutral-900 font-medium text-sm font-mono">{stats.max_depth || 'null'}</span>
                        </div>
                        <div className="flex justify-between gap-3 py-2 border-b border-neutral-200 border-dashed">
                          <span className="text-neutral-600 text-sm">max_iter</span>
                          <span className="text-neutral-900 font-medium text-sm font-mono">{stats.max_iter}</span>
                        </div>
                        <div className="flex justify-between gap-3 py-2 border-b border-neutral-200 border-dashed">
                          <span className="text-neutral-600 text-sm">perc</span>
                          <span className="text-neutral-900 font-medium text-sm font-mono">{stats.perc || 100}</span>
                        </div>
                        <div className="flex justify-between gap-3 py-2 border-b border-neutral-200 border-dashed">
                          <span className="text-neutral-600 text-sm">Alpha</span>
                          <span className="text-neutral-900 font-medium text-sm font-mono">{stats.alpha}</span>
                        </div>
                        <div className="flex justify-between gap-3 py-2 border-b border-neutral-200 border-dashed">
                          <span className="text-neutral-600 text-sm">Class Weight</span>
                          <span className="text-neutral-900 font-medium text-sm font-mono">{stats.class_weight || 'balanced'}</span>
                        </div>
                        <div className="flex justify-between gap-3 py-2 border-b border-neutral-200 border-dashed">
                          <span className="text-neutral-600 text-sm">Random State</span>
                          <span className="text-neutral-900 font-medium text-sm font-mono">{stats.random_state || 42}</span>
                        </div>
                        <div className="flex justify-between gap-3 py-2 border-b border-neutral-200 border-dashed">
                          <span className="text-neutral-600 text-sm">Chế độ chọn đặc trưng</span>
                          <span className="text-brand-700 font-bold text-sm font-mono">{stats.selection_mode || 'confirmed'}</span>
                        </div>
                        {stats.selection_mode === 'top_k' && (
                          <div className="flex justify-between gap-3 py-2 border-b border-neutral-200 border-dashed">
                            <span className="text-neutral-600 text-sm">k</span>
                            <span className="text-neutral-900 font-medium text-sm font-mono">{stats.k}</span>
                          </div>
                        )}
                     </>
                   )}
                </div>
              </div>

              {/* Kết quả */}
              <div className="bg-neutral-50 rounded-xl p-5 border border-neutral-200">
                <h3 className="text-sm font-bold text-neutral-900 mb-4 flex items-center gap-2">
                  <span className="w-1.5 h-4 bg-info-500 rounded-full inline-block"></span>
                  Kết quả phân tích
                </h3>
                <div className="grid grid-cols-1 gap-y-3">
                  <div className="flex justify-between gap-3 py-2 border-b border-neutral-200 border-dashed">
                    <span className="text-neutral-600 text-sm">Dataset</span>
                    <span className="text-neutral-900 font-medium text-sm font-mono">{stats.dataset_name}</span>
                  </div>
                  <div className="flex justify-between gap-3 py-2 border-b border-neutral-200 border-dashed">
                    <span className="text-neutral-600 text-sm">Framework</span>
                    <span className="text-neutral-900 font-medium text-sm">{stats.framework}</span>
                  </div>
                  <div className="flex justify-between gap-3 py-2 border-b border-neutral-200 border-dashed">
                    <span className="text-neutral-600 text-sm">Số lượng mẫu</span>
                    <span className="text-neutral-900 font-medium text-sm font-mono">{stats.n_samples}</span>
                  </div>
{/* 
                  {(stats.feature_selection ?? '').includes('mrmr') && (
                    <div className="flex justify-between gap-3 py-2 border-b border-neutral-200 border-dashed">
                      <span className="text-neutral-600 text-sm">Implementation</span>
                      <span className="text-neutral-900 font-medium text-sm">{stats.implementation}</span>
                    </div>
                  )} */}

                  {stats.feature_selection === 'boruta' && (
                    <>
                      <div className="flex justify-between gap-3 py-2 border-b border-neutral-200 border-dashed">
                        <span className="text-neutral-600 text-sm">Confirmed Features</span>
                        <span className="text-success-600 font-bold text-sm font-mono">{stats.confirmed}</span>
                      </div>
                      <div className="flex justify-between gap-3 py-2 border-b border-neutral-200 border-dashed">
                        <span className="text-neutral-600 text-sm">Tentative Features</span>
                        <span className="text-warning-500 font-bold text-sm font-mono">{stats.tentative}</span>
                      </div>
                      <div className="flex justify-between gap-3 py-2 border-b border-neutral-200 border-dashed">
                        <span className="text-neutral-600 text-sm">Rejected Features</span>
                        <span className="text-danger-500 font-bold text-sm font-mono">{stats.rejected?.toLocaleString()}</span>
                      </div>
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
