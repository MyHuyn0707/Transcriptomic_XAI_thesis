import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { GitMerge, AlertTriangle } from 'lucide-react';
import { ChevronDown, ChevronUp } from '@tailgrids/icons';
import { api, ContentInfo, SplitStats } from '../lib/api';
import { PALETTE, DROPPED_COLOR } from '../lib/palette';
import Panel from './ui/Panel';

/** Right-column display for Bước 2 "Xử lý & Chia Dữ liệu" — moved out of
 * DatasetOverview so that component keeps only the dataset-identity content
 * (title/organism/GEO/original-study/annotation) while everything about the
 * rare-class-drop + train/test split lives here, driven by whatever stats
 * (cached overview OR a live /split/preview) the parent last loaded.
 */
export default function DatasetSplit({ datasetId, stats, collapseSignal }: { datasetId: string, stats: SplitStats | null, collapseSignal?: boolean }) {
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [content, setContent] = useState<ContentInfo>({});

  useEffect(() => {
    setIsCollapsed(!!collapseSignal);
  }, [collapseSignal]);

  useEffect(() => {
    if (!datasetId) return;
    let ignore = false;
    setContent({});
    api.getOverview(datasetId)
      .then(o => { if (!ignore) setContent(o.content || {}); })
      .catch(() => { if (!ignore) setContent({}); });
    return () => { ignore = true; };
  }, [datasetId]);

  if (!datasetId || !stats) return null;

  const allLabels = Object.keys(stats.raw_class_counts);
  const droppedSet = new Set(stats.dropped_classes || []);
  const classesContent = content.classes || {};
  const vieName = (label: string) => classesContent[label]?.display_name_vi;

  const classDistributionData = allLabels.map((label, idx) => ({
    name: droppedSet.has(label) ? `${label} (Đã loại bỏ)` : label,
    count: stats.raw_class_counts[label] || 0,
    fill: droppedSet.has(label) ? DROPPED_COLOR : PALETTE[idx % PALETTE.length],
  }));

  // Colored the SAME way as "Phân phối các lớp" — Train is a solid fill in
  // the class's color, Test is the SAME color but diagonally hatched (via an
  // SVG pattern, see <defs> below) instead of a lighter alpha — an alpha tint
  // reads as "less important" and is hard to tell apart from other classes'
  // Train bars at a glance; a hatch pattern stays visually distinct while
  // keeping the same hue tied to the class.
  const splitData = stats.class_labels.map((label) => {
    const idx = allLabels.indexOf(label);
    const color = idx >= 0 ? PALETTE[idx % PALETTE.length] : DROPPED_COLOR;
    return {
      name: label,
      Train: stats.train_class_counts[label] || 0,
      Test: stats.test_class_counts[label] || 0,
      color,
    };
  });

  return (
    <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div className="bg-white p-6 rounded-2xl shadow-sm border border-success-100 transition-all duration-300">
        <div
          className="flex items-center justify-between cursor-pointer group"
          onClick={() => setIsCollapsed(!isCollapsed)}
        >
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-success-50 flex items-center justify-center text-success-600 border border-success-100">
              <GitMerge size={20} />
            </div>
            <h2 className="text-xl font-bold text-neutral-800 tracking-tight group-hover:text-success-700 transition-colors">
              Chia Dữ liệu (Train/Test)
            </h2>
          </div>
          <button className="p-2 rounded-full hover:bg-success-50 text-neutral-600 hover:text-success-600 transition-colors">
            {isCollapsed ? <ChevronDown size={24} /> : <ChevronUp size={24} />}
          </button>
        </div>

        {!isCollapsed && (
          <div className="mt-6 animate-in fade-in slide-in-from-top-4 duration-300 space-y-8">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <Panel padding="lg" className="space-y-3">
                <div className="flex justify-between gap-3 py-2 border-b border-neutral-200/60">
                  <span className="text-neutral-600 font-medium">Tổng số mẫu ban đầu</span>
                  <span className="text-neutral-800 font-bold font-mono">
                    {Object.values(stats.raw_class_counts).reduce((a, b) => a + b, 0)}
                  </span>
                </div>
                <div className="flex justify-between gap-3 py-2 border-b border-neutral-200/60">
                  <span className="text-neutral-600 font-medium">Số mẫu sau khi loại lớp hiếm</span>
                  <span className="text-neutral-800 font-bold font-mono">{stats.n_samples_total}</span>
                </div>
                <div className="flex justify-between gap-3 py-2 border-b border-neutral-200/60">
                  <span className="text-neutral-600 font-medium">Tổng số lớp ban đầu</span>
                  <span className="text-neutral-800 font-bold font-mono">{allLabels.length}</span>
                </div>
                <div className="flex justify-between gap-3 py-2 border-b border-neutral-200/60">
                  <span className="text-neutral-600 font-medium">Số lớp sau khi loại lớp hiếm</span>
                  <span className="text-neutral-800 font-bold font-mono">{stats.class_labels.length}</span>
                </div>
                <div className="flex justify-between gap-3 py-2 border-b border-neutral-200/60">
                  <span className="text-neutral-600 font-medium">Ngưỡng loại lớp hiếm</span>
                  <span className="text-neutral-800 font-bold font-mono">{"< "}{stats.min_samples_per_class} mẫu</span>
                </div>
              </Panel>
              <Panel padding="lg" className="space-y-3">
                <div className="flex justify-between gap-3 py-2 border-b border-neutral-200/60">
                  <span className="text-neutral-600 font-medium">Huấn luyện / Kiểm thử</span>
                  <span className="text-neutral-800 font-bold font-mono">{stats.n_train} / {stats.n_test}</span>
                </div>
                <div className="flex justify-between gap-3 py-2 border-b border-neutral-200/60">
                  <span className="text-neutral-600 font-medium">Tỷ lệ test_size</span>
                  <span className="text-neutral-800 font-bold font-mono">{stats.test_size}</span>
                </div>
                <div className="flex flex-col py-2 border-b border-neutral-200/60">
                  <span className="text-neutral-600 font-medium mb-1.5">Số lớp ({allLabels.length})</span>
                  <div className="flex flex-wrap gap-1.5">
                    {allLabels.map((label, idx) => {
                      const dropped = droppedSet.has(label);
                      const color = dropped ? DROPPED_COLOR : PALETTE[idx % PALETTE.length];
                      return (
                        <div
                          key={label}
                          className="flex flex-col items-center bg-white border rounded-lg px-2 py-1 leading-tight"
                          style={{ borderColor: color + '55' }}
                          title={dropped ? 'Bị loại do quá ít mẫu' : undefined}
                        >
                          <span className="text-xs font-semibold flex items-center gap-1" style={{ color }}>
                            {dropped && <AlertTriangle size={10} />}
                            {vieName(label) || label}
                          </span>
                          {vieName(label) && (
                            <span className="text-xs text-neutral-600 font-mono">{label}</span>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </div>
              </Panel>
            </div>

            <div className="grid grid-cols-1 xl:grid-cols-2 gap-8">
              <div className="h-72 border border-neutral-100 rounded-xl p-4 pt-6 relative">
                <h3 className="absolute -top-3 left-4 bg-white px-2 text-sm font-semibold text-neutral-700">Phân phối các lớp</h3>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={classDistributionData} layout="vertical" margin={{ top: 5, right: 30, left: 40, bottom: 5 }}>
                    <CartesianGrid strokeDasharray="3 3" horizontal={false} />
                    <XAxis type="number" />
                    <YAxis dataKey="name" type="category" width={140} tick={{ fontSize: 11 }} />
                    <Tooltip cursor={{fill: 'transparent'}} />
                    <Bar dataKey="count" radius={[0, 4, 4, 0]}>
                      {classDistributionData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.fill} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>

              <div className="h-72 border border-neutral-100 rounded-xl p-4 pt-6 relative">
                <div className="absolute -top-3 left-4 right-4 flex items-center justify-between">
                  <h3 className="bg-white px-2 text-sm font-semibold text-neutral-700">Tỷ lệ Train / Test</h3>
                  <div className="flex items-center gap-3 bg-white px-2 text-xs font-medium text-neutral-600">
                    <span className="flex items-center gap-1.5">
                      <span className="w-3 h-3 rounded-sm bg-neutral-400 inline-block" /> Train
                    </span>
                    <span className="flex items-center gap-1.5">
                      <span
                        className="w-3 h-3 rounded-sm inline-block border border-neutral-400"
                        style={{ backgroundImage: 'repeating-linear-gradient(45deg, #94a3b8 0 2px, transparent 2px 4px)' }}
                      />
                      Test
                    </span>
                  </div>
                </div>
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={splitData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                    <defs>
                      {/* Test bars reuse the SAME color as their class's Train bar, just
                          diagonally hatched (white stripes over the solid color) instead
                          of a lighter alpha — keeps the class-color link obvious while
                          staying visually distinct from Train at a glance. */}
                      {splitData.map((entry, index) => (
                        <pattern
                          key={`hatch-def-${index}`}
                          id={`test-hatch-${index}`}
                          patternUnits="userSpaceOnUse"
                          width="6" height="6"
                          patternTransform="rotate(45)"
                        >
                          <rect width="6" height="6" fill={entry.color} />
                          <line x1="0" y1="0" x2="0" y2="6" stroke="#ffffff" strokeWidth="2.5" />
                        </pattern>
                      ))}
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" vertical={false} />
                    <XAxis dataKey="name" tick={{ fontSize: 12 }} />
                    <YAxis />
                    <Tooltip cursor={{fill: '#f1f5f9'}} />
                    <Bar dataKey="Train" name="Train" radius={[4, 4, 0, 0]}>
                      {splitData.map((entry, index) => (
                        <Cell key={`train-${index}`} fill={entry.color} />
                      ))}
                    </Bar>
                    <Bar dataKey="Test" name="Test" radius={[4, 4, 0, 0]}>
                      {splitData.map((entry, index) => (
                        <Cell key={`test-${index}`} fill={`url(#test-hatch-${index})`} stroke={entry.color} strokeWidth={1} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
