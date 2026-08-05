import React, { useState } from 'react';
import { Maximize2, X } from 'lucide-react';
import { cn } from '../lib/utils';

interface ConfusionMatrixProps {
  data: number[][];
  labels: string[];
}

/** Shared cell-color helpers so the compact grid and the zoomed overlay render identically. */
function useColorScale(data: number[][]) {
  const maxVal = Math.max(...data.flat(), 1);
  const getBgColor = (val: number) => {
    const intensity = Math.max(0, Math.min(1, val / maxVal));
    return `rgba(13, 148, 136, ${intensity * 0.8 + 0.1})`;
  };
  const getTextColor = (val: number) => (val / maxVal > 0.5 ? 'text-white' : 'text-slate-700');
  return { getBgColor, getTextColor };
}

/** Compact grid — cells sized by CSS Grid `fr` units so the whole matrix always
 * fits its container width (never triggers a horizontal scrollbar), instead of
 * the old fixed 80px-per-cell layout which overflowed for datasets with many
 * classes. Click/hover to open a full-size, readable zoomed overlay instead. */
function MatrixGrid({
  data, labels, cellClassName, fontSizeClass, labelFontSizeClass,
}: {
  data: number[][], labels: string[], cellClassName?: string, fontSizeClass: string, labelFontSizeClass: string,
}) {
  const { getBgColor, getTextColor } = useColorScale(data);
  const n = labels.length;
  return (
    <div className="flex items-center gap-3">
      <div className="flex flex-col items-center justify-center origin-center -rotate-90 text-sm font-medium text-slate-500 whitespace-nowrap shrink-0">
        Nhãn thật
      </div>
      <div className="min-w-0 flex-1">
        <div className="text-center text-sm font-medium text-slate-500 mb-2">Nhãn dự đoán</div>
        <div className="grid gap-0.5" style={{ gridTemplateColumns: `minmax(3.5rem, auto) repeat(${n}, minmax(0, 1fr))` }}>
          <div />
          {labels.map((label, i) => (
            <div key={`x-${i}`} className={cn("text-center font-medium text-slate-600 mb-1 px-0.5 truncate", labelFontSizeClass)} title={label}>
              {label}
            </div>
          ))}
          {data.map((row, i) => (
            <React.Fragment key={`row-${i}`}>
              <div className={cn("text-right pr-2 font-medium text-slate-600 flex items-center justify-end truncate", labelFontSizeClass)} title={labels[i]}>
                {labels[i]}
              </div>
              {row.map((cell, j) => (
                <div
                  key={`cell-${i}-${j}`}
                  className={cn(
                    "aspect-square flex items-center justify-center border border-white transition-colors cursor-default hover:opacity-80",
                    getTextColor(cell), cellClassName,
                  )}
                  style={{ backgroundColor: getBgColor(cell) }}
                  title={`Thật: ${labels[i]}, Dự đoán: ${labels[j]}`}
                >
                  <span className={cn("font-bold", fontSizeClass)}>{cell}</span>
                </div>
              ))}
            </React.Fragment>
          ))}
        </div>
      </div>
    </div>
  );
}

export default function ConfusionMatrix({ data, labels }: ConfusionMatrixProps) {
  const [zoomed, setZoomed] = useState(false);

  return (
    <>
      <div className="w-full relative group/matrix">
        <button
          onClick={() => setZoomed(true)}
          className="absolute -top-1 right-0 z-10 p-1.5 rounded-full bg-white border border-slate-200 text-slate-400 opacity-0 group-hover/matrix:opacity-100 hover:text-teal-600 hover:border-teal-200 transition-all shadow-sm"
          title="Phóng to ma trận nhầm lẫn"
        >
          <Maximize2 size={14} />
        </button>
        <MatrixGrid data={data} labels={labels} fontSizeClass="text-[11px] sm:text-sm" labelFontSizeClass="text-[9px] sm:text-xs" />
      </div>

      {zoomed && (
        <div
          className="fixed inset-0 z-50 bg-slate-900/60 backdrop-blur-sm flex items-center justify-center p-6 animate-in fade-in duration-150"
          onClick={() => setZoomed(false)}
        >
          <div
            className="bg-white rounded-2xl shadow-2xl p-8 max-w-[90vw] max-h-[90vh] overflow-auto"
            onClick={e => e.stopPropagation()}
          >
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-base font-semibold text-slate-800">Ma trận nhầm lẫn</h3>
              <button
                onClick={() => setZoomed(false)}
                className="p-1.5 rounded-full hover:bg-slate-100 text-slate-500 transition-colors"
              >
                <X size={18} />
              </button>
            </div>
            <div style={{ width: Math.max(480, labels.length * 90) }}>
              <MatrixGrid data={data} labels={labels} cellClassName="min-h-[70px]" fontSizeClass="text-xl" labelFontSizeClass="text-xs" />
            </div>
          </div>
        </div>
      )}
    </>
  );
}
