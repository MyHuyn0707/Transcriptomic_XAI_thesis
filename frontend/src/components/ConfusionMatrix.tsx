import React, { useState } from 'react';
import { ExpandSquare4, Close } from '@tailgrids/icons';
import { cn } from '../lib/utils';

interface ConfusionMatrixProps {
  data: number[][];
  labels: string[];
}

/** Shared cell-color helpers so the compact grid and the zoomed overlay render identically. */
function useColorScale(data: number[][]) {
  const maxVal = Math.max(...data.flat(), 1);
  // sqrt (not linear) intensity: a linear val/maxVal scale makes every small
  // count (0, 1, 2...) look almost the same faint tint next to one dominant
  // cell — sqrt spreads the low end out so "0" and "1" are actually
  // distinguishable instead of both reading as "empty".
  const intensityOf = (val: number) => (val <= 0 ? 0 : Math.sqrt(Math.min(1, val / maxVal)));
  const getBgColor = (val: number) => {
    if (val <= 0) return 'rgba(15, 23, 42, 0.04)';
    return `rgba(13, 148, 136, ${intensityOf(val) * 0.75 + 0.15})`;
  };
  const getTextColor = (val: number) => (intensityOf(val) > 0.6 ? 'text-white' : 'text-neutral-800');
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
      <div className="flex flex-col items-center justify-center origin-center -rotate-90 text-sm font-medium text-neutral-600 whitespace-nowrap shrink-0">
        Nhãn thật
      </div>
      <div className="min-w-0 flex-1">
        <div className="text-center text-sm font-medium text-neutral-600 mb-2">Nhãn dự đoán</div>
        <div className="grid gap-0.5" style={{ gridTemplateColumns: `minmax(3.5rem, auto) repeat(${n}, minmax(0, 1fr))` }}>
          <div />
          {labels.map((label, i) => (
            <div key={`x-${i}`} className={cn("text-center font-medium text-neutral-700 mb-1 px-0.5 truncate", labelFontSizeClass)} title={label}>
              {label}
            </div>
          ))}
          {data.map((row, i) => (
            <React.Fragment key={`row-${i}`}>
              <div className={cn("text-right pr-2 font-medium text-neutral-700 flex items-center justify-end truncate", labelFontSizeClass)} title={labels[i]}>
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
          className="absolute -top-1 right-0 z-10 p-1.5 rounded-full bg-white border border-neutral-200 text-neutral-500 opacity-0 group-hover/matrix:opacity-100 hover:text-brand-600 hover:border-brand-200 transition-all shadow-sm"
          title="Phóng to ma trận nhầm lẫn"
        >
          <ExpandSquare4 size={14} />
        </button>
        <MatrixGrid data={data} labels={labels} fontSizeClass="text-[11px] sm:text-sm" labelFontSizeClass="text-[9px] sm:text-xs" />
      </div>

      {zoomed && (
        <div
          className="fixed inset-0 z-50 bg-neutral-900/60 backdrop-blur-sm flex items-center justify-center p-6 animate-in fade-in duration-150"
          onClick={() => setZoomed(false)}
        >
          <div
            className="bg-white rounded-2xl shadow-2xl p-8 max-w-[90vw] max-h-[90vh] overflow-auto"
            onClick={e => e.stopPropagation()}
          >
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-base font-semibold text-neutral-900">Ma trận nhầm lẫn</h3>
              <button
                onClick={() => setZoomed(false)}
                className="p-1.5 rounded-full hover:bg-neutral-100 text-neutral-600 transition-colors"
              >
                <Close size={18} />
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
