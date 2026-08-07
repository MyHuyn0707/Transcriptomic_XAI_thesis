import React from 'react';
import { History } from 'lucide-react';
import { cn } from '../lib/utils';
import { RunRecord } from '../lib/api';

interface RunHistoryListProps {
  title: string;
  runs: RunRecord[];
  disabled?: boolean;
  isSelected?: (run: RunRecord) => boolean;
  isAvailable?: (run: RunRecord) => boolean;
  unavailableTitle?: string;
  unavailableLabel?: string;
  onSelect: (run: RunRecord) => void;
  renderLabel: (run: RunRecord) => React.ReactNode;
  renderStatus: (run: RunRecord) => React.ReactNode;
}

// Shared "log box + run history list" used by upload/split/feature-selection/
// model steps — was copy-pasted 4x with only the label/status/availability
// logic differing between them.
export default function RunHistoryList({
  title,
  runs,
  disabled,
  isSelected,
  isAvailable,
  unavailableTitle,
  unavailableLabel = 'đã mất',
  onSelect,
  renderLabel,
  renderStatus,
}: RunHistoryListProps) {
  if (runs.length === 0) return null;

  return (
    <div className="mt-4">
      <div className="flex items-center gap-2 text-xs font-semibold text-slate-500 mb-2">
        <History size={14} /> {title} ({runs.length})
      </div>
      <div className="space-y-1.5 max-h-40 overflow-y-auto">
        {runs.map(run => {
          const available = isAvailable ? isAvailable(run) : true;
          const selected = isSelected ? isSelected(run) : false;
          return (
            <button
              key={run.run_id}
              onClick={() => { if (available) onSelect(run); }}
              disabled={disabled || !available}
              title={!available ? unavailableTitle : undefined}
              className={cn(
                "w-full text-left text-xs px-3 py-2 rounded-lg border transition-colors flex items-center justify-between gap-2",
                run.status === 'error'
                  ? "border-rose-100 bg-rose-50/50 text-rose-500"
                  : !available
                    ? "border-slate-100 bg-slate-50 text-slate-400 cursor-not-allowed"
                    : selected
                      ? "border-teal-300 bg-teal-50 text-teal-700"
                      : "border-slate-100 bg-slate-50 text-slate-600 hover:bg-slate-100"
              )}
            >
              <span className="font-mono truncate">{renderLabel(run)}</span>
              <span className="shrink-0">
                {run.status === 'error' ? 'lỗi' : !available ? unavailableLabel : renderStatus(run)}
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
