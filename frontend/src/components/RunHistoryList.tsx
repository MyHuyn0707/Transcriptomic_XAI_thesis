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
  unavailableLabel = '',
  onSelect,
  renderLabel,
  renderStatus,
}: RunHistoryListProps) {
  if (runs.length === 0) return null;

  return (
    <div className="mt-4">
      <div className="flex items-center gap-2 text-sm font-semibold text-neutral-600 mb-2">
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
                "w-full text-left text-sm px-3 py-2 rounded-lg border transition-colors flex items-center justify-between gap-2",
                run.status === 'error'
                  ? "border-danger-100 bg-danger-50 text-danger-600"
                  : !available
                    ? "border-neutral-100 bg-neutral-50 text-neutral-600 cursor-not-allowed"
                    : selected
                      ? "border-brand-300 bg-brand-50 text-brand-700"
                      : "border-neutral-100 bg-neutral-50 text-neutral-700 hover:bg-neutral-100"
              )}
            >
              <span className={cn("font-mono truncate", !available && "line-through")}>{renderLabel(run)}</span>
              <span className={cn("shrink-0", !available && "line-through")}>
                {run.status === 'error' ? 'lỗi' : !available ? unavailableLabel : renderStatus(run)}
              </span>
            </button>
          );
        })}
      </div>
    </div>
  );
}
