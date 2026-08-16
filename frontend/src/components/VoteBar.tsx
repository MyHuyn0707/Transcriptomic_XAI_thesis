import React from 'react';
import Panel from './ui/Panel';

interface VoteBarRow {
  key: string;
  label: React.ReactNode;
  caption: React.ReactNode;
  percentage: number;
  color: string;
}

interface VoteBarProps {
  title: string;
  rows: VoteBarRow[];
  className?: string;
}

/** Row-per-category percentage bar, shared by the class-vote, partial-vote
 * and match-tier breakdowns in the test-results report. */
export default function VoteBar({ title, rows, className }: VoteBarProps) {
  return (
    <Panel padding="lg" className={className}>
      <h4 className="font-semibold tracking-wide text-xs uppercase text-brand-700 mb-4">{title}</h4>
      <div className="space-y-3">
        {rows.map(row => (
          <div key={row.key} className="space-y-1">
            <div className="flex items-center justify-between gap-3">
              <span className="text-sm font-semibold leading-snug" style={{ color: row.color }}>{row.label}</span>
              <span className="shrink-0 text-right text-neutral-500 font-mono text-xs">{row.caption}</span>
            </div>
            <div className="h-2.5 rounded-full bg-neutral-100 overflow-hidden">
              <div className="h-full rounded-full transition-all" style={{ width: `${row.percentage}%`, backgroundColor: row.color }} />
            </div>
          </div>
        ))}
      </div>
    </Panel>
  );
}
