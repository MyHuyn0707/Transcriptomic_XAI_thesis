import React from 'react';
import { ChevronDown, ChevronUp } from '@tailgrids/icons';
import { cn } from '../lib/utils';

interface WizardStepProps {
  number: number;
  title: string;
  active: boolean;
  ready: boolean;
  onOpen: () => void;
  children: React.ReactNode;
}

/** Numbered-circle header + collapsible card shared by all 5 left-column
 * wizard steps. A step stays grayed out and closed to the user until its
 * prerequisite step has produced something (ready) — the same gating each
 * step used to repeat around its own copy of this shell. */
export default function WizardStep({ number, title, active, ready, onOpen, children }: WizardStepProps) {
  return (
    <section className={cn("relative transition-opacity duration-300", !ready && "opacity-50 grayscale")}>
      <div className="flex items-center gap-4 mb-4 cursor-pointer group" onClick={onOpen}>
        <button
          onClick={onOpen}
          disabled={!ready}
          className={cn(
            // w-10 h-10 matches the icon-badge circle size used everywhere
            // else in the app (DatasetOverview, FeatureExtractionOverview,
            // the right-column panel headers) — this was the one place
            // still using its own larger, one-off w-12 h-12.
            "w-10 h-10 shrink-0 bg-white border-2 rounded-full flex items-center justify-center shadow-sm z-10 transition-colors",
            ready ? "border-brand-500 text-brand-600 hover:bg-brand-50" : "border-neutral-200 text-neutral-600 cursor-not-allowed",
          )}
        >
          <span className="font-bold text-sm">{number}</span>
        </button>
        <h2 className="flex-1 text-base font-semibold text-neutral-900 group-hover:text-brand-700 transition-colors">{title}</h2>
        <span className="text-neutral-600 group-hover:text-brand-600 transition-colors">
          {active ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
        </span>
      </div>
      {/* Content row — its own full-width row below the header, instead of
          a second column indented past the number circle, so every step's
          card fills the whole sidebar width. */}
      <div className={cn("bg-white p-5 rounded-2xl shadow-xs border border-neutral-200 relative overflow-hidden", !active && "hidden")}>
        {!ready && <div className="absolute inset-0 z-20 bg-neutral-50/50"></div>}
        {children}
      </div>
    </section>
  );
}
