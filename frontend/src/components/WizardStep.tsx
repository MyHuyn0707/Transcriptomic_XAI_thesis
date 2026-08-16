import React from 'react';
import { ChevronDown, ChevronUp } from 'lucide-react';
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
    <section className="relative">
      <div className={cn("flex items-start gap-4 mb-4 transition-opacity duration-300", !ready && "opacity-50 grayscale")}>
        <button
          onClick={onOpen}
          disabled={!ready}
          className={cn(
            "w-12 h-12 shrink-0 bg-white border-2 rounded-full flex items-center justify-center shadow-sm z-10 transition-colors",
            ready ? "border-brand-500 text-brand-600 hover:bg-brand-50" : "border-neutral-200 text-neutral-500 cursor-not-allowed",
          )}
        >
          <span className="font-bold text-lg">{number}</span>
        </button>
        <div className="pt-2 w-full">
          <div className="flex items-center justify-between cursor-pointer group mb-4" onClick={onOpen}>
            <h2 className="text-lg font-semibold text-neutral-900 group-hover:text-brand-700 transition-colors">{title}</h2>
            <span className="text-neutral-500 group-hover:text-brand-600 transition-colors">
              {active ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
            </span>
          </div>
          <div className={cn("bg-white p-5 rounded-2xl shadow-sm border border-neutral-200 relative overflow-hidden", !active && "hidden")}>
            {!ready && <div className="absolute inset-0 z-20 bg-neutral-50/50"></div>}
            {children}
          </div>
        </div>
      </div>
    </section>
  );
}
