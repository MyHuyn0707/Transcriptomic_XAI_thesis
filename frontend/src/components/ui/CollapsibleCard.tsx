import React, { useState } from 'react';
import { ChevronDown, ChevronUp } from '@tailgrids/icons';
import { cn } from '../../lib/utils';

interface CollapsibleCardProps {
  accent?: 'brand' | 'neutral';
  title: React.ReactNode;
  children: React.ReactNode;
  defaultOpen?: boolean;
}

const ACCENT_STYLES: Record<Required<CollapsibleCardProps>['accent'], { border: string, hover: string }> = {
  brand: { border: 'border-l-brand-600', hover: 'group-hover:text-brand-700' },
  neutral: { border: 'border-l-neutral-500', hover: 'group-hover:text-neutral-700' },
};

/** Bordered, left-accented card with its own collapse toggle — for long
 * content sections that shouldn't force their parent panel to scroll just to
 * see what's below them. */
export default function CollapsibleCard({ accent = 'brand', title, children, defaultOpen = true }: CollapsibleCardProps) {
  const [open, setOpen] = useState(defaultOpen);
  const { border, hover } = ACCENT_STYLES[accent];
  return (
    <div className={cn('bg-white rounded-xl border border-neutral-200 shadow-sm p-5 border-l-4', border)}>
      <div
        className="flex items-center justify-between cursor-pointer group mb-1"
        onClick={() => setOpen(o => !o)}
      >
        <h3 className={cn('text-base font-bold text-neutral-800 flex items-center gap-2 transition-colors', hover)}>
          {title}
        </h3>
        <button className="p-1.5 rounded-full hover:bg-neutral-100 text-neutral-400 transition-colors shrink-0">
          {open ? <ChevronUp size={18} /> : <ChevronDown size={18} />}
        </button>
      </div>
      {open && (
        <div className="mt-4 animate-in fade-in slide-in-from-top-2 duration-200">{children}</div>
      )}
    </div>
  );
}
