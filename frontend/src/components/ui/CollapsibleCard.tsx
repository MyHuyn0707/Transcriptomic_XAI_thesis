import React, { useState } from 'react';
import { ChevronDown, ChevronUp } from '@tailgrids/icons';

interface CollapsibleCardProps {
  color: string;
  title: React.ReactNode;
  children: React.ReactNode;
  defaultOpen?: boolean;
}

/** Bordered, left-accented card with its own collapse toggle — for long
 * content sections that shouldn't force their parent panel to scroll just to
 * see what's below them. */
export default function CollapsibleCard({ color, title, children, defaultOpen = true }: CollapsibleCardProps) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="bg-white rounded-xl border border-neutral-200 shadow-sm p-5 border-l-4" style={{ borderLeftColor: color }}>
      <div
        className="flex items-center justify-between cursor-pointer group mb-1"
        onClick={() => setOpen(o => !o)}
      >
        <h3 className="text-base font-bold text-neutral-800 flex items-center gap-2 group-hover:text-success-700 transition-colors">
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
