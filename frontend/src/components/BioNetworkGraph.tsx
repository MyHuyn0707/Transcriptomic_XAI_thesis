import React from 'react';
import { Microscope } from 'lucide-react';

interface BioNetworkGraphProps {
  graphType?: string;
}

// Biology Graph is intentionally left empty for this iteration (out of scope
// per user request) — no mock network is wired in, so the panel reads as a
// clear "not yet built" placeholder rather than fabricated data.
export default function BioNetworkGraph({ graphType }: BioNetworkGraphProps) {
  return (
    <div className="w-full h-[500px] flex flex-col items-center justify-center text-slate-400 bg-slate-50 border border-slate-100">
      <Microscope size={32} className="text-slate-300 mb-4" />
      <p className="text-sm font-medium text-slate-500">Biology Graph — sắp ra mắt</p>
      <p className="text-xs text-slate-400 mt-1">({graphType})</p>
    </div>
  );
}
