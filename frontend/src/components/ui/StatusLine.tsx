import React from 'react';
import { AlertTriangle } from 'lucide-react';
import { CheckCircle1, InfoCircle, Reload } from '@tailgrids/icons';
import { cn } from '../../lib/utils';

interface StatusLineProps {
  log: string;
  loading?: boolean;
  className?: string;
}

const ERROR_MARKER = '[LỖI]';
const SUCCESS_MARKERS = ['[THÀNH CÔNG]', '[OK]', '[CACHE]'];

/** One-line replacement for the old scrolling dark-terminal log box — shows
 * only the most recent `[MARKER] ...` line accumulated by a step's action
 * handler, with an icon/tone derived from that marker (or a spinner while
 * the action is still running), instead of a multi-line console. */
export default function StatusLine({ log, loading, className }: StatusLineProps) {
  const lines = log.split('\n').map(l => l.trim()).filter(Boolean);
  const lastLine = lines[lines.length - 1];
  if (!lastLine) return null;

  const isError = lastLine.includes(ERROR_MARKER);
  const isSuccess = !loading && !isError && SUCCESS_MARKERS.some(m => lastLine.includes(m));
  const text = lastLine.replace(/^\[[^\]]+\]\s*/, '');

  const Icon = loading ? Reload : isError ? AlertTriangle : isSuccess ? CheckCircle1 : InfoCircle;
  const toneClass = isError ? 'text-danger-600' : isSuccess ? 'text-success-600' : 'text-neutral-600';

  return (
    <div className={cn('flex items-center gap-2 text-sm mt-3', toneClass, className)} title={lastLine}>
      <Icon size={14} className={cn('shrink-0', loading && 'animate-spin')} />
      <span className="truncate">{text}</span>
    </div>
  );
}
