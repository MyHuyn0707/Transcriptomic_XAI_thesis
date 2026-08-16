import React from 'react';
import { cn } from '../../lib/utils';

type BadgeTone = 'neutral' | 'brand' | 'success' | 'warning' | 'danger' | 'info';

interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  tone?: BadgeTone;
  /** Raw hex for a per-class categorical color (see lib/palette.ts) instead
   * of one of the fixed semantic tones — overrides `tone` when given. */
  color?: string;
}

const TONE_CLASSES: Record<BadgeTone, string> = {
  neutral: 'bg-neutral-100 text-neutral-600',
  brand: 'bg-brand-100 text-brand-700',
  success: 'bg-success-100 text-success-700',
  warning: 'bg-warning-100 text-warning-700',
  danger: 'bg-danger-100 text-danger-700',
  info: 'bg-info-100 text-info-700',
};

export default function Badge({ tone = 'neutral', color, className, style, children, ...rest }: BadgeProps) {
  const colorStyle = color ? { backgroundColor: `${color}1f`, color, borderColor: `${color}40` } : undefined;
  return (
    <span
      className={cn(
        'inline-flex items-center px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider',
        color ? 'border' : TONE_CLASSES[tone],
        className,
      )}
      style={{ ...colorStyle, ...style }}
      {...rest}
    >
      {children}
    </span>
  );
}
