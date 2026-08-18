import React from 'react';
import { cn } from '../../lib/utils';

type PanelSurface = 'neutral' | 'brand' | 'success' | 'warning' | 'danger' | 'info';
type PanelPadding = 'sm' | 'md' | 'lg';

interface PanelProps extends React.HTMLAttributes<HTMLDivElement> {
  surface?: PanelSurface;
  padding?: PanelPadding;
}

const SURFACE_CLASSES: Record<PanelSurface, string> = {
  neutral: 'bg-neutral-50/50 border-neutral-100',
  brand: 'bg-brand-50/50 border-brand-100',
  success: 'bg-success-50/50 border-success-100',
  warning: 'bg-warning-50/50 border-warning-200',
  danger: 'bg-danger-50/50 border-danger-200',
  info: 'bg-info-50/50 border-info-200',
};

const PADDING_CLASSES: Record<PanelPadding, string> = {
  sm: 'p-3',
  md: 'p-4',
  lg: 'p-5',
};

export default function Panel({ surface = 'neutral', padding = 'md', className, ...rest }: PanelProps) {
  return (
    <div
      className={cn('rounded-xl border', SURFACE_CLASSES[surface], PADDING_CLASSES[padding], className)}
      {...rest}
    />
  );
}
