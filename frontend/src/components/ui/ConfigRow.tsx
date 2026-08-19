import React from 'react';
import { cn } from '../../lib/utils';

interface ConfigRowProps {
  label: React.ReactNode;
  value: React.ReactNode;
  mono?: boolean;
  labelMono?: boolean;
  valueClassName?: string;
}

/** Read-only "label ... value" row for config/result summaries — the shape
 * repeated across FeatureExtractionOverview's Cấu hình/Kết quả lists and
 * Step4Model's hyperparameter/filter-config display. */
export default function ConfigRow({ label, value, mono = true, labelMono = false, valueClassName }: ConfigRowProps) {
  return (
    <div className="flex justify-between gap-3 py-2 border-b border-neutral-200 border-dashed">
      <span className={cn('text-neutral-800 text-sm', labelMono && 'font-mono')}>{label}</span>
      <span className={cn('text-neutral-800 font-medium text-sm', mono && 'font-mono', valueClassName)}>{value}</span>
    </div>
  );
}
