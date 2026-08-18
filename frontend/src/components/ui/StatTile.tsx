import React from 'react';
import { cn } from '../../lib/utils';
import FieldLabel from './FieldLabel';

interface StatTileProps {
  label: React.ReactNode;
  value: React.ReactNode;
  mono?: boolean;
  truncate?: boolean;
  className?: string;
}

/** Small bordered "eyebrow label + bold value" tile — the dataset-stat shape
 * repeated across Step1Dataset's dataset-info card and DatasetOverview's
 * summary row (platform/samples/features/classes, originally 6 near-identical
 * copies of this same markup). */
export default function StatTile({ label, value, mono, truncate, className }: StatTileProps) {
  return (
    <div className={cn('rounded-lg border border-neutral-200 px-3 py-2.5', className)}>
      <FieldLabel>{label}</FieldLabel>
      <p className={cn('text-lg font-bold text-neutral-900 mt-0.5', mono && 'font-mono', truncate && 'truncate')}>
        {value == null || value === '' ? 'N/A' : value}
      </p>
    </div>
  );
}
