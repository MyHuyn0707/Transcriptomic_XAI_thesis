import React from 'react';
import { cn } from '../../lib/utils';

interface FieldLabelProps {
  as?: 'span' | 'p' | 'div';
  className?: string;
  children: React.ReactNode;
}

/** The small uppercase eyebrow-label shape used above stat values, section
 * captions and metadata rows throughout the app. */
export default function FieldLabel({ as: Tag = 'span', className, children }: FieldLabelProps) {
  return (
    <Tag className={cn('text-[.6875rem] font-bold uppercase tracking-widest text-neutral-500 text-pretty', className)}>
      {children}
    </Tag>
  );
}
