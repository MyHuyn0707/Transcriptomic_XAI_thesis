import React from 'react';
import { cn } from '../../lib/utils';

interface FieldLabelProps {
  as?: 'span' | 'p' | 'div';
  className?: string;
  style?: React.CSSProperties;
  children: React.ReactNode;
}

/** The small uppercase eyebrow-label shape used above stat values, section
 * captions and metadata rows throughout the app. */
export default function FieldLabel({ as: Tag = 'p', className, style, children }: FieldLabelProps) {
  return (
    <Tag className={cn('text-[.6875rem] font-bold uppercase tracking-wider text-neutral-500/90 text-pretty', className)} style={style}>
      {children}
    </Tag>
  );
}
