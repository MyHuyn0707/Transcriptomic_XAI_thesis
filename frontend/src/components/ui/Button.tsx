import React from 'react';
import { cn } from '../../lib/utils';

type ButtonVariant = 'primary' | 'secondary' | 'danger';
type ButtonSize = 'md' | 'lg';

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
}

const VARIANT_CLASSES: Record<ButtonVariant, string> = {
  primary: 'bg-brand-600 hover:bg-brand-700 text-white',
  secondary: 'bg-white hover:bg-neutral-50 text-neutral-700 border border-neutral-200',
  danger: 'bg-danger-600 hover:bg-danger-700 text-white',
};

const SIZE_CLASSES: Record<ButtonSize, string> = {
  md: 'py-2 px-3 rounded-lg text-sm',
  lg: 'py-2.5 px-4 rounded-xl text-sm',
};

export default function Button({ variant = 'primary', size = 'md', className, ...rest }: ButtonProps) {
  return (
    <button
      className={cn(
        'font-medium flex items-center justify-center gap-2 transition-colors shadow-sm disabled:opacity-60 disabled:cursor-not-allowed',
        VARIANT_CLASSES[variant],
        SIZE_CLASSES[size],
        className,
      )}
      {...rest}
    />
  );
}
