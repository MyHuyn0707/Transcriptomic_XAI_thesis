import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

// A caught value isn't necessarily an Error (a thrown string, a rejected
// fetch body, ...) — this is the one place that assumption gets made instead
// of at every catch block's `e.message`.
export function errorMessage(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}
