import React, { useRef, useState } from 'react';
import { UploadCloud, FileText, Close } from '@tailgrids/icons';
import { cn } from '../../lib/utils';

function formatBytes(bytes: number): string {
  if (bytes <= 0) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB'];
  const i = Math.min(units.length - 1, Math.floor(Math.log(bytes) / Math.log(1024)));
  return `${(bytes / Math.pow(1024, i)).toFixed(i === 0 ? 0 : 1)} ${units[i]}`;
}

interface FilePickerProps {
  value: File | null;
  onChange: (file: File | null) => void;
  accept?: string;
  disabled?: boolean;
  placeholder?: string;
  className?: string;
}

/** Tailgrids' component registry has no dropzone/file-upload component, so
 * this one is hand-built (not pulled from their registry like the other
 * components under tailgrids/core/) — reuses the same bridged tokens
 * (border-base-300, bg-input-background, brand/danger scale) so it still
 * reads as part of the same design system. Drag-and-drop is layered on top
 * of a plain hidden <input type="file">, so it still works via click/
 * keyboard when JS drag events aren't available (e.g. some mobile browsers). */
export default function FilePicker({ value, onChange, accept, disabled, placeholder = 'Kéo thả file vào đây hoặc bấm để chọn', className }: FilePickerProps) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [isDragOver, setIsDragOver] = useState(false);

  return (
    <div className={className}>
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        disabled={disabled}
        onChange={e => onChange(e.target.files?.[0] || null)}
        className="hidden"
      />

      {value ? (
        <div className="flex items-center gap-3 rounded-lg border border-base-300 bg-input-background px-3.5 py-2.5">
          <FileText size={18} className="text-brand-600 shrink-0" />
          <div className="flex-1 min-w-0">
            <p className="text-sm font-medium text-neutral-800 truncate" title={value.name}>{value.name}</p>
            <p className="text-xs text-neutral-500">{formatBytes(value.size)}</p>
          </div>
          <button
            type="button"
            onClick={() => { onChange(null); if (inputRef.current) inputRef.current.value = ''; }}
            disabled={disabled}
            title="Bỏ chọn file"
            className="p-1.5 rounded-full text-neutral-400 hover:text-danger-600 hover:bg-danger-50 transition-colors shrink-0 disabled:opacity-60 disabled:pointer-events-none"
          >
            <Close size={14} />
          </button>
        </div>
      ) : (
        <button
          type="button"
          onClick={() => inputRef.current?.click()}
          onDragOver={e => { e.preventDefault(); setIsDragOver(true); }}
          onDragLeave={() => setIsDragOver(false)}
          onDrop={e => {
            e.preventDefault();
            setIsDragOver(false);
            const file = e.dataTransfer.files?.[0];
            if (file) onChange(file);
          }}
          disabled={disabled}
          className={cn(
            'w-full flex flex-col items-center gap-1.5 rounded-lg border-2 border-dashed px-4 py-5 text-center transition-colors',
            isDragOver ? 'border-brand-500 bg-brand-50' : 'border-base-300 hover:border-brand-400 hover:bg-neutral-50',
            disabled && 'opacity-60 cursor-not-allowed pointer-events-none',
          )}
        >
          <UploadCloud size={20} className="text-neutral-400" />
          <span className="text-sm text-neutral-600">{placeholder}</span>
          {accept && <span className="text-xs text-neutral-500">{accept}</span>}
        </button>
      )}
    </div>
  );
}
