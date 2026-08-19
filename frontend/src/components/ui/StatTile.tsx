import React from "react";
import { cn } from "../../lib/utils";
import FieldLabel from "./FieldLabel";

interface StatTileProps extends React.HTMLAttributes<HTMLDivElement> {
  label: React.ReactNode;
  value: React.ReactNode;
  mono?: boolean;
  truncate?: boolean;
  valueClassName?: string;
  /** Inline color override for the label — for a per-class categorical hue
   * (from lib/palette.ts) that isn't one of the app's semantic tokens, so a
   * Tailwind className can't express it. */
  labelStyle?: React.CSSProperties;
}

/** Small bordered "eyebrow label + bold value" tile — the dataset-stat shape
 * repeated across Step1Dataset's dataset-info card and DatasetOverview's
 * summary row (platform/samples/features/classes, originally 6 near-identical
 * copies of this same markup). */
export default function StatTile({ label, value, mono, truncate, className, valueClassName, labelStyle, ...rest }: StatTileProps) {
  return (
    <div className={cn("rounded-lg border border-neutral-200 px-3 py-2.5", className)} {...rest}>
      <FieldLabel style={labelStyle}>{label}</FieldLabel>
      <p
        className={cn(
          "text-lg font-bold text-neutral-800 mt-0.5",
          mono && "font-mono",
          truncate && "truncate",
          valueClassName,
        )}
      >
        {value == null || value === "" ? "N/A" : value}
      </p>
    </div>
  );
}
