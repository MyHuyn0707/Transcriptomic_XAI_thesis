// Shared categorical palette so a given class label gets the SAME color
// everywhere it appears (DatasetOverview class pills/chart, rule-class
// badges in the prediction report, etc.) — index by position in the
// dataset's canonical class_labels array, not by first-seen order per
// component, so the color assignment stays stable across the whole app.
export const PALETTE = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'];
export const DROPPED_COLOR = '#cbd5e1';

export function classColor(label: string, allLabels: string[]): string {
  const idx = allLabels.indexOf(label);
  return idx >= 0 ? PALETTE[idx % PALETTE.length] : DROPPED_COLOR;
}
