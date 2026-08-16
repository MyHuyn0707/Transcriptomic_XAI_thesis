// A class's Vietnamese display name if the dataset description defines one
// (same name shown in Dataset Overview), else just the raw label — avoids
// showing only the raw internal class label in the prediction report.
export function displayLabel(raw: string, displayNames?: Record<string, string>) {
  return displayNames?.[raw] || raw;
}

export interface ClassificationReportRow {
  precision: number;
  recall: number;
  f1: number;
  support: number;
}

export function classificationReport(cm: number[][]): ClassificationReportRow[] {
  return cm.map((row, i) => {
    const support = row.reduce((a, b) => a + b, 0);
    const colSum = cm.reduce((acc, r) => acc + r[i], 0);
    const tp = cm[i][i];
    const precision = colSum > 0 ? tp / colSum : 0;
    const recall = support > 0 ? tp / support : 0;
    const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0;
    return { precision, recall, f1, support };
  });
}
