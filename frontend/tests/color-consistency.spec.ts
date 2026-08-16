import { test, expect } from '@playwright/test';
import { mockApi } from './fixtures/mockApi';
import { openApp, selectDataset, loadSplit, loadFeatureSelection, trainModel, predictSample } from './fixtures/wizard';

// Regression guard for a real bug found during design review: classColor()
// used to be indexed against modelStats.labels (sklearn's alphabetical
// order) in Step 4/5, while Step 1/2 indexed the SAME palette against
// raw_class_counts' insertion order — so a class could render blue in the
// split chart and orange in the prediction card. The mock's MODEL_STATS
// deliberately uses a different label order than SPLIT_STATS specifically
// to catch this if it ever regresses.
test('a class keeps the same color from Bước 2 through Bước 5', async ({ page }) => {
  await mockApi(page);
  await openApp(page);
  await selectDataset(page);
  await loadSplit(page);

  // Step 2's "Số lớp" chip renders the class label text colored via
  // classColor(label, raw_class_counts-keys) — see DatasetSplit.tsx.
  const step2Chip = page.locator('span', { hasText: /^epi$/ }).first();
  await expect(step2Chip).toBeVisible();
  const step2Color = await step2Chip.evaluate((el) => getComputedStyle(el).color);

  await loadFeatureSelection(page);
  await trainModel(page);
  await predictSample(page, 'GSM725578');

  // Step 5's "Dự đoán theo Mô hình" value renders the SAME classification
  // ("epi") colored via classColor(label, canonicalClassLabels(state)).
  const step5Value = page.getByText('epi', { exact: true }).last();
  await expect(step5Value).toBeVisible();
  const step5Color = await step5Value.evaluate((el) => getComputedStyle(el).color);

  expect(step5Color, 'epi should render the same color in Bước 5 as it does in Bước 2').toBe(step2Color);
});
