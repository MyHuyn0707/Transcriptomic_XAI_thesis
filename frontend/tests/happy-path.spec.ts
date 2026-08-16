import { test, expect } from '@playwright/test';
import { mockApi } from './fixtures/mockApi';
import { openApp, selectDataset, loadSplit, loadFeatureSelection, trainModel, predictSample } from './fixtures/wizard';

// End-to-end walk through all 5 wizard steps with a mocked backend — guards
// against the kind of regression that's easy to introduce silently while
// refactoring one step (e.g. Phase C/D's reducer/component split): a step
// further down the wizard quietly stops receiving the data it needs.
test.describe('Wizard happy path', () => {
  test.beforeEach(async ({ page }) => {
    await mockApi(page);
  });

  test('walks dataset -> split -> feature selection -> model -> prediction with no console errors', async ({ page }) => {
    const consoleErrors: string[] = [];
    page.on('console', (msg) => {
      if (msg.type() === 'error' && !msg.text().includes('favicon')) consoleErrors.push(msg.text());
    });
    page.on('pageerror', (err) => consoleErrors.push(err.message));

    await openApp(page);
    await selectDataset(page);

    // Step 1: dataset info card populated.
    await expect(page.getByText('GEO-Mesothelioma-29354')).toBeVisible();
    await expect(page.getByText('22,215').first()).toBeVisible();

    // Step 2: split stats.
    await loadSplit(page);
    await expect(page.getByText('44 / 9')).toBeVisible();
    await expect(page.getByText('0.15', { exact: true })).toBeVisible();

    // Step 3: feature-selection stats.
    await loadFeatureSelection(page);
    await expect(page.getByText('Tổng quan Trích xuất Đặc trưng')).toBeVisible();
    await expect(page.getByText('16', { exact: true }).first()).toBeVisible();

    // Step 4: trained model overview + confusion matrix + rules.
    await trainModel(page);
    await expect(page.getByText('Tổng Quan Mô Hình')).toBeVisible();
    await expect(page.getByText('88.9%')).toBeVisible();
    await expect(page.getByText('86.3%')).toBeVisible();
    await expect(page.getByText('Ma trận nhầm lẫn')).toBeVisible();

    // Step 5: prediction result.
    await predictSample(page, 'GSM725578');
    await expect(page.getByText('Báo cáo Phân loại')).toBeVisible();
    await expect(page.getByText('Dự đoán theo Mô hình (RF/DT)')).toBeVisible();
    await expect(page.getByText('Khớp 13 rules')).toBeVisible();

    expect(consoleErrors, `Unexpected console errors:\n${consoleErrors.join('\n')}`).toEqual([]);
  });

  test('changing the dataset resets every downstream step', async ({ page }) => {
    await openApp(page);
    await selectDataset(page);
    await loadSplit(page);
    await expect(page.getByText('44 / 9')).toBeVisible();

    // Re-selecting "no dataset" should clear the split panel instead of
    // leaving stale data from the previous dataset on screen. Bước 1's own
    // panel is collapsed right now (Bước 2 is the active step), so reopen
    // it first — WizardStep hides an inactive step's content entirely.
    await page.getByRole('heading', { name: 'Chọn Dữ Liệu' }).click();
    await page.getByRole('button', { name: 'Chọn dataset' }).click();
    await page.getByRole('option', { name: '-- Chọn Dữ liệu --' }).click();
    await expect(page.getByText('44 / 9')).not.toBeVisible();
  });
});
