import { test, expect } from '@playwright/test';
import { mockApi } from './fixtures/mockApi';
import { openApp, selectDataset, loadSplit, loadFeatureSelection, trainModel, predictSample } from './fixtures/wizard';

// The user's core ask for this suite: when the backend fails, the app must
// surface that failure visibly (not go blank, not leave a button stuck
// disabled forever) instead of silently swallowing it.
test.describe('Backend failures surface visibly instead of being swallowed', () => {
  test('split failure shows an error status and re-enables the action button', async ({ page }) => {
    await mockApi(page, { fail: ['overview', 'split'] });
    await openApp(page);
    await selectDataset(page);
    await page.getByRole('heading', { name: 'Xử lý & Chia Dữ liệu' }).click();

    const loadButton = page.getByRole('button', { name: 'Tải dữ liệu có sẵn' });
    await loadButton.click();

    await expect(page.getByText(/mock lỗi/)).toBeVisible();
    await expect(loadButton).toBeEnabled();
  });

  test('feature-selection failure shows an error status and re-enables the action button', async ({ page }) => {
    await mockApi(page, { fail: ['fs'] });
    await openApp(page);
    await selectDataset(page);
    await loadSplit(page);
    await page.getByRole('heading', { name: 'Trích xuất đặc trưng' }).click();

    const loadButton = page.getByRole('button', { name: 'Tải kết quả có sẵn' });
    await loadButton.click();

    await expect(page.getByText(/mock lỗi/)).toBeVisible();
    await expect(loadButton).toBeEnabled();
  });

  test('model training failure shows an error status and re-enables the action button', async ({ page }) => {
    await mockApi(page, { fail: ['modelTrain'] });
    await openApp(page);
    await selectDataset(page);
    await loadSplit(page);
    await loadFeatureSelection(page);
    await page.getByRole('heading', { name: 'Mô Hình' }).click();

    const trainButton = page.getByRole('button', { name: 'Huấn luyện lại' });
    await trainButton.click();

    await expect(page.getByText(/mock lỗi/)).toBeVisible();
    await expect(trainButton).toBeEnabled();
  });

  test('prediction failure renders the dedicated error card, not a blank panel', async ({ page }) => {
    await mockApi(page, { fail: ['predict'] });
    await openApp(page);
    await selectDataset(page);
    await loadSplit(page);
    await loadFeatureSelection(page);
    await trainModel(page);
    await predictSample(page, 'GSM725578');

    await expect(page.getByText('Lỗi khi thực nghiệm')).toBeVisible();
    await expect(page.getByText(/mock lỗi/)).toBeVisible();
    // The predict button itself must recover too, not stay stuck spinning.
    await expect(page.getByRole('button', { name: 'Dự Đoán Kết Quả' })).toBeEnabled();
  });
});
