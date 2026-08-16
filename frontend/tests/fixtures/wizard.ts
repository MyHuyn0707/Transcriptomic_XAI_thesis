import { Page, expect } from '@playwright/test';
import { DATASET_NAME } from './mockApi';

// Small step-by-step helpers shared across specs — each mirrors exactly the
// clicks a real user makes, in the same order as the manual QA pass this
// suite is meant to replace/guard.

export async function openApp(page: Page) {
  await page.goto('/');
}

export async function selectDataset(page: Page) {
  await page.getByRole('combobox').first().selectOption({ label: DATASET_NAME });
}

export async function openStep(page: Page, title: string | RegExp) {
  await page.getByRole('heading', { name: title }).click();
}

export async function loadSplit(page: Page) {
  await openStep(page, 'Xử lý & Chia Dữ liệu');
  await page.getByRole('button', { name: 'Tải dữ liệu có sẵn' }).click();
}

export async function loadFeatureSelection(page: Page) {
  await openStep(page, 'Trích xuất đặc trưng');
  await page.getByRole('button', { name: 'Tải kết quả có sẵn' }).click();
}

export async function trainModel(page: Page) {
  await openStep(page, 'Mô Hình');
  await page.getByRole('button', { name: 'Huấn luyện lại' }).click();
}

/** @param sampleId the GEO sample_id — also the <option value>, so this
 * matches regardless of the "(nhãn thật: ...)" label text next to it. */
export async function predictSample(page: Page, sampleId: string) {
  await openStep(page, 'Kiểm Thử (Thực Nghiệm)');
  await page.getByRole('combobox').last().selectOption(sampleId);
  await page.getByRole('button', { name: 'Dự Đoán Kết Quả' }).click();
}

/** Walks the whole wizard through to a completed prediction — the shared
 * "golden path" setup most specs start from before asserting on one
 * specific thing. */
export async function runFullWizard(page: Page) {
  await openApp(page);
  await selectDataset(page);
  await loadSplit(page);
  await expect(page.getByText('44 / 9')).toBeVisible();
  await loadFeatureSelection(page);
  await expect(page.getByText('16', { exact: true }).first()).toBeVisible();
  await trainModel(page);
  await expect(page.getByText('88.9%')).toBeVisible();
  await predictSample(page, 'GSM725578');
}
