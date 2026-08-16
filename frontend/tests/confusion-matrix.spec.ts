import { test, expect } from '@playwright/test';
import { mockApi } from './fixtures/mockApi';
import { runFullWizard } from './fixtures/wizard';

function parseAlpha(rgba: string): number {
  const match = rgba.match(/rgba?\(([^)]+)\)/);
  if (!match) return NaN;
  const parts = match[1].split(',').map((p) => parseFloat(p.trim()));
  return parts.length === 4 ? parts[3] : 1;
}

// Regression guard for the confusion-matrix contrast fix: a linear
// val/maxVal color scale made every small count (0, 1) render as almost the
// same faint tint next to one dominant cell. The fixture's matrix
// ([[1,1,0],[0,6,0],[0,0,1]]) has exactly that shape — one dominant 6 next
// to a mix of 0s and 1s — so this asserts the cell shading still tells them
// apart instead of only telling apart "empty" vs "the max".
test('confusion matrix cells with different counts have visibly different shading', async ({ page }) => {
  await mockApi(page);
  await runFullWizard(page);

  // Reaching Bước 5 auto-collapses the "Tổng Quan Mô Hình" panel (see
  // active_step_set in workspaceReducer.ts) — reopen it to see the matrix.
  await page.getByRole('heading', { name: 'Tổng Quan Mô Hình' }).click();

  const matrixCells = page.locator('[title^="Thật:"]');
  await expect(matrixCells.first()).toBeVisible();

  const cellColors = await matrixCells.evaluateAll((els) =>
    els.map((el) => ({
      title: el.getAttribute('title'),
      bg: getComputedStyle(el).backgroundColor,
    })),
  );

  const zeroCell = cellColors.find((c) => c.title === 'Thật: biphasic, Dự đoán: sarc'); // value 0
  const oneCell = cellColors.find((c) => c.title === 'Thật: biphasic, Dự đoán: biphasic'); // value 1
  const maxCell = cellColors.find((c) => c.title === 'Thật: epi, Dự đoán: epi'); // value 6

  expect(zeroCell, 'expected a 0-value cell in the matrix').toBeTruthy();
  expect(oneCell, 'expected a 1-value cell in the matrix').toBeTruthy();
  expect(maxCell, 'expected the max-value cell in the matrix').toBeTruthy();

  const zeroAlpha = parseAlpha(zeroCell!.bg);
  const oneAlpha = parseAlpha(oneCell!.bg);
  const maxAlpha = parseAlpha(maxCell!.bg);

  // Monotonically increasing with the underlying count — a 0 cell must be
  // meaningfully lighter than a 1 cell, which must be meaningfully lighter
  // than the max cell (not all three collapsing to a near-identical tint).
  expect(oneAlpha - zeroAlpha).toBeGreaterThan(0.15);
  expect(maxAlpha - oneAlpha).toBeGreaterThan(0.15);
});
