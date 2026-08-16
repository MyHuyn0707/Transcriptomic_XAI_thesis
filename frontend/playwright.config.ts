import { defineConfig, devices } from '@playwright/test';

// E2E suite runs against a mocked backend (see tests/fixtures/mockApi.ts) —
// no real /api/* calls ever leave the browser, so these tests never depend
// on a live model-training backend or a specific dataset actually existing
// on disk. That's what makes them safe to run standalone with `npx
// playwright test` instead of only interactively through Claude.
export default defineConfig({
  testDir: './tests',
  fullyParallel: true,
  retries: process.env.CI ? 1 : 0,
  reporter: 'list',
  use: {
    baseURL: 'http://localhost:3000',
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
  },
  projects: [
    { name: 'chromium', use: { ...devices['Desktop Chrome'] } },
  ],
  webServer: {
    command: 'npm run dev',
    url: 'http://localhost:3000',
    reuseExistingServer: true,
    timeout: 30_000,
  },
});
