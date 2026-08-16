# E2E test suite

Runs the wizard end-to-end against a **mocked backend** (`tests/fixtures/mockApi.ts`)
instead of a real one — every `/api/*` call is intercepted and answered with
fixture data modeled on a real session (dataset GSE29354). That means:

- Tests never depend on the real Python backend, a trained model, or any
  dataset actually existing on disk.
- Tests are fast and deterministic — no real model training, no waiting on
  job polling.
- Tests can also simulate a **failing** backend (see `mockApi(page, { fail: [...] })`)
  to check the UI surfaces errors instead of silently swallowing them.

## Running

```bash
npm run dev          # in one terminal — must be running on :3000
npm test             # in another terminal — headless run
npm run test:ui      # interactive UI mode (see each step, time-travel)
```

`playwright.config.ts` will also auto-start `npm run dev` for you if port
3000 isn't already listening (`reuseExistingServer: true` — if you already
have the dev server running, it reuses that one instead of starting a
second).

## What's covered

| File | Covers |
|---|---|
| `happy-path.spec.ts` | Full wizard walkthrough (dataset → split → feature selection → model → prediction), zero console errors; changing the dataset resets every downstream step. |
| `color-consistency.spec.ts` | Regression guard: a class must render the same color in Bước 2 (split chart) and Bước 5 (prediction) — this exact bug was found and fixed during design review. |
| `confusion-matrix.spec.ts` | Regression guard: 0 / 1 / max count cells in the confusion matrix must have visibly different shading, not collapse to the same faint tint. |
| `error-handling.spec.ts` | Each step's action (split/feature-selection/model/predict) surfaces a visible error and re-enables its button when the backend fails — nothing gets silently swallowed or left stuck. |

## Adding a new case

1. If it needs new fixture data or a new endpoint, add it to `tests/fixtures/mockApi.ts`.
2. Reuse the step helpers in `tests/fixtures/wizard.ts` (`selectDataset`,
   `loadSplit`, `loadFeatureSelection`, `trainModel`, `predictSample`,
   `runFullWizard`) instead of re-deriving selectors per test.
3. Prefer asserting on visible text/roles a user would actually see, not
   implementation details (class names, internal state).
