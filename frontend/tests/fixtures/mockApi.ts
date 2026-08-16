import type { Page, Route } from '@playwright/test';

// Realistic fixture data modeled on a real session against GSE29354
// (Pleural mesothelioma subtypes) — same dataset/numbers used throughout
// the manual design review, so behavior asserted here matches what a real
// backend actually returns.
export const DATASET_ID = 'GEO-Mesothelioma-29354';
export const DATASET_NAME = 'Pleural mesothelioma subtypes (GSE29354)';

export const DATASETS = [
  {
    id: DATASET_ID,
    name: DATASET_NAME,
    platform: 'GPL96',
    n_samples: 53,
    n_features: 22215,
    n_classes: 3,
    class_labels: ['epi', 'biphasic', 'sarc'],
    description: 'Mesothelioma tumor gene expression profiles',
    fs_models: { boruta: ['rf', 'dt'], mrmr_k50: ['rf', 'dt'] },
  },
];

// class_labels keys deliberately ordered epi -> biphasic -> sarc, matching
// raw_class_counts insertion order — this is the ONE canonical ordering
// every classColor() call should key off (see state/types.ts
// canonicalClassLabels). MODEL_STATS below deliberately uses a DIFFERENT
// order to make the color-consistency test meaningful.
export const SPLIT_STATS = {
  class_labels: ['epi', 'biphasic', 'sarc'],
  train_class_counts: { epi: 32, biphasic: 8, sarc: 4 },
  test_class_counts: { epi: 6, biphasic: 2, sarc: 1 },
  raw_class_counts: { epi: 38, biphasic: 10, sarc: 5 },
  dropped_classes: [],
  n_samples_total: 53,
  n_train: 44,
  n_test: 9,
  test_size: 0.15,
  min_samples_per_class: 4,
};

export const OVERVIEW = {
  ...SPLIT_STATS,
  dataset: DATASET_ID,
  visualizations: {},
  origin: {
    geo_accession: 'GSE29354',
    platform: 'GPL96',
    samples_kept: 53,
    annotation_rows: 22215,
    probes_with_gene_symbol: 21146,
    probes_from_annotation_file: 21146,
    probes_from_mygene: 0,
    probes_without_gene_symbol: 1069,
    affx_control_removed: 0,
    unique_probes: 22215,
    unique_genes: 14170,
    na_dropped: 0,
  },
  content: {
    title_vi: 'Hồ sơ biểu hiện gene của khối u trung biểu mô',
    organism: 'Homo sapiens',
    description_vi: 'Nghiên cứu xây dựng hồ sơ biểu hiện gene từ các khối u trung biểu mô màng phổi.',
    // No display_name_vi on purpose — keeps class chips rendering the raw
    // "epi"/"biphasic"/"sarc" code directly, which is what the
    // color-consistency test locates by text.
    classes: {
      epi: {},
      biphasic: {},
      sarc: {},
    },
  },
};

export const FS_STATS_BORUTA = {
  n_original_features: 22215,
  n_selected_features: 16,
  runtime_seconds: 282,
  feature_selection: 'boruta',
  dataset_name: DATASET_ID,
  framework: 'Classification Transcriptomic with XAI',
  n_samples: 45,
  selection_mode: 'confirmed',
  confirmed: 5,
  tentative: 11,
  rejected: 22188,
  n_estimators: 'auto',
  rf_n_estimators: 500,
  max_iter: 100,
  perc: 100,
  alpha: 0.05,
  class_weight: 'balanced',
  random_state: 42,
};

// Deliberately NOT epi/biphasic/sarc order (that's SPLIT_STATS' order) —
// mirrors sklearn's alphabetical classes_ order, which is exactly the
// mismatch that used to make classColor() paint "epi" a different color
// in Step 4/5 than in Step 1/2. See tests/color-consistency.spec.ts.
export const MODEL_STATS = {
  best_run_test_metrics: { accuracy: 0.889, f1_macro: 0.863 },
  n_rules: 60,
  confusion_matrix: [
    [1, 1, 0],
    [0, 6, 0],
    [0, 0, 1],
  ],
  class_labels: ['biphasic', 'epi', 'sarc'],
  hyperparams: { n_estimators: 200, max_depth: 5, min_samples_leaf: 2, class_weight: 'balanced_subsample' },
  rules_summary: {
    n_rules_raw: 1178,
    n_rules_passed_filter: 499,
    n_rules_kept: 60,
    n_dropped_by_filter: 679,
    n_dropped_by_cap: 439,
    filter_config: {
      min_confidence: 0.8, min_fidelity: 0, min_support: 0.05, min_abs_support: 3,
      max_conditions: 5, merge_same_gene: true, dedup: true, dedup_sig_figs: 2,
      merge_generalization: true, max_rules_per_class: 20, max_rules_total: 100,
    },
  },
};

export const RULES = {
  rules: [
    {
      rule_id: 1, text: 'IF GLRX>7.982 AND FCGR2A>8.055 THEN Class=sarc', matched: true,
      explanation: 'Mo ta sinh hoc mau.', metrics: { fidelity: 0.9, lift: 1.2, confidence: 0.95, support: 0.1 },
      consequent: { class_label: 'sarc' },
    },
    {
      rule_id: 2, text: 'IF FADS3<=7.973 AND FADS3<=7.574 THEN Class=epi', matched: true,
      explanation: 'Mo ta sinh hoc mau.', metrics: { fidelity: 0.92, lift: 1.1, confidence: 0.97, support: 0.3 },
      consequent: { class_label: 'epi' },
    },
    {
      rule_id: 3, text: 'IF MIR943>7.381 AND FAM57A>7.708 AND HLA-DPB1>9.9 THEN Class=biphasic', matched: true,
      explanation: 'Mo ta sinh hoc mau.', metrics: { fidelity: 0.85, lift: 1.3, confidence: 0.9, support: 0.08 },
      consequent: { class_label: 'biphasic' },
    },
  ],
};

export const GENES = [
  { gene: 'GLRX', gene_title: 'glutaredoxin', n_rules: 3, classes: 'sarc' },
  { gene: 'FADS3', gene_title: 'fatty acid desaturase 3', n_rules: 6, classes: 'epi' },
];

export const TEST_SAMPLES = [
  { sample_id: 'GSM725578', true_label: 'epi', row_index: 0 },
  { sample_id: 'GSM725567', true_label: 'biphasic', row_index: 1 },
  { sample_id: 'GSM725617', true_label: 'sarc', row_index: 2 },
];

export const PREDICT_RESPONSE = {
  matched_count: 13,
  rules: RULES.rules.map(r => ({
    rule_id: r.rule_id, text: r.text, matched: r.matched, explanation: r.explanation,
    sample_values: { GLRX: 9.76, FADS3: 7.39 }, consequent_label: r.consequent.class_label,
  })),
  classification: 'epi',
  class_votes: {
    epi: { count: 6, percentage: 46.2 },
    biphasic: { count: 4, percentage: 30.8 },
    sarc: { count: 3, percentage: 23.1 },
  },
  class_votes_over50: {
    biphasic: { count: 16, percentage: 44.4 },
    epi: { count: 12, percentage: 33.3 },
    sarc: { count: 8, percentage: 22.2 },
  },
  rule_prediction: 'epi',
  partial_matches: [],
  n_partial_matches_total: 36,
  n_rules_total: 60,
  true_label: 'epi',
  biomedical_summary: 'Mo hinh du doan mau thuoc lop epi.',
  llm_used: true,
};

export interface MockOptions {
  /** Endpoint keys to force into a 500 error response, to exercise
   * error-handling paths (see tests/error-handling.spec.ts). */
  fail?: Array<'overview' | 'split' | 'fsTrain' | 'fs' | 'modelTrain' | 'model' | 'predict'>;
}

async function json(route: Route, body: unknown, status = 200) {
  await route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(body) });
}

async function errorJson(route: Route, status: number, detail: string) {
  await route.fulfill({ status, contentType: 'application/json', body: JSON.stringify({ detail }) });
}

/** Intercepts every /api/* call the app makes and answers with fixture data
 * instead of hitting a real backend — makes the suite deterministic, fast,
 * and runnable with no model-training backend present at all. */
export async function mockApi(page: Page, opts: MockOptions = {}) {
  const fails = new Set(opts.fail || []);

  await page.route('**/api/**', async (route) => {
    const request = route.request();
    const { pathname } = new URL(request.url());
    const method = request.method();

    if (pathname === '/api/datasets' && method === 'GET') {
      return json(route, DATASETS);
    }
    if (/\/api\/datasets\/[^/]+\/overview$/.test(pathname)) {
      if (fails.has('overview')) return errorJson(route, 500, 'Không tải được overview (mock lỗi).');
      return json(route, OVERVIEW);
    }
    if (/\/api\/datasets\/[^/]+\/split\/preview$/.test(pathname)) {
      if (fails.has('split')) return errorJson(route, 500, 'Không chia được dữ liệu (mock lỗi).');
      return json(route, SPLIT_STATS);
    }
    if (/\/api\/datasets\/[^/]+\/feature-selection\/[^/]+\/train$/.test(pathname)) {
      if (fails.has('fsTrain')) return errorJson(route, 500, 'Trích xuất đặc trưng thất bại (mock lỗi).');
      return json(route, { job_id: 'job-fs-1' });
    }
    if (/\/api\/datasets\/[^/]+\/feature-selection\/[^/]+$/.test(pathname)) {
      if (fails.has('fs')) return errorJson(route, 500, 'Không tải được kết quả trích xuất (mock lỗi).');
      return json(route, FS_STATS_BORUTA);
    }
    if (/\/api\/datasets\/[^/]+\/model\/[^/]+\/[^/]+\/train$/.test(pathname)) {
      if (fails.has('modelTrain')) return errorJson(route, 500, 'Huấn luyện mô hình thất bại (mock lỗi).');
      return json(route, { job_id: 'job-model-1' });
    }
    if (/\/api\/datasets\/[^/]+\/model\/[^/]+\/[^/]+$/.test(pathname)) {
      if (fails.has('model')) return errorJson(route, 500, 'Không tải được mô hình (mock lỗi).');
      return json(route, MODEL_STATS);
    }
    if (/^\/api\/jobs\//.test(pathname)) {
      // Always resolves "done" on the first poll — keeps tests fast; the
      // polling loop itself (pollJob in lib/api.ts) isn't what's under test.
      return json(route, {
        job_id: pathname.split('/').pop(),
        status: 'done',
        log: ['[HỆ THỐNG] Bắt đầu...', '[THÀNH CÔNG] Xong.'],
        result: {},
        error: null,
      });
    }
    if (/\/api\/datasets\/[^/]+\/rules\//.test(pathname)) {
      return json(route, RULES);
    }
    if (/\/api\/datasets\/[^/]+\/genes\//.test(pathname)) {
      return json(route, GENES);
    }
    if (/\/api\/datasets\/[^/]+\/predict$/.test(pathname) && method === 'POST') {
      if (fails.has('predict')) return errorJson(route, 500, 'Dự đoán thất bại (mock lỗi).');
      return json(route, PREDICT_RESPONSE);
    }
    if (/\/api\/datasets\/[^/]+\/runs$/.test(pathname)) {
      return json(route, []);
    }
    if (/\/api\/datasets\/[^/]+\/test-samples\/[^/]+$/.test(pathname)) {
      return json(route, TEST_SAMPLES);
    }
    if (pathname === '/api/datasets/upload/history') {
      return json(route, []);
    }
    // Any endpoint this suite doesn't know about yet — fail loudly instead
    // of silently falling through to the real backend.
    return json(route, { detail: `unmocked path: ${pathname}` }, 404);
  });
}
