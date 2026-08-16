import React, { useEffect } from 'react';
import { Activity, AlertTriangle, ListChecks } from 'lucide-react';
import { ChevronDown, ChevronUp, UploadCloud, CheckCircle1, Download1, InfoCircle, Reload } from '@tailgrids/icons';
import { cn, errorMessage } from '../../lib/utils';
import { classColor } from '../../lib/palette';
import { displayLabel } from '../../lib/metrics';
import { api, PredictResponse } from '../../lib/api';
import { canonicalClassLabels, fsMethodKeyOf, WorkspaceAction, WorkspaceState } from '../../state/types';
import { Button } from '../tailgrids/core/button';
import Panel from '../ui/Panel';
import VoteBar from '../VoteBar';
import GeneChipList from '../GeneChipList';

interface Props {
  state: WorkspaceState;
  dispatch: React.Dispatch<WorkspaceAction>;
}

// Not in lucide-react — a fixed-size 24px glyph to match its other icons;
// accepts (and ignores) `size` only so call sites can use it the same way.
function Brain(props: React.SVGProps<SVGSVGElement> & { size?: number }) {
  return (
    <svg {...props} xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M12 5a3 3 0 1 0-5.997.125 4 4 0 0 0-2.526 5.77 4 4 0 0 0 .556 6.588A4 4 0 1 0 12 18Z"/>
      <path d="M12 5a3 3 0 1 1 5.997.125 4 4 0 0 1 2.526 5.77 4 4 0 0 1-.556 6.588A4 4 0 1 1 12 18Z"/>
      <path d="M15 13a4.5 4.5 0 0 1-3-4 4.5 4.5 0 0 1-3 4"/>
      <path d="M17.599 6.5a3 3 0 0 0 .399-1.375"/>
      <path d="M6.003 5.125A3 3 0 0 0 6.401 6.5"/>
      <path d="M3.477 10.896a4 4 0 0 1 .585-.396"/>
      <path d="M19.938 10.5a4 4 0 0 1 .585.396"/>
      <path d="M6 18a4 4 0 0 1-1.967-.516"/>
      <path d="M19.967 17.484A4 4 0 0 1 18 18"/>
    </svg>
  );
}

/** Bước 5 "Kiểm Thử (Thực Nghiệm)" — predict a held-out sample (or an
 * uploaded one) against Bước 4's model and matched rules. */
export default function Step5Testing({ state, dispatch }: Props) {
  const { datasetId, modelStats, modelType, modelRunId, splitParams, testMode, testSamples, testSampleId, testUploadFile, isTesting } = state;
  const fsMethodKey = fsMethodKeyOf(state);

  // Load the real held-out test samples once a model is trained/loaded.
  useEffect(() => {
    if (!datasetId || !modelStats) return;
    let ignore = false;
    api.getTestSamplesWithSplit(datasetId, fsMethodKey, splitParams)
      .then(samples => { if (!ignore) dispatch({ type: 'test_samples_loaded', samples }); })
      .catch(() => { if (!ignore) dispatch({ type: 'test_samples_loaded', samples: [] }); });
    return () => { ignore = true; };
  }, [datasetId, modelStats, fsMethodKey, splitParams, dispatch]);

  const applyPredictionResult = (result: PredictResponse) => {
    dispatch({
      type: 'test_computed',
      result: {
        matchedCount: result.matched_count,
        rules: result.rules.map(r => ({
          id: r.rule_id,
          text: r.text,
          matched: r.matched,
          desc: r.explanation || '',
          sampleValues: r.sample_values || {},
          class: r.consequent_label,
        })),
        classification: result.classification,
        classDescription: result.class_description,
        classDisplayName: result.class_display_name || null,
        classDisplayNames: result.class_display_names || {},
        rulePrediction: result.rule_prediction || null,
        rulePredictionDescription: result.rule_prediction_description || null,
        rulePredictionDisplayName: result.rule_prediction_display_name || null,
        classVotes: result.class_votes || {},
        classVotesOver50: result.class_votes_over50 || {},
        partialMatches: (result.partial_matches || []).map(p => ({
          ruleId: p.rule_id,
          text: p.text,
          class: p.consequent_label,
          satisfied: p.satisfied,
          total: p.total,
          ratio: p.ratio,
          conditions: p.conditions || [],
        })),
        nPartialMatchesTotal: result.n_partial_matches_total ?? (result.partial_matches || []).length,
        nRulesTotal: result.n_rules_total ?? null,
        trueLabel: result.true_label,
        biomedicalSummary: result.biomedical_summary,
        biomedicalRationale: result.biomedical_rationale,
        biomedicalModelVsRule: result.biomedical_model_vs_rule,
        biomedicalDisclaimer: result.biomedical_disclaimer,
        llmUsed: result.llm_used,
      },
    });
  };

  const handleTestSample = async () => {
    if (!datasetId) return;
    if (testMode === 'sample' && !testSampleId) return;
    if (testMode === 'upload' && !testUploadFile) return;
    dispatch({ type: 'test_started' });
    try {
      const result = testMode === 'sample'
        ? await api.predict(datasetId, fsMethodKey, modelType, testSampleId, modelRunId, splitParams)
        : await api.predictUpload(datasetId, fsMethodKey, modelType, testUploadFile as File, modelRunId);
      applyPredictionResult(result);
    } catch (e) {
      dispatch({ type: 'test_failed', result: { matchedCount: 0, rules: [], classification: 'Loi', explanation: errorMessage(e) } });
    }
  };

  return (
    <fieldset disabled={isTesting} className="border-0 p-0 m-0 min-w-0 disabled:opacity-60">
      <div className="flex flex-col gap-4">
         <div className="flex gap-2 p-1 bg-neutral-100 rounded-lg text-sm">
           <button
             onClick={() => dispatch({ type: 'test_mode_changed', mode: 'sample' })}
             className={cn("flex-1 py-1.5 rounded-md flex items-center justify-center gap-1.5 transition-colors", testMode === 'sample' ? "bg-white shadow-sm text-brand-700 font-medium" : "text-neutral-600")}
           >
             <ListChecks size={14} /> Chọn mẫu có sẵn
           </button>
           <button
             onClick={() => dispatch({ type: 'test_mode_changed', mode: 'upload' })}
             className={cn("flex-1 py-1.5 rounded-md flex items-center justify-center gap-1.5 transition-colors", testMode === 'upload' ? "bg-white shadow-sm text-brand-700 font-medium" : "text-neutral-600")}
           >
             <UploadCloud size={14} /> Tải lên file
           </button>
         </div>

         {/* Download the exact held-out test set (test_set.csv +
             manifest.csv + samples/*.json) — for transparency
             (audit which rows were held out) and for re-testing
             via "Tải lên file" above. Works for both a cached
             dataset and a fresh upload / custom "Thực hiện lại"
             split (backend recomputes live when there's no cache
             yet) — the URL already carries the current splitParams. */}
         {datasetId && (
           <a
             href={api.getTestSetDownloadUrl(datasetId, fsMethodKey, splitParams)}
             download
             className="w-full py-2 border border-neutral-200 hover:bg-neutral-50 text-neutral-700 rounded-lg flex items-center justify-center gap-1.5 transition-colors text-xs font-medium"
           >
             <Download1 size={14} /> Tải xuống test set (ZIP) — minh bạch dữ liệu đánh giá
           </a>
         )}

         {testMode === 'sample' ? (
           <select
             value={testSampleId}
             onChange={(e) => dispatch({ type: 'test_sample_selected', sampleId: e.target.value })}
             className="w-full bg-neutral-50 border border-neutral-200 text-neutral-800 rounded-lg px-4 py-2.5 focus:outline-none focus:ring-2 focus:ring-brand-500/30 focus:border-brand-500 transition-colors shadow-sm"
           >
             <option value="">-- Chọn mẫu bệnh phẩm (test set thật) --</option>
             {testSamples.map(s => (
               <option key={s.sample_id} value={s.sample_id}>{s.sample_id} (nhãn thật: {s.true_label})</option>
             ))}
           </select>
         ) : (
           <label className="w-full border-2 border-dashed border-brand-200 bg-brand-50/30 hover:bg-brand-50/80 text-brand-700 rounded-xl px-4 py-6 flex flex-col items-center justify-center cursor-pointer transition-colors text-center">
             <UploadCloud size={24} className="mb-2 text-brand-500" />
             <span className="font-medium text-sm">{testUploadFile ? testUploadFile.name : 'Tải lên mẫu bệnh phẩm'}</span>
             <span className="text-xs text-neutral-500 mt-1">
               Định dạng <span className="font-mono">.json</span> (giống mẫu test_set — chỉ cần giữ lại dữ liệu microarray, các trường khác có thể lược bỏ),
               {' '}<span className="font-mono">.csv</span> (cột "samples,type,{'{probe}'}...") hoặc <span className="font-mono">.txt</span> (2 cột probe,giá trị)
             </span>
             <input type="file" accept=".json,.csv,.txt" className="hidden" onChange={(e) => dispatch({ type: 'test_upload_file_changed', file: e.target.files?.[0] || null })} />
           </label>
         )}

          <Button
            onClick={handleTestSample}
            disabled={(testMode === 'sample' ? !testSampleId : !testUploadFile) || isTesting}
            size="lg"
            className="w-full"
          >
            {isTesting ? <Reload size={18} className="animate-spin" /> : <Activity size={18} />}
            Dự Đoán Kết Quả
          </Button>
      </div>
    </fieldset>
  );
}

interface PanelProps extends Props {
  onJumpToGene: (gene: string) => void;
}

/** Right-column "Kết quả Thực nghiệm" card — the model's own prediction vs.
 * the matched rules' majority vote, the matched/partially-matched rules and
 * genes behind it, and the biomedical write-up. */
export function TestResultsPanel({ state, dispatch, onJumpToGene }: PanelProps) {
  const { modelStats, testResults, isTestResultsCollapsed, expandedRule, isMatchedRulesCollapsed, isPartialMatchesCollapsed } = state;
  if (!modelStats) return null;

  return (
    <div>
      <div
        className="flex items-center justify-between mb-6 cursor-pointer group"
        onClick={() => dispatch({ type: 'test_results_toggled' })}
      >
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 shrink-0 rounded-full bg-success-50 flex items-center justify-center text-success-600 border border-success-100">
            <Activity size={20} />
          </div>
          <h2 className="text-2xl font-bold text-neutral-900 tracking-tight group-hover:text-success-700 transition-colors">Kết quả Thực nghiệm</h2>
        </div>
        <button className="p-2 rounded-full hover:bg-success-50 text-neutral-600 hover:text-success-600 transition-colors shrink-0">
          {isTestResultsCollapsed ? <ChevronDown size={24} /> : <ChevronUp size={24} />}
        </button>
      </div>

      {!isTestResultsCollapsed && (
      <div className="w-full">
        {testResults ? (
          testResults.classification === 'Loi' ? (
          <div className="bg-white p-6 rounded-2xl shadow-sm animate-in fade-in slide-in-from-right-4 duration-500 border border-danger-200 h-full">
            <div className="flex items-center gap-3 mb-4 pb-4 border-b border-neutral-200">
              <AlertTriangle className="text-danger-500" size={24} />
              <h3 className="text-lg font-bold text-neutral-800">Lỗi khi thực nghiệm</h3>
            </div>
            <p className="text-sm text-danger-700 bg-danger-50 border border-danger-200 rounded-lg px-3 py-2">
              {testResults.explanation || 'Đã có lỗi xảy ra, vui lòng thử lại.'}
            </p>
          </div>
          ) : (
          <div className="bg-white p-6 rounded-2xl shadow-sm animate-in fade-in slide-in-from-right-4 duration-500 border border-neutral-200 h-full">
            <div className="flex items-center gap-3 mb-6 pb-4 border-b border-neutral-200">
              <CheckCircle1 className="text-brand-600" size={24} />
              <h3 className="text-lg font-bold text-neutral-800">Báo cáo Phân loại</h3>
              <span className="ml-auto bg-brand-100 text-brand-700 border border-brand-200 text-xs py-1 px-3 rounded-full font-medium">
                Khớp {testResults.matchedCount} rules
              </span>
            </div>

            {testResults.trueLabel ? (
              <p className="text-sm text-neutral-500 mb-4">
                Nhãn thật: <span className="font-semibold text-neutral-700">
                  {displayLabel(testResults.trueLabel, testResults.classDisplayNames)}
                </span>
                {testResults.classDisplayNames?.[testResults.trueLabel] && (
                  <span className="text-neutral-500 font-mono text-xs ml-1.5">({testResults.trueLabel})</span>
                )}
              </p>
            ) : (
              <p className="text-sm text-neutral-500 mb-4 italic">Dữ liệu upload không có nhãn</p>
            )}

            {/* Two independent predictions shown side by side: the model's
                own predict() call vs. the majority label among matched rules —
                these can disagree (see class_votes note above), so both are
                surfaced explicitly instead of only the model's result. */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 mb-6">
              <div className="rounded-xl p-4 border" style={{
                backgroundColor: classColor(testResults.classification, canonicalClassLabels(state)) + '14',
                borderColor: classColor(testResults.classification, canonicalClassLabels(state)) + '55',
              }}>
                <span className="text-neutral-500 text-xs uppercase tracking-wider font-semibold block mb-1.5">Dự đoán theo Mô hình (RF/DT)</span>
                <span className="font-bold text-xl" style={{ color: classColor(testResults.classification, canonicalClassLabels(state)) }}>
                  {testResults.classDisplayName || testResults.classification}
                </span>
                {testResults.classDisplayName && (
                  <span className="text-neutral-500 font-mono text-xs block mt-0.5">{testResults.classification}</span>
                )}
              </div>
              <div className="rounded-xl p-4 border" style={{
                backgroundColor: testResults.rulePrediction ? classColor(testResults.rulePrediction, canonicalClassLabels(state)) + '14' : 'rgba(100,116,139,0.1)',
                borderColor: testResults.rulePrediction ? classColor(testResults.rulePrediction, canonicalClassLabels(state)) + '55' : 'rgba(100,116,139,0.3)',
              }}>
                <span className="text-neutral-500 text-xs uppercase tracking-wider font-semibold block mb-1.5">Dự đoán theo Tập luật (rule-based)</span>
                {testResults.rulePrediction ? (
                  <>
                    <span className="font-bold text-xl" style={{ color: classColor(testResults.rulePrediction, canonicalClassLabels(state)) }}>
                      {testResults.rulePredictionDisplayName || testResults.rulePrediction}
                    </span>
                    {testResults.rulePredictionDisplayName && (
                      <span className="text-neutral-500 font-mono text-xs block mt-0.5">{testResults.rulePrediction}</span>
                    )}
                  </>
                ) : (
                  <span className="font-bold text-xl text-neutral-500">Không có luật khớp</span>
                )}
              </div>
            </div>
            {testResults.rulePrediction && testResults.rulePrediction !== testResults.classification && (
              <p className="text-xs text-warning-800 bg-warning-50 border border-warning-200 rounded-lg px-3 py-2 mb-6">
                Hai dự đoán không trùng nhau — mô hình quyết định dựa trên toàn bộ đặc trưng, còn tập luật chỉ phản ánh các luật đơn giản tình cờ khớp với mẫu này (xem "Tỷ lệ Luật Khớp theo Lớp" bên dưới).
              </p>
            )}

            {/* Matched-rules list — same bordered-box level as the vote breakdown
                and biomedical assessment below it (previously this list floated
                without its own box, unlike its siblings). */}
            <div className="bg-white rounded-xl p-5 border border-neutral-200 mb-6">
              <div
                className="flex items-center justify-between mb-4 cursor-pointer group/rules"
                onClick={() => dispatch({ type: 'matched_rules_toggled' })}
              >
                <h4 className="font-semibold tracking-wide text-xs uppercase text-brand-700 group-hover/rules:text-brand-800 transition-colors">
                  Danh sách Luật Khớp ({testResults.rules.length})
                </h4>
                <button className="text-neutral-400 hover:text-brand-600 transition-colors">
                  {isMatchedRulesCollapsed ? <ChevronDown size={18} /> : <ChevronUp size={18} />}
                </button>
              </div>
              {!isMatchedRulesCollapsed && (
              <div className="space-y-3">
                {testResults.rules.map((rule) => {
                  const ruleColor = classColor(rule.class, canonicalClassLabels(state));
                  return (
                  <div key={rule.id} className={cn("rounded-xl overflow-hidden transition-all", rule.matched ? "bg-brand-50 border border-brand-200" : "bg-white border border-neutral-200 opacity-60")}>
                    <div className="px-4 py-3 flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <span
                          className="px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider border"
                          style={{ backgroundColor: ruleColor + '2a', color: ruleColor, borderColor: ruleColor + '55' }}
                        >
                          {displayLabel(rule.class, testResults.classDisplayNames)}
                        </span>
                        <code className="font-mono text-xs md:text-sm text-brand-800">{rule.text}</code>
                      </div>
                      <button
                        onClick={() => dispatch({ type: 'expanded_rule_toggled', ruleId: rule.id })}
                        className="text-brand-700 hover:text-brand-900 transition-colors shrink-0 ml-4 pl-3 border-l border-brand-200"
                        title="Giải thích Y sinh"
                      >
                        <InfoCircle size={16} />
                      </button>
                    </div>
                    {expandedRule === rule.id && (
                      <div className="px-4 py-3 bg-neutral-50 text-sm text-neutral-800 border-t border-neutral-200 leading-relaxed">
                        <p className="mb-3">{rule.desc || 'Chưa có mô tả sinh học cho luật này (chạy scripts/generate_bio_descriptions.py để sinh).'}</p>
                        <div className="flex flex-wrap gap-2">
                          {Object.entries(rule.sampleValues).map(([gene, val]) => (
                            <span key={gene} className="bg-white px-2 py-1 rounded-md text-xs font-mono border border-neutral-200">
                              <span className="text-brand-700">{gene}</span> = {val}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                  );
                })}
              </div>
              )}
            </div>

            {/* Genes appearing in the matched rules — just the list (no
                redefinition, that already lives in "Gene xuất hiện trong Danh
                sách Luật" above); click jumps straight to that gene's own card. */}
            {(() => {
              const matchedGenes = Array.from(new Set(
                testResults.rules.flatMap((rule) => Object.keys(rule.sampleValues)),
              )) as string[];
              return (
                <GeneChipList
                  className="mb-6"
                  title="Danh sách Gene trong Luật Khớp"
                  genes={matchedGenes}
                  onJumpToGene={onJumpToGene}
                />
              );
            })()}

            {/* Rules that satisfy >=50% of their conditions but didn't fully match —
                transparency into "near misses", sorted by match ratio.
                Monochrome styling matches "Danh sách Luật Khớp" above (colored class
                badge only — rule text and condition chips stay a single neutral tone,
                the ✓/✗ mark is just appended after the value instead of color-coding
                the whole chip); the ratio badge sits on its own header row so it never
                crowds out the rule text like it did when squeezed inline before. */}
            {testResults.partialMatches && testResults.partialMatches.length > 0 && (
              <Panel padding="lg" className="mb-6">
                <div
                  className="flex items-center justify-between mb-4 cursor-pointer group/partial"
                  onClick={() => dispatch({ type: 'partial_matches_toggled' })}
                >
                  <h4 className="font-semibold tracking-wide text-xs uppercase text-brand-700 group-hover/partial:text-brand-800 transition-colors">
                    Luật Khớp Một Phần ({testResults.partialMatches.length})
                  </h4>
                  <button className="text-neutral-400 hover:text-brand-600 transition-colors">
                    {isPartialMatchesCollapsed ? <ChevronDown size={18} /> : <ChevronUp size={18} />}
                  </button>
                </div>
                {!isPartialMatchesCollapsed && (
                <div className="space-y-3">
                  {testResults.partialMatches.map((p) => {
                    const ruleColor = classColor(p.class, canonicalClassLabels(state));
                    return (
                      <div key={p.ruleId} className="rounded-xl overflow-hidden bg-white border border-neutral-200 p-4">
                        <div className="flex items-center justify-between gap-3 mb-2">
                          <span
                            className="px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider border shrink-0"
                            style={{ backgroundColor: ruleColor + '2a', color: ruleColor, borderColor: ruleColor + '55' }}
                          >
                            {displayLabel(p.class, testResults.classDisplayNames)}
                          </span>
                          <span className="shrink-0 bg-warning-100 text-warning-700 border border-warning-200 text-[11px] py-1 px-2.5 rounded-full font-semibold">
                            {p.satisfied}/{p.total} điều kiện — {Math.round(p.ratio * 100)}%
                          </span>
                        </div>
                        <code className="font-mono text-xs md:text-sm text-neutral-700 block break-words">{p.text}</code>
                        <div className="mt-3 flex flex-wrap gap-2">
                          {p.conditions.map((c, ci) => (
                            <span
                              key={ci}
                              className="px-2 py-1 rounded-md text-xs font-mono border border-neutral-200 bg-neutral-50 text-neutral-600"
                            >
                              {c.gene}={c.actual} {c.ok ? '✓' : '✗'}
                            </span>
                          ))}
                        </div>
                      </div>
                    );
                  })}
                </div>
                )}
              </Panel>
            )}

            {/* Genes that actually SATISFIED a condition in a partial-match rule —
                not every gene referenced (a failed condition's gene isn't "in" the
                rule the way a matched one is), mirroring "Danh sách Gene trong Luật
                Khớp" above but scoped to every visible partial match. */}
            {(() => {
              const partialMatchedGenes = Array.from(new Set(
                (testResults.partialMatches || []).flatMap(
                  (p) => p.conditions.filter(c => c.ok).map(c => c.gene),
                ),
              )) as string[];
              return (
                <GeneChipList
                  className="mb-6"
                  title="Danh sách Gene trong Luật Khớp Một Phần"
                  genes={partialMatchedGenes}
                  onJumpToGene={onJumpToGene}
                />
              );
            })()}

            {/* Per-class matched-rule vote breakdown — how many/what % of the
                matched rules point to each class, independent of the model's
                own predict() call (a transparency/sanity-check signal). */}
            {Object.keys(testResults.classVotes || {}).length > 0 && (
              <VoteBar
                className="mb-6"
                title="Tỷ lệ Luật Khớp hoàn toàn theo Lớp"
                rows={Object.entries(testResults.classVotes as Record<string, { count: number, percentage: number }>).map(([label, v]) => ({
                  key: label,
                  label: displayLabel(label, testResults.classDisplayNames),
                  caption: `${v.count} luật (${v.percentage}%)`,
                  percentage: v.percentage,
                  color: classColor(label, canonicalClassLabels(state)),
                }))}
              />
            )}

            {/* Per-class breakdown of PARTIAL matches only (>=50% but <100% of
                conditions) — the per-class counterpart of "Danh sách Luật Khớp một Phần",
                complementing "Tỷ lệ Luật Khớp theo Lớp" above (full matches only)
                instead of duplicating it with matched rules mixed back in. */}
            {Object.keys(testResults.classVotesOver50 || {}).length > 0 && (
              <VoteBar
                className="mb-6"
                title="Tỷ lệ Luật Khớp một phần theo Lớp"
                rows={Object.entries(testResults.classVotesOver50 as Record<string, { count: number, percentage: number }>).map(([label, v]) => ({
                  key: label,
                  label: displayLabel(label, testResults.classDisplayNames),
                  caption: `${v.count} luật (${v.percentage}%)`,
                  percentage: v.percentage,
                  color: classColor(label, canonicalClassLabels(state)),
                }))}
              />
            )}

            {/* Comparison of match TIERS (đủ 100% vs một phần >=50%) out of every
                rule this model has — same row-per-category bar layout as "Tỷ lệ
                Luật Khớp theo Lớp" above, just comparing match tiers instead of
                classes. Uses the aggregate count returned by the API. */}
            {!!testResults.nRulesTotal && (() => {
              const total = testResults.nRulesTotal;
              const tiers = [
                { key: 'full', label: 'Khớp đủ (100%)', count: testResults.matchedCount, color: '#2dd4bf' },
                { key: 'partial', label: 'Khớp một phần (≥50%)', count: testResults.nPartialMatchesTotal || 0, color: '#fbbf24' },
              ];
              return (
                <VoteBar
                  className="mb-6"
                  title="Tỷ lệ Luật Khớp Trên 50%"
                  rows={tiers.map(t => {
                    const pct = Math.round((t.count / total) * 1000) / 10;
                    return { key: t.key, label: t.label, color: t.color, percentage: pct, caption: `${t.count}/${total} (${pct}%)` };
                  })}
                />
              );
            })()}

            <Panel padding="lg">
              <div className="flex items-center gap-2 mb-4 text-brand-700">
                <Brain size={18} />
                <h4 className="font-semibold tracking-wide text-xs uppercase">
                  Đánh Giá Y Sinh {testResults.llmUsed === false && <span className="text-neutral-500 normal-case">(template — Gemini không khả dụng)</span>}
                </h4>
              </div>
              <div className="space-y-5">
                {testResults.biomedicalSummary && (
                  <div>
                    <span className="text-neutral-500 text-xs uppercase tracking-wider font-semibold block mb-1.5">Tóm tắt</span>
                    <p className="text-neutral-800 leading-relaxed text-base font-medium">{testResults.biomedicalSummary}</p>
                  </div>
                )}
                {testResults.biomedicalRationale && (
                  <div>
                    <span className="text-neutral-500 text-xs uppercase tracking-wider font-semibold block mb-1.5">Cơ sở Sinh học</span>
                    <p className="text-neutral-600 leading-relaxed text-base">{testResults.biomedicalRationale}</p>
                  </div>
                )}
                {testResults.biomedicalModelVsRule && (
                  <Panel
                    surface={testResults.rulePrediction && testResults.rulePrediction !== testResults.classification ? 'warning' : 'brand'}
                  >
                    <span className={cn(
                      "text-xs uppercase tracking-wider font-semibold block mb-1.5",
                      testResults.rulePrediction && testResults.rulePrediction !== testResults.classification ? "text-warning-700" : "text-brand-700"
                    )}>
                      So sánh Mô hình &harr; Tập luật
                    </span>
                    <p className="text-neutral-700 leading-relaxed text-base">{testResults.biomedicalModelVsRule}</p>
                  </Panel>
                )}
                {testResults.biomedicalDisclaimer && (
                  <div className="flex items-start gap-2 bg-warning-50 border border-warning-200 rounded-lg px-3 py-2.5 mt-1">
                    <AlertTriangle size={14} className="text-warning-500 shrink-0 mt-0.5" />
                    <p className="text-warning-800 text-xs leading-relaxed">{testResults.biomedicalDisclaimer}</p>
                  </div>
                )}
              </div>
            </Panel>
          </div>
          )
        ) : (
          <div className="h-full bg-neutral-100/50 border border-neutral-200 border-dashed rounded-2xl flex flex-col items-center justify-center text-neutral-500 p-8 text-center min-h-[300px]">
             <div className="w-16 h-16 bg-white rounded-full flex items-center justify-center mb-4 border border-neutral-200 shadow-sm">
               <Activity size={24} className="text-neutral-400" />
             </div>
             <p className="text-sm max-w-sm font-medium text-neutral-600">
               Chọn một mẫu bệnh phẩm và nhấn "Dự Đoán Kết Quả" ở Bước 5 để bắt đầu.
             </p>
          </div>
        )}
      </div>
      )}
    </div>
  );
}
