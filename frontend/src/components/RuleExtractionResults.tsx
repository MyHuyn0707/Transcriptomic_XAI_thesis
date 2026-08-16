import React, { useState, useEffect, useRef } from 'react';
import { GitMerge } from 'lucide-react';
import { ChevronDown, ChevronUp, FileText, ArrowRight, Bolt1, InfoCircle, Filter, Search1 } from '@tailgrids/icons';
import { cn } from '../lib/utils';
import { classColor } from '../lib/palette';
import { api } from '../lib/api';

interface RuleItem {
  rule_id: number;
  text: string;
  explanation?: string | null;
  metrics: Record<string, number>;
  consequent?: { class_idx?: number, class_label?: string };
}

interface GeneItem {
  gene: string;
  gene_title?: string;
  entrez_id?: string;
  genbank_acc?: string;
  refseq?: string;
  species?: string;
  go_biological_process?: string;
  go_cellular_component?: string;
  go_molecular_function?: string;
  go_terms?: string;
  go_format?: 'split' | 'combined';
  n_rules?: number;
  classes?: string;
  probes?: string;
  description_vn?: string | null;
}

interface RulesSummary {
  n_rules_raw?: number;
  n_rules_passed_filter?: number;
  n_rules_kept?: number;
  n_dropped_by_filter?: number;
  n_dropped_by_cap?: number;
}

interface Props {
  datasetId: string;
  fsMethod: string;
  model: string;
  runId?: string | null;
  rulesSummary?: RulesSummary | null;
  labels?: string[];
}

// Exactly these 4 metrics, in this order — showing all of support/confidence/
// lift/class_specificity/fidelity (5 badges) wrapped to a 2nd row on longer
// class names (e.g. "sarc" in GSE29354's mesothelioma classes); dropping
// class_specificity keeps every rule card to one tidy row.
const DISPLAYED_METRICS: { key: string; label: string }[] = [
  { key: 'fidelity', label: 'Fidelity' },
  { key: 'lift', label: 'Lift' },
  { key: 'confidence', label: 'Confidence' },
  { key: 'support', label: 'Support' },
];

/** DOM id for a gene's card — exported so other components (the test-results
 * "Danh sách Gene trong Luật Khớp" panel) can jump straight to it instead of
 * re-describing the same gene a second time. */
export function geneAnchorId(gene: string) {
  return `gene-card-${gene.replace(/[^a-zA-Z0-9_-]/g, '_')}`;
}

/** One funnel stage — same emerald family as "Danh sách Luật Trích xuất"
 * (not a rainbow of unrelated colors), progressively smaller/lighter as the
 * rule count shrinks stage to stage, connected by a centered arrow. */
function FunnelStage({
  value, label, rejected, size,
}: { value: number, label: string, rejected?: string, size: 'lg' | 'md' | 'sm' }) {
  const sizing = {
    lg: { flex: 'flex-[3]', pad: 'py-8 px-6', num: 'text-4xl', bg: 'bg-success-50', border: 'border-success-200', text: 'text-success-600', lbl: 'text-success-900/50' },
    md: { flex: 'flex-[2]', pad: 'py-6 px-5', num: 'text-3xl', bg: 'bg-success-100/70', border: 'border-success-300', text: 'text-success-700', lbl: 'text-success-900/55' },
    sm: { flex: 'flex-[1.3]', pad: 'py-5 px-4', num: 'text-2xl', bg: 'bg-success-200/70', border: 'border-success-400', text: 'text-success-800', lbl: 'text-success-900/60' },
  }[size];
  return (
    <div className={cn(sizing.flex, "min-w-[140px] rounded-2xl border-2 flex flex-col items-center justify-center text-center shadow-sm", sizing.pad, sizing.bg, sizing.border)}>
      <span className={cn("font-extrabold mb-1.5", sizing.num, sizing.text)}>{value.toLocaleString()}</span>
      <span className={cn("text-[11px] font-bold uppercase tracking-wider mb-2", sizing.lbl)}>{label}</span>
      {rejected && (
        <span className="inline-flex items-center gap-1 text-[11px] font-semibold text-danger-600 bg-danger-100 border border-danger-200 px-2.5 py-1 rounded-md">
          {rejected}
        </span>
      )}
    </div>
  );
}

export default function RuleExtractionResults({ datasetId, fsMethod, model, runId, rulesSummary, labels }: Props) {
  const [isRulesCollapsed, setIsRulesCollapsed] = useState(true);
  const [isGenesCollapsed, setIsGenesCollapsed] = useState(true);
  const [collapsedGroups, setCollapsedGroups] = useState<Record<string, boolean>>({});
  const [expandedGenes, setExpandedGenes] = useState<Record<number, boolean>>({});
  const [expandedRules, setExpandedRules] = useState<Record<number, boolean>>({});
  const [rules, setRules] = useState<RuleItem[]>([]);
  const [genes, setGenes] = useState<GeneItem[]>([]);
  const [error, setError] = useState('');
  const [ruleSearch, setRuleSearch] = useState('');
  const [geneSearch, setGeneSearch] = useState('');
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!datasetId) return;
    let ignore = false;
    setRules([]); setGenes([]); setError('');
    // allSettled (not all): rules and genes are independent endpoints — a
    // dataset with no annotation_file never gets a gene_description.csv, so
    // getGenes 404s while getRules succeeds. Don't let that failure blank
    // out rules that DID load successfully.
    Promise.allSettled([
      api.getRules(datasetId, fsMethod, model, runId),
      api.getGenes(datasetId, fsMethod, model, runId),
    ]).then(([rulesResult, genesResult]) => {
      if (ignore) return;
      if (rulesResult.status === 'fulfilled') {
        setRules(rulesResult.value.rules || []);
      } else {
        setError(rulesResult.reason?.message || String(rulesResult.reason));
      }
      if (genesResult.status === 'fulfilled') {
        setGenes(genesResult.value || []);
      }
    });
    return () => { ignore = true; };
  }, [datasetId, fsMethod, model, runId]);

  // Let other components (test-results "Danh sách Gene trong Luật Khớp")
  // jump straight to a gene's card here instead of re-describing it — opens
  // both the genes panel and that gene's own detail, then scrolls to it.
  useEffect(() => {
    const handler = (e: Event) => {
      const gene = (e as CustomEvent<{ gene: string }>).detail?.gene;
      if (!gene) return;
      setIsGenesCollapsed(false);
      const idx = genes.findIndex(g => g.gene === gene);
      if (idx >= 0) setExpandedGenes(prev => ({ ...prev, [idx]: true }));
      setTimeout(() => {
        document.getElementById(geneAnchorId(gene))?.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }, 150);
    };
    window.addEventListener('jump-to-gene', handler);
    return () => window.removeEventListener('jump-to-gene', handler);
  }, [genes]);

  const toggleGene = (idx: number) => {
    setExpandedGenes(prev => ({ ...prev, [idx]: !prev[idx] }));
  };

  const toggleRule = (idx: number) => {
    setExpandedRules(prev => ({ ...prev, [idx]: !prev[idx] }));
  };

  const toggleGroup = (label: string) => {
    setCollapsedGroups(prev => ({ ...prev, [label]: !prev[label] }));
  };

  // Group rules by predicted class so the list reads as "here's what the
  // model learned for class X" rather than one long undifferentiated stream.
  const ruleGroups: Record<string, RuleItem[]> = {};
  for (const r of rules) {
    const label = r.consequent?.class_label || 'Không xác định';
    (ruleGroups[label] ||= []).push(r);
  }
  const groupLabels = Object.keys(ruleGroups).sort((a, b) => ruleGroups[b].length - ruleGroups[a].length);
  const allLabels = labels && labels.length ? labels : groupLabels;

  const ruleQuery = ruleSearch.trim().toLowerCase();
  const filteredGroups: Record<string, RuleItem[]> = {};
  let filteredRuleCount = 0;
  for (const label of groupLabels) {
    const matched = ruleQuery
      ? ruleGroups[label].filter(r =>
          r.text.toLowerCase().includes(ruleQuery) || label.toLowerCase().includes(ruleQuery))
      : ruleGroups[label];
    if (matched.length) {
      filteredGroups[label] = matched;
      filteredRuleCount += matched.length;
    }
  }

  const geneQuery = geneSearch.trim().toLowerCase();
  const filteredGenes = geneQuery
    ? genes.filter(g => g.gene.toLowerCase().includes(geneQuery) || (g.gene_title || '').toLowerCase().includes(geneQuery))
    : genes;

  return (
    <div ref={containerRef} className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500 mt-6">
      <div className="bg-white p-6 rounded-2xl shadow-sm border border-success-100">
        <div className="flex items-center gap-2 mb-6">
          <div className="w-8 h-8 rounded-full bg-success-50 flex items-center justify-center text-success-600">
            <Bolt1 size={18} />
          </div>
          <h2 className="text-2xl font-bold text-neutral-900 tracking-tight">
            Kết quả Trích xuất Luật sinh học
          </h2>
        </div>

        {error && <p className="text-sm text-danger-600 mb-4">Không tải được rules/genes: {error}</p>}

        {/* Rule-extraction log — what actually happened during mining, not just
            the final kept count: how many candidate rules were mined, how many
            passed the confidence/support filter, and how many were then capped.
            Same emerald family as the rules list below (not an unrelated
            indigo/amber scheme), and blocks visibly shrink stage to stage
            since the rule set itself shrinks — arrows sit centered between them. */}
        {rulesSummary && (rulesSummary.n_rules_raw != null) && (
          <div className="mb-6 bg-white border border-neutral-200 rounded-2xl p-6 shadow-sm">
            <div className="flex items-center gap-2 mb-6">
              <Filter size={16} className="text-neutral-500" />
              <span className="text-xs font-bold uppercase tracking-widest text-neutral-600">Nhật ký trích xuất luật</span>
            </div>
            <div className="flex items-center gap-3 sm:gap-4">
              <FunnelStage value={rulesSummary.n_rules_raw ?? 0} label="Luật khai thác" size="lg" />
              <ArrowRight size={22} className="text-success-300 shrink-0" />
              <FunnelStage
                value={rulesSummary.n_rules_passed_filter ?? 0}
                label="Qua bộ lọc"
                size="md"
                rejected={rulesSummary.n_dropped_by_filter ? `-${rulesSummary.n_dropped_by_filter} bị loại` : undefined}
              />
              <ArrowRight size={20} className="text-success-300 shrink-0" />
              <FunnelStage
                value={rulesSummary.n_rules_kept ?? 0}
                label="Luật giữ lại"
                size="sm"
                rejected={rulesSummary.n_dropped_by_cap ? `-${rulesSummary.n_dropped_by_cap} do giới hạn` : undefined}
              />
            </div>
          </div>
        )}

        <div className="space-y-6">
          {/* List of Rules — grouped by predicted class, each group tinted with
              that class's own color (same classColor() used everywhere else),
              each individually collapsible, with a search box to filter. */}
          <div className="border border-success-100 rounded-2xl overflow-hidden shadow-sm">
            <div
              className="bg-success-50/50 p-5 flex items-center justify-between cursor-pointer hover:bg-success-50 transition-colors"
              onClick={() => setIsRulesCollapsed(!isRulesCollapsed)}
            >
              <div className="flex items-center gap-2">
                <GitMerge size={20} className="text-success-600" />
                <h3 className="font-bold text-neutral-900 text-lg">Danh sách Luật Trích xuất ({rules.length})</h3>
              </div>
              <button className="text-neutral-600 p-1 hover:bg-success-100 rounded-full transition-colors">
                {isRulesCollapsed ? <ChevronDown size={22} /> : <ChevronUp size={22} />}
              </button>
            </div>

            {!isRulesCollapsed && (
              <div className="p-5 bg-white space-y-5">
                <div className="relative">
                  <Search1 size={15} className="absolute left-3 top-1/2 -tranneutral-y-1/2 text-neutral-500" />
                  <input
                    type="text"
                    value={ruleSearch}
                    onChange={e => setRuleSearch(e.target.value)}
                    placeholder="Tìm luật theo gene, điều kiện hoặc tên lớp..."
                    className="w-full pl-9 pr-3 py-2 text-sm bg-neutral-50 border border-neutral-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-success-500/30 focus:border-success-500 transition-colors"
                  />
                </div>
                {ruleQuery && (
                  <p className="text-xs text-neutral-500 -mt-2">Tìm thấy {filteredRuleCount}/{rules.length} luật khớp với "{ruleSearch}"</p>
                )}
                {Object.keys(filteredGroups).map(label => {
                  const color = classColor(label, allLabels);
                  const groupCollapsed = collapsedGroups[label];
                  return (
                    <div key={label} className="rounded-2xl border overflow-hidden" style={{ borderColor: color + '40', backgroundColor: color + '0a' }}>
                      <div
                        className="flex items-center gap-2.5 p-4 cursor-pointer"
                        onClick={() => toggleGroup(label)}
                      >
                        <span className="w-2.5 h-2.5 rounded-full inline-block shrink-0" style={{ backgroundColor: color }} />
                        <h4 className="font-extrabold text-lg flex-1" style={{ color }}>{label}</h4>
                        <span className="text-sm text-neutral-500 font-medium">({filteredGroups[label].length} luật)</span>
                        <button className="p-1 rounded-full transition-colors shrink-0" style={{ color }}>
                          {groupCollapsed ? <ChevronDown size={18} /> : <ChevronUp size={18} />}
                        </button>
                      </div>
                      {!groupCollapsed && (
                        <div className="px-4 pb-4 space-y-3 animate-in fade-in slide-in-from-top-2 duration-200">
                          {filteredGroups[label].map((ruleItem) => {
                            const condition = ruleItem.text.split(' THEN ')[0].replace(/^IF /, '');
                            const isExpanded = expandedRules[ruleItem.rule_id];
                            const metricEntries = DISPLAYED_METRICS
                              .filter(({ key }) => ruleItem.metrics?.[key] != null)
                              .map(({ key, label }) => [label, ruleItem.metrics[key]] as [string, number]);

                            return (
                              <div key={ruleItem.rule_id} className="bg-white rounded-xl border shadow-sm overflow-hidden transition-shadow hover:shadow-md" style={{ borderColor: color + '35' }}>
                                <div className="p-4">
                                  <div className="flex items-start justify-between gap-3">
                                    <code className="font-mono text-[13px] md:text-sm leading-relaxed text-neutral-800 flex-1 pt-0.5">
                                      <span className="font-bold mr-2" style={{ color }}>IF</span>
                                      {condition}
                                    </code>
                                    <button
                                      onClick={(e) => { e.stopPropagation(); toggleRule(ruleItem.rule_id); }}
                                      className="p-1.5 rounded-full transition-colors shrink-0 border"
                                      style={isExpanded
                                        ? { backgroundColor: color + '15', color, borderColor: color + '40' }
                                        : { backgroundColor: 'white', color: '#94a3b8', borderColor: '#e2e8f0' }}
                                      title="Giải thích luật"
                                    >
                                      <InfoCircle size={16} />
                                    </button>
                                  </div>

                                  {isExpanded && (
                                    <div className="mt-4 pt-4 border-t animate-in fade-in slide-in-from-top-2" style={{ borderColor: color + '25' }}>
                                      <div className="p-4 rounded-xl" style={{ backgroundColor: color + '0d' }}>
                                        <span className="font-bold uppercase tracking-wider text-xs flex items-center gap-1.5 mb-2" style={{ color }}>
                                          <InfoCircle size={13} /> Mô tả sinh học
                                        </span>
                                        <p className="text-sm leading-relaxed text-neutral-800">
                                          {ruleItem.explanation || 'Chưa có mô tả (chạy scripts/generate_bio_descriptions.py để sinh).'}
                                        </p>
                                      </div>
                                    </div>
                                  )}

                                  {metricEntries.length > 0 && (
                                    <div className="grid grid-cols-4 gap-2 mt-3 pt-3 border-t border-neutral-100">
                                      {metricEntries.map(([label, val]) => (
                                        <span key={label} className="text-[11px] px-2 py-1 bg-neutral-50 border border-neutral-200 rounded-md text-neutral-600 font-mono text-center truncate">
                                          {label}: <span className="text-neutral-900 font-bold">{typeof val === 'number' ? val.toFixed(2) : String(val)}</span>
                                        </span>
                                      ))}
                                    </div>
                                  )}
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      )}
                    </div>
                  );
                })}
                {rules.length === 0 && !error && (
                  <p className="text-sm text-neutral-500 text-center py-6">Không có luật nào để hiển thị.</p>
                )}
                {rules.length > 0 && Object.keys(filteredGroups).length === 0 && (
                  <p className="text-sm text-neutral-500 text-center py-6">Không có luật nào khớp với tìm kiếm.</p>
                )}
              </div>
            )}
          </div>

          {/* List of Genes */}
          <div className="border border-brand-100 rounded-2xl overflow-hidden shadow-sm">
            <div
              className="bg-brand-50/50 p-5 flex items-center justify-between cursor-pointer hover:bg-brand-50 transition-colors"
              onClick={() => setIsGenesCollapsed(!isGenesCollapsed)}
            >
              <div className="flex items-center gap-2">
                <FileText size={20} className="text-brand-600" />
                <h3 className="font-bold text-neutral-900 text-lg">Gene xuất hiện trong Danh sách Luật ({genes.length})</h3>
              </div>
              <button className="text-neutral-600 p-1 hover:bg-brand-100 rounded-full transition-colors">
                {isGenesCollapsed ? <ChevronDown size={22} /> : <ChevronUp size={22} />}
              </button>
            </div>

            {!isGenesCollapsed && (
              <div className="p-5 space-y-3">
                <div className="relative">
                  <Search1 size={15} className="absolute left-3 top-1/2 -tranneutral-y-1/2 text-neutral-500" />
                  <input
                    type="text"
                    value={geneSearch}
                    onChange={e => setGeneSearch(e.target.value)}
                    placeholder="Tìm gene theo ký hiệu hoặc tên..."
                    className="w-full pl-9 pr-3 py-2 text-sm bg-neutral-50 border border-neutral-200 rounded-lg focus:outline-none focus:ring-2 focus:ring-brand-500/30 focus:border-brand-500 transition-colors"
                  />
                </div>
                {geneQuery && (
                  <p className="text-xs text-neutral-500">Tìm thấy {filteredGenes.length}/{genes.length} gene khớp với "{geneSearch}"</p>
                )}
                {filteredGenes.map((gene) => {
                  const idx = genes.indexOf(gene);
                  const isExpanded = expandedGenes[idx];
                  const geneClasses = gene.classes ? gene.classes.split(';').map(c => c.trim()).filter(Boolean) : [];
                  return (
                    <div key={gene.gene} id={geneAnchorId(gene.gene)} className="bg-white border border-neutral-200 rounded-xl hover:border-brand-300 hover:shadow-md transition-all scroll-mt-4">
                      <div className="p-4">
                        <div className="flex justify-between items-start gap-4">
                          {/* Row 1: symbol + rule count — the identifying, always-short
                              part, kept on its own line so it never competes for space
                              with the (often long) title or the class badge list below. */}
                          <div className="min-w-0 flex-1 space-y-1.5">
                            <div className="flex items-center gap-2.5 flex-wrap">
                              <span className="font-extrabold text-brand-900 text-lg tracking-tight">{gene.gene}</span>
                              <span className="px-2 py-0.5 bg-brand-50 text-brand-700 text-[10px] uppercase tracking-wider rounded-md font-bold border border-brand-100 shrink-0">
                                {gene.n_rules} luật
                              </span>
                            </div>
                            {/* Row 2: full gene name — its own line so a long name
                                (e.g. "family with sequence similarity 57 member A")
                                never forces the class badges below to wrap awkwardly
                                mid-sentence. */}
                            {gene.gene_title && (
                              <p className="text-sm text-neutral-600 leading-snug">{gene.gene_title}</p>
                            )}
                            {/* Row 3: classes this gene's rules fire for — its own row
                                with a label prefix, so wrapping to a 2nd line here (e.g.
                                many classes) reads as a normal tag list, not as broken
                                layout mixed in with the name. */}
                            {geneClasses.length > 0 && (
                              <div className="flex flex-wrap items-center gap-1.5 pt-0.5">
                                <span className="text-[10px] font-semibold text-neutral-500 uppercase tracking-wider">Xuất hiện ở lớp:</span>
                                {geneClasses.map(c => {
                                  const color = classColor(c, allLabels);
                                  return (
                                    <span
                                      key={c}
                                      className="px-2 py-0.5 text-[10px] uppercase tracking-wider rounded-md font-bold border"
                                      style={{ backgroundColor: color + '14', color, borderColor: color + '40' }}
                                    >
                                      {c}
                                    </span>
                                  );
                                })}
                              </div>
                            )}
                          </div>
                          <button
                            onClick={(e) => { e.stopPropagation(); toggleGene(idx); }}
                            className={cn(
                              "p-2 rounded-full transition-colors shrink-0 flex items-center justify-center border",
                              isExpanded ? "bg-brand-50 text-brand-600 border-brand-100" : "bg-white text-neutral-500 border-neutral-200 hover:text-brand-600 hover:border-brand-200"
                            )}
                            title="Thông tin chi tiết Gene"
                          >
                            <InfoCircle size={18} />
                          </button>
                        </div>

                        {isExpanded && (
                          <div className="mt-4 pt-4 border-t border-neutral-100 animate-in fade-in slide-in-from-top-2 space-y-3">
                            <div className="p-3.5 bg-brand-50/40 border border-brand-100 rounded-xl">
                              <span className="font-bold text-brand-700 uppercase tracking-wider text-[10px] block mb-1">Vai trò sinh học</span>
                              <span className="text-brand-900/90 text-sm leading-relaxed block">
                                {gene.description_vn || 'Chưa có mô tả (chạy scripts/generate_bio_descriptions.py để sinh).'}
                              </span>
                            </div>

                            {/* Compact single-row meta strip instead of a tall left column —
                                avoids the large empty gap that used to appear next to short
                                fields when the GO text on the other side ran much longer. */}
                            <div className="flex flex-wrap gap-2 text-xs">
                              <span className="px-2.5 py-1 rounded-md bg-neutral-50 border border-neutral-200 text-neutral-700">
                                <span className="text-neutral-500 font-semibold">Species:</span> {gene.species || '—'}
                              </span>
                              <span className="px-2.5 py-1 rounded-md bg-neutral-50 border border-neutral-200 text-neutral-700 font-mono">
                                <span className="text-neutral-500 font-semibold font-sans">GenBank:</span> {gene.genbank_acc || '—'}
                              </span>
                              <span className="px-2.5 py-1 rounded-md bg-neutral-50 border border-neutral-200 text-neutral-700 font-mono">
                                <span className="text-neutral-500 font-semibold font-sans">RefSeq:</span> {gene.refseq || '—'}
                              </span>
                              <span className="px-2.5 py-1 rounded-md bg-brand-50 border border-brand-100 text-brand-800 font-mono font-semibold">
                                <span className="text-brand-500 font-semibold font-sans">Probes:</span> {gene.probes || '—'}
                              </span>
                            </div>

                            {gene.go_format === 'combined' ? (
                              <div className="p-3 rounded-lg bg-neutral-50 border border-neutral-100">
                                <span
                                  className="font-bold text-neutral-500 uppercase tracking-wider text-[10px] block mb-1"
                                  title="Nền tảng annotation này (Agilent) không tách BP/CC/MF riêng — toàn bộ GO term gộp chung vào đây."
                                >
                                  GO Terms
                                </span>
                                <span className="text-neutral-800 leading-relaxed text-sm">{gene.go_terms || '—'}</span>
                              </div>
                            ) : (
                              // One category per full-width row (not 3 narrow columns side
                              // by side) — GO term text is often long, and squeezing 3
                              // columns into this panel's width forces it to wrap across
                              // many lines per box, which reads worse than a single wide row.
                              <div className="space-y-2">
                                <div className="p-3 rounded-lg bg-neutral-50 border border-neutral-100">
                                  <span className="font-bold text-neutral-500 uppercase tracking-wider text-[10px] block mb-1">Biological Process</span>
                                  <span className="text-neutral-800 leading-relaxed text-xs">{gene.go_biological_process || '—'}</span>
                                </div>
                                <div className="p-3 rounded-lg bg-neutral-50 border border-neutral-100">
                                  <span className="font-bold text-neutral-500 uppercase tracking-wider text-[10px] block mb-1">Cellular Component</span>
                                  <span className="text-neutral-800 leading-relaxed text-xs">{gene.go_cellular_component || '—'}</span>
                                </div>
                                <div className="p-3 rounded-lg bg-neutral-50 border border-neutral-100">
                                  <span className="font-bold text-neutral-500 uppercase tracking-wider text-[10px] block mb-1">Molecular Function</span>
                                  <span className="text-neutral-800 leading-relaxed text-xs">{gene.go_molecular_function || '—'}</span>
                                </div>
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    </div>
                  );
                })}
                {genes.length === 0 && !error && (
                  <p className="text-sm text-neutral-500 text-center py-6">Không có gene nào để hiển thị.</p>
                )}
                {genes.length > 0 && filteredGenes.length === 0 && (
                  <p className="text-sm text-neutral-500 text-center py-6">Không có gene nào khớp với tìm kiếm.</p>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
