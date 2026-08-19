import React, { useState, useEffect, useRef } from "react";
import { GitMerge } from "lucide-react";
import { ChevronDown, ChevronUp, FileText, ArrowRight, Bolt1, InfoCircle, Filter, Search1 } from "@tailgrids/icons";
import { cn } from "../lib/utils";
import { classColor } from "../lib/palette";
import { api } from "../lib/api";
import { Input } from "./tailgrids/core/input";
import FieldLabel from "./ui/FieldLabel";
import ConfigRow from "./ui/ConfigRow";

interface RuleItem {
  rule_id: number;
  text: string;
  explanation?: string | null;
  metrics: Record<string, number>;
  consequent?: { class_idx?: number; class_label?: string };
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
  go_format?: "split" | "combined";
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
  { key: "fidelity", label: "Fidelity" },
  { key: "lift", label: "Lift" },
  { key: "confidence", label: "Confidence" },
  { key: "support", label: "Support" },
];

/** DOM id for a gene's card — exported so other components (the test-results
 * "Danh sách Gene trong Luật Khớp" panel) can jump straight to it instead of
 * re-describing the same gene a second time. */
export function geneAnchorId(gene: string) {
  return `gene-card-${gene.replace(/[^a-zA-Z0-9_-]/g, "_")}`;
}

/** One funnel stage — same size and shade as its siblings (the shrinking
 * count itself carries the "funnel" story, not a graduated box size), each
 * connected by a centered arrow. */
function FunnelStage({ value, label, rejected }: { value: number; label: string; rejected?: string }) {
  return (
    <div className="flex-1 min-w-[140px] rounded-xl border border-neutral-200 py-6 px-4 flex flex-col items-center justify-center text-center">
      <span className="font-bold font-mono mb-1.5 text-3xl text-brand-600">{value.toLocaleString()}</span>
      <span className="text-xs font-bold uppercase tracking-wider mb-1 text-neutral-600">{label}</span>
      {rejected && <span className="text-xs font-bold uppercase text-danger-600">{rejected}</span>}
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
  const [error, setError] = useState("");
  const [ruleSearch, setRuleSearch] = useState("");
  const [geneSearch, setGeneSearch] = useState("");
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!datasetId) return;
    let ignore = false;
    setRules([]);
    setGenes([]);
    setError("");
    // allSettled (not all): rules and genes are independent endpoints — a
    // dataset with no annotation_file never gets a gene_description.csv, so
    // getGenes 404s while getRules succeeds. Don't let that failure blank
    // out rules that DID load successfully.
    Promise.allSettled([
      api.getRules(datasetId, fsMethod, model, runId),
      api.getGenes(datasetId, fsMethod, model, runId),
    ]).then(([rulesResult, genesResult]) => {
      if (ignore) return;
      if (rulesResult.status === "fulfilled") {
        setRules(rulesResult.value.rules || []);
      } else {
        setError(rulesResult.reason?.message || String(rulesResult.reason));
      }
      if (genesResult.status === "fulfilled") {
        setGenes(genesResult.value || []);
      }
    });
    return () => {
      ignore = true;
    };
  }, [datasetId, fsMethod, model, runId]);

  // Let other components (test-results "Danh sách Gene trong Luật Khớp")
  // jump straight to a gene's card here instead of re-describing it — opens
  // both the genes panel and that gene's own detail, then scrolls to it.
  useEffect(() => {
    const handler = (e: Event) => {
      const gene = (e as CustomEvent<{ gene: string }>).detail?.gene;
      if (!gene) return;
      setIsGenesCollapsed(false);
      const idx = genes.findIndex((g) => g.gene === gene);
      if (idx >= 0) setExpandedGenes((prev) => ({ ...prev, [idx]: true }));
      setTimeout(() => {
        document.getElementById(geneAnchorId(gene))?.scrollIntoView({ behavior: "smooth", block: "center" });
      }, 150);
    };
    window.addEventListener("jump-to-gene", handler);
    return () => window.removeEventListener("jump-to-gene", handler);
  }, [genes]);

  const toggleGene = (idx: number) => {
    setExpandedGenes((prev) => ({ ...prev, [idx]: !prev[idx] }));
  };

  const toggleRule = (idx: number) => {
    setExpandedRules((prev) => ({ ...prev, [idx]: !prev[idx] }));
  };

  const toggleGroup = (label: string) => {
    setCollapsedGroups((prev) => ({ ...prev, [label]: !prev[label] }));
  };

  // Group rules by predicted class so the list reads as "here's what the
  // model learned for class X" rather than one long undifferentiated stream.
  const ruleGroups: Record<string, RuleItem[]> = {};
  for (const r of rules) {
    const label = r.consequent?.class_label || "Không xác định";
    (ruleGroups[label] ||= []).push(r);
  }
  const groupLabels = Object.keys(ruleGroups).sort((a, b) => ruleGroups[b].length - ruleGroups[a].length);
  const allLabels = labels && labels.length ? labels : groupLabels;

  const ruleQuery = ruleSearch.trim().toLowerCase();
  const filteredGroups: Record<string, RuleItem[]> = {};
  let filteredRuleCount = 0;
  for (const label of groupLabels) {
    const matched = ruleQuery
      ? ruleGroups[label].filter(
          (r) => r.text.toLowerCase().includes(ruleQuery) || label.toLowerCase().includes(ruleQuery),
        )
      : ruleGroups[label];
    if (matched.length) {
      filteredGroups[label] = matched;
      filteredRuleCount += matched.length;
    }
  }

  const geneQuery = geneSearch.trim().toLowerCase();
  const filteredGenes = geneQuery
    ? genes.filter(
        (g) => g.gene.toLowerCase().includes(geneQuery) || (g.gene_title || "").toLowerCase().includes(geneQuery),
      )
    : genes;

  return (
    <div ref={containerRef} className="animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div className="flex items-center gap-3 mb-6">
        <div className="w-10 h-10 rounded-full bg-brand-50 flex items-center justify-center text-brand-600 border border-brand-100">
          <Bolt1 size={20} />
        </div>
        <h2 className="text-xl font-bold text-neutral-800">Kết quả trích xuất luật sinh học</h2>
      </div>

      {error && <p className="text-sm text-danger-600 mb-4">Không tải được rules/genes: {error}</p>}

      {/* Rule-extraction log — what actually happened during mining, not just
          the final kept count: how many candidate rules were mined, how many
          passed the confidence/support filter, and how many were then capped.
          Blocks visibly shrink stage to stage since the rule set itself
          shrinks — arrows sit centered between them. */}
      {rulesSummary && rulesSummary.n_rules_raw != null && (
        <div className="mb-6">
          <div className="flex items-stretch gap-3 sm:gap-4 overflow-auto">
            <FunnelStage value={rulesSummary.n_rules_raw ?? 0} label="Luật khai thác" />
            <ArrowRight size={20} className="text-neutral-300 shrink-0 self-center" />
            <FunnelStage
              value={rulesSummary.n_rules_passed_filter ?? 0}
              label="Qua bộ lọc"
              rejected={rulesSummary.n_dropped_by_filter ? `-${rulesSummary.n_dropped_by_filter} bị loại` : undefined}
            />
            <ArrowRight size={20} className="text-neutral-300 shrink-0 self-center" />
            <FunnelStage
              value={rulesSummary.n_rules_kept ?? 0}
              label="Luật giữ lại"
              rejected={rulesSummary.n_dropped_by_cap ? `-${rulesSummary.n_dropped_by_cap} do giới hạn` : undefined}
            />
          </div>
        </div>
      )}

      <div className="divide-y divide-neutral-200">
        {/* List of Rules — grouped by predicted class, each group tinted with
            that class's own color (same classColor() used everywhere else)
            via a small dot + colored label only (not a full tinted box), each
            individually collapsible, with a search box to filter. */}
        <div className="py-6 first:pt-0">
          <div
            className="flex items-center justify-between cursor-pointer group/rules"
            onClick={() => setIsRulesCollapsed(!isRulesCollapsed)}
          >
            <div className="flex items-center gap-2">
              <GitMerge className="text-brand-600" size={20} />
              <h3 className="text-base font-semibold text-neutral-800 group-hover/rules:text-brand-700 transition-colors">
                Danh sách luật trích xuất ({rules.length})
              </h3>
            </div>
            <button className="p-1.5 rounded-full hover:bg-brand-50 text-neutral-600 transition-colors shrink-0">
              {isRulesCollapsed ? <ChevronDown size={20} /> : <ChevronUp size={20} />}
            </button>
          </div>

          {!isRulesCollapsed && (
            <div className="mt-6 space-y-5 animate-in fade-in slide-in-from-top-2 duration-200">
              <div className="relative">
                <Search1 size={15} className="absolute left-3 top-1/2 -translate-y-1/2 text-neutral-600 z-10" />
                <Input
                  value={ruleSearch}
                  onChange={(e) => setRuleSearch(e.target.value)}
                  placeholder="Tìm luật theo gene, điều kiện hoặc tên lớp..."
                  className="w-full pl-9 text-sm"
                />
              </div>
              {ruleQuery && (
                <p className="text-xs text-neutral-600 -mt-2">
                  Tìm thấy {filteredRuleCount}/{rules.length} luật khớp với "{ruleSearch}"
                </p>
              )}
              <div className="divide-y divide-neutral-100">
                {Object.keys(filteredGroups).map((label) => {
                  const color = classColor(label, allLabels);
                  const groupCollapsed = collapsedGroups[label];
                  return (
                    <div key={label} className="py-4 first:pt-0 last:pb-0">
                      <div className="flex items-center gap-2.5 cursor-pointer" onClick={() => toggleGroup(label)}>
                        <span
                          className="w-2.5 h-2.5 rounded-full inline-block shrink-0"
                          style={{ backgroundColor: color }}
                        />
                        <h4 className="font-bold text-base flex-1" style={{ color }}>
                          {label}
                        </h4>
                        <span className="text-sm text-neutral-600 font-medium">
                          ({filteredGroups[label].length} luật)
                        </span>
                        <button className="p-1 rounded-full hover:bg-neutral-100 text-neutral-600 transition-colors shrink-0">
                          {groupCollapsed ? <ChevronDown size={18} /> : <ChevronUp size={18} />}
                        </button>
                      </div>
                      {!groupCollapsed && (
                        <div className="mt-3 space-y-3 animate-in fade-in slide-in-from-top-2 duration-200">
                          {filteredGroups[label].map((ruleItem) => {
                            const condition = ruleItem.text.split(" THEN ")[0].replace(/^IF /, "");
                            const clauses = condition.split(" AND ");
                            const isExpanded = expandedRules[ruleItem.rule_id];
                            const metricEntries = DISPLAYED_METRICS.filter(
                              ({ key }) => ruleItem.metrics?.[key] != null,
                            ).map(({ key, label }) => [label, ruleItem.metrics[key]] as [string, number]);

                            return (
                              <div key={ruleItem.rule_id} className="bg-white rounded-lg border border-neutral-200 p-4">
                                <div className="flex items-start justify-between gap-3">
                                  <code className="font-mono text-sm leading-relaxed text-neutral-800 flex-1 pt-0.5">
                                    <span className="font-bold mr-2 tracking-wide" style={{ color }}>
                                      IF
                                    </span>
                                    {clauses.map((clause, i) => (
                                      <React.Fragment key={i}>
                                        {i > 0 && (
                                          <span className="font-bold not-italic mx-1.5 tracking-wide" style={{ color }}>
                                            AND
                                          </span>
                                        )}
                                        {clause}
                                      </React.Fragment>
                                    ))}
                                  </code>
                                  <button
                                    onClick={(e) => {
                                      e.stopPropagation();
                                      toggleRule(ruleItem.rule_id);
                                    }}
                                    className={cn(
                                      "p-1.5 rounded-full transition-colors shrink-0 border",
                                      isExpanded
                                        ? "bg-brand-50 text-brand-600 border-brand-100"
                                        : "bg-white text-neutral-500 border-neutral-200 hover:text-brand-600 hover:border-brand-200",
                                    )}
                                    title="Giải thích luật"
                                  >
                                    <InfoCircle size={16} />
                                  </button>
                                </div>

                                {isExpanded && (
                                  <div className="mt-4 pt-4 border-t border-neutral-100 animate-in fade-in slide-in-from-top-2">
                                    <FieldLabel className="text-neutral-800 text-[0.6875rem] flex items-center gap-1.5 mb-1.5">
                                      {" "}
                                      Mô tả sinh học
                                    </FieldLabel>
                                    <p className="text-sm leading-relaxed text-neutral-800">
                                      {ruleItem.explanation ||
                                        "Chưa có mô tả (chạy scripts/generate_bio_descriptions.py để sinh)."}
                                    </p>
                                  </div>
                                )}

                                {metricEntries.length > 0 && (
                                  <p className="mt-3 pt-3 border-t border-neutral-100 text-xs font-mono text-neutral-600">
                                    {metricEntries.map(([metricLabel, val], i) => (
                                      <React.Fragment key={metricLabel}>
                                        {i > 0 && <span className="text-neutral-300 mx-1.5">·</span>}
                                        {metricLabel}{" "}
                                        <span className="text-neutral-800 font-bold">
                                          {typeof val === "number" ? val.toFixed(2) : String(val)}
                                        </span>
                                      </React.Fragment>
                                    ))}
                                  </p>
                                )}
                              </div>
                            );
                          })}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
              {rules.length === 0 && !error && (
                <p className="text-sm text-neutral-600 text-center py-6">Không có luật nào để hiển thị.</p>
              )}
              {rules.length > 0 && Object.keys(filteredGroups).length === 0 && (
                <p className="text-sm text-neutral-600 text-center py-6">Không có luật nào khớp với tìm kiếm.</p>
              )}
            </div>
          )}
        </div>

        {/* List of Genes */}
        <div className="py-6 last:pb-0">
          <div
            className="flex items-center justify-between cursor-pointer group/genes"
            onClick={() => setIsGenesCollapsed(!isGenesCollapsed)}
          >
            <div className="flex items-center gap-2">
              <FileText className="text-brand-600" size={20} />
              <h3 className="text-base font-semibold text-neutral-800 group-hover/genes:text-brand-700 transition-colors">
                Gene xuất hiện trong danh sách luật ({genes.length})
              </h3>
            </div>
            <button className="p-1.5 rounded-full hover:bg-brand-50 text-neutral-600 transition-colors shrink-0">
              {isGenesCollapsed ? <ChevronDown size={20} /> : <ChevronUp size={20} />}
            </button>
          </div>

          {!isGenesCollapsed && (
            <div className="mt-6 space-y-4 animate-in fade-in slide-in-from-top-2 duration-200">
              <div className="relative">
                <Search1 size={15} className="absolute left-3 top-1/2 -translate-y-1/2 text-neutral-600 z-10" />
                <Input
                  value={geneSearch}
                  onChange={(e) => setGeneSearch(e.target.value)}
                  placeholder="Tìm gene theo ký hiệu hoặc tên..."
                  className="w-full pl-9 text-sm"
                />
              </div>
              {geneQuery && (
                <p className="text-xs text-neutral-600">
                  Tìm thấy {filteredGenes.length}/{genes.length} gene khớp với "{geneSearch}"
                </p>
              )}
              {filteredGenes.map((gene) => {
                const idx = genes.indexOf(gene);
                const isExpanded = expandedGenes[idx];
                const geneClasses = gene.classes
                  ? gene.classes
                      .split(";")
                      .map((c) => c.trim())
                      .filter(Boolean)
                  : [];
                return (
                  <div
                    key={gene.gene}
                    id={geneAnchorId(gene.gene)}
                    className="bg-white border border-neutral-200 rounded-xl hover:border-brand-300 transition-colors scroll-mt-4"
                  >
                    <div className="p-4">
                      <div className="flex justify-between items-start gap-4">
                        {/* Row 1: symbol + rule count — the identifying, always-short
                            part, kept on its own line so it never competes for space
                            with the (often long) title or the class badge list below. */}
                        <div className="min-w-0 flex-1 space-y-1.5">
                          <div className="flex items-center gap-2.5 flex-wrap">
                            <span className="font-bold text-brand-900 text-lg tracking-tight">{gene.gene}</span>
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
                              <FieldLabel as="span" className="text-neutral-600">Xuất hiện ở lớp:</FieldLabel>
                              {geneClasses.map((c) => {
                                const color = classColor(c, allLabels);
                                return (
                                  <span
                                    key={c}
                                    className="px-2 py-0.5 text-[10px] uppercase tracking-wider rounded-md font-bold border"
                                    style={{ backgroundColor: color + "14", color, borderColor: color + "40" }}
                                  >
                                    {c}
                                  </span>
                                );
                              })}
                            </div>
                          )}
                        </div>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            toggleGene(idx);
                          }}
                          className={cn(
                            "p-2 rounded-full transition-colors shrink-0 flex items-center justify-center border",
                            isExpanded
                              ? "bg-brand-50 text-brand-600 border-brand-100"
                              : "bg-white text-neutral-600 border-neutral-200 hover:text-brand-600 hover:border-brand-200",
                          )}
                          title="Thông tin chi tiết Gene"
                        >
                          <InfoCircle size={18} />
                        </button>
                      </div>

                      {isExpanded && (
                        <div className="mt-4 pt-4 border-t border-neutral-100 animate-in fade-in slide-in-from-top-2 space-y-4">
                          <div>
                            <FieldLabel className="text-[0.6875rem] font-bold text-neutral-800">Vai trò sinh học</FieldLabel>
                            <p className="text-sm text-neutral-800 leading-relaxed mt-1">
                              {gene.description_vn ||
                                "Chưa có mô tả (chạy scripts/generate_bio_descriptions.py để sinh)."}
                            </p>
                          </div>

                          <div className="grid grid-cols-1 gap-y-1">
                            <ConfigRow label="Species" value={gene.species || "—"} mono={false} />
                            <ConfigRow label="GenBank" value={gene.genbank_acc || "—"} />
                            <ConfigRow label="RefSeq" value={gene.refseq || "—"} />
                            <ConfigRow label="Probes" value={gene.probes || "—"} />
                          </div>

                          {gene.go_format === "combined" ? (
                            <div>
                              <FieldLabel>
                                <span title="Nền tảng annotation này (Agilent) không tách BP/CC/MF riêng — toàn bộ GO term gộp chung vào đây.">
                                  GO Terms
                                </span>
                              </FieldLabel>
                              <p className="text-sm text-neutral-800 leading-relaxed mt-1">{gene.go_terms || "—"}</p>
                            </div>
                          ) : (
                            // One category per full-width row (not 3 narrow columns side
                            // by side) — GO term text is often long, and squeezing 3
                            // columns into this panel's width forces it to wrap across
                            // many lines per box, which reads worse than a single wide row.
                            <div className="space-y-3">
                              <div>
                                <FieldLabel className="text-[0.6875rem] font-bold text-neutral-800">Biological Process</FieldLabel>
                                <p className="text-sm text-neutral-800 leading-relaxed mt-1">
                                  {gene.go_biological_process || "—"}
                                </p>
                              </div>
                              <div>
                                <FieldLabel className="text-[0.6875rem] font-bold text-neutral-800">Cellular Component</FieldLabel>
                                <p className="text-sm text-neutral-800 leading-relaxed mt-1">
                                  {gene.go_cellular_component || "—"}
                                </p>
                              </div>
                              <div>
                                <FieldLabel className="text-[0.6875rem] font-bold text-neutral-800">Molecular Function</FieldLabel>
                                <p className="text-sm text-neutral-800 leading-relaxed mt-1">
                                  {gene.go_molecular_function || "—"}
                                </p>
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
                <p className="text-sm text-neutral-600 text-center py-6">Không có gene nào để hiển thị.</p>
              )}
              {genes.length > 0 && filteredGenes.length === 0 && (
                <p className="text-sm text-neutral-600 text-center py-6">Không có gene nào khớp với tìm kiếm.</p>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
