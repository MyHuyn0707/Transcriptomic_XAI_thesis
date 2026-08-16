import React, { useState, useEffect } from 'react';
import { ChevronDown, ChevronUp, Database, ExternalLink, Info, AlertTriangle } from 'lucide-react';
import { api } from '../lib/api';
import { PALETTE, DROPPED_COLOR } from '../lib/palette';
import { cn } from '../lib/utils';
import CollapsibleCard from './ui/CollapsibleCard';

interface OriginInfo {
  geo_accession?: string;
  platform?: string;
  title?: string;
  source_cumida_csv?: string;
  class_characteristic?: string;
  classes_raw?: Record<string, number>;
  annotation_rows?: number;
  probes_with_gene_symbol?: number;
  probes_from_annotation_file?: number;
  probes_from_mygene?: number;
  probes_without_gene_symbol?: number;
  affx_control_removed?: number;
  unique_probes?: number;
  unique_genes?: number;
  samples_kept?: number;
  na_dropped?: number;
}

interface ClassContent {
  display_name_vi?: string;
  description_vi?: string;
  canonical_name_en?: string;
  entity_type?: string;
  excluded_from_model?: boolean;
  label_note_vi?: string;
  source_ids?: string[];
}

interface Reference {
  title: string;
  publisher: string;
  url: string;
  source_type?: string;
}

interface DiseaseContextEntity {
  entity_id: string;
  name_vi: string;
  canonical_name_en?: string;
  description_vi: string;
  source_ids?: string[];
  media_asset_ids?: string[];
}

interface DiseaseContext {
  scope_type?: string;
  primary_display_entity_id?: string | null;
  ui_title_vi?: string;
  primary_entities?: DiseaseContextEntity[];
  dataset_relationship_vi?: string;
  media_asset_ids?: string[];
  media_disclaimer_vi?: string;
}

interface MediaAsset {
  title_vi?: string;
  caption_vi?: string;
  alt_text_vi?: string;
  credit?: string;
  source_page_url?: string;
  source_publisher?: string;
  usage_note_vi?: string;
  url?: string | null;
}

interface DatasetProvenance {
  local_folder_group?: string;
  download_provider?: {
    provider_id?: string;
    name?: string;
    canonical_name?: string;
    source_ids?: string[];
  };
  provenance_note_vi?: string;
  source_ids?: string[];
}

interface DatasetStudy {
  arrayexpress_accession?: string;
  original_study_title_en?: string;
  study_objective_vi?: string;
  original_design_vi?: string;
  local_ml_relation_vi?: string;
  source_ids?: string[];
}

interface ContentInfo {
  title_vi?: string;
  description_vi?: string;
  disease_context?: DiseaseContext;
  dataset_provenance?: DatasetProvenance;
  dataset_study?: DatasetStudy;
  organism?: string;
  experiment_type_vi?: string;
  dataset_note_vi?: string;
  class_descriptions_vi?: Record<string, string>;
  classes?: Record<string, ClassContent>;
  references?: Record<string, Reference>;
  ui_disclaimer_vi?: string;
}

interface OverviewData {
  dataset: string;
  class_labels: string[];
  train_class_counts: Record<string, number>;
  test_class_counts: Record<string, number>;
  raw_class_counts: Record<string, number>;
  dropped_classes: string[];
  n_samples_total: number;
  n_train: number;
  n_test: number;
  test_size: number;
  min_samples_per_class: number;
  visualizations: Record<string, string>;
  origin: OriginInfo;
  content: ContentInfo;
  media_assets?: Record<string, MediaAsset>;
}

/** Reference-citation list nested inside "Nguồn gốc & Chú giải" — its own
 * "Thu gọn/Mở rộng" toggle since the list can get long (many NCI sources)
 * and shouldn't force the whole annotation card to scroll. */
function ReferencesList({ referenceList, disclaimer }: { referenceList: Reference[], disclaimer?: string }) {
  const [open, setOpen] = useState(true);
  return (
    <div className="mt-5 pt-4 border-t border-neutral-100">
      <div className="flex items-center justify-between mb-2">
        <p className="text-[10px] uppercase tracking-wider text-neutral-500 font-bold">Nguồn tham khảo</p>
        <button
          onClick={() => setOpen(o => !o)}
          className="flex items-center gap-1 text-[11px] font-semibold text-brand-700 border border-brand-200 bg-brand-50 hover:bg-brand-100 rounded-md px-2 py-1 transition-colors"
        >
          {open ? 'Thu gọn' : 'Mở rộng'}
          {open ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
        </button>
      </div>
      {open && (
        <div className="animate-in fade-in slide-in-from-top-2 duration-200">
          <ul className="space-y-1">
            {referenceList.map(ref => (
              <li key={ref.url} className="text-xs text-neutral-600 flex items-center gap-1.5">
                <ExternalLink size={11} className="shrink-0 text-neutral-500" />
                <a href={ref.url} target="_blank" rel="noreferrer" className="hover:text-brand-600 hover:underline">
                  {ref.title}
                </a>
                <span className="text-neutral-400">·</span>
                <span>{ref.publisher}</span>
              </li>
            ))}
          </ul>
          {disclaimer && (
            <p className="text-[11px] text-neutral-500 italic mt-3">{disclaimer}</p>
          )}
        </div>
      )}
    </div>
  );
}

export default function DatasetOverview({ datasetId, collapseSignal, isTemp }: { datasetId: string, collapseSignal?: boolean, isTemp?: boolean }) {
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [overview, setOverview] = useState<OverviewData | null>(null);
  const [error, setError] = useState('');

  useEffect(() => {
    setIsCollapsed(!!collapseSignal);
  }, [collapseSignal]);

  useEffect(() => {
    // Session-only uploaded datasets never get a Dataset_Description entry or
    // an outputs_holdout/{id}/split_info.json (no offline holdout run ever
    // touches them) — /overview 404s unconditionally for them, so skip the
    // fetch entirely instead of surfacing that as a raw backend error.
    if (!datasetId || isTemp) { setOverview(null); setError(''); return; }
    let ignore = false;
    setOverview(null);
    setError('');
    api.getOverview(datasetId)
      .then(o => { if (!ignore) setOverview(o); })
      .catch(e => { if (!ignore) setError(e.message); });
    return () => { ignore = true; };
  }, [datasetId, isTemp]);

  if (!datasetId) return null;

  if (isTemp) {
    return (
      <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
        <div className="bg-white p-6 rounded-2xl shadow-sm border border-success-100">
          <div className="flex items-center gap-3 mb-2">
            <div className="w-10 h-10 rounded-full bg-success-50 flex items-center justify-center text-success-600 border border-success-100">
              <Database size={20} />
            </div>
            <h2 className="text-2xl font-bold text-neutral-900 tracking-tight">Tổng quan Dữ liệu</h2>
          </div>
          <div className="mt-4 flex items-start gap-2.5 bg-warning-50 border border-warning-200 rounded-lg p-4 text-sm text-warning-800 leading-relaxed">
            <AlertTriangle size={16} className="text-warning-500 shrink-0 mt-0.5" />
            <p>
              Dataset này được tải lên trong phiên làm việc hiện tại và chưa qua kiểm duyệt hoặc
              tìm kiếm thông tin chi tiết (nghiên cứu gốc, chú giải sinh học, hình ảnh minh họa),
              nên tạm thời chưa có thông tin tổng quan để hiển thị.
            </p>
          </div>
        </div>
      </div>
    );
  }

  const origin: OriginInfo = overview?.origin || {};
  const content: ContentInfo = overview?.content || {};
  const diseaseContext = content.disease_context;
  const primaryEntity = diseaseContext?.primary_entities?.find(
    e => e.entity_id === diseaseContext.primary_display_entity_id,
  );
  const classesContent = content.classes || {};
  // Same order/PALETTE indexing DatasetSplit's "Phân phối các lớp" chart uses
  // (both key off overview.raw_class_counts) so a class's color here matches
  // its color in Bước 2 exactly, even though the two live in different steps.
  const allLabels = overview ? Object.keys(overview.raw_class_counts) : [];
  const droppedSet = new Set(overview?.dropped_classes || []);
  const references = content.references || {};
  const datasetStudy = content.dataset_study;
  const datasetProvenance = content.dataset_provenance;
  const usedReferenceIds = new Set<string>();
  primaryEntity?.source_ids?.forEach(id => usedReferenceIds.add(id));
  Object.values(classesContent).forEach(c => {
    c.source_ids?.forEach(id => usedReferenceIds.add(id));
  });
  datasetStudy?.source_ids?.forEach(id => usedReferenceIds.add(id));
  datasetProvenance?.download_provider?.source_ids?.forEach(id => usedReferenceIds.add(id));
  const referenceList = [...usedReferenceIds].map(id => references[id]).filter(Boolean);

  return (
    <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div className="bg-white p-6 rounded-2xl shadow-sm border border-success-100 transition-all duration-300">
        <div
          className="flex items-center justify-between cursor-pointer group"
          onClick={() => setIsCollapsed(!isCollapsed)}
        >
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-success-50 flex items-center justify-center text-success-600 border border-success-100">
              <Database size={20} />
            </div>
            <h2 className="text-2xl font-bold text-neutral-900 tracking-tight group-hover:text-success-700 transition-colors">
              Tổng quan Dữ liệu {origin.geo_accession ? `(${origin.geo_accession})` : ''}
            </h2>
          </div>
          <button className="p-2 rounded-full hover:bg-success-50 text-neutral-600 hover:text-success-600 transition-colors">
            {isCollapsed ? <ChevronDown size={24} /> : <ChevronUp size={24} />}
          </button>
        </div>

        {!isCollapsed && (
          <div className="mt-6 animate-in fade-in slide-in-from-top-4 duration-300 space-y-8">
            {error && <p className="text-sm text-danger-600">Không tải được overview: {error}</p>}
            {overview && (
              <>
                {/* Summary card — GEO accession / platform left, title / organism right.
                    Rare-class-drop stats, train/test split, class-distribution charts and
                    per-class biology now live in Bước 2 "Xử lý & Chia Dữ liệu" (DatasetSplit). */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                  <div className="bg-neutral-50 p-5 rounded-xl border border-neutral-100 space-y-3 shadow-sm">
                    {origin.geo_accession && (
                      <div className="flex justify-between py-2 border-b border-neutral-200/60">
                        <span className="text-neutral-600 font-medium">Mã truy cập GEO</span>
                        <span className="text-success-700 font-bold font-mono bg-success-100/50 px-2 py-0.5 rounded">{origin.geo_accession}</span>
                      </div>
                    )}
                    {origin.platform && (
                      <div className="flex justify-between py-2 border-b border-neutral-200/60">
                        <span className="text-neutral-600 font-medium">Nền tảng vi mảng</span>
                        <span className="text-success-700 font-bold font-mono bg-success-100/50 px-2 py-0.5 rounded">{origin.platform}</span>
                      </div>
                    )}
                    {origin.samples_kept != null && (
                      <div className="flex justify-between py-2 border-b border-neutral-200/60">
                        <span className="text-neutral-600 font-medium">Tổng số mẫu ban đầu</span>
                        <span className="text-neutral-900 font-bold font-mono">{origin.samples_kept}</span>
                      </div>
                    )}
                    {overview.raw_class_counts && (
                      <div className="flex justify-between py-2 border-b border-neutral-200/60">
                        <span className="text-neutral-600 font-medium">Tổng số lớp ban đầu</span>
                        <span className="text-neutral-900 font-bold font-mono">{Object.keys(overview.raw_class_counts).length}</span>
                      </div>
                    )}
                  </div>
                  <div className="bg-neutral-50 p-5 rounded-xl border border-neutral-100 space-y-3 shadow-sm">
                    <div className="flex flex-col py-2 border-b border-neutral-200/60">
                      <span className="text-neutral-600 font-medium mb-1">Tiêu đề</span>
                      {content.title_vi && (
                        <span className="text-neutral-900 font-semibold leading-relaxed">{content.title_vi}</span>
                      )}
                      {/* dataset_study.original_study_title_en (cross-checked against ArrayExpress+GEO)
                          is populated for every dataset, CuMiDa included — origin.title only comes
                          from the annotation report's "Title" line, which CuMiDa reports don't have. */}
                      {(datasetStudy?.original_study_title_en || origin.title) && (
                        <span className="text-neutral-500 text-xs leading-relaxed mt-0.5">
                          {datasetStudy?.original_study_title_en || origin.title}
                        </span>
                      )}
                      {!content.title_vi && !datasetStudy?.original_study_title_en && !origin.title && (
                        <span className="text-neutral-900 font-semibold">—</span>
                      )}
                    </div>
                    <div className="flex justify-between py-2 border-b border-neutral-200/60">
                      <span className="text-neutral-600 font-medium">Organism</span>
                      <span className="text-neutral-900 font-semibold text-right">{content.organism || '—'}</span>
                    </div>
                  </div>
                </div>

                {/* Dataset description — concise summary */}
                {content.description_vi && (
                  <div className="bg-neutral-50 rounded-xl border border-neutral-100 p-4">
                    <p className="text-[10px] uppercase tracking-wider text-neutral-500 font-bold mb-1.5">Mô tả tập dữ liệu</p>
                    <p className="text-sm text-neutral-700 leading-relaxed">{content.description_vi}</p>
                  </div>
                )}

                {/* Disease context: what this disease/family actually is, in plain Vietnamese */}
                {diseaseContext && (primaryEntity || diseaseContext.dataset_relationship_vi) && (() => {
                  const assetId = (primaryEntity?.media_asset_ids || diseaseContext.media_asset_ids || [])[0];
                  const asset = assetId ? overview.media_assets?.[assetId] : undefined;
                  return (
                    <CollapsibleCard color="#0d9488" title={diseaseContext.ui_title_vi || 'Bối cảnh bệnh lý'}>
                      <div className={cn("gap-5", asset?.url ? "grid grid-cols-1 sm:grid-cols-[220px_1fr]" : "")}>
                        {asset?.url && (
                          <figure className="shrink-0">
                            <img
                              src={asset.url}
                              alt={asset.alt_text_vi || asset.title_vi || ''}
                              className="w-full sm:w-[220px] h-auto rounded-lg border border-neutral-200 shadow-sm object-cover"
                            />
                            <figcaption className="mt-2 text-[11px] text-neutral-500 leading-snug">
                              {asset.caption_vi}
                              {asset.credit && <span className="block mt-0.5">Ảnh: {asset.credit}</span>}
                              {asset.source_page_url && (
                                <a href={asset.source_page_url} target="_blank" rel="noreferrer" className="text-brand-600 hover:underline block mt-0.5">
                                  Nguồn: {asset.source_publisher || 'NCI'}
                                </a>
                              )}
                            </figcaption>
                          </figure>
                        )}
                        <div className="space-y-3 min-w-0 bg-neutral-50/70 rounded-xl border border-neutral-100 p-4">
                          {primaryEntity && (
                            <p className="text-sm text-neutral-800 leading-relaxed">{primaryEntity.description_vi}</p>
                          )}
                          {diseaseContext.dataset_relationship_vi && (
                            <p className="text-sm text-neutral-700 leading-relaxed italic">{diseaseContext.dataset_relationship_vi}</p>
                          )}
                          {asset && diseaseContext.media_disclaimer_vi && (
                            <p className="text-[11px] text-neutral-500 leading-relaxed border-t border-neutral-200 pt-2">{diseaseContext.media_disclaimer_vi}</p>
                          )}
                        </div>
                      </div>
                    </CollapsibleCard>
                  );
                })()}

                {/* Original study — cross-checked against ArrayExpress + NCBI GEO, separate from
                    where the data FILE was downloaded (dataset_provenance, below) */}
                {datasetStudy && (
                  <CollapsibleCard color="#0d9488" title="Nghiên cứu gốc (đối chiếu ArrayExpress & NCBI GEO)">
                    <div className="space-y-5 text-sm">
                      {datasetStudy.original_study_title_en && (
                        <div>
                          <span className="text-neutral-500 text-xs uppercase tracking-wider font-semibold block mb-1">Tiêu đề nghiên cứu gốc</span>
                          <span className="text-neutral-800 italic">{datasetStudy.original_study_title_en}</span>
                          {datasetStudy.arrayexpress_accession && (
                            <span className="text-neutral-500 text-xs font-mono ml-2">({datasetStudy.arrayexpress_accession})</span>
                          )}
                        </div>
                      )}
                      {datasetStudy.study_objective_vi && (
                        <div>
                          <span className="text-neutral-500 text-xs uppercase tracking-wider font-semibold block mb-1">Mục tiêu nghiên cứu</span>
                          <span className="text-neutral-800 leading-relaxed">{datasetStudy.study_objective_vi}</span>
                        </div>
                      )}
                      {datasetStudy.original_design_vi && (
                        <div>
                          <span className="text-neutral-500 text-xs uppercase tracking-wider font-semibold block mb-1">Thiết kế nghiên cứu gốc</span>
                          <span className="text-neutral-800 leading-relaxed">{datasetStudy.original_design_vi}</span>
                        </div>
                      )}
                    </div>
                    {datasetStudy.local_ml_relation_vi && (
                      <div className="mt-5 pt-5 border-t border-neutral-100">
                        <div className="bg-info-50/60 rounded-lg p-4 border border-info-200 flex gap-2.5">
                          <Info size={16} className="text-info-500 shrink-0 mt-0.5" />
                          <div>
                            <span className="text-info-700 text-xs uppercase tracking-wider font-semibold block mb-1">Liên hệ với bài toán học máy ở đây</span>
                            <span className="text-neutral-800 leading-relaxed text-sm">{datasetStudy.local_ml_relation_vi}</span>
                          </div>
                        </div>
                      </div>
                    )}
                    {datasetProvenance?.download_provider?.name && (
                      <p className="text-xs text-neutral-500 mt-4 pt-3 border-t border-neutral-100">
                        Nguồn tải file dữ liệu: <span className="font-semibold text-neutral-600">{datasetProvenance.download_provider.name}</span>
                        {datasetProvenance.provenance_note_vi && ` — ${datasetProvenance.provenance_note_vi}`}
                      </p>
                    )}
                  </CollapsibleCard>
                )}

                {/* Per-class biological definitions — placed after "Nghiên cứu gốc" since it
                    elaborates on the classes that study defines. Colors match Bước 2's
                    "Phân phối các lớp" chart exactly (both index PALETTE by position in
                    raw_class_counts), so a class keeps the same color across both steps. */}
                {Object.keys(classesContent).length > 0 && (
                  <CollapsibleCard color="#0d9488" title="Đặc tính Sinh học các Phân lớp">
                    {content.dataset_note_vi && (
                      <div className="bg-warning-50 border border-warning-200 rounded-lg p-3.5 text-sm text-warning-800 leading-relaxed mb-4 flex gap-2.5">
                        <AlertTriangle size={16} className="text-warning-500 shrink-0 mt-0.5" />
                        <div><span className="font-semibold">Lưu ý: </span>{content.dataset_note_vi}</div>
                      </div>
                    )}
                    {(() => {
                      const entries = Object.entries(classesContent);
                      const isExcluded = (label: string, cls: ClassContent) => cls.excluded_from_model || droppedSet.has(label);
                      const active = entries.filter(([label, cls]) => !isExcluded(label, cls));
                      const excluded = entries.filter(([label, cls]) => isExcluded(label, cls));
                      const classCard = (label: string, cls: ClassContent, color: string) => (
                        <div
                          key={label}
                          className="bg-white p-4 rounded-lg border border-neutral-200 shadow-sm border-l-4 flex flex-col items-start text-left h-full"
                          style={{ borderLeftColor: color }}
                        >
                          <div className="flex items-start gap-2 mb-2 w-full">
                            <span className="w-2 h-2 rounded-full inline-block mt-1.5 shrink-0" style={{ backgroundColor: color }}></span>
                            <div className="flex flex-col items-start text-left">
                              <h4 className="font-bold text-neutral-900 leading-tight text-left">{cls.display_name_vi || label}</h4>
                              {cls.display_name_vi && (
                                <span className="text-neutral-500 text-[11px] uppercase tracking-wide font-semibold text-left">{label}</span>
                              )}
                            </div>
                          </div>
                          <p className="text-sm text-neutral-700 leading-relaxed text-left">{cls.description_vi}</p>
                        </div>
                      );
                      return (
                        <>
                          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                            {active.map(([label, cls]) => {
                              const idx = allLabels.indexOf(label);
                              return classCard(label, cls, PALETTE[(idx >= 0 ? idx : 0) % PALETTE.length]);
                            })}
                          </div>
                          {/* Excluded classes get their own strip — same visual weight as
                              a single grid cell would look lopsided when there's an odd
                              count left over after the active-classes 2-column grid. */}
                          {excluded.length > 0 && (
                            <div className={cn("space-y-3", active.length > 0 && "mt-4 pt-4 border-t border-neutral-100")}>
                              {excluded.map(([label, cls]) => (
                                <div
                                  key={label}
                                  className="bg-warning-50/60 p-4 rounded-lg border border-warning-100 border-l-4 flex flex-col items-start text-left"
                                  style={{ borderLeftColor: DROPPED_COLOR }}
                                >
                                  <div className="flex items-start gap-2 mb-2 w-full">
                                    <span className="w-2 h-2 rounded-full inline-block mt-1.5 shrink-0" style={{ backgroundColor: DROPPED_COLOR }}></span>
                                    <div className="flex flex-col items-start text-left">
                                      <h4 className="font-bold text-neutral-900 leading-tight text-left">{cls.display_name_vi || label}</h4>
                                      {cls.display_name_vi && (
                                        <span className="text-neutral-500 text-[11px] uppercase tracking-wide font-semibold text-left">{label}</span>
                                      )}
                                    </div>
                                  </div>
                                  <p className="text-sm text-neutral-700 leading-relaxed text-left mb-2.5">{cls.description_vi}</p>
                                  <div className="w-full flex items-start gap-1.5 text-warning-700">
                                    <AlertTriangle size={13} className="shrink-0 mt-0.5 text-warning-500" />
                                    <p className="text-xs leading-relaxed text-left">
                                      Bị loại khỏi tập huấn luyện do quá ít mẫu.
                                    </p>
                                  </div>
                                </div>
                              ))}
                            </div>
                          )}
                        </>
                      );
                    })()}
                  </CollapsibleCard>
                )}

                {/* Build-time annotation/provenance report + reference citations, in one card
                    (references get their own nested collapse toggle since the list can be long) */}
                {origin.annotation_rows != null && (
                  <CollapsibleCard color="#64748b" title="Nguồn gốc & Chú giải">
                    <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 text-sm">
                      {origin.class_characteristic && (
                        <div className="bg-neutral-50 rounded-lg p-3 border border-neutral-100">
                          <div className="text-[10px] uppercase tracking-wider text-neutral-500 font-bold mb-0.5">Class characteristic</div>
                          <div className="font-mono font-semibold text-neutral-800">{origin.class_characteristic}</div>
                        </div>
                      )}
                      <div className="bg-neutral-50 rounded-lg p-3 border border-neutral-100">
                        <div className="text-[10px] uppercase tracking-wider text-neutral-500 font-bold mb-0.5">Annotation rows</div>
                        <div className="font-mono font-semibold text-neutral-800">{origin.annotation_rows?.toLocaleString()}</div>
                      </div>
                      <div className="bg-neutral-50 rounded-lg p-3 border border-neutral-100">
                        <div className="text-[10px] uppercase tracking-wider text-neutral-500 font-bold mb-0.5">Probes có gene symbol</div>
                        <div className="font-mono font-semibold text-success-600">{origin.probes_with_gene_symbol?.toLocaleString()}</div>
                        <div className="text-[11px] text-neutral-500 mt-1">
                          từ annotation file: {origin.probes_from_annotation_file?.toLocaleString()} · MyGene.info: {origin.probes_from_mygene?.toLocaleString()}
                        </div>
                      </div>
                      <div className="bg-neutral-50 rounded-lg p-3 border border-neutral-100">
                        <div className="text-[10px] uppercase tracking-wider text-neutral-500 font-bold mb-0.5">Probes không có gene symbol</div>
                        <div className="font-mono font-semibold text-danger-500">{origin.probes_without_gene_symbol?.toLocaleString()}</div>
                      </div>
                      <div className="bg-neutral-50 rounded-lg p-3 border border-neutral-100">
                        <div className="text-[10px] uppercase tracking-wider text-neutral-500 font-bold mb-0.5">AFFX control đã loại</div>
                        <div className="font-mono font-semibold text-neutral-800">{origin.affx_control_removed?.toLocaleString()}</div>
                      </div>
                      <div className="bg-neutral-50 rounded-lg p-3 border border-neutral-100">
                        <div className="text-[10px] uppercase tracking-wider text-neutral-500 font-bold mb-0.5">Probes / Genes duy nhất</div>
                        <div className="font-mono font-semibold text-neutral-800">{origin.unique_probes?.toLocaleString()} / {origin.unique_genes?.toLocaleString()}</div>
                      </div>
                      <div className="bg-neutral-50 rounded-lg p-3 border border-neutral-100">
                        <div className="text-[10px] uppercase tracking-wider text-neutral-500 font-bold mb-0.5">Mẫu giữ lại</div>
                        <div className="font-mono font-semibold text-neutral-800">{origin.samples_kept?.toLocaleString()} <span className="text-neutral-500 font-normal">(NA dropped: {origin.na_dropped})</span></div>
                      </div>
                    </div>

                    {referenceList.length > 0 && (
                      <ReferencesList referenceList={referenceList} disclaimer={content.ui_disclaimer_vi} />
                    )}
                  </CollapsibleCard>
                )}
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
