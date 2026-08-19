import React, { useState, useEffect } from "react";
import { Database, AlertTriangle } from "lucide-react";
import { ChevronDown, ChevronUp, Link1AngularRight, InfoCircle } from "@tailgrids/icons";
import { api, ClassContent, ContentInfo, OriginInfo, OverviewData, Reference } from "../lib/api";
import { PALETTE, DROPPED_COLOR } from "../lib/palette";
import CollapsibleCard from "./ui/CollapsibleCard";
import Panel from "./ui/Panel";
import StatTile from "./ui/StatTile";
import FieldLabel from "./ui/FieldLabel";

/** Reference-citation list nested inside "Nguồn gốc & Chú giải" — its own
 * "Thu gọn/Mở rộng" toggle since the list can get long (many NCI sources)
 * and shouldn't force the whole annotation card to scroll. */
function ReferencesList({ referenceList, disclaimer }: { referenceList: Reference[]; disclaimer?: string }) {
  return (
    <div className="mt-5 pt-4 border-t border-neutral-100">
      <div className="flex items-center justify-between mb-2">
        <p className="text-xs uppercase tracking-wider text-neutral-800 font-bold">Nguồn tham khảo</p>
      </div>
      <div className="animate-in fade-in slide-in-from-top-2 duration-200">
        <ul className="space-y-1.5">
          {referenceList.map((ref) => (
            <li key={ref.url} className="text-xs text-neutral-600 flex items-center gap-2">
              <Link1AngularRight size={11} className="shrink-0 text-neutral-600" />
              <a href={ref.url} target="_blank" rel="noreferrer" className="hover:text-brand-600 underline">
                {ref.title}&nbsp;&nbsp;<span className="text-neutral-400">·</span>&nbsp;&nbsp;
                <span>{ref.publisher}</span>
              </a>
            </li>
          ))}
        </ul>
        {disclaimer && <p className="text-xs text-neutral-600 italic mt-3">DISCLAIMER: {disclaimer}</p>}
      </div>
    </div>
  );
}

export default function DatasetOverview({
  datasetId,
  collapseSignal,
  isTemp,
}: {
  datasetId: string;
  collapseSignal?: boolean;
  isTemp?: boolean;
}) {
  const [isCollapsed, setIsCollapsed] = useState(false);
  const [overview, setOverview] = useState<OverviewData | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    setIsCollapsed(!!collapseSignal);
  }, [collapseSignal]);

  useEffect(() => {
    // Session-only uploaded datasets never get a Dataset_Description entry or
    // an outputs_holdout/{id}/split_info.json (no offline holdout run ever
    // touches them) — /overview 404s unconditionally for them, so skip the
    // fetch entirely instead of surfacing that as a raw backend error.
    if (!datasetId || isTemp) {
      setOverview(null);
      setError("");
      return;
    }
    let ignore = false;
    setOverview(null);
    setError("");
    api
      .getOverview(datasetId)
      .then((o) => {
        if (!ignore) setOverview(o);
      })
      .catch((e) => {
        if (!ignore) setError(e.message);
      });
    return () => {
      ignore = true;
    };
  }, [datasetId, isTemp]);

  if (!datasetId) return null;

  if (isTemp) {
    return (
      <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
        <div className="bg-white p-6 rounded-2xl shadow-xs border border-neutral-200 transition-all duration-300">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-success-50 flex items-center justify-center text-success-600 border border-success-100">
              <Database size={20} />
            </div>
            <h2 className="text-lg font-semibold text-neutral-800 group-hover:text-success-700 transition-colors">
              Tổng quan dữ liệu
            </h2>
          </div>
          <Panel surface="warning" padding="md" className="my-5 flex gap-2.5">
            <AlertTriangle size={16} className="text-warning-800 shrink-0 mt-0.5" />
            <p className="text-sm text-warning-800">
              Dataset này được tải lên trong phiên làm việc hiện tại và chưa qua kiểm duyệt hoặc tìm kiếm thông tin chi
              tiết (nghiên cứu gốc, chú giải sinh học, hình ảnh minh họa), nên tạm thời chưa có thông tin tổng quan để
              hiển thị.
            </p>
          </Panel>
        </div>
      </div>
    );
  }

  const origin: OriginInfo = overview?.origin || {};
  const content: ContentInfo = overview?.content || {};
  const diseaseContext = content.disease_context;
  const primaryEntity = diseaseContext?.primary_entities?.find(
    (e) => e.entity_id === diseaseContext.primary_display_entity_id,
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
  primaryEntity?.source_ids?.forEach((id) => usedReferenceIds.add(id));
  Object.values(classesContent).forEach((c) => {
    c.source_ids?.forEach((id) => usedReferenceIds.add(id));
  });
  datasetStudy?.source_ids?.forEach((id) => usedReferenceIds.add(id));
  datasetProvenance?.download_provider?.source_ids?.forEach((id) => usedReferenceIds.add(id));
  const referenceList = [...usedReferenceIds].map((id) => references[id]).filter(Boolean);

  return (
    <div className="space-y-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div className="bg-white p-6 rounded-2xl shadow-xs border border-neutral-200 transition-all duration-300">
        <div
          className="flex items-center justify-between cursor-pointer group"
          onClick={() => setIsCollapsed(!isCollapsed)}
        >
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-success-50 flex items-center justify-center text-success-600 border border-success-100">
              <Database size={20} />
            </div>
            <h2 className="text-lg font-semibold text-neutral-800 group-hover:text-success-700 transition-colors">
              Tổng quan dữ liệu
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
                {/* Summary — title comes first (it's the dataset's actual identity),
                    then original-scale stats below as their own row. Platform is
                    dropped entirely — it's already shown in Bước 1's own dataset-info
                    tiles, right next to this panel. GEO accession lives as a quiet
                    identity chip alongside Organism, under the title, instead of
                    living on the card header (tried, read as a stray tag) or as its
                    own subtitle line (tried, still competed with the title for
                    attention). Rare-class-drop stats, train/test split,
                    class-distribution charts and per-class biology now live in
                    Bước 2 "Xử lý & Chia Dữ liệu" (DatasetSplit). */}
                {/* dataset_study.original_study_title_en (cross-checked against ArrayExpress+GEO)
                    is populated for every dataset, CuMiDa included — origin.title only comes
                    from the annotation report's "Title" line, which CuMiDa reports don't have.
                    Largest text always shows something, falling back vi -> en -> a plain
                    "unavailable" label instead of ever rendering blank; the EN line only
                    shows as a subtitle when VI got to be the headline and EN also exists —
                    otherwise EN itself is already the headline and repeating it would be
                    redundant. */}
                {(() => {
                  const enTitle = datasetStudy?.original_study_title_en || origin.title;
                  const headline = content.title_vi || enTitle;
                  return (
                    <div>
                      <h2 className="text-xl text-pretty font-bold text-neutral-800 leading-snug">
                        {headline || "Tiêu đề không khả dụng"}
                      </h2>
                      {content.title_vi && enTitle && (
                        <p className="text-neutral-600 text-base leading-relaxed mt-1">– {enTitle}</p>
                      )}
                    </div>
                  );
                })()}

                <div className="grid grid-cols-2 gap-3">
                  {origin.geo_accession && <StatTile mono label="GEO" value={origin.geo_accession} />}
                  {content.organism && <StatTile mono label="Organism" value={content.organism} />}
                  {origin.samples_kept != null && (
                    <StatTile mono label="Tổng số mẫu ban đầu" value={origin.samples_kept} />
                  )}
                  {overview.raw_class_counts && (
                    <StatTile mono label="Tổng số lớp ban đầu" value={Object.keys(overview.raw_class_counts).length} />
                  )}
                </div>

                {/* Dataset description — concise summary */}
                {content.description_vi && (
                  <div>
                    <FieldLabel className="text-sm font-bold text-neutral-800">Mô tả tập dữ liệu</FieldLabel>
                    <p className="text-base text-neutral-700 leading-relaxed mt-1.5">{content.description_vi}</p>
                  </div>
                )}

                {/* Disease context: what this disease/family actually is, in plain Vietnamese */}
                {diseaseContext &&
                  (primaryEntity || diseaseContext.dataset_relationship_vi) &&
                  (() => {
                    const assetId = (primaryEntity?.media_asset_ids || diseaseContext.media_asset_ids || [])[0];
                    const asset = assetId ? overview.media_assets?.[assetId] : undefined;
                    return (
                      <CollapsibleCard title={diseaseContext.ui_title_vi || "Bối cảnh bệnh lý"}>
                        <div className="space-y-5">
                          {asset?.url && (
                            <figure>
                              <img
                                src={asset.url}
                                alt={asset.alt_text_vi || asset.title_vi || ""}
                                className="w-full max-w-xs h-auto rounded-lg border border-neutral-200 shadow-xs object-cover"
                              />
                              <figcaption className="mt-2 text-sm text-neutral-500 leading-snug">
                                {asset.caption_vi}
                                {asset.credit && <span className="block mt-0.5">Ảnh: {asset.credit}</span>}
                                {asset.source_page_url && (
                                  <a
                                    href={asset.source_page_url}
                                    target="_blank"
                                    rel="noreferrer"
                                    className="text-brand-700 underline block mt-0.5"
                                  >
                                    Nguồn: {asset.source_publisher || "NCI"}
                                  </a>
                                )}
                              </figcaption>
                            </figure>
                          )}
                          <div className="space-y-3">
                            {primaryEntity && (
                              <p className="text-base text-neutral-700 leading-relaxed">
                                {primaryEntity.description_vi}
                              </p>
                            )}
                            {diseaseContext.dataset_relationship_vi && (
                              <p className="text-base text-neutral-700 leading-relaxed">
                                {diseaseContext.dataset_relationship_vi}
                              </p>
                            )}
                            {asset && diseaseContext.media_disclaimer_vi && (
                              <p className="text-xs text-neutral-500 leading-relaxed mt-4 pt-3 border-t border-neutral-100 italic">
                                DISCLAIMER: {diseaseContext.media_disclaimer_vi}
                              </p>
                            )}
                          </div>
                        </div>
                      </CollapsibleCard>
                    );
                  })()}

                {/* Original study — cross-checked against ArrayExpress + NCBI GEO, separate from
                    where the data FILE was downloaded (dataset_provenance, below) */}
                {datasetStudy && (
                  <CollapsibleCard title="Nghiên cứu gốc (đối chiếu ArrayExpress & NCBI GEO)">
                    <div className="space-y-5">
                      {datasetStudy.original_study_title_en && (
                        <div>
                          <FieldLabel className="text-sm font-bold tracking-wider text-neutral-800">
                            Tiêu đề nghiên cứu gốc
                          </FieldLabel>
                          <p className="text-neutral-700 leading-relaxed">
                            {datasetStudy.original_study_title_en}
                            {datasetStudy.arrayexpress_accession && (
                              <span className="text-sm font-mono ml-2">({datasetStudy.arrayexpress_accession})</span>
                            )}
                          </p>
                        </div>
                      )}
                      {datasetStudy.study_objective_vi && (
                        <div>
                          <FieldLabel className="text-sm font-bold tracking-wider text-neutral-800">
                            Mục tiêu nghiên cứu
                          </FieldLabel>
                          <p className="text-neutral-700 leading-relaxed">{datasetStudy.study_objective_vi}</p>
                        </div>
                      )}
                      {datasetStudy.original_design_vi && (
                        <div>
                          <FieldLabel className="text-sm font-bold tracking-wider text-neutral-800">
                            Thiết kế nghiên cứu gốc
                          </FieldLabel>
                          <p className="text-neutral-700 leading-relaxed">{datasetStudy.original_design_vi}</p>
                        </div>
                      )}
                    </div>
                    {datasetStudy.local_ml_relation_vi && (
                      <div className="mt-5">
                        <Panel surface="info" padding="md" className="flex gap-2.5">
                          <InfoCircle size={16} className="text-info-700 shrink-0 mt-0.5" />
                          <div>
                            <p className="text-info-700 text-xs uppercase tracking-wider font-bold mb-1">
                              Liên hệ với bài toán học máy ở đây
                            </p>
                            <span className="text-info-700 leading-relaxed text-sm">
                              {datasetStudy.local_ml_relation_vi}
                            </span>
                          </div>
                        </Panel>
                      </div>
                    )}
                    {datasetProvenance?.download_provider?.name && (
                      <p className="text-xs text-neutral-600 mt-4 pt-3 border-t border-neutral-100">
                        Nguồn tải file dữ liệu:{" "}
                        <span className="font-bold text-neutral-800">{datasetProvenance.download_provider.name}</span>
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
                  <CollapsibleCard title="Đặc tính sinh học các phân lớp">
                    {content.dataset_note_vi && (
                      <Panel surface="warning" padding="md" className="mb-5 flex gap-2.5">
                        <AlertTriangle size={16} className="text-warning-800 shrink-0 mt-0.5" />
                        <div>
                          <p className="text-warning-800 text-xs uppercase tracking-wider font-bold mb-1">Lưu ý</p>
                          <span className="text-warning-800 leading-relaxed text-sm">{content.dataset_note_vi}</span>
                        </div>
                      </Panel>
                    )}
                    {(() => {
                      const entries = Object.entries(classesContent);
                      const isExcluded = (label: string, cls: ClassContent) =>
                        cls.excluded_from_model || droppedSet.has(label);
                      const active = entries.filter(([label, cls]) => !isExcluded(label, cls));
                      const excluded = entries.filter(([label, cls]) => isExcluded(label, cls));
                      const classCard = (label: string, cls: ClassContent, color: string) => (
                        <div
                          key={label}
                          className="bg-white p-4 rounded-lg border border-neutral-200 shadow-xs border-l-3 flex flex-col items-start text-left h-full"
                          style={{ borderLeftColor: color }}
                        >
                          <h4 className="font-bold text-neutral-800 leading-snug text-left text-pretty mb-1">
                            {cls.display_name_vi || label}
                          </h4>
                          {cls.display_name_vi && (
                            <FieldLabel className="text-xs font-bold tracking-wider text-neutral-500/90 break-all">
                              {label}
                            </FieldLabel>
                          )}
                          <p className="text-sm text-neutral-700 leading-relaxed text-left mt-2">
                            {cls.description_vi}
                          </p>
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
                            <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-4">
                              {excluded.map(([label, cls]) => (
                                <div
                                  key={label}
                                  className="bg-white p-4 rounded-lg border border-neutral-200 shadow-xs border-l-3 flex flex-col items-start text-left"
                                  style={{ borderLeftColor: DROPPED_COLOR }}
                                >
                                  <h4 className="font-bold text-neutral-800 leading-snug text-left text-pretty mb-1">
                                    {cls.display_name_vi || label}
                                  </h4>
                                  {cls.display_name_vi && (
                                    <FieldLabel className="text-xs font-bold tracking-wider text-neutral-500/90 break-all">
                                      {label}
                                    </FieldLabel>
                                  )}
                                  <p className="text-sm text-neutral-700 leading-relaxed text-left mt-2 mb-2.5">
                                    {cls.description_vi}
                                  </p>
                                  <Panel surface="warning" padding="xs" className="flex gap-2.5">
                                    <AlertTriangle size={13} className="shrink-0 mt-0.5 text-warning-700" />
                                    <p className="text-xs leading-relaxed text-left text-warning-800">
                                      Bị loại khỏi tập huấn luyện do quá ít mẫu.
                                    </p>
                                  </Panel>
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
                  <CollapsibleCard accent="neutral" title="Nguồn gốc & Chú giải">
                    <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
                      {origin.class_characteristic && (
                        <StatTile label="Class characteristic" value={origin.class_characteristic} mono />
                      )}
                      <StatTile label="Annotation rows" value={origin.annotation_rows?.toLocaleString()} mono />
                      <StatTile
                        label="Probes có gene symbol"
                        mono
                        valueClassName="text-success-600"
                        value={
                          <>
                            {origin.probes_with_gene_symbol?.toLocaleString()}{" "}
                            <span className="text-xs font-normal text-neutral-600">
                              (từ annotation file: {origin.probes_from_annotation_file?.toLocaleString()} · MyGene.info:{" "}
                              {origin.probes_from_mygene?.toLocaleString()})
                            </span>
                          </>
                        }
                      />
                      <StatTile
                        label="Probes không có gene symbol"
                        mono
                        valueClassName="text-danger-500"
                        value={origin.probes_without_gene_symbol?.toLocaleString()}
                      />
                      <StatTile
                        label="AFFX control đã loại"
                        value={origin.affx_control_removed?.toLocaleString()}
                        mono
                      />
                      <StatTile
                        label="Probes / Genes duy nhất"
                        mono
                        value={`${origin.unique_probes?.toLocaleString()} / ${origin.unique_genes?.toLocaleString()}`}
                      />
                      <StatTile
                        label="Mẫu giữ lại"
                        mono
                        value={
                          <>
                            {origin.samples_kept?.toLocaleString()}{" "}
                            <span className="text-xs font-normal text-neutral-600">
                              (NA dropped: {origin.na_dropped})
                            </span>
                          </>
                        }
                      />
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
