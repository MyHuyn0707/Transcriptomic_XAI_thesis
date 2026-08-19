import React from 'react';

interface GeneChipListProps {
  title: string;
  genes: string[];
  onJumpToGene: (gene: string) => void;
  className?: string;
}

/** Gene symbols as clickable chips that jump to that gene's own card in the
 * "Gene xuất hiện trong Danh sách Luật" panel above — shared by the full-match
 * and partial-match gene lists in the test-results report. */
export default function GeneChipList({ title, genes, onJumpToGene, className }: GeneChipListProps) {
  if (genes.length === 0) return null;
  return (
    <div className={className}>
      <h4 className="font-semibold tracking-wide text-xs uppercase text-brand-700 mb-4">{title}</h4>
      <div className="flex flex-wrap gap-2">
        {genes.map(gene => (
          <button
            key={gene}
            onClick={() => onJumpToGene(gene)}
            className="px-2.5 py-1 rounded-md text-xs font-mono bg-white border border-neutral-200 text-brand-700 hover:bg-brand-50 hover:border-brand-200 hover:text-brand-800 transition-colors"
            title="Xem chi tiết gene này ở danh sách Gene phía trên"
          >
            {gene}
          </button>
        ))}
      </div>
    </div>
  );
}
