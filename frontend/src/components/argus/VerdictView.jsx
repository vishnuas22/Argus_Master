import { useState } from "react";
import { VerdictHeader } from "./VerdictHeader";
import { ScoreDials } from "./ScoreDials";
import { EvidenceCard } from "./EvidenceCard";
import { ArtifactViewer } from "./ArtifactViewer";
import { ContradictionsPanel, DegradationStrip, ExplanationPanel, UnavailablePanel } from "./Panels";

export const VerdictView = ({ verdict, originalUrl }) => {
  const [viewer, setViewer] = useState(null);

  return (
    <div className="space-y-4 md:space-y-6" data-testid="verdict-dashboard">
      <VerdictHeader verdict={verdict} />
      <ScoreDials verdict={verdict} />
      <DegradationStrip verdict={verdict} />
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 md:gap-6">
        <div className="lg:col-span-2 space-y-4">
          <div className="text-xs font-mono uppercase tracking-[0.2em] text-zinc-500">
            Evidence ranking [{verdict.evidence_ranking.length}]
          </div>
          {verdict.evidence_ranking.length === 0 && (
            <p data-testid="evidence-empty" className="text-xs font-mono text-zinc-500 border border-zinc-800 p-4">
              No evidence stream survived the reliability gate.
            </p>
          )}
          {verdict.evidence_ranking.map((entry) => (
            <EvidenceCard key={entry.module} entry={entry} onViewArtifact={setViewer} />
          ))}
        </div>
        <div className="space-y-4">
          <ContradictionsPanel contradictions={verdict.contradictions} />
          <UnavailablePanel unavailable={verdict.unavailable_evidence} />
          <ExplanationPanel verdict={verdict} />
        </div>
      </div>
      {viewer && <ArtifactViewer artifact={viewer} originalUrl={originalUrl} onClose={() => setViewer(null)} />}
    </div>
  );
};
