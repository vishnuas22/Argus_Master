import { useState } from "react";
import { CaretDown, CaretUp, ImageSquare } from "@phosphor-icons/react";

const DIR = {
  authentic: "text-emerald-500 bg-emerald-500/10 border-emerald-500/20",
  synthetic: "text-red-500 bg-red-500/10 border-red-500/20",
  manipulated: "text-red-400 bg-red-500/10 border-red-500/20",
  neutral: "text-zinc-400 bg-zinc-500/10 border-zinc-500/20",
};

const Bar = ({ label, value, testId }) => (
  <div data-testid={testId}>
    <div className="flex justify-between text-[10px] font-mono uppercase tracking-[0.15em] text-zinc-500 mb-1">
      <span>{label}</span>
      <span className="text-zinc-300">{value.toFixed(2)}</span>
    </div>
    <div className="h-1 w-full bg-zinc-800">
      <div className="h-full bg-zinc-400" style={{ width: `${value * 100}%` }} />
    </div>
  </div>
);

const EvidenceBar = ({ value }) => (
  <div data-testid="evidence-score-bar">
    <div className="flex justify-between text-[10px] font-mono uppercase tracking-[0.15em] text-zinc-500 mb-1">
      <span>Evidence [-1 syn · +1 auth]</span>
      <span className="text-zinc-300">{value >= 0 ? "+" : ""}{value.toFixed(2)}</span>
    </div>
    <div className="relative h-1 w-full bg-zinc-800">
      <div className="absolute left-1/2 top-[-2px] h-2 w-px bg-zinc-600" />
      <div
        className={`absolute h-full ${value >= 0 ? "bg-emerald-500" : "bg-red-500"}`}
        style={
          value >= 0
            ? { left: "50%", width: `${value * 50}%` }
            : { right: "50%", width: `${-value * 50}%` }
        }
      />
    </div>
  </div>
);

export const EvidenceCard = ({ entry, onViewArtifact }) => {
  const [open, setOpen] = useState(false);
  return (
    <div data-testid={`evidence-card-${entry.module}`} className="border border-zinc-800 bg-zinc-900/40 fade-up">
      <button
        data-testid={`evidence-card-toggle-${entry.module}`}
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between gap-3 p-4 text-left hover:bg-zinc-900/80 transition-colors duration-200"
      >
        <div className="flex items-center gap-3 min-w-0">
          <span className="text-xs font-mono text-zinc-600">#{entry.rank}</span>
          <span className="text-sm font-medium text-zinc-100 truncate">{entry.module}</span>
          <span className={`px-2 py-0.5 text-[10px] font-mono uppercase tracking-[0.1em] border ${DIR[entry.direction]}`}>
            {entry.direction}
          </span>
        </div>
        <div className="flex items-center gap-4 shrink-0">
          <span className="hidden sm:inline text-[11px] font-mono text-zinc-500" title="heuristic likelihood ratio (v0)">
            LR≈{entry.likelihood_ratio}
          </span>
          {open ? <CaretUp size={14} className="text-zinc-500" /> : <CaretDown size={14} className="text-zinc-500" />}
        </div>
      </button>
      <div className="px-4 pb-3 grid grid-cols-3 gap-4">
        <EvidenceBar value={entry.evidence_score} />
        <Bar label="Reliability" value={entry.reliability} testId="reliability-bar" />
        <Bar label="Confidence" value={entry.confidence} testId="confidence-bar" />
      </div>
      {open && (
        <div data-testid={`evidence-detail-${entry.module}`} className="border-t border-zinc-800/60 p-4 space-y-3">
          <div className="text-[11px] font-mono text-zinc-500">
            fusion contribution: <span className="text-zinc-300">{entry.shap_contribution >= 0 ? "+" : ""}{entry.shap_contribution.toFixed(4)}</span>
          </div>
          {entry.artifacts.map((a, i) => (
            <div key={i} className="border border-zinc-800/80 bg-zinc-950 p-3">
              <div className="flex items-center justify-between gap-3 mb-2">
                <span className="text-xs font-mono text-zinc-300">{a.type} · strength {a.strength.toFixed(2)}</span>
                {a.visual && (
                  <button
                    data-testid={`view-artifact-${entry.module}-${i}`}
                    onClick={() => onViewArtifact({ ...a, module: entry.module })}
                    className="inline-flex items-center gap-1.5 px-2 py-1 text-[10px] font-mono border border-zinc-700 text-zinc-300 hover:border-emerald-500/50 hover:text-emerald-400 transition-colors duration-200"
                  >
                    <ImageSquare size={12} weight="light" /> VIEW OVERLAY
                  </button>
                )}
              </div>
              <p className="text-xs text-zinc-500 mb-1">{a.description}</p>
              <p data-testid={`checkable-claim-${entry.module}-${i}`} className="text-[11px] font-mono text-zinc-400 leading-relaxed border-l-2 border-zinc-700 pl-2">
                {a.checkable_claim}
              </p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
