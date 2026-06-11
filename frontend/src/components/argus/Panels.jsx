import { Warning, Prohibit } from "@phosphor-icons/react";

export const DegradationStrip = ({ verdict }) => {
  const d = verdict.input.degradation_state;
  const items = [
    ["format", verdict.input.format.toUpperCase()],
    ["dimensions", verdict.input.dimensions.join("×")],
    ["jpeg quality", d.jpeg_quality_est ?? "n/a"],
    ["recompressions", d.recompression_generations],
    ["resize factor", d.resize_factor_est.toFixed(2)],
    ["screenshot p", d.screenshot_probability.toFixed(2)],
    ["eff. resolution", `${d.effective_resolution}px`],
  ];
  const capCls =
    d.evidence_capacity === "HIGH"
      ? "text-emerald-500 border-emerald-500/30"
      : d.evidence_capacity === "MODERATE"
      ? "text-amber-500 border-amber-500/30"
      : "text-red-500 border-red-500/30";
  return (
    <div data-testid="degradation-strip" className="border border-zinc-800 bg-zinc-900/40 px-4 py-3 flex flex-wrap items-center gap-x-6 gap-y-2">
      <span className="text-[10px] font-mono uppercase tracking-[0.2em] text-zinc-500">Tier 1 · Degradation state d</span>
      {items.map(([k, v]) => (
        <span key={k} className="text-[11px] font-mono text-zinc-400">
          {k}: <span className="text-zinc-200">{v}</span>
        </span>
      ))}
      <span data-testid="evidence-capacity" className={`px-2 py-0.5 text-[10px] font-mono border ${capCls}`}>
        CAPACITY {d.evidence_capacity}
      </span>
    </div>
  );
};

export const ContradictionsPanel = ({ contradictions }) => (
  <div data-testid="contradictions-section" className="border border-amber-500/30 bg-amber-500/5 p-4">
    <div className="text-sm font-mono text-amber-500 flex items-center gap-2 mb-3">
      <Warning size={16} weight="light" /> CONTRADICTIONS [{contradictions.length}]
    </div>
    {contradictions.length === 0 ? (
      <p data-testid="contradictions-empty" className="text-xs text-zinc-500 font-mono">
        No reliable evidence streams disagree.
      </p>
    ) : (
      <div className="space-y-3">
        {contradictions.map((c, i) => (
          <div key={i} data-testid={`contradiction-${i}`} className="border-l-2 border-amber-500/40 pl-3">
            <div className="text-[11px] font-mono text-zinc-300">{c.modules.join(" ⇄ ")} · conflict {c.conflict_contribution.toFixed(3)}</div>
            <p className="text-xs text-zinc-400 mt-1">{c.description}</p>
            <p className="text-xs text-amber-200/70 mt-1 italic">{c.interpretation}</p>
          </div>
        ))}
      </div>
    )}
  </div>
);

export const UnavailablePanel = ({ unavailable }) => (
  <div data-testid="unavailable-section" className="border border-zinc-800 bg-zinc-900/30 p-4">
    <div className="text-sm font-mono text-zinc-500 flex items-center gap-2 mb-3">
      <Prohibit size={16} weight="light" /> UNAVAILABLE EVIDENCE [{unavailable.length}]
    </div>
    {unavailable.length === 0 ? (
      <p data-testid="unavailable-empty" className="text-xs text-zinc-500 font-mono">
        All registered modules produced usable evidence.
      </p>
    ) : (
      <div className="space-y-2">
        {unavailable.map((u, i) => (
          <div key={i} data-testid={`unavailable-${u.module}`} className="text-xs font-mono">
            <span className="text-zinc-300">{u.module}</span>
            <span className="text-zinc-500"> — {u.reason}</span>
          </div>
        ))}
      </div>
    )}
  </div>
);

export const ExplanationPanel = ({ verdict }) => (
  <div data-testid="explanation-panel" className="border border-zinc-800 bg-zinc-900/30 p-4">
    <div className="text-sm font-mono text-zinc-400 mb-2">ANALYST EXPLANATION</div>
    <p className="text-xs text-zinc-400 leading-relaxed">{verdict.explanation.detail}</p>
    <div className="mt-3 pt-3 border-t border-zinc-800/60 flex flex-wrap gap-x-4 gap-y-1 text-[10px] font-mono text-zinc-600">
      <span>pipeline {verdict.schema_version}</span>
      <span>fusion {verdict.meta.fusion_model}</span>
      <span>curves {verdict.meta.reliability_curves}</span>
      <span>{verdict.meta.total_compute_ms}ms</span>
      <span className="break-all">sha256 {verdict.input.sha256.slice(0, 16)}…</span>
    </div>
  </div>
);
