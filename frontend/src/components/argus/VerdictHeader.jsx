import { DownloadSimple, ShieldWarning } from "@phosphor-icons/react";

const LABEL = {
  camera_original: { text: "CAMERA ORIGINAL", cls: "text-emerald-500" },
  ai_generated: { text: "AI GENERATED", cls: "text-red-500" },
  manipulated: { text: "MANIPULATED", cls: "text-red-400" },
};

export const VerdictHeader = ({ verdict }) => {
  const v = verdict.verdict;
  const top = Object.entries(v.probabilities).sort((a, b) => b[1] - a[1])[0];
  const label = v.abstained ? { text: "ABSTAIN", cls: "text-amber-500" } : LABEL[top[0]];

  const download = () => {
    const blob = new Blob([JSON.stringify(verdict, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = `${verdict.verdict_id}.json`;
    a.click();
    URL.revokeObjectURL(a.href);
  };

  return (
    <div data-testid="verdict-header" className="border border-zinc-800 bg-zinc-900/50 p-4 md:p-6 fade-up">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="min-w-0">
          <div className="text-xs font-mono uppercase tracking-[0.2em] text-zinc-500 mb-1">Assessment</div>
          <h1 data-testid="verdict-label" className={`text-2xl sm:text-3xl font-semibold tracking-tight ${label.cls}`}>
            {label.text}
            {!v.abstained && <span className="ml-3 text-lg font-mono text-zinc-400">{(top[1] * 100).toFixed(0)}%</span>}
          </h1>
          <p data-testid="verdict-summary" className="mt-2 text-sm text-zinc-400 leading-relaxed max-w-3xl">
            {verdict.explanation.summary}
          </p>
        </div>
        <div className="flex flex-col items-end gap-2 shrink-0">
          <div
            data-testid="conformal-badge"
            title={`Conformal prediction set at alpha=${v.conformal.alpha} (stratum: ${v.conformal.calibration_stratum})`}
            className="inline-flex items-center gap-1.5 px-2.5 py-1 text-xs font-mono border border-zinc-700 bg-zinc-800 text-zinc-300"
          >
            {Math.round((1 - v.conformal.alpha) * 100)}% SET · {"{"}
            {v.conformal.set.join(", ")}
            {"}"}
          </div>
          {v.abstained && (
            <div
              data-testid="abstention-badge"
              className="inline-flex items-center gap-1.5 px-2.5 py-1 text-xs font-mono border border-amber-500/40 bg-amber-500/10 text-amber-500"
            >
              <ShieldWarning size={14} weight="light" /> ABSTAINED — set not singleton
            </div>
          )}
          <button
            data-testid="download-verdict-json"
            onClick={download}
            className="inline-flex items-center gap-2 px-3 py-1.5 text-xs font-mono border border-zinc-700 text-zinc-300 hover:border-zinc-500 hover:text-zinc-50 transition-colors duration-200"
          >
            <DownloadSimple size={14} weight="light" /> VERDICT JSON
          </button>
        </div>
      </div>
      <div className="mt-4 pt-4 border-t border-zinc-800/60 grid grid-cols-3 gap-3 max-w-xl">
        {Object.entries(v.probabilities).map(([h, p]) => (
          <div key={h} data-testid={`prob-${h}`}>
            <div className="text-[10px] font-mono uppercase tracking-[0.15em] text-zinc-500">{h.replace("_", " ")}</div>
            <div className="text-sm font-mono text-zinc-200">{(p * 100).toFixed(1)}%</div>
            <div className="h-1 w-full bg-zinc-800 mt-1">
              <div className="h-full bg-zinc-400" style={{ width: `${p * 100}%`, transition: "width 600ms ease" }} />
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
