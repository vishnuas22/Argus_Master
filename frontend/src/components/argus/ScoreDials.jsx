const SEMANTICS = {
    authenticity:
      "Calibrated probability that the image is a camera original (docs 7.2). 1.0 = certainly authentic, 0.0 = certainly synthetic/manipulated.",
    trust:
      "How much this verdict itself should be trusted: evidence capacity of the degraded input, panel coverage, conformal set size and unresolved conflict (docs 7.2). Low trust = treat the verdict as weak regardless of direction.",
    risk:
      "Adversarial-posture indicators: laundering patterns, metadata forgery signals, concealed compression history (docs 7.2). High risk = someone may be gaming the analysis.",
  };
  
  const dialColor = (id, v) => {
    if (id === "risk") return v >= 0.5 ? "#ef4444" : v >= 0.25 ? "#f59e0b" : "#10b981";
    return v >= 0.65 ? "#10b981" : v >= 0.35 ? "#f59e0b" : "#ef4444";
  };
  
  const Dial = ({ id, label, value }) => {
    const r = 52;
    const c = 2 * Math.PI * r;
    const pct = Math.max(0, Math.min(1, value));
    const color = dialColor(id, pct);
    return (
      <div
        data-testid={`${id}-dial`}
        className="group relative flex flex-col items-center border border-zinc-800 bg-zinc-900/40 p-6"
      >
        <div className="relative w-[132px] h-[132px]">
          <svg width="132" height="132" viewBox="0 0 132 132" className="-rotate-90">
            <circle cx="66" cy="66" r={r} fill="none" stroke="#27272a" strokeWidth="5" />
            <circle
              cx="66"
              cy="66"
              r={r}
              fill="none"
              stroke={color}
              strokeWidth="5"
              strokeDasharray={`${c * pct} ${c}`}
              style={{ transition: "stroke-dasharray 700ms ease" }}
            />
          </svg>
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <span data-testid={`${id}-dial-value`} className="text-3xl font-mono tracking-tighter" style={{ color }}>
              {value.toFixed(2)}
            </span>
          </div>
        </div>
        <span className="mt-3 text-xs font-mono uppercase tracking-[0.2em] text-zinc-500">{label}</span>
        <div className="pointer-events-none absolute left-1/2 -translate-x-1/2 bottom-full mb-2 w-72 z-30 hidden group-hover:block">
          <div className="border border-zinc-700 bg-zinc-900 p-3 text-[11px] leading-relaxed text-zinc-300 shadow-xl">
            {SEMANTICS[id]}
          </div>
        </div>
      </div>
    );
  };
  
  export const ScoreDials = ({ verdict }) => (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 md:gap-6" data-testid="score-dials">
      <Dial id="authenticity" label="Authenticity" value={verdict.verdict.authenticity_score} />
      <Dial id="trust" label="Trust" value={verdict.verdict.trust_score} />
      <Dial id="risk" label="Gaming Risk" value={verdict.verdict.risk_score} />
    </div>
  );
  