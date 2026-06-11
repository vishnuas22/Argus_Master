import { useEffect, useState } from "react";
import { Check } from "@phosphor-icons/react";

const TIERS = [
  { label: "TIER 0 · INTAKE", desc: "decode · sha-256 · container parse" },
  { label: "TIER 1 · TRIAGE", desc: "degradation state d — quality, ghosts, resampling" },
  { label: "TIER 2 · EVIDENCE PANEL", desc: "metadata · compression · spectral · DINOv2 probes" },
  { label: "TIER 3 · FUSION COURT", desc: "reliability gate · weighted vote · conformal set" },
];

// /api/assess is a single synchronous call: stages advance on a client-side
// timeline (see DECISIONS.md), completing when the response lands.
export const TierProgress = ({ done }) => {
  const [stage, setStage] = useState(0);

  useEffect(() => {
    const timers = [600, 1800, 4500].map((ms, i) => setTimeout(() => setStage(i + 1), ms));
    return () => timers.forEach(clearTimeout);
  }, []);

  return (
    <div data-testid="tier-progress" className="border border-zinc-800 bg-zinc-900/40 p-6 space-y-4 fade-up">
      {TIERS.map((t, i) => {
        const complete = done || i < stage;
        const active = !done && i === stage;
        return (
          <div key={t.label} data-testid={`tier-step-${i}`} className="flex items-center gap-4">
            <div
              className={`w-5 h-5 flex items-center justify-center border text-[10px] font-mono ${
                complete
                  ? "border-emerald-500/50 text-emerald-500"
                  : active
                  ? "border-zinc-400 text-zinc-300 pulse-line"
                  : "border-zinc-800 text-zinc-700"
              }`}
            >
              {complete ? <Check size={12} weight="bold" /> : i}
            </div>
            <div className="flex-1 min-w-0">
              <div
                className={`text-xs font-mono tracking-[0.15em] ${
                  complete ? "text-zinc-300" : active ? "text-zinc-100 pulse-line" : "text-zinc-600"
                }`}
              >
                {t.label}
              </div>
              <div className="text-[11px] text-zinc-600 truncate">{t.desc}</div>
            </div>
          </div>
        );
      })}
    </div>
  );
};
