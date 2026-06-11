import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import api from "../lib/api";

const verdictLabel = (row) => {
  if (row.verdict.abstained) return ["ABSTAIN", "text-amber-500"];
  const top = row.verdict.hypothesis_set[0];
  if (top === "camera_original") return ["AUTHENTIC", "text-emerald-500"];
  if (top === "manipulated") return ["MANIPULATED", "text-red-400"];
  return ["AI GENERATED", "text-red-500"];
};

export default function History() {
  const [rows, setRows] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    api
      .get("/verdicts")
      .then(({ data }) => setRows(data))
      .catch((e) => setError(e.message));
  }, []);

  return (
    <div className="space-y-6" data-testid="history-page">
      <div>
        <div className="text-xs font-mono uppercase tracking-[0.2em] text-zinc-500 mb-2">Case archive</div>
        <h1 className="text-2xl sm:text-3xl font-semibold tracking-tight">Verdict history</h1>
      </div>
      {error && <div className="border border-red-500/30 p-4 text-sm font-mono text-red-400">{error}</div>}
      {rows === null && !error && <div className="text-sm font-mono text-zinc-500">loading…</div>}
      {rows && rows.length === 0 && (
        <div data-testid="history-empty" className="border border-zinc-800 p-8 text-center text-sm font-mono text-zinc-500">
          No assessments yet.
        </div>
      )}
      {rows && rows.length > 0 && (
        <div className="border border-zinc-800 overflow-x-auto">
          <table className="w-full text-left" data-testid="history-table">
            <thead>
              <tr className="border-b border-zinc-800 text-[10px] font-mono uppercase tracking-[0.2em] text-zinc-500">
                <th className="px-4 py-3">File</th>
                <th className="px-4 py-3">Verdict</th>
                <th className="px-4 py-3">Auth</th>
                <th className="px-4 py-3">Trust</th>
                <th className="px-4 py-3">Risk</th>
                <th className="px-4 py-3">Capacity</th>
                <th className="px-4 py-3">When</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => {
                const [label, cls] = verdictLabel(row);
                return (
                  <tr key={row.verdict_id} className="border-b border-zinc-800/50 hover:bg-zinc-900/60 transition-colors duration-200">
                    <td className="px-4 py-3">
                      <Link
                        to={`/verdict/${row.verdict_id}`}
                        data-testid={`history-row-${row.verdict_id}`}
                        className="text-sm font-mono text-zinc-200 hover:text-emerald-400 transition-colors duration-200"
                      >
                        {row.filename}
                      </Link>
                      <div className="text-[10px] font-mono text-zinc-600">
                        {row.input.format.toUpperCase()} · {row.input.dimensions.join("×")}
                      </div>
                    </td>
                    <td className={`px-4 py-3 text-xs font-mono ${cls}`}>{label}</td>
                    <td className="px-4 py-3 text-xs font-mono text-zinc-300">{row.verdict.authenticity_score.toFixed(2)}</td>
                    <td className="px-4 py-3 text-xs font-mono text-zinc-300">{row.verdict.trust_score.toFixed(2)}</td>
                    <td className="px-4 py-3 text-xs font-mono text-zinc-300">{row.verdict.risk_score.toFixed(2)}</td>
                    <td className="px-4 py-3 text-xs font-mono text-zinc-400">{row.input.degradation_state.evidence_capacity}</td>
                    <td className="px-4 py-3 text-[11px] font-mono text-zinc-500">
                      {row.created_at ? new Date(row.created_at).toLocaleString() : "—"}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
