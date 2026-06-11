import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { ArrowLeft } from "@phosphor-icons/react";
import api from "../lib/api";
import { VerdictView } from "../components/argus/VerdictView";

export default function VerdictPage() {
  const { id } = useParams();
  const [verdict, setVerdict] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    api
      .get(`/verdicts/${id}`)
      .then(({ data }) => setVerdict(data))
      .catch((e) => setError(e.response?.status === 404 ? "verdict not found" : e.message));
  }, [id]);

  return (
    <div className="space-y-4" data-testid="verdict-page">
      <Link
        to="/history"
        data-testid="back-to-history"
        className="inline-flex items-center gap-2 text-xs font-mono text-zinc-500 hover:text-zinc-200 transition-colors duration-200"
      >
        <ArrowLeft size={14} /> HISTORY
      </Link>
      {error && <div className="border border-red-500/30 p-4 text-sm font-mono text-red-400">{error}</div>}
      {!verdict && !error && <div className="text-sm font-mono text-zinc-500">loading…</div>}
      {verdict && <VerdictView verdict={verdict} originalUrl={null} />}
    </div>
  );
}
