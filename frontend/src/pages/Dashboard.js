import { useState } from "react";
import { ArrowCounterClockwise, Play } from "@phosphor-icons/react";
import api from "../lib/api";
import { UploadZone } from "../components/argus/UploadZone";
import { TierProgress } from "../components/argus/TierProgress";
import { VerdictView } from "../components/argus/VerdictView";

export default function Dashboard() {
  const [file, setFile] = useState(null);
  const [url, setUrl] = useState(null);
  const [verdict, setVerdict] = useState(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState(null);

  const onFile = (f) => {
    setFile(f);
    setUrl(URL.createObjectURL(f));
    setVerdict(null);
    setError(null);
  };

  const reset = () => {
    setFile(null);
    setUrl(null);
    setVerdict(null);
    setError(null);
  };

  const assess = async () => {
    setRunning(true);
    setError(null);
    try {
      const fd = new FormData();
      fd.append("file", file);
      const { data } = await api.post("/assess", fd, { timeout: 180000 });
      setVerdict(data);
    } catch (e) {
      setError(e.response?.data?.detail || e.message || "assessment failed");
    } finally {
      setRunning(false);
    }
  };

  if (verdict) {
    return (
      <div className="space-y-4">
        <button
          data-testid="new-assessment-btn"
          onClick={reset}
          className="inline-flex items-center gap-2 px-3 py-1.5 text-xs font-mono border border-zinc-700 text-zinc-300 hover:border-zinc-500 hover:text-zinc-50 transition-colors duration-200"
        >
          <ArrowCounterClockwise size={14} weight="light" /> NEW ASSESSMENT
        </button>
        <VerdictView verdict={verdict} originalUrl={url} />
      </div>
    );
  }

  return (
    <div className="max-w-3xl mx-auto space-y-6 pt-8">
      <div>
        <div className="text-xs font-mono uppercase tracking-[0.2em] text-zinc-500 mb-2">Tiered evidence court</div>
        <h1 className="text-2xl sm:text-3xl font-semibold tracking-tight">Submit an image for authenticity assessment</h1>
        <p className="text-sm text-zinc-400 mt-2 leading-relaxed">
          Five forensic modules — metadata, compression history, spectral probe, DINOv2 realness and perturbation
          probes — assessed under a degradation-aware reliability gate, fused into a calibrated verdict.
        </p>
      </div>
      <UploadZone onFile={onFile} disabled={running} />
      {url && (
        <div className="border border-zinc-800 bg-zinc-900/40 p-4 flex items-center gap-4" data-testid="upload-preview">
          <img src={url} alt="preview" className="h-20 w-20 object-cover border border-zinc-800" />
          <div className="min-w-0 flex-1">
            <div className="text-sm font-mono text-zinc-200 truncate">{file.name}</div>
            <div className="text-xs font-mono text-zinc-500">{(file.size / 1024).toFixed(0)} KB</div>
          </div>
          <button
            data-testid="run-assessment-btn"
            onClick={assess}
            disabled={running}
            className="inline-flex items-center gap-2 px-4 py-2 text-xs font-mono border border-emerald-500/50 text-emerald-400 hover:bg-emerald-500/10 transition-colors duration-200 disabled:opacity-40"
          >
            <Play size={14} weight="light" /> {running ? "RUNNING…" : "RUN ASSESSMENT"}
          </button>
        </div>
      )}
      {running && <TierProgress done={false} />}
      {error && (
        <div data-testid="assess-error" className="border border-red-500/30 bg-red-500/5 p-4 text-sm font-mono text-red-400">
          {String(error)}
        </div>
      )}
    </div>
  );
}
