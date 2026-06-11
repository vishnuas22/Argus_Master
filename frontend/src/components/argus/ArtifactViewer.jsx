import { useState } from "react";
import { X } from "@phosphor-icons/react";
import { artifactUrl } from "../../lib/api";

export const ArtifactViewer = ({ artifact, originalUrl, onClose }) => {
  const [overlayOn, setOverlayOn] = useState(Boolean(originalUrl));
  const [opacity, setOpacity] = useState(0.7);
  const src = artifactUrl(artifact.visual);

  return (
    <div
      data-testid="artifact-viewer"
      className="fixed inset-0 z-50 bg-zinc-950/90 backdrop-blur-md flex items-center justify-center p-4 md:p-8"
      onClick={onClose}
    >
      <div
        className="w-full max-w-4xl border border-zinc-800 bg-zinc-950"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between border-b border-zinc-800 px-4 py-3">
          <div className="text-xs font-mono uppercase tracking-[0.15em] text-zinc-300">
            {artifact.module} · {artifact.type}
          </div>
          <button
            data-testid="artifact-viewer-close"
            onClick={onClose}
            className="text-zinc-500 hover:text-zinc-50 transition-colors duration-200"
          >
            <X size={18} />
          </button>
        </div>
        <div className="relative w-full bg-zinc-950 overflow-hidden flex items-center justify-center" style={{ minHeight: 240 }}>
          {overlayOn && originalUrl ? (
            <div className="relative">
              <img src={originalUrl} alt="original" className="w-full h-auto object-contain max-h-[65vh]" />
              <img
                src={src}
                alt={artifact.type}
                className="absolute inset-0 w-full h-full object-fill mix-blend-screen pointer-events-none transition-opacity duration-300"
                style={{ opacity }}
              />
            </div>
          ) : (
            <img src={src} alt={artifact.type} className="w-full h-auto object-contain max-h-[65vh]" />
          )}
        </div>
        <div className="border-t border-zinc-800 px-4 py-3 flex flex-wrap items-center gap-4">
          {originalUrl && (
            <label className="flex items-center gap-2 text-xs font-mono text-zinc-400 cursor-pointer">
              <input
                data-testid="artifact-overlay-toggle"
                type="checkbox"
                checked={overlayOn}
                onChange={(e) => setOverlayOn(e.target.checked)}
              />
              OVERLAY ON ORIGINAL
            </label>
          )}
          {overlayOn && originalUrl && (
            <label className="flex items-center gap-2 text-xs font-mono text-zinc-400">
              OPACITY
              <input
                data-testid="artifact-opacity-slider"
                type="range"
                min="0.1"
                max="1"
                step="0.05"
                value={opacity}
                onChange={(e) => setOpacity(parseFloat(e.target.value))}
              />
            </label>
          )}
          <p className="w-full text-[11px] font-mono text-zinc-500 leading-relaxed">{artifact.checkable_claim}</p>
        </div>
      </div>
    </div>
  );
};
