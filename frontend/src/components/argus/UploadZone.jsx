import { useRef, useState } from "react";
import { UploadSimple } from "@phosphor-icons/react";

export const UploadZone = ({ onFile, disabled }) => {
  const inputRef = useRef(null);
  const [dragOver, setDragOver] = useState(false);

  const handleFiles = (files) => {
    if (files && files[0]) onFile(files[0]);
  };

  return (
    <div
      data-testid="upload-zone"
      role="button"
      tabIndex={0}
      onClick={() => !disabled && inputRef.current?.click()}
      onKeyDown={(e) => e.key === "Enter" && !disabled && inputRef.current?.click()}
      onDragOver={(e) => {
        e.preventDefault();
        setDragOver(true);
      }}
      onDragLeave={() => setDragOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setDragOver(false);
        if (!disabled) handleFiles(e.dataTransfer.files);
      }}
      className={`border-2 border-dashed p-12 flex flex-col items-center justify-center transition-colors duration-200 cursor-pointer bg-zinc-950 ${
        dragOver ? "border-emerald-500/60" : "border-zinc-800 hover:border-zinc-600"
      } ${disabled ? "opacity-50 pointer-events-none" : ""}`}
    >
      <UploadSimple size={28} weight="light" className="text-zinc-500 mb-4" />
      <p className="text-sm font-mono text-zinc-400">DROP IMAGE OR CLICK TO SELECT</p>
      <p className="text-xs font-mono text-zinc-600 mt-2">JPEG · PNG · WEBP — max 25MB · never stored beyond processing</p>
      <input
        ref={inputRef}
        data-testid="upload-input"
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(e) => handleFiles(e.target.files)}
      />
    </div>
  );
};
