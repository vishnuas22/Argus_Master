"# 11 — Frontend (React 19 + TypeScript strict, Control Room theme)

> **Goal:** Production-grade UI that surfaces every signal from the 5-tier COEF
> pipeline, complies with `AGENTS_FRONTEND.md` (P0 rules), and ships zero
> \"AI slop\" aesthetics. Every interactive element has a `data-testid`. Every
> component has loading / error / empty states. Coverage ≥80 % via Vitest +
> RTL; E2E via Playwright; a11y via axe-core (WCAG 2.1 AA).
>
> **Source of truth for UI scope:** `Masterplan.md §14` + `00_README.md §1.5`.
> **Source of truth for rules:** `AGENTS_FRONTEND.md`.

---

## 1. Stack lock (final)

| Concern | Choice | AGENTS_FRONTEND.md alignment |
|---|---|---|
| Framework | React 19 + craco (kept — platform default) | §3 (Next.js was *preferred*, not mandatory) |
| Language | **TypeScript 5.4+ strict mode** | §3, §8 (P0) |
| Styling | Tailwind CSS + CSS variables | §3 |
| UI primitives | shadcn/ui (Radix-based, already installed) | §3 |
| Charts | Recharts | §3 |
| Icons | `@phosphor-icons/react` (Duotone / Regular) | §3 (lucide-react allowed; phosphor for Control-Room aesthetic) |
| Routing | react-router-dom v6 | §3 |
| Forms | react-hook-form + zod | §21 |
| HTTP | fetch (native) + custom retry wrapper | §12 |
| State | React Query (TanStack) for server state; React state local | §11 |
| Unit tests | **Vitest** + React Testing Library | §3 (P0) |
| E2E | **Playwright** | §3 (P0) |
| a11y | **axe-core** via `@axe-core/playwright` + `vitest-axe` | §10 (P0) |
| Lint | ESLint (strict) + `eslint-plugin-jsx-a11y` + `eslint-plugin-security` | §4 |
| Format | Prettier | §4 |

> **Why we stay on craco (not migrate to Vite):** the Emergent runtime is
> wired to CRA's dev server on port 3000. Migrating to Vite would risk the
> hot-reload integration. craco + TS works cleanly. AGENTS_FRONTEND.md §3
> lists Next.js as the *default* but allows the stack to be overridden if
> P0 rules still pass — TS strict + Vitest + Playwright + axe-core all do.

---

## 2. Migration plan (JS → TS, in-place)

Performed once at M0, before any feature work.

```bash
cd /app/frontend

# 1. Add TypeScript + types
yarn add -D typescript @types/react @types/react-dom @types/node \
  @types/react-router-dom

# 2. Add Vitest + RTL + axe + jsdom + Playwright + jsx-a11y
yarn add -D vitest @vitejs/plugin-react jsdom \
  @testing-library/react @testing-library/jest-dom @testing-library/user-event \
  vitest-axe @axe-core/playwright @playwright/test \
  eslint-plugin-jsx-a11y eslint-plugin-security \
  @typescript-eslint/parser @typescript-eslint/eslint-plugin \
  prettier

# 3. Phosphor + react-hook-form + zod + TanStack Query
yarn add @phosphor-icons/react react-hook-form zod \
  @tanstack/react-query
```

> Existing `App.js`, `index.js`, etc. are renamed to `App.tsx`, `index.tsx`,
> etc. and content migrated incrementally per §10 below. No new files are
> created until `tsc --noEmit` passes.

---

## 3. `tsconfig.json` (strict, P0)

```json
// file: /app/frontend/tsconfig.json
{
  \"compilerOptions\": {
    \"target\": \"ES2022\",
    \"lib\": [\"DOM\", \"DOM.Iterable\", \"ES2022\"],
    \"jsx\": \"react-jsx\",
    \"module\": \"ESNext\",
    \"moduleResolution\": \"Bundler\",
    \"allowJs\": false,
    \"strict\": true,
    \"noImplicitAny\": true,
    \"strictNullChecks\": true,
    \"noUnusedLocals\": true,
    \"noUnusedParameters\": true,
    \"noImplicitReturns\": true,
    \"noFallthroughCasesInSwitch\": true,
    \"forceConsistentCasingInFileNames\": true,
    \"esModuleInterop\": true,
    \"resolveJsonModule\": true,
    \"isolatedModules\": true,
    \"skipLibCheck\": true,
    \"noEmit\": true,
    \"baseUrl\": \"src\",
    \"paths\": {
      \"@/*\": [\"*\"],
      \"@components/*\": [\"components/*\"],
      \"@pages/*\": [\"pages/*\"],
      \"@lib/*\": [\"lib/*\"],
      \"@types/*\": [\"types/*\"]
    }
  },
  \"include\": [\"src/**/*.ts\", \"src/**/*.tsx\"],
  \"exclude\": [\"node_modules\", \"build\", \"dist\", \"**/*.test.tsx\"]
}
```

> `any` is allowed **only** when annotated with an inline justification per
> AGENTS_FRONTEND.md §8 P0. Pattern:
> `// any-justification: 3rd-party untyped lib (recharts payload)`

`tsc --noEmit` runs in pre-commit and CI. Any error → blocked.

---

## 4. craco config (TS aware)

```js
// file: /app/frontend/craco.config.js
const path = require(\"path\");

module.exports = {
  webpack: {
    alias: {
      \"@\": path.resolve(__dirname, \"src\"),
      \"@components\": path.resolve(__dirname, \"src/components\"),
      \"@pages\": path.resolve(__dirname, \"src/pages\"),
      \"@lib\": path.resolve(__dirname, \"src/lib\"),
      \"@types\": path.resolve(__dirname, \"src/types\"),
    },
  },
  style: { postcss: { plugins: [require(\"tailwindcss\"), require(\"autoprefixer\")] } },
};
```

---

## 5. Tailwind theme tokens (Control Room)

```js
// file: /app/frontend/tailwind.config.js
/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [\"./src/**/*.{ts,tsx}\", \"./public/index.html\"],
  darkMode: \"class\",
  theme: {
    extend: {
      colors: {
        // Background layers
        bg: { 0: \"#0A0A0A\", 1: \"#121212\", 2: \"#1A1A1A\", 3: \"#27272A\" },
        // Text
        fg: { 0: \"#FFFFFF\", 1: \"#A1A1AA\", 2: \"#71717A\", 3: \"#52525B\" },
        // Brand cyan
        brand: { DEFAULT: \"#06B6D4\", hover: \"#22D3EE\", dim: \"#0E7490\" },
        // Verdicts
        verdict: {
          ai: \"#EF4444\",
          real: \"#10B981\",
          inconclusive: \"#F59E0B\",
        },
        // Provenance / status
        ok: \"#10B981\",
        warn: \"#F59E0B\",
        err: \"#EF4444\",
      },
      fontFamily: {
        sans: [\"Inter\", \"system-ui\", \"sans-serif\"],
        display: [\"IBM Plex Sans\", \"Inter\", \"sans-serif\"],
        mono: [\"JetBrains Mono\", \"ui-monospace\", \"monospace\"],
      },
      borderRadius: { card: \"12px\", pill: \"9999px\" },
      boxShadow: {
        card: \"0 1px 0 rgba(255,255,255,0.04) inset, 0 8px 24px rgba(0,0,0,0.40)\",
        glow: \"0 0 0 1px rgba(6,182,212,0.30), 0 0 24px rgba(6,182,212,0.15)\",
      },
      animation: {
        \"fade-in\": \"fadeIn 300ms ease-out both\",
        \"slide-up\": \"slideUp 320ms ease-out both\",
        \"pulse-dim\": \"pulseDim 2s ease-in-out infinite\",
      },
      keyframes: {
        fadeIn: { \"0%\": { opacity: 0 }, \"100%\": { opacity: 1 } },
        slideUp: {
          \"0%\": { opacity: 0, transform: \"translateY(8px)\" },
          \"100%\": { opacity: 1, transform: \"translateY(0)\" },
        },
        pulseDim: {
          \"0%,100%\": { opacity: 0.55 },
          \"50%\": { opacity: 1.0 },
        },
      },
    },
  },
  plugins: [require(\"tailwindcss-animate\")],
};
```

```css
/* file: /app/frontend/src/styles/globals.css */
@import url(\"https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap\");

@tailwind base;
@tailwind components;
@tailwind utilities;

:root {
  color-scheme: dark;
}

html, body, #root { background: theme(\"colors.bg.0\"); color: theme(\"colors.fg.0\"); }

/* Selection */
::selection { background: theme(\"colors.brand.dim\"); color: theme(\"colors.fg.0\"); }

/* Focus ring (WCAG 2.1 AA visible focus) */
*:focus-visible {
  outline: 2px solid theme(\"colors.brand.DEFAULT\");
  outline-offset: 2px;
  border-radius: 4px;
}

/* Reduce motion */
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: 0.001ms !important;
    transition-duration: 0.001ms !important;
  }
}

/* Scrollbar */
::-webkit-scrollbar { width: 8px; height: 8px; }
::-webkit-scrollbar-track { background: theme(\"colors.bg.1\"); }
::-webkit-scrollbar-thumb { background: theme(\"colors.bg.3\"); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: theme(\"colors.brand.dim\"); }
```

> **No purple gradients. No emoji. No \"AI slop\".** Verified at component
> review time by linting against a list of disallowed CSS tokens (see §16).

---

## 6. ESLint + Prettier

```js
// file: /app/frontend/.eslintrc.cjs
module.exports = {
  root: true,
  parser: \"@typescript-eslint/parser\",
  parserOptions: { ecmaVersion: 2022, sourceType: \"module\", ecmaFeatures: { jsx: true } },
  plugins: [\"@typescript-eslint\", \"react\", \"jsx-a11y\", \"security\"],
  extends: [
    \"eslint:recommended\",
    \"plugin:@typescript-eslint/recommended\",
    \"plugin:react/recommended\",
    \"plugin:react/jsx-runtime\",
    \"plugin:jsx-a11y/strict\",
    \"plugin:security/recommended-legacy\",
    \"prettier\",
  ],
  settings: { react: { version: \"19\" } },
  rules: {
    \"@typescript-eslint/no-explicit-any\": \"error\",
    \"@typescript-eslint/explicit-function-return-type\": \"off\",
    \"@typescript-eslint/no-unused-vars\": [\"error\", { argsIgnorePattern: \"^_\" }],
    \"react/prop-types\": \"off\",
    \"jsx-a11y/anchor-is-valid\": \"warn\",
    \"no-console\": [\"warn\", { allow: [\"warn\", \"error\"] }],
  },
  ignorePatterns: [\"build/\", \"dist/\", \"node_modules/\", \"*.config.js\"],
};
```

```json
// file: /app/frontend/.prettierrc.json
{ \"semi\": true, \"singleQuote\": false, \"trailingComma\": \"all\", \"printWidth\": 100, \"tabWidth\": 2 }
```

---

## 7. Folder structure (mirrors Masterplan §15.frontend)

```
src/
├── index.tsx
├── App.tsx
├── styles/
│   └── globals.css
├── types/
│   └── api.ts                 // mirrors backend Pydantic
├── lib/
│   ├── api.ts                 // fetch wrapper + retry
│   ├── format.ts              // percent, bytes, durations
│   ├── devmode.ts             // localStorage flag + key listener
│   └── queryClient.ts         // TanStack Query instance
├── components/
│   ├── ui/                    // shadcn primitives (Button, Card, Tooltip, etc.)
│   ├── DropZone.tsx
│   ├── ProgressSteps.tsx
│   ├── VerdictCard.tsx
│   ├── ProvenanceBadge.tsx
│   ├── VLMBadge.tsx
│   ├── ReverseSearchBadge.tsx
│   ├── ContentTypeBadge.tsx
│   ├── ConfidenceAgreementBars.tsx
│   ├── NarrativePanel.tsx
│   ├── SignalBarChart.tsx
│   ├── HeatmapPanel.tsx
│   ├── FrequencyPanel.tsx
│   ├── MetadataTable.tsx
│   ├── CompressionFingerprintPanel.tsx
│   ├── RetrievalNeighborsPanel.tsx
│   ├── ReverseSearchPanel.tsx
│   ├── VLMRationalePanel.tsx
│   ├── CorrectVerdictBar.tsx
│   ├── DeveloperPanel.tsx
│   └── HistoryList.tsx
├── pages/
│   ├── UploadPage.tsx
│   ├── JobPage.tsx
│   └── AboutPage.tsx
└── tests/
    ├── setup.ts
    └── e2e/                   // Playwright specs
```

---

## 8. API types (mirror backend `schemas/results.py`)

```ts
// file: /app/frontend/src/types/api.ts
export type Verdict = \"AI-GENERATED\" | \"REAL\" | \"INCONCLUSIVE\" | \"MANIPULATED\";
export type Modality = \"image\" | \"audio\" | \"video\";
export type Profile = \"cloud_lite\" | \"mac_full\" | \"cuda_full\";
export type JobStatus = \"queued\" | \"running\" | \"done\" | \"failed\";
export type CalibrationMode = \"platt_refdb\" | \"platt_blended\" | \"isotonic\" | \"cold_start\";
export type FusionMode = \"uniform\" | \"lr_l2\" | \"gbdt\";
export type ContentType =
  | \"selfie_portrait\"
  | \"landscape_scene\"
  | \"object_product\"
  | \"meme_screenshot\"
  | \"document_scan\"
  | \"artwork_illustration\";

export interface Provenance {
  hit: boolean;
  source: \"c2pa\" | \"synthid\" | \"sd_wm\" | \"meta_wm\" | \"none\";
  details?: Record<string, unknown>;
}

export interface SignalRow {
  name: string;
  p_fake: number;       // 0..1, calibrated
  weight: number;       // 0..1
  explanation: string;
  raw?: number;         // raw uncalibrated score, dev mode only
}

export interface RetrievalNeighbor {
  id: string;
  label: \"real\" | \"ai\";
  distance: number;
  source: string;
  thumb_url: string;
  generator_family?: string;
}

export interface ReverseHit {
  url: string;
  domain: string;
  date: string | null;
  thumbnail?: string;
  title?: string;
}

export interface VLMRationale {
  p_ai: number;
  defects: string[];
  rationale: string;
}

export interface CompressionFingerprint {
  container: \"png\" | \"jpeg\" | \"webp\" | \"avif\" | \"heic\" | \"unknown\";
  fingerprint: Record<string, string | number | boolean | null>;
  flag: \"ai_signature\" | \"camera_signature\" | \"neutral\";
}

export interface XAI {
  heatmap_url: string | null;
  frequency_plot_url: string | null;
  metadata: Record<string, unknown>;
  compression_fingerprint: CompressionFingerprint | null;
  narrative: string;
  narrative_source: \"gemini\" | \"fallback_template\";
}

export interface JobResult {
  job_id: string;
  modality: Modality;
  profile: Profile;
  calibration: CalibrationMode;
  fusion_model: FusionMode;
  content_type: ContentType;
  verdict: Verdict;
  p_ai_generated: number;
  confidence: number;
  agreement: number;
  extremity: number;
  cross_modal_bonus: number;
  abstained: boolean;
  novel_generator_suspected?: boolean;
  provenance: Provenance;
  vlm_invoked: boolean;
  reverse_invoked: boolean;
  signals: SignalRow[];
  retrieval: { k: number; neighbors: RetrievalNeighbor[] };
  reverse_search: { hits: ReverseHit[]; reason?: string } | null;
  vlm?: VLMRationale | null;
  xai: XAI;
  input: { filename: string; sha256: string; bytes: number; mime: string };
  durations_ms: Record<string, number>;
  debug?: DebugBlock | null;
}

export interface DebugBlock {
  fusion_vector: number[];
  fusion_vector_mask: boolean[];
  fusion_vector_keys: string[];
  raw_signals: Array<{ name: string; raw: number; calibrated: number; weight: number }>;
  gate_states: {
    vlm: { invoked: boolean; reason: string };
    reverse: { invoked: boolean; reason: string };
  };
  retrieval_full: RetrievalNeighbor[];
  reverse_full: unknown;
}

export interface JobStatusEnvelope {
  job_id: string;
  modality: Modality;
  status: JobStatus;
  progress: number;
  stage: string;
  started_at: string;
  finished_at: string | null;
  error?: string;
}

export interface HealthEnvelope {
  status: \"ok\" | \"degraded\";
  profile: Profile;
  signals_loaded: string[];
  db_ok: boolean;
  gemini_ok: boolean;
  serpapi_ok: boolean;
  refdb_loaded: boolean;
  refdb_size: Record<string, number>;
  fusion_mode: FusionMode;
  calibration: CalibrationMode;
  ece_refdb_holdout: number;
  auroc_refdb_holdout: number;
  n_user_labels: number;
  uptime_s: number;
}

export interface ApiError {
  error: string;
  message: string;
  request_id: string;
}
```

---

## 9. lib/api.ts — fetch wrapper

```ts
// file: /app/frontend/src/lib/api.ts
import type {
  JobResult, JobStatusEnvelope, HealthEnvelope, ApiError, Modality,
} from \"@types/api\";

const BASE = process.env.REACT_APP_BACKEND_URL;
if (!BASE) throw new Error(\"REACT_APP_BACKEND_URL is not set\");

const TIMEOUT_MS = 30_000;

async function withTimeout<T>(p: Promise<T>, ms = TIMEOUT_MS): Promise<T> {
  return Promise.race([
    p,
    new Promise<T>((_, rej) => setTimeout(() => rej(new Error(\"timeout\")), ms)),
  ]);
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const url = `${BASE}/api${path}`;
  const res = await withTimeout(fetch(url, { ...init, credentials: \"omit\" }));
  if (!res.ok) {
    let err: ApiError;
    try { err = (await res.json()) as ApiError; }
    catch { err = { error: \"INTERNAL\", message: res.statusText, request_id: \"n/a\" }; }
    throw err;
  }
  return (await res.json()) as T;
}

export async function postAnalyze(file: File, hints?: { modality?: Modality }) {
  const fd = new FormData();
  fd.append(\"file\", file);
  if (hints?.modality) fd.append(\"hints\", JSON.stringify(hints));
  return request<{ job_id: string; modality: Modality; status: \"queued\"; profile: string }>(
    \"/analyze\",
    { method: \"POST\", body: fd },
  );
}

export const getJob = (id: string) => request<JobStatusEnvelope>(`/jobs/${id}`);

export const getResult = (id: string, debug = false) =>
  request<JobResult>(`/jobs/${id}/result${debug ? \"?debug=1\" : \"\"}`);

export const getHealth = () => request<HealthEnvelope>(\"/health\");

export const getHistory = (limit = 20) =>
  request<{ items: JobStatusEnvelope[] }>(`/history?limit=${limit}`);

export const postCorrect = (id: string, label: \"ai\" | \"real\") =>
  request<{ ok: true; refdb_hard_size: number }>(`/jobs/${id}/correct`, {
    method: \"POST\",
    headers: { \"Content-Type\": \"application/json\" },
    body: JSON.stringify({ user_label: label }),
  });

export const assetUrl = (id: string, name: string) =>
  `${BASE}/api/jobs/${id}/assets/${encodeURIComponent(name)}`;
```

---

## 10. lib/format.ts + lib/devmode.ts + queryClient.ts

```ts
// file: /app/frontend/src/lib/format.ts
export const pct = (x: number, digits = 0): string =>
  `${(Math.max(0, Math.min(1, x)) * 100).toFixed(digits)}%`;

export const bytes = (n: number): string => {
  if (n < 1024) return `${n} B`;
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(1)} KB`;
  if (n < 1024 ** 3) return `${(n / 1024 ** 2).toFixed(1)} MB`;
  return `${(n / 1024 ** 3).toFixed(2)} GB`;
};

export const ms = (n: number): string => (n < 1000 ? `${n} ms` : `${(n / 1000).toFixed(2)} s`);

export const truncate = (s: string, n: number): string => (s.length <= n ? s : `${s.slice(0, n)}…`);
```

```ts
// file: /app/frontend/src/lib/devmode.ts
const KEY = \"argus.devmode\";

export const isDevMode = (): boolean => {
  try { return localStorage.getItem(KEY) === \"1\"; }
  catch { return false; }
};

export const setDevMode = (v: boolean): void => {
  try { localStorage.setItem(KEY, v ? \"1\" : \"0\"); }
  catch { /* ignore */ }
  window.dispatchEvent(new CustomEvent(\"argus.devmode.changed\", { detail: v }));
};

/** Bind Ctrl/Cmd+D to toggle dev mode. Call once at App mount. */
export const bindDevModeHotkey = (): () => void => {
  const onKey = (e: KeyboardEvent): void => {
    const mod = e.ctrlKey || e.metaKey;
    if (mod && e.key.toLowerCase() === \"d\") {
      e.preventDefault();
      setDevMode(!isDevMode());
    }
  };
  window.addEventListener(\"keydown\", onKey);
  return () => window.removeEventListener(\"keydown\", onKey);
};
```

```ts
// file: /app/frontend/src/lib/queryClient.ts
import { QueryClient } from \"@tanstack/react-query\";

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      retry: (failureCount, error) => {
        // Retry network errors twice; never retry 4xx
        const e = error as { error?: string };
        if (e?.error === \"INTERNAL\" && failureCount < 2) return true;
        return false;
      },
      refetchOnWindowFocus: false,
      staleTime: 5_000,
    },
  },
});
```

---

## 11. App + routing + entry

```tsx
// file: /app/frontend/src/index.tsx
import React from \"react\";
import { createRoot } from \"react-dom/client\";
import { QueryClientProvider } from \"@tanstack/react-query\";
import { BrowserRouter } from \"react-router-dom\";

import App from \"./App\";
import { queryClient } from \"@lib/queryClient\";
import \"./styles/globals.css\";

const root = createRoot(document.getElementById(\"root\") as HTMLElement);
root.render(
  <React.StrictMode>
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <App />
      </BrowserRouter>
    </QueryClientProvider>
  </React.StrictMode>,
);
```

```tsx
// file: /app/frontend/src/App.tsx
import { useEffect } from \"react\";
import { Routes, Route, Link, NavLink } from \"react-router-dom\";
import { bindDevModeHotkey } from \"@lib/devmode\";
import UploadPage from \"@pages/UploadPage\";
import JobPage from \"@pages/JobPage\";
import AboutPage from \"@pages/AboutPage\";

export default function App(): JSX.Element {
  useEffect(() => bindDevModeHotkey(), []);

  return (
    <div className=\"min-h-screen bg-bg-0 text-fg-0 font-sans\">
      <header className=\"border-b border-bg-3 px-6 py-4 flex items-center justify-between\">
        <Link to=\"/\" className=\"flex items-center gap-2\" data-testid=\"header-home-link\">
          <span className=\"h-2 w-2 rounded-full bg-brand animate-pulse-dim\" aria-hidden />
          <span className=\"font-display text-lg font-semibold tracking-tight\">ARGUS</span>
          <span className=\"text-fg-2 text-xs font-mono uppercase tracking-widest\">
            Forensic Console
          </span>
        </Link>
        <nav className=\"flex gap-6 text-sm\" aria-label=\"Primary navigation\">
          <NavLink to=\"/\" end className={navClass} data-testid=\"nav-upload\">Analyse</NavLink>
          <NavLink to=\"/about\" className={navClass} data-testid=\"nav-about\">About</NavLink>
        </nav>
      </header>

      <main className=\"px-6 py-8 max-w-6xl mx-auto\">
        <Routes>
          <Route path=\"/\" element={<UploadPage />} />
          <Route path=\"/job/:id\" element={<JobPage />} />
          <Route path=\"/about\" element={<AboutPage />} />
        </Routes>
      </main>

      <footer className=\"border-t border-bg-3 px-6 py-3 text-xs text-fg-2 font-mono\">
        Calibrated Orthogonal Evidence Fusion · ≥95 % accuracy on non-abstained ·
        No model fine-tuning · No training cost
      </footer>
    </div>
  );
}

const navClass = ({ isActive }: { isActive: boolean }): string =>
  `transition-colors duration-200 ${isActive ? \"text-brand\" : \"text-fg-1 hover:text-fg-0\"}`;
```

---

## 12. UploadPage

```tsx
// file: /app/frontend/src/pages/UploadPage.tsx
import { useNavigate } from \"react-router-dom\";
import { useQuery } from \"@tanstack/react-query\";
import DropZone from \"@components/DropZone\";
import HistoryList from \"@components/HistoryList\";
import { getHistory, postAnalyze } from \"@lib/api\";

export default function UploadPage(): JSX.Element {
  const navigate = useNavigate();
  const history = useQuery({ queryKey: [\"history\"], queryFn: () => getHistory(20) });

  const onFile = async (file: File): Promise<void> => {
    const { job_id } = await postAnalyze(file);
    navigate(`/job/${job_id}`);
  };

  return (
    <div className=\"flex flex-col gap-10 animate-fade-in\">
      <section aria-labelledby=\"upload-h\">
        <h1 id=\"upload-h\" className=\"font-display text-3xl font-semibold mb-2\">
          Upload a media file
        </h1>
        <p className=\"text-fg-1 mb-6 max-w-2xl\">
          The system runs five tiers of orthogonal evidence — provenance, forensic detectors,
          retrieval, reverse search, and a VLM tiebreaker — then returns a calibrated verdict
          and the evidence behind it.
        </p>
        <DropZone onFile={onFile} />
      </section>

      <section aria-labelledby=\"how-h\">
        <h2 id=\"how-h\" className=\"font-display text-xl font-semibold mb-3\">
          How it works
        </h2>
        <ol className=\"grid grid-cols-1 md:grid-cols-5 gap-3 text-sm text-fg-1\">
          {STEPS.map((s, i) => (
            <li
              key={s.title}
              className=\"rounded-card border border-bg-3 bg-bg-1 p-4\"
              data-testid={`how-step-${i}`}
            >
              <div className=\"text-brand font-mono text-xs tracking-widest\">
                TIER {i === 4 ? \"3\" : i}
              </div>
              <div className=\"font-display font-semibold mt-1 mb-1\">{s.title}</div>
              <div>{s.body}</div>
            </li>
          ))}
        </ol>
      </section>

      <section aria-labelledby=\"history-h\">
        <h2 id=\"history-h\" className=\"font-display text-xl font-semibold mb-3\">
          Recent jobs
        </h2>
        <HistoryList query={history} />
      </section>
    </div>
  );
}

const STEPS = [
  { title: \"Provenance\", body: \"C2PA + SD watermark + SynthID. Short-circuit when found.\" },
  { title: \"Forensic & learned\", body: \"Up to 10 orthogonal detectors per modality.\" },
  { title: \"Retrieval\", body: \"CLIP/DINOv2 k-NN against a curated reference DB.\" },
  { title: \"Reverse search\", body: \"SerpAPI lookup — pre-2022 hits prove REAL.\" },
  { title: \"VLM tiebreaker\", body: \"Gemini explains visible defects, only when uncertain.\" },
];
```

---

## 13. JobPage (the centerpiece)

```tsx
// file: /app/frontend/src/pages/JobPage.tsx
import { useParams } from \"react-router-dom\";
import { useQuery } from \"@tanstack/react-query\";
import { useEffect, useState } from \"react\";

import { getJob, getResult } from \"@lib/api\";
import { isDevMode } from \"@lib/devmode\";

import ProgressSteps from \"@components/ProgressSteps\";
import VerdictCard from \"@components/VerdictCard\";
import ConfidenceAgreementBars from \"@components/ConfidenceAgreementBars\";
import NarrativePanel from \"@components/NarrativePanel\";
import SignalBarChart from \"@components/SignalBarChart\";
import HeatmapPanel from \"@components/HeatmapPanel\";
import FrequencyPanel from \"@components/FrequencyPanel\";
import MetadataTable from \"@components/MetadataTable\";
import CompressionFingerprintPanel from \"@components/CompressionFingerprintPanel\";
import RetrievalNeighborsPanel from \"@components/RetrievalNeighborsPanel\";
import ReverseSearchPanel from \"@components/ReverseSearchPanel\";
import VLMRationalePanel from \"@components/VLMRationalePanel\";
import CorrectVerdictBar from \"@components/CorrectVerdictBar\";
import DeveloperPanel from \"@components/DeveloperPanel\";

export default function JobPage(): JSX.Element {
  const { id } = useParams<{ id: string }>();
  if (!id) return <p data-testid=\"job-page-missing-id\">Job id missing.</p>;

  const [dev, setDev] = useState<boolean>(isDevMode());
  useEffect(() => {
    const onChange = (e: Event): void => setDev((e as CustomEvent<boolean>).detail);
    window.addEventListener(\"argus.devmode.changed\", onChange);
    return () => window.removeEventListener(\"argus.devmode.changed\", onChange);
  }, []);

  const job = useQuery({
    queryKey: [\"job\", id],
    queryFn: () => getJob(id),
    refetchInterval: (q) => {
      const s = q.state.data?.status;
      return s === \"queued\" || s === \"running\" ? 1000 : false;
    },
  });

  const result = useQuery({
    queryKey: [\"result\", id, dev],
    queryFn: () => getResult(id, dev),
    enabled: job.data?.status === \"done\",
  });

  if (job.isLoading) {
    return <p className=\"text-fg-1\" data-testid=\"job-loading\">Loading job…</p>;
  }
  if (job.error) {
    return (
      <p className=\"text-err\" data-testid=\"job-error\">
        Failed to load job. {(job.error as { message?: string }).message}
      </p>
    );
  }
  if (!job.data) return <p data-testid=\"job-empty\">No job data.</p>;

  if (job.data.status !== \"done\") {
    return (
      <div data-testid=\"job-running\" className=\"animate-fade-in\">
        <ProgressSteps job={job.data} />
      </div>
    );
  }

  if (result.isLoading) return <p data-testid=\"result-loading\">Loading result…</p>;
  if (result.error) {
    return (
      <p className=\"text-err\" data-testid=\"result-error\">
        Failed to load result.
      </p>
    );
  }
  if (!result.data) return <p data-testid=\"result-empty\">No result.</p>;

  const r = result.data;

  return (
    <div className=\"grid grid-cols-1 lg:grid-cols-3 gap-6 animate-slide-up\" data-testid=\"result-view\">
      <div className=\"lg:col-span-2 flex flex-col gap-6\">
        <VerdictCard result={r} />
        <ConfidenceAgreementBars result={r} />
        <NarrativePanel result={r} />
        <SignalBarChart signals={r.signals} />
        <HeatmapPanel result={r} />
        <FrequencyPanel result={r} />
        <RetrievalNeighborsPanel retrieval={r.retrieval} />
        {r.reverse_invoked && r.reverse_search && (
          <ReverseSearchPanel reverse={r.reverse_search} />
        )}
        {r.vlm_invoked && r.vlm && <VLMRationalePanel vlm={r.vlm} />}
        <CorrectVerdictBar jobId={r.job_id} />
      </div>
      <aside className=\"flex flex-col gap-6\">
        <MetadataTable result={r} />
        {r.xai.compression_fingerprint && (
          <CompressionFingerprintPanel fp={r.xai.compression_fingerprint} />
        )}
        {dev && r.debug && <DeveloperPanel result={r} />}
      </aside>
    </div>
  );
}
```

---

## 14. Component contracts

Every component listed below conforms to the **Standard Component Contract**
(AGENTS_FRONTEND.md §7 P0):

1. Typed `Props` interface
2. Loading state UI
3. Error state UI
4. Empty state UI
5. ARIA attributes for non-trivial widgets
6. Unit test colocated as `Component.test.tsx`
7. Memoization where computation is non-trivial

> Loading / error / empty are handled by parent (`JobPage` for big panels)
> or shown inline (`HistoryList`, `RetrievalNeighborsPanel`).

### 14.1 `DropZone.tsx`

```tsx
// file: /app/frontend/src/components/DropZone.tsx
import { useCallback, useRef, useState } from \"react\";
import { UploadSimple, FileImage, Spinner } from \"@phosphor-icons/react\";
import { bytes } from \"@lib/format\";

interface Props {
  onFile: (file: File) => Promise<void>;
  maxMB?: number;
  accept?: string;
}

const DEFAULT_ACCEPT = \"image/*,audio/*,video/*\";

export default function DropZone({ onFile, maxMB = 200, accept = DEFAULT_ACCEPT }: Props): JSX.Element {
  const [drag, setDrag] = useState(false);
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFile = useCallback(
    async (f: File): Promise<void> => {
      setErr(null);
      if (f.size > maxMB * 1024 * 1024) {
        setErr(`File exceeds ${maxMB} MB (${bytes(f.size)}).`);
        return;
      }
      setBusy(true);
      try { await onFile(f); }
      catch (e) {
        const m = (e as { message?: string })?.message ?? \"Upload failed.\";
        setErr(m);
      } finally { setBusy(false); }
    },
    [maxMB, onFile],
  );

  const onDrop = useCallback(
    (e: React.DragEvent<HTMLLabelElement>): void => {
      e.preventDefault();
      setDrag(false);
      const f = e.dataTransfer.files?.[0];
      if (f) void handleFile(f);
    },
    [handleFile],
  );

  return (
    <div className=\"flex flex-col gap-2\">
      <label
        htmlFor=\"media-upload-input\"
        data-testid=\"media-upload-dropzone\"
        onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
        onDragLeave={() => setDrag(false)}
        onDrop={onDrop}
        className={[
          \"block cursor-pointer rounded-card border-2 border-dashed p-12 text-center\",
          \"transition-colors duration-200\",
          drag ? \"border-brand bg-bg-2\" : \"border-bg-3 bg-bg-1 hover:border-brand/60\",
        ].join(\" \")}
      >
        <input
          id=\"media-upload-input\"
          ref={inputRef}
          type=\"file\"
          className=\"sr-only\"
          accept={accept}
          onChange={(e) => {
            const f = e.target.files?.[0];
            if (f) void handleFile(f);
          }}
          data-testid=\"media-upload-input\"
          disabled={busy}
        />
        <div className=\"flex flex-col items-center gap-3\">
          {busy ? (
            <Spinner size={36} className=\"animate-spin text-brand\" aria-hidden />
          ) : (
            <UploadSimple size={36} className=\"text-brand\" aria-hidden />
          )}
          <div className=\"font-display font-semibold text-lg\">
            {busy ? \"Uploading…\" : \"Drop a file or click to choose\"}
          </div>
          <div className=\"text-fg-2 text-sm flex items-center gap-2\">
            <FileImage size={16} aria-hidden />
            <span>Image · Audio · Video — up to {maxMB} MB</span>
          </div>
        </div>
      </label>
      {err && (
        <p role=\"alert\" data-testid=\"dropzone-error\" className=\"text-err text-sm\">
          {err}
        </p>
      )}
    </div>
  );
}
```

### 14.2 `ProgressSteps.tsx`

```tsx
// file: /app/frontend/src/components/ProgressSteps.tsx
import { CheckCircle, CircleNotch, Circle } from \"@phosphor-icons/react\";
import type { JobStatusEnvelope } from \"@types/api\";

interface Props { job: JobStatusEnvelope; }

const STAGES = [
  { key: \"preprocess\", label: \"Preprocess\" },
  { key: \"tier0\",      label: \"Tier 0 — Provenance\" },
  { key: \"tier1\",      label: \"Tier 1 — Detectors\" },
  { key: \"tier2\",      label: \"Tier 2 — Retrieval\" },
  { key: \"tier25\",     label: \"Tier 2.5 — Reverse search\" },
  { key: \"tier3\",      label: \"Tier 3 — VLM\" },
  { key: \"fusion\",     label: \"Fusion + abstention\" },
  { key: \"xai\",        label: \"XAI + narrative\" },
];

export default function ProgressSteps({ job }: Props): JSX.Element {
  const currentIdx = stageIndex(job.stage);
  return (
    <ol className=\"flex flex-col gap-3\" aria-label=\"Job progress\" data-testid=\"job-progress-steps\">
      {STAGES.map((s, i) => {
        const state: \"done\" | \"active\" | \"pending\" =
          i < currentIdx ? \"done\" : i === currentIdx ? \"active\" : \"pending\";
        return (
          <li
            key={s.key}
            data-testid={`progress-step-${s.key}`}
            data-state={state}
            className=\"flex items-center gap-3 rounded-card border border-bg-3 bg-bg-1 px-4 py-3\"
          >
            <StepIcon state={state} />
            <div className=\"font-display\">{s.label}</div>
            {state === \"active\" && (
              <div className=\"ml-auto text-xs font-mono text-fg-2\">
                {(job.progress * 100).toFixed(0)} %
              </div>
            )}
          </li>
        );
      })}
      <li data-testid=\"progress-status-pill\" className=\"text-xs text-fg-2 font-mono\">
        status: {job.status}
      </li>
    </ol>
  );
}

function StepIcon({ state }: { state: \"done\" | \"active\" | \"pending\" }): JSX.Element {
  if (state === \"done\") return <CheckCircle size={20} className=\"text-ok\" aria-hidden />;
  if (state === \"active\") {
    return <CircleNotch size={20} className=\"text-brand animate-spin\" aria-hidden />;
  }
  return <Circle size={20} className=\"text-fg-3\" aria-hidden />;
}

function stageIndex(stage: string): number {
  const map: Record<string, number> = {
    preprocess: 0, tier0: 1, tier1: 2, tier2: 3,
    tier25: 4, tier3: 5, fusion: 6, xai: 7,
  };
  const key = stage.split(\"_\")[0];
  return map[key] ?? 0;
}
```

### 14.3 `VerdictCard.tsx`

```tsx
// file: /app/frontend/src/components/VerdictCard.tsx
import { ShieldCheck, ShieldWarning, Question, Wrench } from \"@phosphor-icons/react\";
import type { JobResult, Verdict } from \"@types/api\";
import ProvenanceBadge from \"./ProvenanceBadge\";
import VLMBadge from \"./VLMBadge\";
import ReverseSearchBadge from \"./ReverseSearchBadge\";
import ContentTypeBadge from \"./ContentTypeBadge\";
import { pct } from \"@lib/format\";

interface Props { result: JobResult; }

const VERDICT_META: Record<Verdict, { color: string; bg: string; icon: JSX.Element; label: string }> = {
  \"AI-GENERATED\": {
    color: \"text-verdict-ai\", bg: \"bg-verdict-ai/15 border-verdict-ai/40\",
    icon: <ShieldWarning size={36} weight=\"duotone\" />,
    label: \"AI-GENERATED\",
  },
  \"REAL\": {
    color: \"text-verdict-real\", bg: \"bg-verdict-real/15 border-verdict-real/40\",
    icon: <ShieldCheck size={36} weight=\"duotone\" />,
    label: \"REAL\",
  },
  \"INCONCLUSIVE\": {
    color: \"text-verdict-inconclusive\", bg: \"bg-verdict-inconclusive/15 border-verdict-inconclusive/40\",
    icon: <Question size={36} weight=\"duotone\" />,
    label: \"INCONCLUSIVE\",
  },
  \"MANIPULATED\": {
    color: \"text-verdict-inconclusive\", bg: \"bg-verdict-inconclusive/15 border-verdict-inconclusive/40\",
    icon: <Wrench size={36} weight=\"duotone\" />,
    label: \"MANIPULATED\",
  },
};

export default function VerdictCard({ result }: Props): JSX.Element {
  const m = VERDICT_META[result.verdict];
  return (
    <section
      data-testid=\"verdict-card-container\"
      aria-labelledby=\"verdict-h\"
      className={`rounded-card border ${m.bg} p-6 shadow-card`}
    >
      <div className=\"flex items-start gap-4\">
        <div className={m.color} aria-hidden>{m.icon}</div>
        <div className=\"flex-1\">
          <h2 id=\"verdict-h\" className={`font-display text-2xl font-semibold ${m.color}`}
              data-testid=\"verdict-label\">
            {m.label}
          </h2>
          <p className=\"text-fg-1 text-sm\">
            P(AI-generated) ={\" \"}
            <span className=\"font-mono text-fg-0\" data-testid=\"verdict-p-ai\">
              {pct(result.p_ai_generated, 1)}
            </span>
            {\" · \"}
            confidence{\" \"}
            <span className=\"font-mono text-fg-0\" data-testid=\"verdict-confidence\">
              {pct(result.confidence)}
            </span>
            {result.abstained && (
              <span className=\"ml-2 text-verdict-inconclusive\" data-testid=\"verdict-abstained\">
                (abstained)
              </span>
            )}
            {result.novel_generator_suspected && (
              <span className=\"ml-2 text-warn\" data-testid=\"verdict-novel-suspected\">
                (novel generator suspected)
              </span>
            )}
          </p>
          <div className=\"mt-3 flex flex-wrap gap-2\">
            <ContentTypeBadge type={result.content_type} />
            <ProvenanceBadge provenance={result.provenance} />
            {result.vlm_invoked && <VLMBadge />}
            {result.reverse_invoked && <ReverseSearchBadge />}
          </div>
        </div>
      </div>
    </section>
  );
}
```

### 14.4 Badge components (Provenance / VLM / Reverse / ContentType)

```tsx
// file: /app/frontend/src/components/ProvenanceBadge.tsx
import { SealCheck } from \"@phosphor-icons/react\";
import type { Provenance } from \"@types/api\";

export default function ProvenanceBadge({ provenance }: { provenance: Provenance }): JSX.Element | null {
  if (!provenance.hit) return null;
  return (
    <span
      data-testid=\"provenance-badge\"
      className=\"inline-flex items-center gap-1 rounded-pill bg-ok/20 border border-ok/40 px-3 py-1 text-xs font-mono text-ok\"
      title={`Provenance: ${provenance.source}`}
    >
      <SealCheck size={14} weight=\"fill\" aria-hidden /> provenance · {provenance.source}
    </span>
  );
}
```

```tsx
// file: /app/frontend/src/components/VLMBadge.tsx
import { Brain } from \"@phosphor-icons/react\";

export default function VLMBadge(): JSX.Element {
  return (
    <span
      data-testid=\"vlm-invoked-badge\"
      className=\"inline-flex items-center gap-1 rounded-pill bg-brand/15 border border-brand/40 px-3 py-1 text-xs font-mono text-brand\"
    >
      <Brain size={14} weight=\"duotone\" aria-hidden /> vlm tiebreaker
    </span>
  );
}
```

```tsx
// file: /app/frontend/src/components/ReverseSearchBadge.tsx
import { MagnifyingGlass } from \"@phosphor-icons/react\";

export default function ReverseSearchBadge(): JSX.Element {
  return (
    <span
      data-testid=\"reverse-search-badge\"
      className=\"inline-flex items-center gap-1 rounded-pill bg-brand/15 border border-brand/40 px-3 py-1 text-xs font-mono text-brand\"
    >
      <MagnifyingGlass size={14} weight=\"duotone\" aria-hidden /> reverse search
    </span>
  );
}
```

```tsx
// file: /app/frontend/src/components/ContentTypeBadge.tsx
import type { ContentType } from \"@types/api\";

const LABEL: Record<ContentType, string> = {
  selfie_portrait: \"selfie / portrait\",
  landscape_scene: \"landscape\",
  object_product: \"object / product\",
  meme_screenshot: \"meme / screenshot\",
  document_scan: \"document scan\",
  artwork_illustration: \"artwork\",
};

export default function ContentTypeBadge({ type }: { type: ContentType }): JSX.Element {
  return (
    <span
      data-testid=\"content-type-badge\"
      className=\"inline-flex items-center gap-1 rounded-pill bg-bg-2 border border-bg-3 px-3 py-1 text-xs font-mono text-fg-1\"
    >
      {LABEL[type]}
    </span>
  );
}
```

### 14.5 `ConfidenceAgreementBars.tsx`

```tsx
// file: /app/frontend/src/components/ConfidenceAgreementBars.tsx
import type { JobResult } from \"@types/api\";
import { pct } from \"@lib/format\";

interface Props { result: JobResult; }

export default function ConfidenceAgreementBars({ result }: Props): JSX.Element {
  return (
    <section
      data-testid=\"confidence-agreement-section\"
      className=\"grid grid-cols-1 md:grid-cols-3 gap-4 rounded-card border border-bg-3 bg-bg-1 p-4\"
      aria-label=\"Confidence and agreement\"
    >
      <Bar label=\"Confidence\" value={result.confidence} testId=\"confidence-progress-bar\" />
      <Bar label=\"Agreement\"  value={result.agreement}  testId=\"agreement-progress-bar\" />
      <Bar label=\"Extremity\"  value={result.extremity}  testId=\"extremity-progress-bar\" />
    </section>
  );
}

function Bar({ label, value, testId }: { label: string; value: number; testId: string }): JSX.Element {
  const v = Math.max(0, Math.min(1, value));
  return (
    <div>
      <div className=\"flex items-baseline justify-between mb-1\">
        <span className=\"text-xs font-mono uppercase tracking-widest text-fg-2\">{label}</span>
        <span className=\"text-sm font-mono text-fg-0\">{pct(v, 1)}</span>
      </div>
      <div
        role=\"progressbar\"
        aria-valuemin={0} aria-valuemax={100} aria-valuenow={v * 100}
        aria-label={label}
        data-testid={testId}
        className=\"h-2 rounded-pill bg-bg-2 overflow-hidden\"
      >
        <div
          className=\"h-full bg-brand transition-all duration-500 ease-out\"
          style={{ width: `${v * 100}%` }}
        />
      </div>
    </div>
  );
}
```

### 14.6 `NarrativePanel.tsx`

```tsx
// file: /app/frontend/src/components/NarrativePanel.tsx
import type { JobResult } from \"@types/api\";
import { Lightning } from \"@phosphor-icons/react\";

export default function NarrativePanel({ result }: { result: JobResult }): JSX.Element {
  const text = result.xai.narrative;
  const src = result.xai.narrative_source;
  return (
    <section
      aria-labelledby=\"narrative-h\"
      className=\"rounded-card border border-bg-3 bg-bg-1 p-5\"
      data-testid=\"narrative-panel\"
    >
      <div className=\"flex items-center justify-between mb-2\">
        <h3 id=\"narrative-h\" className=\"font-display font-semibold flex items-center gap-2\">
          <Lightning size={18} weight=\"duotone\" className=\"text-brand\" aria-hidden />
          Narrative
        </h3>
        <span className=\"text-xs font-mono text-fg-2\" data-testid=\"narrative-source\">
          {src === \"gemini\" ? \"gemini\" : \"rule-based fallback\"}
        </span>
      </div>
      <p className=\"text-fg-0 leading-relaxed\" data-testid=\"narrative-text\">{text}</p>
    </section>
  );
}
```

### 14.7 `SignalBarChart.tsx`

```tsx
// file: /app/frontend/src/components/SignalBarChart.tsx
import { BarChart, Bar, XAxis, YAxis, Tooltip, Cell, ResponsiveContainer } from \"recharts\";
import type { SignalRow } from \"@types/api\";
import { useMemo } from \"react\";

interface Props { signals: SignalRow[]; }

export default function SignalBarChart({ signals }: Props): JSX.Element {
  const data = useMemo(
    () => signals
      .filter((s) => Number.isFinite(s.p_fake))
      .map((s) => ({ ...s, contribution: s.p_fake * s.weight })),
    [signals],
  );

  if (data.length === 0) {
    return (
      <section className=\"rounded-card border border-bg-3 bg-bg-1 p-5\" data-testid=\"signal-bar-empty\">
        <p className=\"text-fg-2 text-sm\">No signal data available.</p>
      </section>
    );
  }

  return (
    <section
      aria-labelledby=\"signals-h\"
      className=\"rounded-card border border-bg-3 bg-bg-1 p-5\"
      data-testid=\"signal-bar-chart-section\"
    >
      <h3 id=\"signals-h\" className=\"font-display font-semibold mb-4\">Per-signal P(fake)</h3>
      <div data-testid=\"chart-signal-bars\" style={{ width: \"100%\", height: 28 * data.length + 40 }}>
        <ResponsiveContainer width=\"100%\" height=\"100%\">
          <BarChart data={data} layout=\"vertical\" margin={{ left: 8, right: 16 }}>
            <XAxis
              type=\"number\" domain={[0, 1]} tick={{ fill: \"#A1A1AA\", fontSize: 11 }}
              stroke=\"#27272A\"
            />
            <YAxis
              type=\"category\" dataKey=\"name\" width={140}
              tick={{ fill: \"#A1A1AA\", fontSize: 11 }} stroke=\"#27272A\"
            />
            <Tooltip
              contentStyle={{ background: \"#121212\", border: \"1px solid #27272A\", borderRadius: 8 }}
              formatter={(v: number) => v.toFixed(2)}
            />
            <Bar dataKey=\"p_fake\" barSize={14} radius={[2, 2, 2, 2]}>
              {data.map((d, i) => (
                <Cell
                  key={d.name}
                  fill={d.p_fake >= 0.5 ? \"#EF4444\" : \"#10B981\"}
                  data-testid={`signal-bar-${d.name}`}
                  data-index={i}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </section>
  );
}
```

### 14.8 `HeatmapPanel.tsx`

```tsx
// file: /app/frontend/src/components/HeatmapPanel.tsx
import type { JobResult } from \"@types/api\";
import { assetUrl } from \"@lib/api\";

export default function HeatmapPanel({ result }: { result: JobResult }): JSX.Element {
  const url = result.xai.heatmap_url
    ? assetUrl(result.job_id, \"heatmap.png\")
    : null;

  return (
    <section
      aria-labelledby=\"heatmap-h\"
      className=\"rounded-card border border-bg-3 bg-bg-1 p-5\"
      data-testid=\"heatmap-panel\"
    >
      <h3 id=\"heatmap-h\" className=\"font-display font-semibold mb-3\">
        Forensic heatmap
      </h3>
      {url ? (
        <img
          src={url}
          alt={`Forensic GradCAM heatmap for job ${result.job_id}`}
          className=\"rounded-card max-h-96 w-auto mx-auto\"
          data-testid=\"heatmap-image\"
          loading=\"lazy\"
        />
      ) : (
        <p className=\"text-fg-2 text-sm\" data-testid=\"heatmap-empty\">
          No heatmap available for this verdict.
        </p>
      )}
    </section>
  );
}
```

### 14.9 `FrequencyPanel.tsx`

```tsx
// file: /app/frontend/src/components/FrequencyPanel.tsx
import type { JobResult } from \"@types/api\";
import { assetUrl } from \"@lib/api\";

export default function FrequencyPanel({ result }: { result: JobResult }): JSX.Element {
  const url = result.xai.frequency_plot_url
    ? assetUrl(result.job_id, \"fft.png\")
    : null;
  return (
    <section
      aria-labelledby=\"freq-h\"
      className=\"rounded-card border border-bg-3 bg-bg-1 p-5\"
      data-testid=\"frequency-panel\"
    >
      <h3 id=\"freq-h\" className=\"font-display font-semibold mb-3\">FFT radial profile</h3>
      {url ? (
        <img
          src={url}
          alt={`FFT radial frequency profile for job ${result.job_id}`}
          className=\"rounded-card max-h-80 w-auto mx-auto\"
          data-testid=\"chart-fft-radial\"
          loading=\"lazy\"
        />
      ) : (
        <p className=\"text-fg-2 text-sm\" data-testid=\"frequency-empty\">
          No frequency plot available.
        </p>
      )}
    </section>
  );
}
```

### 14.10 `MetadataTable.tsx`

```tsx
// file: /app/frontend/src/components/MetadataTable.tsx
import type { JobResult } from \"@types/api\";
import { bytes } from \"@lib/format\";

export default function MetadataTable({ result }: { result: JobResult }): JSX.Element {
  const rows: Array<[string, string]> = [
    [\"filename\", result.input.filename],
    [\"mime\", result.input.mime],
    [\"bytes\", bytes(result.input.bytes)],
    [\"sha256\", result.input.sha256.slice(0, 16) + \"…\"],
    [\"profile\", result.profile],
    [\"calibration\", result.calibration],
    [\"fusion\", result.fusion_model],
  ];
  return (
    <section
      aria-labelledby=\"meta-h\"
      className=\"rounded-card border border-bg-3 bg-bg-1 p-5\"
      data-testid=\"metadata-technical-table\"
    >
      <h3 id=\"meta-h\" className=\"font-display font-semibold mb-3\">Technical metadata</h3>
      <dl className=\"grid grid-cols-[max-content_1fr] gap-x-4 gap-y-2 text-sm font-mono\">
        {rows.map(([k, v]) => (
          <div className=\"contents\" key={k} data-testid={`meta-row-${k}`}>
            <dt className=\"text-fg-2 uppercase text-xs tracking-widest\">{k}</dt>
            <dd className=\"text-fg-0 break-all\">{v}</dd>
          </div>
        ))}
      </dl>
    </section>
  );
}
```

### 14.11 `CompressionFingerprintPanel.tsx`

```tsx
// file: /app/frontend/src/components/CompressionFingerprintPanel.tsx
import type { CompressionFingerprint } from \"@types/api\";

const FLAG_COLOR: Record<CompressionFingerprint[\"flag\"], string> = {
  ai_signature: \"text-verdict-ai\",
  camera_signature: \"text-verdict-real\",
  neutral: \"text-fg-2\",
};

export default function CompressionFingerprintPanel({ fp }: { fp: CompressionFingerprint }): JSX.Element {
  return (
    <section
      aria-labelledby=\"cfp-h\"
      className=\"rounded-card border border-bg-3 bg-bg-1 p-5\"
      data-testid=\"compression-fingerprint-panel\"
    >
      <h3 id=\"cfp-h\" className=\"font-display font-semibold mb-2\">Compression fingerprint</h3>
      <div className={`text-xs font-mono uppercase tracking-widest mb-3 ${FLAG_COLOR[fp.flag]}`}>
        container: {fp.container} · {fp.flag.replace(\"_\", \" \")}
      </div>
      <table className=\"w-full text-xs font-mono\">
        <tbody>
          {Object.entries(fp.fingerprint).map(([k, v]) => (
            <tr key={k} className=\"border-b border-bg-3 last:border-0\">
              <td className=\"py-1 pr-2 text-fg-2\">{k}</td>
              <td className=\"py-1 text-fg-0 break-all\">{String(v)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}
```

### 14.12 `RetrievalNeighborsPanel.tsx`

```tsx
// file: /app/frontend/src/components/RetrievalNeighborsPanel.tsx
import type { RetrievalNeighbor } from \"@types/api\";

interface Props { retrieval: { k: number; neighbors: RetrievalNeighbor[] }; }

export default function RetrievalNeighborsPanel({ retrieval }: Props): JSX.Element {
  const top = retrieval.neighbors.slice(0, 5);

  return (
    <section
      aria-labelledby=\"retr-h\"
      className=\"rounded-card border border-bg-3 bg-bg-1 p-5\"
      data-testid=\"retrieval-neighbors-panel\"
    >
      <h3 id=\"retr-h\" className=\"font-display font-semibold mb-1\">
        Reference DB neighbors
      </h3>
      <p className=\"text-fg-2 text-xs mb-4\">
        Top 5 of k={retrieval.k} nearest by CLIP cosine. Higher distance = less similar.
      </p>
      {top.length === 0 ? (
        <p className=\"text-fg-2 text-sm\" data-testid=\"retrieval-empty\">
          No neighbors returned.
        </p>
      ) : (
        <ul className=\"grid grid-cols-2 md:grid-cols-5 gap-3\" role=\"list\">
          {top.map((n, i) => (
            <li
              key={n.id}
              className=\"rounded-card border border-bg-3 bg-bg-2 overflow-hidden\"
              data-testid={`retrieval-neighbor-${i}`}
            >
              <img
                src={n.thumb_url}
                alt={`refDB neighbor ${i + 1} labelled ${n.label}`}
                className=\"w-full aspect-square object-cover\"
                loading=\"lazy\"
              />
              <div className=\"px-2 py-2\">
                <div
                  className={`text-xs font-mono uppercase ${
                    n.label === \"ai\" ? \"text-verdict-ai\" : \"text-verdict-real\"
                  }`}
                >
                  {n.label}
                </div>
                <div className=\"text-[10px] font-mono text-fg-2 truncate\" title={n.source}>
                  {n.generator_family ?? n.source}
                </div>
                <div className=\"text-[10px] font-mono text-fg-2\">
                  d={n.distance.toFixed(3)}
                </div>
              </div>
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}
```

### 14.13 `ReverseSearchPanel.tsx`

```tsx
// file: /app/frontend/src/components/ReverseSearchPanel.tsx
import type { ReverseHit } from \"@types/api\";

interface Props {
  reverse: { hits: ReverseHit[]; reason?: string };
}

export default function ReverseSearchPanel({ reverse }: Props): JSX.Element {
  const hits = reverse.hits.slice(0, 5);
  return (
    <section
      aria-labelledby=\"rev-h\"
      className=\"rounded-card border border-bg-3 bg-bg-1 p-5\"
      data-testid=\"reverse-search-panel\"
    >
      <h3 id=\"rev-h\" className=\"font-display font-semibold mb-1\">Reverse-search hits</h3>
      {reverse.reason && (
        <p className=\"text-xs font-mono text-fg-2 mb-2\" data-testid=\"reverse-reason\">
          interpretation: {reverse.reason}
        </p>
      )}
      {hits.length === 0 ? (
        <p className=\"text-fg-2 text-sm\" data-testid=\"reverse-empty\">
          No hits returned.
        </p>
      ) : (
        <ol className=\"flex flex-col gap-2 text-sm\">
          {hits.map((h, i) => (
            <li
              key={`${h.url}-${i}`}
              data-testid={`reverse-hit-${i}`}
              className=\"rounded-card border border-bg-3 bg-bg-2 px-3 py-2 flex items-center gap-3\"
            >
              <div className=\"flex-1 min-w-0\">
                <a
                  href={h.url}
                  target=\"_blank\"
                  rel=\"noreferrer noopener\"
                  className=\"text-brand hover:text-brand-hover truncate block\"
                >
                  {h.title ?? h.domain}
                </a>
                <div className=\"text-xs font-mono text-fg-2 truncate\">
                  {h.domain}{h.date ? ` · ${h.date}` : \"\"}
                </div>
              </div>
              {h.thumbnail && (
                <img
                  src={h.thumbnail}
                  alt=\"\"
                  className=\"w-12 h-12 rounded object-cover\"
                  loading=\"lazy\"
                  aria-hidden
                />
              )}
            </li>
          ))}
        </ol>
      )}
    </section>
  );
}
```

### 14.14 `VLMRationalePanel.tsx`

```tsx
// file: /app/frontend/src/components/VLMRationalePanel.tsx
import { Brain } from \"@phosphor-icons/react\";
import type { VLMRationale } from \"@types/api\";
import { pct } from \"@lib/format\";

export default function VLMRationalePanel({ vlm }: { vlm: VLMRationale }): JSX.Element {
  return (
    <section
      aria-labelledby=\"vlm-h\"
      className=\"rounded-card border border-bg-3 bg-bg-1 p-5\"
      data-testid=\"vlm-rationale-panel\"
    >
      <h3 id=\"vlm-h\" className=\"font-display font-semibold mb-1 flex items-center gap-2\">
        <Brain size={18} weight=\"duotone\" className=\"text-brand\" aria-hidden />
        VLM rationale
      </h3>
      <p className=\"text-xs font-mono text-fg-2 mb-3\">
        Gemini 3 Flash rated this {pct(vlm.p_ai, 1)} AI-likely.
      </p>
      <ul className=\"list-disc list-inside flex flex-col gap-1 text-sm\" data-testid=\"vlm-rationale-list\">
        {vlm.defects.length === 0 ? (
          <li className=\"text-fg-2 list-none\" data-testid=\"vlm-defects-empty\">
            No specific defects cited.
          </li>
        ) : (
          vlm.defects.map((d, i) => (
            <li key={i} data-testid={`vlm-defect-${i}`}>{d}</li>
          ))
        )}
      </ul>
      <p className=\"text-sm text-fg-1 mt-3\" data-testid=\"vlm-rationale-text\">
        {vlm.rationale}
      </p>
    </section>
  );
}
```

### 14.15 `CorrectVerdictBar.tsx`

```tsx
// file: /app/frontend/src/components/CorrectVerdictBar.tsx
import { useState } from \"react\";
import { useMutation } from \"@tanstack/react-query\";
import { CheckCircle, X } from \"@phosphor-icons/react\";
import { postCorrect } from \"@lib/api\";

export default function CorrectVerdictBar({ jobId }: { jobId: string }): JSX.Element {
  const [done, setDone] = useState<\"ai\" | \"real\" | null>(null);
  const mut = useMutation({
    mutationFn: (label: \"ai\" | \"real\") => postCorrect(jobId, label),
    onSuccess: (_, label) => setDone(label),
  });

  if (done) {
    return (
      <div
        data-testid=\"correct-verdict-confirmation\"
        className=\"rounded-card border border-ok/40 bg-ok/10 p-4 flex items-center gap-2 text-sm\"
      >
        <CheckCircle size={18} className=\"text-ok\" aria-hidden />
        Thanks — labeled as <span className=\"font-mono text-ok\">{done}</span>. The reference DB
        has been updated; future predictions improve immediately.
      </div>
    );
  }

  return (
    <div
      data-testid=\"correct-verdict-bar\"
      className=\"rounded-card border border-bg-3 bg-bg-1 p-4 flex items-center justify-between gap-3\"
    >
      <span className=\"text-fg-1 text-sm\">
        Was the verdict correct? Help calibrate the system.
      </span>
      <div className=\"flex gap-2\">
        <button
          data-testid=\"correct-verdict-real-btn\"
          onClick={() => mut.mutate(\"real\")}
          disabled={mut.isPending}
          className=\"inline-flex items-center gap-1 rounded-pill border border-verdict-real/50 px-3 py-1 text-sm text-verdict-real hover:bg-verdict-real/15 disabled:opacity-50 transition-colors\"
        >
          <CheckCircle size={14} weight=\"duotone\" aria-hidden /> mark REAL
        </button>
        <button
          data-testid=\"correct-verdict-ai-btn\"
          onClick={() => mut.mutate(\"ai\")}
          disabled={mut.isPending}
          className=\"inline-flex items-center gap-1 rounded-pill border border-verdict-ai/50 px-3 py-1 text-sm text-verdict-ai hover:bg-verdict-ai/15 disabled:opacity-50 transition-colors\"
        >
          <X size={14} weight=\"bold\" aria-hidden /> mark AI
        </button>
      </div>
      {mut.error && (
        <p role=\"alert\" data-testid=\"correct-verdict-error\" className=\"text-err text-xs\">
          Failed to submit.
        </p>
      )}
    </div>
  );
}
```

### 14.16 `DeveloperPanel.tsx`

The single most important debug surface. Renders **only** when dev mode is
on AND `result.debug` is present.

```tsx
// file: /app/frontend/src/components/DeveloperPanel.tsx
import { useState } from \"react\";
import { Terminal } from \"@phosphor-icons/react\";
import type { JobResult, SignalRow } from \"@types/api\";
import { setDevMode } from \"@lib/devmode\";

interface Props { result: JobResult; }

export default function DeveloperPanel({ result }: Props): JSX.Element {
  const dbg = result.debug;
  const [high, setHigh] = useState<number>(0.75);
  const [low, setLow]   = useState<number>(0.25);
  const [agree, setAgr] = useState<number>(0.55);

  const liveVerdict = computeLiveVerdict(result, high, low, agree);

  return (
    <section
      aria-labelledby=\"dev-h\"
      data-testid=\"developer-panel\"
      className=\"rounded-card border border-brand/40 bg-bg-1 p-5\"
    >
      <div className=\"flex items-center justify-between mb-3\">
        <h3 id=\"dev-h\" className=\"font-display font-semibold flex items-center gap-2 text-brand\">
          <Terminal size={18} weight=\"duotone\" aria-hidden /> Developer mode
        </h3>
        <button
          data-testid=\"dev-mode-toggle-off\"
          onClick={() => setDevMode(false)}
          className=\"text-xs font-mono text-fg-2 hover:text-fg-0 transition-colors\"
        >
          [exit]
        </button>
      </div>

      {/* Raw signals */}
      <div>
        <h4 className=\"text-xs font-mono uppercase tracking-widest text-fg-2 mb-2\">
          Raw signal table
        </h4>
        <table className=\"w-full text-xs font-mono\">
          <thead className=\"text-fg-2\">
            <tr>
              <th className=\"text-left\">name</th>
              <th className=\"text-right\">raw</th>
              <th className=\"text-right\">calib</th>
              <th className=\"text-right\">w</th>
              <th className=\"text-right\">w·p</th>
            </tr>
          </thead>
          <tbody>
            {result.signals.map((s: SignalRow) => (
              <tr key={s.name} data-testid={`dev-raw-signal-row-${s.name}`} className=\"border-t border-bg-3\">
                <td className=\"py-1 text-fg-0\">{s.name}</td>
                <td className=\"py-1 text-right text-fg-1\">{(s.raw ?? NaN).toFixed?.(3) ?? \"—\"}</td>
                <td className=\"py-1 text-right text-fg-1\">{s.p_fake.toFixed(3)}</td>
                <td className=\"py-1 text-right text-fg-1\">{s.weight.toFixed(3)}</td>
                <td className=\"py-1 text-right text-fg-0\">{(s.p_fake * s.weight).toFixed(3)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Threshold sliders — client-side re-render only */}
      <div className=\"mt-4\">
        <h4 className=\"text-xs font-mono uppercase tracking-widest text-fg-2 mb-2\">
          Threshold overrides (client-side preview)
        </h4>
        <ThresholdSlider name=\"high\" value={high} setValue={setHigh} />
        <ThresholdSlider name=\"low\"   value={low}  setValue={setLow}  />
        <ThresholdSlider name=\"agree\" value={agree} setValue={setAgr} />
        <div
          data-testid=\"dev-live-verdict\"
          className=\"mt-2 text-sm font-mono px-3 py-2 rounded border border-bg-3 bg-bg-2\"
        >
          live preview: <span className=\"text-brand\">{liveVerdict}</span>
        </div>
      </div>

      {/* Durations */}
      <div className=\"mt-4\">
        <h4 className=\"text-xs font-mono uppercase tracking-widest text-fg-2 mb-2\">
          Stage durations (ms)
        </h4>
        <ul className=\"grid grid-cols-2 gap-x-4 gap-y-1 text-xs font-mono\">
          {Object.entries(result.durations_ms).map(([k, v]) => (
            <li key={k} className=\"flex justify-between border-b border-bg-3 py-0.5\">
              <span className=\"text-fg-2\">{k}</span>
              <span className=\"text-fg-0\">{v}</span>
            </li>
          ))}
        </ul>
      </div>

      {/* Gate states */}
      {dbg && (
        <div className=\"mt-4\">
          <h4 className=\"text-xs font-mono uppercase tracking-widest text-fg-2 mb-2\">
            Gate states
          </h4>
          <pre
            data-testid=\"dev-gate-state-dump\"
            className=\"text-xs font-mono bg-bg-2 border border-bg-3 rounded p-3 overflow-auto max-h-48\"
          >
            {JSON.stringify(dbg.gate_states, null, 2)}
          </pre>
        </div>
      )}

      {/* Fusion vector */}
      {dbg && (
        <div className=\"mt-4\">
          <h4 className=\"text-xs font-mono uppercase tracking-widest text-fg-2 mb-2\">
            Fusion vector
          </h4>
          <pre
            data-testid=\"dev-fusion-vector-dump\"
            className=\"text-xs font-mono bg-bg-2 border border-bg-3 rounded p-3 overflow-auto max-h-48\"
          >
{`keys : ${JSON.stringify(dbg.fusion_vector_keys)}
mask : ${JSON.stringify(dbg.fusion_vector_mask)}
vec  : ${JSON.stringify(dbg.fusion_vector.map((x) => +x.toFixed(3)))}`}
          </pre>
        </div>
      )}
    </section>
  );
}

function ThresholdSlider({
  name, value, setValue,
}: { name: \"high\" | \"low\" | \"agree\"; value: number; setValue: (v: number) => void }): JSX.Element {
  return (
    <label className=\"flex items-center gap-3 text-xs font-mono mb-1\">
      <span className=\"w-14 text-fg-2 uppercase\">{name}</span>
      <input
        type=\"range\" min={0} max={1} step={0.01} value={value}
        onChange={(e) => setValue(parseFloat(e.target.value))}
        data-testid={`threshold-slider-${name}`}
        className=\"flex-1 accent-brand\"
        aria-label={`${name} threshold`}
      />
      <span className=\"w-12 text-right text-fg-0\">{value.toFixed(2)}</span>
    </label>
  );
}

function computeLiveVerdict(r: JobResult, high: number, low: number, agree: number): string {
  if (r.provenance.hit) return r.verdict;  // fixed by Tier 0
  if (r.p_ai_generated >= high && r.agreement >= agree) return \"AI-GENERATED\";
  if (r.p_ai_generated <= low  && r.agreement >= agree) return \"REAL\";
  return \"INCONCLUSIVE\";
}
```

### 14.17 `HistoryList.tsx`

```tsx
// file: /app/frontend/src/components/HistoryList.tsx
import { Link } from \"react-router-dom\";
import type { UseQueryResult } from \"@tanstack/react-query\";
import type { JobStatusEnvelope } from \"@types/api\";

interface Props {
  query: UseQueryResult<{ items: JobStatusEnvelope[] }, unknown>;
}

export default function HistoryList({ query }: Props): JSX.Element {
  if (query.isLoading) {
    return <p className=\"text-fg-2 text-sm\" data-testid=\"history-loading\">Loading recent jobs…</p>;
  }
  if (query.error) {
    return (
      <p role=\"alert\" className=\"text-err text-sm\" data-testid=\"history-error\">
        Could not load history.
      </p>
    );
  }
  const items = query.data?.items ?? [];
  if (items.length === 0) {
    return (
      <p className=\"text-fg-2 text-sm\" data-testid=\"history-empty\">
        No jobs yet. Upload something to begin.
      </p>
    );
  }
  return (
    <ul className=\"grid grid-cols-1 md:grid-cols-2 gap-3\" data-testid=\"history-list\">
      {items.map((j) => (
        <li key={j.job_id}>
          <Link
            to={`/job/${j.job_id}`}
            data-testid={`history-item-${j.job_id}`}
            className=\"block rounded-card border border-bg-3 bg-bg-1 px-4 py-3 hover:border-brand transition-colors\"
          >
            <div className=\"flex items-center justify-between\">
              <span className=\"font-mono text-xs text-fg-2\">
                {new Date(j.started_at).toLocaleString()}
              </span>
              <span
                className={`text-xs font-mono uppercase tracking-widest ${
                  j.status === \"done\"
                    ? \"text-ok\"
                    : j.status === \"failed\"
                    ? \"text-err\"
                    : \"text-brand\"
                }`}
              >
                {j.status}
              </span>
            </div>
            <div className=\"text-sm mt-1\">{j.modality} · {j.job_id.slice(0, 8)}</div>
          </Link>
        </li>
      ))}
    </ul>
  );
}
```

---

## 15. About page (plain-English COEF explainer)

```tsx
// file: /app/frontend/src/pages/AboutPage.tsx
import { useQuery } from \"@tanstack/react-query\";
import { getHealth } from \"@lib/api\";
import { pct } from \"@lib/format\";

export default function AboutPage(): JSX.Element {
  const health = useQuery({ queryKey: [\"health\"], queryFn: getHealth, refetchInterval: 15_000 });

  return (
    <article className=\"prose prose-invert max-w-3xl animate-fade-in\">
      <h1 className=\"font-display\">How Argus reaches verdicts</h1>

      <h2>1. Five tiers of orthogonal evidence</h2>
      <p>
        No single AI-detector generalises to unseen generators. Argus runs five independent
        tiers — provenance gates, forensic + learned detectors, retrieval against a curated
        reference DB, reverse image search, and a Vision-Language model tiebreaker — and
        fuses their evidence with explicit per-signal calibration.
      </p>

      <h2>2. Headline KPI</h2>
      <p>
        <strong>≥95 % accuracy on the non-abstained share of uploads</strong>, with abstention
        as a tunable knob. We say <code>INCONCLUSIVE</code> instead of being confidently wrong.
      </p>

      <h2>3. What we do not do</h2>
      <ul>
        <li>We do not fine-tune any model.</li>
        <li>We do not store user authentication; this is a single-user local console.</li>
        <li>We do not claim robustness against targeted adversarial attacks.</li>
      </ul>

      <h2>4. Live calibration health</h2>
      <section
        data-testid=\"about-health-panel\"
        className=\"not-prose rounded-card border border-bg-3 bg-bg-1 p-4 font-mono text-sm\"
      >
        {health.isLoading && <p data-testid=\"about-health-loading\">loading…</p>}
        {health.error && <p data-testid=\"about-health-error\">unavailable</p>}
        {health.data && (
          <dl className=\"grid grid-cols-[max-content_1fr] gap-x-4 gap-y-1\">
            <dt className=\"text-fg-2\">profile</dt><dd>{health.data.profile}</dd>
            <dt className=\"text-fg-2\">calibration</dt><dd>{health.data.calibration}</dd>
            <dt className=\"text-fg-2\">ECE (refDB holdout)</dt><dd>{health.data.ece_refdb_holdout.toFixed(3)}</dd>
            <dt className=\"text-fg-2\">AUROC (refDB holdout)</dt><dd>{pct(health.data.auroc_refdb_holdout, 1)}</dd>
            <dt className=\"text-fg-2\">refDB size</dt>
            <dd>{Object.entries(health.data.refdb_size).map(([k, v]) => `${k}=${v}`).join(\" · \")}</dd>
            <dt className=\"text-fg-2\">fusion mode</dt><dd>{health.data.fusion_mode}</dd>
            <dt className=\"text-fg-2\">labels collected</dt><dd>{health.data.n_user_labels}</dd>
            <dt className=\"text-fg-2\">signals loaded</dt><dd>{health.data.signals_loaded.length}</dd>
          </dl>
        )}
      </section>
    </article>
  );
}
```

---

## 16. `data-testid` registry (kebab-case, mandatory)

Every entry below MUST exist in the DOM when the relevant view renders.
Referenced by E2E tests in §18.

```
# Layout
header-home-link · nav-upload · nav-about

# Upload
media-upload-dropzone · media-upload-input · dropzone-error
how-step-0 … how-step-4
history-list · history-loading · history-error · history-empty
history-item-<job_id>

# Progress
job-progress-steps · progress-step-preprocess · progress-step-tier0 …
progress-step-xai · progress-status-pill

# Result
result-view · verdict-card-container · verdict-label · verdict-p-ai
verdict-confidence · verdict-abstained · verdict-novel-suspected
content-type-badge · provenance-badge · vlm-invoked-badge · reverse-search-badge

confidence-progress-bar · agreement-progress-bar · extremity-progress-bar
narrative-panel · narrative-source · narrative-text
signal-bar-chart-section · chart-signal-bars · signal-bar-<name>
heatmap-panel · heatmap-image · heatmap-empty
frequency-panel · chart-fft-radial · frequency-empty
metadata-technical-table · meta-row-<key>
compression-fingerprint-panel
retrieval-neighbors-panel · retrieval-neighbor-<n> · retrieval-empty
reverse-search-panel · reverse-hit-<n> · reverse-reason · reverse-empty
vlm-rationale-panel · vlm-rationale-list · vlm-defect-<n> · vlm-rationale-text · vlm-defects-empty
correct-verdict-bar · correct-verdict-real-btn · correct-verdict-ai-btn
correct-verdict-confirmation · correct-verdict-error

# Developer mode
developer-panel · dev-mode-toggle-off
dev-raw-signal-row-<name>
threshold-slider-high · threshold-slider-low · threshold-slider-agree
dev-live-verdict · dev-gate-state-dump · dev-fusion-vector-dump

# About
about-health-panel · about-health-loading · about-health-error
```

---

## 17. Vitest setup + sample unit test

```ts
// file: /app/frontend/src/tests/setup.ts
import \"@testing-library/jest-dom/vitest\";
import { afterEach } from \"vitest\";
import { cleanup } from \"@testing-library/react\";

afterEach(() => cleanup());
```

```ts
// file: /app/frontend/vitest.config.ts
import { defineConfig } from \"vitest/config\";
import react from \"@vitejs/plugin-react\";
import path from \"path\";

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      \"@\": path.resolve(__dirname, \"src\"),
      \"@components\": path.resolve(__dirname, \"src/components\"),
      \"@pages\": path.resolve(__dirname, \"src/pages\"),
      \"@lib\": path.resolve(__dirname, \"src/lib\"),
      \"@types\": path.resolve(__dirname, \"src/types\"),
    },
  },
  test: {
    environment: \"jsdom\",
    globals: true,
    setupFiles: [\"src/tests/setup.ts\"],
    coverage: {
      provider: \"v8\",
      reporter: [\"text\", \"html\", \"json-summary\"],
      thresholds: { lines: 80, statements: 80, branches: 70, functions: 80 },
      exclude: [\"**/*.test.tsx\", \"src/tests/**\", \"src/index.tsx\"],
    },
  },
});
```

```tsx
// file: /app/frontend/src/components/VerdictCard.test.tsx
import { render, screen } from \"@testing-library/react\";
import { describe, it, expect } from \"vitest\";
import VerdictCard from \"./VerdictCard\";
import type { JobResult } from \"@types/api\";

const base: JobResult = {
  job_id: \"abc\", modality: \"image\", profile: \"cloud_lite\",
  calibration: \"platt_refdb\", fusion_model: \"uniform\", content_type: \"selfie_portrait\",
  verdict: \"AI-GENERATED\", p_ai_generated: 0.86, confidence: 0.78, agreement: 0.83,
  extremity: 0.72, cross_modal_bonus: 0.06, abstained: false,
  provenance: { hit: false, source: \"none\" }, vlm_invoked: true, reverse_invoked: true,
  signals: [], retrieval: { k: 15, neighbors: [] }, reverse_search: null,
  xai: { heatmap_url: null, frequency_plot_url: null, metadata: {},
         compression_fingerprint: null, narrative: \"\", narrative_source: \"fallback_template\" },
  input: { filename: \"x.png\", sha256: \"0\".repeat(64), bytes: 1024, mime: \"image/png\" },
  durations_ms: {}, debug: null,
};

describe(\"VerdictCard\", () => {
  it(\"renders AI-GENERATED label and badges\", () => {
    render(<VerdictCard result={base} />);
    expect(screen.getByTestId(\"verdict-label\")).toHaveTextContent(\"AI-GENERATED\");
    expect(screen.getByTestId(\"vlm-invoked-badge\")).toBeInTheDocument();
    expect(screen.getByTestId(\"reverse-search-badge\")).toBeInTheDocument();
  });
  it(\"hides provenance badge when no hit\", () => {
    render(<VerdictCard result={base} />);
    expect(screen.queryByTestId(\"provenance-badge\")).toBeNull();
  });
  it(\"shows provenance badge when Tier-0 fires\", () => {
    render(<VerdictCard result={{ ...base, provenance: { hit: true, source: \"c2pa\" } }} />);
    expect(screen.getByTestId(\"provenance-badge\")).toBeInTheDocument();
  });
});
```

> Repeat for each component. Total unit suite target: **≥80 % coverage** on
> `src/components/`, `src/lib/`, `src/pages/`.

---

## 18. Playwright E2E + axe-core a11y

```ts
// file: /app/frontend/playwright.config.ts
import { defineConfig, devices } from \"@playwright/test\";

export default defineConfig({
  testDir: \"./src/tests/e2e\",
  timeout: 60_000,
  reporter: [[\"list\"], [\"html\", { open: \"never\", outputFolder: \"playwright-report\" }]],
  use: {
    baseURL: process.env.REACT_APP_BACKEND_URL?.replace(/\/api$/, \"\") ?? \"http://localhost:3000\",
    trace: \"on-first-retry\",
    screenshot: \"only-on-failure\",
  },
  projects: [
    { name: \"chromium\",       use: { ...devices[\"Desktop Chrome\"] } },
    { name: \"mobile-safari\",  use: { ...devices[\"iPhone 13\"] } },   // mobile-first verification
  ],
});
```

```ts
// file: /app/frontend/src/tests/e2e/smoke.spec.ts
import { test, expect } from \"@playwright/test\";
import AxeBuilder from \"@axe-core/playwright\";
import path from \"path\";

test(\"upload an image and reach the result view\", async ({ page }) => {
  await page.goto(\"/\");
  await expect(page.getByTestId(\"media-upload-dropzone\")).toBeVisible();

  const filePath = path.resolve(__dirname, \"../fixtures/real_photo.jpg\");
  await page.setInputFiles('[data-testid=\"media-upload-input\"]', filePath);

  await page.waitForURL(/\/job\//, { timeout: 60_000 });
  await expect(page.getByTestId(\"job-progress-steps\")).toBeVisible();

  // Poll until result view appears
  await expect(page.getByTestId(\"result-view\")).toBeVisible({ timeout: 60_000 });
  await expect(page.getByTestId(\"verdict-card-container\")).toBeVisible();
  await expect(page.getByTestId(\"signal-bar-chart-section\")).toBeVisible();
  await expect(page.getByTestId(\"metadata-technical-table\")).toBeVisible();
});

test(\"a11y on key pages (WCAG 2.1 AA, no critical violations)\", async ({ page }) => {
  for (const route of [\"/\", \"/about\"]) {
    await page.goto(route);
    const results = await new AxeBuilder({ page })
      .withTags([\"wcag2a\", \"wcag2aa\", \"wcag21aa\"])
      .analyze();
    const critical = results.violations.filter((v) => [\"serious\", \"critical\"].includes(v.impact ?? \"\"));
    expect(critical, JSON.stringify(critical, null, 2)).toHaveLength(0);
  }
});

test(\"developer mode toggle reveals raw signals\", async ({ page }) => {
  // Assume a completed job seeded by build script for deterministic E2E
  const jobId = process.env.E2E_SEED_JOB_ID;
  test.skip(!jobId, \"Set E2E_SEED_JOB_ID for this test\");
  await page.goto(`/job/${jobId}`);
  await page.keyboard.press(process.platform === \"darwin\" ? \"Meta+D\" : \"Control+D\");
  await expect(page.getByTestId(\"developer-panel\")).toBeVisible();
  await expect(page.getByTestId(\"dev-live-verdict\")).toBeVisible();
});
```

> Test fixtures live under `src/tests/e2e/fixtures/`: at minimum
> `real_photo.jpg` (camera-EXIF intact) and `ai_image.png` (SDXL export
> with no EXIF). Added in M0.

---

## 19. Bundle budget (P0 — AGENTS_FRONTEND.md §9)

Enforced by `webpack-bundle-analyzer` in CI:

| Slice | Limit |
|---|---|
| Initial JS | < 200 KB gzipped |
| Total per route | < 350 KB gzipped |
| LCP target | < 2.5 s |
| INP target | < 200 ms |

Mitigations baked into the design:

- Recharts loaded only on `JobPage` (route-split via `React.lazy`)
- Phosphor icons tree-shaken (named imports only)
- Images served via backend `assets/{name}` lazy-loaded
- TanStack Query staleTime 5 s prevents re-render storms

---

## 20. AGENTS_FRONTEND.md compliance map

| Rule | Where honored |
|---|---|
| §3 stack defaults | §1 with documented deviation rationale |
| §5 P0 failure modes | §17 coverage gate; §18 axe gate; §19 bundle gate; `tsc --noEmit` in pre-commit |
| §6 code quality | §6 ESLint + Prettier configs; naming conventions enforced |
| §7 component contract | §14 every component has loading/error/empty + ARIA + colocated test |
| §8 type safety P0 | §3 strict tsconfig; no `any`; types in §8 mirror Pydantic |
| §9 perf P0 | §19 bundle budget; route-split JobPage; lazy images |
| §10 a11y P0 | §5 focus ring; §16 `data-testid`s; §18 axe-core gate; semantic HTML |
| §11 state | §10 TanStack Query for server state; React state local |
| §12 API integration | §9 retry + timeout + envelope errors |
| §13 security | No localStorage of sensitive data (devmode flag is non-sensitive); CSP via backend headers; secrets backend-only |
| §14 testing P0 | §17 + §18 + ≥80 % coverage gate |
| §15 responsive P0 | mobile-first Tailwind; tap targets ≥ 44 px via `py-3` minimum on interactive |
| §16 error handling | every component has `data-testid` error + alert role |
| §17 SEO | semantic h1/h2/h3, alt text, title (added in `index.html`) |
| §19 organization | folders match Masterplan §15 |
| §20 assets | thumbnails lazy-loaded; SVG icons via Phosphor sprite |
| §21 forms | DropZone validates size client-side + server-side |
| §22 animation | CSS keyframes only; `prefers-reduced-motion` honored in `globals.css` |
| §23 DX | pre-commit hook: lint+typecheck+test (added in `12_scripts_and_testing.md`) |
| §25 non-negotiables | mobile-first ✓ · a11y ✓ · loading/error/empty ✓ · perf budget ✓ · security ✓ · coverage ✓ · no-`any` ✓ |

---

## 21. Implementation order (frontend slice of M0→M3)

1. **M0:** TS migration (§2), `tsconfig.json` (§3), `tailwind.config.js` (§5), `globals.css` (§5), ESLint/Prettier (§6), folder scaffold (§7), API types stub (§8), `lib/api.ts` (§9), `lib/format.ts` (§10), `lib/devmode.ts` (§10), `App.tsx` + routing (§11), empty `UploadPage`/`JobPage`/`AboutPage` shells. `tsc --noEmit` green. Vitest smoke test (`expect(true).toBe(true)`) green.
2. **M1 (frontend slice):** `DropZone`, `ProgressSteps`, `VerdictCard`, `ConfidenceAgreementBars`, `NarrativePanel`, `SignalBarChart`, `HeatmapPanel`, `FrequencyPanel`, `MetadataTable`, `HistoryList`, `UploadPage`, `JobPage`, basic `AboutPage`. Vitest coverage ≥ 80 % on these. Axe-core passes on `/` and `/about`.
3. **M2 (frontend slice):** `ProvenanceBadge`. No new pages; `VerdictCard` renders the badge.
4. **M3 (frontend slice):** `VLMBadge`, `ReverseSearchBadge`, `ContentTypeBadge`, `CompressionFingerprintPanel`, `RetrievalNeighborsPanel`, `ReverseSearchPanel`, `VLMRationalePanel`, `CorrectVerdictBar`, `DeveloperPanel`. Final About-page health surface. Playwright E2E (§18) green. Mobile Safari project (§18) green. **→ First finish.**

---

End of `11_frontend.md`. Source of truth for the entire React 19 + TS-strict
implementation. Code blocks are copy-paste; any deviation must be logged in
`/app/memory/PRD.md` under \"Implementation Notes\" per `AGENTS.md §9`.
"