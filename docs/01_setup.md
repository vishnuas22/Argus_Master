"# 01 — Setup & Dependencies (outline, no boilerplate)

> Goal: minimum viable setup so M0 (`server.py` boots, `/api/health` returns 200) works. Everything beyond M0 is in the file-specific docs.

---

## 1. Tech stack (locked)

| Layer | Choice | Why |
|---|---|---|
| Backend | FastAPI (async) + Uvicorn + Motor (Mongo) | matches platform default; async everywhere |
| Detectors | PyTorch 2.2+, transformers, faiss-cpu, scikit-learn, lightgbm | CPU-first; MPS/CUDA toggled by `services/device.py` |
| Forensics | numpy, scipy, opencv-python-headless, Pillow, exifread, pywavelets | training-free signals |
| Provenance | c2pa-python, invisible-watermark, synthid-text (guarded) | watermark / C2PA gate |
| LLM | `emergentintegrations` (single library for Gemini text+vision) | one wrapper, two use-cases |
| Reverse search | `requests` direct call to SerpAPI | no extra SDK needed |
| Frontend | React 19 + craco + Tailwind + shadcn/ui + Recharts + @phosphor-icons/react | Control Room aesthetic |
| Storage | MongoDB (jobs, results, labels, serpapi_cache) + filesystem (uploads, models, refDB) | platform default |

---

## 2. Install order (one shell per block)

### 2.1 Backend Python deps
```bash
cd /app/backend
pip install \
  torch torchvision torchaudio \
  transformers huggingface_hub safetensors accelerate \
  scikit-learn lightgbm faiss-cpu \
  opencv-python-headless Pillow imageio pywavelets matplotlib \
  exifread c2pa invisible-watermark \
  onnxruntime optimum \
  requests emergentintegrations \
  beautifulsoup4 \
  ruff mypy pytest pytest-cov pytest-asyncio httpx python-magic
pip freeze > /app/backend/requirements.txt
```

> `synthid-text` is guarded — import inside the detector module with `try/except ImportError`. Same for `mediapipe`, `insightface`, `scenedetect` when audio/video added later.

### 2.2 emergentintegrations install (mandatory if pip cannot find it)
```bash
pip install emergentintegrations --extra-index-url https://d33sy5i8bnduwe.cloudfront.net/simple/
```

### 2.3 Frontend deps
```bash
cd /app/frontend
yarn add @phosphor-icons/react
# recharts, sonner, lucide-react, shadcn/ui are already present in the template
```

### 2.4 Supervisor restart
```bash
sudo supervisorctl restart backend frontend
```

---

## 3. `.env` outline (structure only — no values)

### 3.1 `/app/backend/.env`
```
# --- Protected: do not edit ---
MONGO_URL=
DB_NAME=
CORS_ORIGINS=

# --- LLM (Gemini) ---
# User chooses ONE of the two next lines (both supported by emergentintegrations):
GEMINI_API_KEY=          # user's own key (preferred — no shared quota)
EMERGENT_LLM_KEY=        # Universal Key fallback (sk-emergent-...)

# --- Reverse image search ---
SERPAPI_KEY=             # SerpAPI free-tier key (100 searches/month)

# --- Profile / device ---
DETECTOR_PROFILE=auto    # auto | cloud_lite | mac_full | cuda_full
TORCH_DEVICE=auto
HF_HOME=/app/backend/storage/models
HF_TOKEN=                # optional, only for HF Inference API fallback on cloud_lite

# --- Feature flags ---
ENABLE_VLM_TIEBREAKER=true
ENABLE_VLM_SECOND_OPINION=true   # NEW v1.3.1
ENABLE_REVERSE_SEARCH=true
ENABLE_DIRE_MPS=false

# --- Gates ---
VLM_EXTREMITY_THRESHOLD=0.25
VLM_AGREEMENT_THRESHOLD=0.63
REVERSE_EXTREMITY_THRESHOLD=0.30
REVERSE_AGREEMENT_THRESHOLD=0.70

# --- Abstention defaults (per-content-type override in code) ---
ABSTAIN_HIGH=0.75
ABSTAIN_LOW=0.25
ABSTAIN_AGREE=0.55

# --- Upload limits ---
MAX_UPLOAD_MB=200
VIDEO_MAX_SECONDS=120
```

### 3.2 `/app/frontend/.env`
```
REACT_APP_BACKEND_URL=
WDS_SOCKET_PORT=443
```
> `REACT_APP_BACKEND_URL` is pre-populated by the platform; do not modify.

---

## 4. Directory tree to create up-front (M0)

```bash
cd /app/backend && mkdir -p \
  routes schemas services provenance detectors detectors/image \
  retrieval reverse_search vlm fusion abstention xai db utils scripts \
  calibration storage/models storage/refdb/thumbs storage/cache/serpapi \
  storage/jobs tests/unit tests/integration tests/fixtures
```

```bash
cd /app/frontend/src && mkdir -p \
  components pages lib styles
```

---

## 5. Where to obtain API keys (paste this in README)

| Key | Where | Free quota |
|---|---|---|
| `GEMINI_API_KEY` | https://aistudio.google.com/apikey | ~1500 requests/day on Flash |
| `EMERGENT_LLM_KEY` | Emergent profile → Universal Key | shared platform budget |
| `SERPAPI_KEY` | https://serpapi.com/manage-api-key | 100 searches/month free |
| `HF_TOKEN` *(optional)* | https://huggingface.co/settings/tokens | 30k/mo on Inference API |

---

## 6. First-boot smoke test (after M0 skeleton)

```bash
# Backend up
curl -s http://localhost:8001/api/health | python3 -m json.tool

# Expected (partial):
# {
#   \"status\": \"ok\",
#   \"profile\": \"cloud_lite\",
#   \"db_ok\": true,
#   \"gemini_ok\": false,        # until GEMINI_API_KEY set
#   \"serpapi_ok\": false,       # until SERPAPI_KEY set
#   \"refdb_loaded\": false      # until build_reference_db.py run
# }
```

If `db_ok=false` → `sudo supervisorctl restart mongodb` and re-curl.

---

## 7. AGENTS.md mapping for this file

| Standard | Where honored |
|---|---|
| 12-factor config | All runtime knobs in `.env`; no hardcoded constants |
| Secrets management | `.env` only; never in code, never in logs (`utils/logs.py` redacts) |
| Type safety | `mypy` ships in requirements; per-module `[mypy]` strictness in `pyproject.toml` (M0 deliverable) |
| Dependency management | `pip freeze > requirements.txt` after every install — never hand-edit |
"