"""ARGUS API — FastAPI on 0.0.0.0:8001, all routes under /api.

POST /api/assess            multipart image -> Verdict JSON (docs 7.1)
GET  /api/verdicts          history summaries from MongoDB
GET  /api/verdicts/{id}     full stored verdict
GET  /api/health            pipeline/module status
/api/artifacts/*            saved PNG overlays (static)
"""
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")

from fastapi import APIRouter, FastAPI, File, HTTPException, UploadFile  # noqa: E402
from fastapi.staticfiles import StaticFiles  # noqa: E402
from motor.motor_asyncio import AsyncIOMotorClient  # noqa: E402
from starlette.concurrency import run_in_threadpool  # noqa: E402
from starlette.middleware.cors import CORSMiddleware  # noqa: E402

import registry  # noqa: E402
from dino_service import DinoService  # noqa: E402
from pipeline import ARTIFACT_ROOT, InvalidImage, run_assessment  # noqa: E402
from schemas import PIPELINE_VERSION, Verdict  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("argus.api")

mongo_url = os.environ["MONGO_URL"]
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ["DB_NAME"]]

app = FastAPI(title="ARGUS", version=PIPELINE_VERSION)
api_router = APIRouter(prefix="/api")

app.mount("/api/artifacts", StaticFiles(directory=str(ARTIFACT_ROOT)), name="artifacts")

MAX_UPLOAD = 25 * 1024 * 1024

_HISTORY_PROJECTION = {
    "_id": 0, "verdict_id": 1, "filename": 1, "created_at": 1,
    "input.format": 1, "input.dimensions": 1, "input.degradation_state.evidence_capacity": 1,
    "verdict.hypothesis_set": 1, "verdict.abstained": 1,
    "verdict.authenticity_score": 1, "verdict.trust_score": 1, "verdict.risk_score": 1,
    "meta.total_compute_ms": 1,
}


@api_router.get("/health")
async def health():
    modules = registry.get_modules()
    return {
        "status": "ok",
        "pipeline_version": PIPELINE_VERSION,
        "modules": [m.module_id for m in modules],
        "dino_ready": DinoService.get().ready(),
    }


@api_router.post("/assess", response_model=Verdict)
async def assess(file: UploadFile = File(...)):
    raw = await file.read()
    if len(raw) > MAX_UPLOAD:
        raise HTTPException(status_code=413, detail="file exceeds 25MB limit")
    if not raw:
        raise HTTPException(status_code=400, detail="empty upload")
    try:
        verdict = await run_in_threadpool(run_assessment, raw, file.filename or "upload")
    except InvalidImage as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    doc = verdict.model_dump()
    doc["created_at"] = datetime.now(timezone.utc).isoformat()
    doc["filename"] = file.filename or "upload"
    await db.verdicts.replace_one({"verdict_id": verdict.verdict_id}, doc, upsert=True)
    return verdict


@api_router.get("/verdicts")
async def list_verdicts(limit: int = 50):
    cursor = db.verdicts.find({}, _HISTORY_PROJECTION).sort("created_at", -1).limit(min(limit, 200))
    return await cursor.to_list(length=min(limit, 200))


@api_router.get("/verdicts/{verdict_id}")
async def get_verdict(verdict_id: str):
    doc = await db.verdicts.find_one({"verdict_id": verdict_id}, {"_id": 0})
    if doc is None:
        raise HTTPException(status_code=404, detail="verdict not found")
    return doc


app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def warm_backbone():
    if os.environ.get("ARGUS_WARM", "1") == "1":
        threading.Thread(target=DinoService.get().warm, daemon=True).start()


@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()
