"""Pipeline runner: triage -> evidence panel -> fusion -> verdict.

Determinism: verdict_id = argus-{sha256[:16]}; artifacts regenerated into the
same directory; all randomness seeded; same image in -> same forensic content
out (timing fields excluded — see DECISIONS.md).
"""
import hashlib
import io
import logging
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

import registry
from base import ImageContext
from fusion import FUSION_MODEL, fuse
from explanation import generate as generate_explanation
from reliability import RELIABILITY_VERSION
from schemas import PIPELINE_VERSION, Verdict, VerdictInput, VerdictMeta
from triage import DegradationEstimator

logger = logging.getLogger("argus.pipeline")

ARTIFACT_ROOT = Path(__file__).parent / "artifacts"
ARTIFACT_ROOT.mkdir(exist_ok=True)

MAX_PIXELS = 40_000_000


class InvalidImage(Exception):
    pass


def run_assessment(raw: bytes, filename: str = "upload") -> Verdict:
    t_start = time.perf_counter()
    try:
        pil = Image.open(io.BytesIO(raw))
        pil.load()
    except Exception as exc:
        raise InvalidImage(f"cannot decode image: {exc}")
    if pil.width * pil.height > MAX_PIXELS:
        raise InvalidImage("image too large")

    sha = hashlib.sha256(raw).hexdigest()
    verdict_id = f"argus-{sha[:16]}"
    fmt = (pil.format or "unknown").lower()

    artifact_dir = ARTIFACT_ROOT / verdict_id
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)  # deterministic regeneration
    artifact_dir.mkdir(parents=True)

    ext = {"jpeg": ".jpg", "png": ".png", "webp": ".webp"}.get(fmt, ".bin")
    src_path = Path("/tmp") / f"argus_src_{verdict_id}{ext}"
    src_path.write_bytes(raw)

    ctx = ImageContext(
        pil=pil, raw_bytes=raw, sha256=sha, fmt=fmt,
        verdict_id=verdict_id, artifact_dir=artifact_dir, src_path=src_path,
    )
    try:
        d = DegradationEstimator().estimate(ctx)
        modules = registry.get_modules()
        with ThreadPoolExecutor(max_workers=3) as pool:
            futures = [pool.submit(m.run, ctx, d) for m in modules]
            outputs = [f.result() for f in futures]  # registration order kept

        core, ranking, contradictions, unavailable, k = fuse(outputs, d)
        explanation = generate_explanation(core, d, ranking, contradictions, unavailable, k)
        total_ms = int((time.perf_counter() - t_start) * 1000)

        return Verdict(
            verdict_id=verdict_id,
            input=VerdictInput(sha256=sha, dimensions=[pil.width, pil.height],
                               format=fmt, degradation_state=d),
            verdict=core,
            evidence_ranking=ranking,
            contradictions=contradictions,
            unavailable_evidence=unavailable,
            explanation=explanation,
            meta=VerdictMeta(
                module_versions={m.module_id: m.version for m in modules},
                fusion_model=FUSION_MODEL,
                reliability_curves=RELIABILITY_VERSION,
                total_compute_ms=total_ms,
            ),
        )
    finally:
        src_path.unlink(missing_ok=True)  # never store images beyond processing
