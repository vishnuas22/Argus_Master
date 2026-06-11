"""ARGUS M6 regression & hardening suite.

Covers: verdict schema validation, module crash isolation, determinism,
laundering monotonicity, and the regression-set manifest with the
confident-wrong = 0 gate (abstentions allowed, wrong-with-high-trust not).
"""
import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import registry
from pipeline import run_assessment
from schemas import Verdict

REG = Path(__file__).resolve().parent / "regression_set"
MANIFEST = REG / "manifest.json"

CONFIDENT_TRUST = 0.5  # wrong verdicts at/above this trust fail the gate


def _noise_png(seed=0, size=320):
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, "PNG")
    return buf.getvalue()


def _strip_timing(v: Verdict) -> dict:
    d = v.model_dump()
    d["meta"].pop("total_compute_ms", None)
    return d


# ---------------------------------------------------------------- schema
def test_verdict_schema_valid():
    v = run_assessment(_noise_png())
    Verdict.model_validate(v.model_dump())  # round-trips the 7.1 contract
    assert 0.0 <= v.verdict.authenticity_score <= 1.0
    assert 0.0 <= v.verdict.trust_score <= 1.0
    assert 0.0 <= v.verdict.risk_score <= 1.0
    assert abs(sum(v.verdict.probabilities.values()) - 1.0) < 0.02
    assert v.verdict.abstained == (len(v.verdict.conformal.set) > 1)
    for e in v.evidence_ranking:
        assert -1.0 <= e.evidence_score <= 1.0
        assert 0.0 <= e.reliability <= 1.0
        assert 0.0 <= e.confidence <= 1.0
        for a in e.artifacts:
            assert a.checkable_claim  # every artifact is checkable
    assert v.meta.module_versions  # versioning embedded


# ---------------------------------------------------- crash isolation
def test_module_crash_isolation(monkeypatch):
    monkeypatch.setenv("ARGUS_ENABLE_STUB", "1")
    monkeypatch.setenv("ARGUS_STUB_MODE", "crash")
    registry.get_modules(force_reload=True)
    try:
        v = run_assessment(_noise_png(seed=1))
        stub = [u for u in v.unavailable_evidence if u.module == "stub"]
        assert stub and stub[0].reason == "internal_error"
        assert "stub" in v.meta.module_versions  # never hidden
    finally:
        monkeypatch.delenv("ARGUS_ENABLE_STUB")
        registry.get_modules(force_reload=True)


# -------------------------------------------------------- determinism
def test_determinism_same_image_identical_verdict():
    reals = sorted((REG / "real").glob("*.jpg"))
    raw = reals[0].read_bytes() if reals else _noise_png(seed=2)
    v1 = run_assessment(raw)
    v2 = run_assessment(raw)
    assert _strip_timing(v1) == _strip_timing(v2)


# ------------------------------------------- laundering monotonicity
def test_laundering_monotonicity():
    reals = sorted((REG / "real").glob("*.jpg"))
    assert reals, "regression reals missing — run prepare_corpus.py"
    img = Image.open(reals[0]).convert("RGB")

    def rung(quality, scale):
        im = img
        if scale < 1.0:
            im = im.resize((max(64, int(im.width * scale)), max(64, int(im.height * scale))), Image.BILINEAR)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=quality)
        return buf.getvalue()

    ladder = [img_bytes for img_bytes in (rung(95, 1.0), rung(70, 1.0), rung(50, 0.6), rung(35, 0.45))]
    trusts = [run_assessment(b).verdict.trust_score for b in ladder]
    eps = 0.02
    for a, b in zip(trusts, trusts[1:]):
        assert b <= a + eps, f"trust increased under laundering: {trusts}"


# ------------------------------------------------- regression manifest
def _load_manifest():
    if not MANIFEST.exists():
        pytest.skip("manifest.json not built yet")
    return json.loads(MANIFEST.read_text())["items"]


def test_regression_directions_and_confident_wrong_zero():
    items = _load_manifest()
    assert len(items) >= 25  # >=10 real, >=10 ai, >=5 laundered
    confident_wrong = []
    results = []
    for item in items:
        raw = (REG / item["path"]).read_bytes()
        v = run_assessment(raw)
        top = max(v.verdict.probabilities, key=v.verdict.probabilities.get)
        correct = (
            top == "camera_original" if item["expected"] == "authentic"
            else top in ("ai_generated", "manipulated")
        )
        results.append((item["path"], item["expected"], top, v.verdict.abstained,
                        v.verdict.trust_score, correct))
        if not correct and not v.verdict.abstained and v.verdict.trust_score >= CONFIDENT_TRUST:
            confident_wrong.append(results[-1])
    for r in results:
        print(r)
    assert not confident_wrong, f"confident-wrong verdicts: {confident_wrong}"
