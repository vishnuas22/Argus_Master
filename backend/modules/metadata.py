"""Module A — Metadata & container forensics (docs 8.2 Module A).

pyexiftool dump + 15 rule checks. Asymmetric contribution per docs 2.2:
anomalies are strong evidence, cleanliness is weak evidence.
"""
import re
from datetime import datetime, timezone

from base import EvidenceModule, ImageContext
from schemas import Artifact, DegradationState, ModuleOutput

AI_PAT = re.compile(
    r"midjourney|dall[\s·\-]?e|stable.?diffusion|sdxl|flux|firefly|imagen|leonardo"
    r"|runway|ideogram|recraft|gpt-?image|openai|comfyui|automatic1111|novelai|invokeai",
    re.I,
)
EDITOR_PAT = re.compile(r"photoshop|lightroom|gimp|affinity|pixelmator|snapseed|canva|paint\.net|luminar", re.I)
SCREENSHOT_PAT = re.compile(r"screenshot|screen.?capture|grab", re.I)
PROMPT_PAT = re.compile(
    r"highly detailed|octane render|unreal engine|8k|trending on|artstation|hyper.?realistic"
    r"|cinematic lighting|negative.?prompt|cfg.?scale|sampler|denoising strength|steps: \d+",
    re.I,
)
_DATE_FMT = "%Y:%m:%d %H:%M:%S"


def _parse_date(val):
    if not isinstance(val, str):
        return None
    try:
        return datetime.strptime(val[:19], _DATE_FMT).replace(tzinfo=timezone.utc)
    except ValueError:
        return None


class MetadataModule(EvidenceModule):
    module_id = "metadata"
    version = "0.1.0"

    def assess(self, ctx: ImageContext, d: DegradationState, base_reliability: float) -> ModuleOutput:
        try:
            from exiftool import ExifToolHelper
        except ImportError:
            return self._unavailable("pyexiftool_not_installed")
        try:
            with ExifToolHelper() as et:
                meta = et.get_metadata(str(ctx.src_path))[0]
        except FileNotFoundError:
            return self._unavailable("exiftool_binary_missing")

        def vals(*needles):
            found = []
            for k, v in meta.items():
                if any(n.lower() in k.lower() for n in needles):
                    found.append((k, str(v)))
            return found

        findings = []  # (type, weight, description, claim)

        def add(a_type, weight, desc, claim):
            findings.append((a_type, weight, desc, claim))

        software = vals("Software", "CreatorTool", "ProcessingSoftware", "HistorySoftwareAgent", "Creator")
        sw_text = " | ".join(f"{k}={v}" for k, v in software)

        # 1. AI generator software string
        if AI_PAT.search(sw_text):
            add("ai_software_tag", -0.9, "Software/creator tag names a known AI generator",
                f"Metadata field contains AI-generator string: {sw_text[:200]}")
        # 2. IPTC/XMP declared algorithmic media
        dst = vals("DigitalSourceType")
        if any("trainedalgorithmicmedia" in v.lower().replace(" ", "") for _, v in dst):
            add("ai_provenance_declaration", -0.95, "XMP DigitalSourceType declares trained algorithmic media",
                f"XMP DigitalSourceType = {dst[0][1]} (IPTC code for AI-generated media)")
        # 3. PNG prompt chunks / generation parameters
        png_chunks = vals("PNG:Parameters", "PNG:Prompt", "PNG:Workflow", "PNG:Description")
        prompt_hits = [(k, v) for k, v in png_chunks if len(v) > 40 or PROMPT_PAT.search(v)]
        if prompt_hits:
            add("prompt_residue", -0.85, "PNG text chunk carries generation parameters/prompt",
                f"PNG chunk {prompt_hits[0][0]} contains: \"{prompt_hits[0][1][:160]}\"")
        # 4. Editor software
        if EDITOR_PAT.search(sw_text) and not AI_PAT.search(sw_text):
            add("editor_software", -0.3, "Image processed by an editing application",
                f"Editing software recorded in metadata: {sw_text[:160]}")
        # 5. Full camera EXIF block
        make = vals("EXIF:Make")
        model = vals("EXIF:Model")
        expo = vals("ExposureTime")
        iso = vals("EXIF:ISO")
        fnum = vals("FNumber")
        has_camera = bool(make and model)
        if has_camera and expo and iso and fnum:
            add("camera_exif_present", +0.45, "Complete capture EXIF block present",
                f"Camera EXIF: Make={make[0][1]}, Model={model[0][1]}, ExposureTime={expo[0][1]}, "
                f"ISO={iso[0][1]}, FNumber={fnum[0][1]}")
        # 6. GPS
        if vals("GPSLatitude"):
            add("gps_present", +0.15, "GPS coordinates present", "EXIF GPSLatitude/GPSLongitude tags are populated")
        # 7. Missing camera metadata on a JPEG
        has_dto = bool(vals("DateTimeOriginal"))
        if ctx.fmt == "jpeg" and not has_camera and not has_dto:
            add("missing_camera_metadata", -0.25, "JPEG carries no capture metadata (stripped or born-digital)",
                "No EXIF Make/Model/DateTimeOriginal present in a JPEG container")
            # 8. Implausible cleanliness
            if (d.jpeg_quality_est or 0) >= 95 and d.recompression_generations <= 1:
                add("implausible_cleanliness", -0.2,
                    "Pristine high-quality JPEG with zero capture metadata is atypical of camera pipelines",
                    f"JPEG quality ≈{d.jpeg_quality_est} with no capture metadata and "
                    f"{d.recompression_generations} compression generation(s)")
        # 9/10. Date logic
        dto = _parse_date(vals("DateTimeOriginal")[0][1]) if has_dto else None
        mod_d = vals("EXIF:ModifyDate") or vals("ModifyDate")
        mdt = _parse_date(mod_d[0][1]) if mod_d else None
        now = datetime.now(timezone.utc)
        if dto and dto > now:
            add("date_anomaly", -0.4, "Capture date lies in the future",
                f"DateTimeOriginal={dto.isoformat()} is later than assessment time {now.date().isoformat()}")
        if dto and mdt and (dto - mdt).total_seconds() > 60:
            add("date_anomaly", -0.35, "File was 'modified' before it was 'captured'",
                f"DateTimeOriginal={dto.isoformat()} is after ModifyDate={mdt.isoformat()}")
        # 11. EXIF dimension mismatch
        ew = vals("ExifImageWidth")
        if has_camera and ew:
            try:
                if int(float(ew[0][1])) != ctx.pil.width:
                    add("dimension_mismatch", -0.2, "EXIF capture dimensions differ from actual pixels (resized after capture)",
                        f"ExifImageWidth={ew[0][1]} vs actual width={ctx.pil.width}")
            except ValueError:
                pass
        # 12. Camera EXIF but no embedded thumbnail
        if has_camera and not vals("ThumbnailImage", "ThumbnailLength"):
            add("thumbnail_missing", -0.1, "Camera EXIF present but embedded thumbnail absent",
                "EXIF block claims camera capture yet contains no ThumbnailImage tag")
        # 13. Prompt-like free text in JPEG description fields
        desc = vals("ImageDescription", "UserComment", "XPComment")
        prompt_desc = [(k, v) for k, v in desc if len(v) > 60 and PROMPT_PAT.search(v)]
        if prompt_desc:
            add("prompt_residue", -0.3, "Description field reads like a generation prompt",
                f"{prompt_desc[0][0]} contains: \"{prompt_desc[0][1][:160]}\"")
        # 14. Screenshot software marker
        if SCREENSHOT_PAT.search(sw_text):
            add("screenshot_software", -0.05, "Screenshot tool recorded in metadata", f"Software tag: {sw_text[:120]}")
        # 15. Camera claimed but no ICC profile
        if has_camera and not vals("ICC_Profile"):
            add("icc_missing", -0.05, "Camera claimed but no ICC color profile embedded",
                f"Make={make[0][1]} present without ICC_Profile group")

        neg = max(-0.95, sum(w for _, w, _, _ in findings if w < 0))
        pos = min(0.55, sum(w for _, w, _, _ in findings if w > 0))
        e = max(-1.0, min(1.0, neg + pos))

        artifacts = [
            Artifact(type=t, description=desc, strength=min(1.0, abs(w)), checkable_claim=claim)
            for t, w, desc, claim in findings
        ]
        if not artifacts:
            artifacts.append(Artifact(
                type="metadata_neutral", description="No rule check fired", strength=0.1,
                checkable_claim=f"exiftool dump contains {len(meta)} tags; none matched the 15-rule anomaly pack",
            ))

        direction = "synthetic" if e < -0.15 else ("authentic" if e > 0.15 else "neutral")
        if direction == "synthetic" and any(t == "editor_software" for t, w, _, _ in findings) and not AI_PAT.search(sw_text):
            direction = "manipulated"
        confidence = min(0.9, 0.35 + 0.004 * len(meta) + 0.05 * len(findings))

        return ModuleOutput(
            module_id=self.module_id, version=self.version,
            evidence_score=round(e, 4), reliability_score=base_reliability,
            confidence_score=round(confidence, 4), verdict_direction=direction,
            artifacts=artifacts,
        )


MODULE = MetadataModule
