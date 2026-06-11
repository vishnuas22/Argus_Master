"""Deterministic corpus preparation from the COCO val2017 download.

Splits (sorted filename order, fixed slices — fully reproducible):
  reference  2500  -> /tmp/argus_corpus/reference   (embedded into ref-v0)
  calib       400  -> /tmp/argus_corpus/calib       (kNN/perturb/envelope calibration)
  smoke real   20  -> /app/tests/smoke_set/real
  regression   12  -> /app/tests/regression_set/real
Laundered variants (JPEG q50 + 0.5x resize, then a heavier q35+0.4x rung)
are generated from the first regression reals.
"""
import shutil
import sys
from pathlib import Path

from PIL import Image

SRC = Path("/tmp/val2017")
CORPUS = Path("/tmp/argus_corpus")
SMOKE = Path("/app/tests/smoke_set")
REG = Path("/app/tests/regression_set")


def launder(src: Path, dst: Path, q: int, scale: float):
    img = Image.open(src).convert("RGB")
    img = img.resize((max(64, int(img.width * scale)), max(64, int(img.height * scale))), Image.BILINEAR)
    img.save(dst, "JPEG", quality=q)


def main():
    files = sorted(SRC.glob("*.jpg"))
    assert len(files) >= 3000, f"expected >=3000 COCO images, got {len(files)}"
    splits = {
        CORPUS / "reference": files[0:2500],
        CORPUS / "calib": files[2500:2900],
        SMOKE / "real": files[2900:2920],
        REG / "real": files[2920:2932],
    }
    for dest, chunk in splits.items():
        dest.mkdir(parents=True, exist_ok=True)
        for f in chunk:
            shutil.copy(f, dest / f.name)
        print(f"{dest}: {len(chunk)} images")

    laund_dir = REG / "laundered"
    laund_dir.mkdir(parents=True, exist_ok=True)
    reg_reals = sorted((REG / "real").glob("*.jpg"))[:6]
    for f in reg_reals[:3]:
        launder(f, laund_dir / f"{f.stem}_q50_r05.jpg", q=50, scale=0.5)
    for f in reg_reals[3:6]:
        launder(f, laund_dir / f"{f.stem}_q35_r04.jpg", q=35, scale=0.4)
    print(f"{laund_dir}: {len(list(laund_dir.glob('*.jpg')))} laundered variants")


if __name__ == "__main__":
    sys.exit(main())
