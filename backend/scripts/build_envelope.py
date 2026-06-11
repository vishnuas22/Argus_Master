"""Fit the spectral real-image envelope (module C calibration, real data only).

Uses the exact radial_profile/standardize transform from modules.spectral so
calibration and inference are bitwise-consistent. numpy-only (no torch).
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modules.spectral import radial_profile, standardize  # noqa: E402

CORPUS = Path("/tmp/argus_corpus/reference")
OUT = Path(__file__).resolve().parent.parent / "data" / "spectral_envelope.npz"
N = 500


def main():
    files = sorted(CORPUS.glob("*.jpg"))[:N]
    assert files, "reference corpus missing — run prepare_corpus.py first"
    profiles = []
    for i, f in enumerate(files):
        try:
            profiles.append(radial_profile(standardize(Image.open(f))))
        except Exception as exc:  # noqa: BLE001
            print(f"skip {f.name}: {exc}")
        if (i + 1) % 100 == 0:
            print(f"{i + 1}/{len(files)}")
    arr = np.stack(profiles)
    OUT.parent.mkdir(exist_ok=True)
    np.savez(OUT, mean=arr.mean(axis=0), std=arr.std(axis=0), n=len(profiles))
    print(f"saved {OUT} (n={len(profiles)})")


if __name__ == "__main__":
    main()
