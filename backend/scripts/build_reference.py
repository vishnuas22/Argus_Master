"""Build the DINOv2 reference set (ref-v0) and calibrations for modules E & F.

Outputs into backend/data/:
  reference_embeddings.npy   (~2500 x 768 float32, L2-normalized)
  knn_calib_distances.npy    kNN distances of held-out reals vs the reference
  perturb_calib.npz          noise/blur similarity drops on held-out reals
  reference_meta.json        provenance (version, n, source, date)
CPU-only; ~15-30 min. Run in background, then restart backend.
"""
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dino_service import DinoService  # noqa: E402
from modules.perturbation import perturb_drops  # noqa: E402

CORPUS = Path("/tmp/argus_corpus")
DATA = Path(__file__).resolve().parent.parent / "data"
N_REF = 2500
N_CAL = 300
N_PERTURB = 150


def main():
    DATA.mkdir(exist_ok=True)
    service = DinoService.get()
    service.warm()
    assert service.ready(), "DINOv2 failed to load"

    ref_files = sorted((CORPUS / "reference").glob("*.jpg"))[:N_REF]
    cal_files = sorted((CORPUS / "calib").glob("*.jpg"))

    embs = []
    for i, f in enumerate(ref_files):
        try:
            embs.append(service.embed(Image.open(f)))
        except Exception as exc:  # noqa: BLE001
            print(f"skip {f.name}: {exc}")
        if (i + 1) % 200 == 0:
            print(f"ref {i + 1}/{len(ref_files)}", flush=True)
    ref = np.stack(embs).astype(np.float32)
    np.save(DATA / "reference_embeddings.npy", ref)
    print(f"reference: {ref.shape}")

    k = 5
    dists = []
    for i, f in enumerate(cal_files[:N_CAL]):
        try:
            e = service.embed(Image.open(f))
            d = 1.0 - ref @ e
            dists.append(float(np.sort(d)[:k].mean()))
        except Exception as exc:  # noqa: BLE001
            print(f"skip {f.name}: {exc}")
        if (i + 1) % 50 == 0:
            print(f"calib {i + 1}/{N_CAL}", flush=True)
    np.save(DATA / "knn_calib_distances.npy", np.array(dists, dtype=np.float32))
    print(f"knn calib: {len(dists)}")

    noise_drops, blur_drops = [], []
    for i, f in enumerate(cal_files[:N_PERTURB]):
        try:
            pil = Image.open(f)
            base = service.embed(pil)
            dn, db = perturb_drops(pil, service, base)
            noise_drops.append(dn)
            blur_drops.append(db)
        except Exception as exc:  # noqa: BLE001
            print(f"skip {f.name}: {exc}")
        if (i + 1) % 50 == 0:
            print(f"perturb {i + 1}/{N_PERTURB}", flush=True)
    np.savez(DATA / "perturb_calib.npz",
             noise_drops=np.array(noise_drops, dtype=np.float32),
             blur_drops=np.array(blur_drops, dtype=np.float32))
    print(f"perturb calib: {len(noise_drops)}")

    (DATA / "reference_meta.json").write_text(json.dumps({
        "version": "ref-v0", "n": int(ref.shape[0]), "source": "COCO val2017 (deterministic slice)",
        "embed_size": 224, "built": date.today().isoformat(),
        "caveat": "small-N reference (2-5k); module E confidence capped at 0.65",
    }, indent=2))
    print("done")


if __name__ == "__main__":
    main()
