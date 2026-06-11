"""Write /app/tests/regression_set/manifest.json from the on-disk corpus.

Expected directions:
  real/       -> authentic
  ai/         -> synthetic
  laundered/  -> authentic (laundered real photos; abstention allowed,
                 confident-wrong is not)
"""
import json
from pathlib import Path

REG = Path(__file__).resolve().parent / "regression_set"

EXPECT = {"real": "authentic", "ai": "synthetic", "laundered": "authentic"}


def main():
    items = []
    for sub, expected in EXPECT.items():
        for f in sorted((REG / sub).glob("*")):
            if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"):
                items.append({"path": f"{sub}/{f.name}", "expected": expected})
    manifest = {"version": 1, "items": items}
    (REG / "manifest.json").write_text(json.dumps(manifest, indent=2))
    counts = {s: sum(1 for i in items if i["path"].startswith(s)) for s in EXPECT}
    print(f"manifest: {len(items)} items {counts}")


if __name__ == "__main__":
    main()
