"""Generate the AI test corpus via gpt-image-1 (Emergent LLM key).

COCO-caption-style prompts (docs 10 S4 de-confounding: everyday scenes that
match the real corpus content). Outputs:
  /app/tests/smoke_set/ai        20 images (M3 AUROC gate)
  /app/tests/regression_set/ai   10 images (M6 regression suite)
Skips files that already exist (resumable). Run in background; ~1 min/image.
"""
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from emergentintegrations.llm.openai.image_generation import OpenAIImageGeneration  # noqa: E402

SMOKE = Path("/app/tests/smoke_set/ai")
REG = Path("/app/tests/regression_set/ai")

PROMPTS = [
    "A man riding a skateboard down a city street in the afternoon",
    "Two dogs playing with a frisbee in a grassy park",
    "A plate of spaghetti with tomato sauce on a wooden restaurant table",
    "A woman holding an umbrella waiting at a bus stop in light rain",
    "A group of people sitting around a table eating pizza",
    "A brown horse grazing in a fenced field near a barn",
    "A kitchen counter with a bowl of fruit and a coffee maker",
    "A man surfing a small wave at a crowded beach",
    "A double-decker bus driving past a row of shops",
    "A cat sleeping on a windowsill next to a potted plant",
    "A baseball player swinging a bat during a game",
    "A bathroom with a white sink, mirror and folded towels",
    "Several boats docked in a small harbor in the morning",
    "A child flying a colorful kite in an open field",
    "A train arriving at an outdoor station platform",
    "A vendor selling vegetables at an outdoor street market",
    "A living room with a grey couch, bookshelf and television",
    "A cyclist riding across a bridge with traffic behind",
    "A giraffe standing near trees in a zoo enclosure",
    "A laptop, notebook and cup of coffee on an office desk",
    "A pizza with pepperoni and basil on a metal tray",
    "An elderly man reading a newspaper on a park bench",
    "A stop sign at a suburban intersection with parked cars",
    "Two sheep standing on a hillside on an overcast day",
    "A bowl of soup with bread on a checkered tablecloth",
    "A skier going down a snowy slope past pine trees",
    "A motorcycle parked on the side of a narrow street",
    "A flock of pigeons on a plaza with people walking",
    "A tennis player serving on an outdoor clay court",
    "A refrigerator and stove in a small apartment kitchen",
]

SUFFIX = ", photorealistic photograph, natural lighting"
CONCURRENCY = 3


async def gen_one(gen, sem, prompt, dest: Path):
    async with sem:
        for attempt in range(3):
            try:
                images = await gen.generate_images(prompt=prompt + SUFFIX, model="gpt-image-1", number_of_images=1)
                dest.write_bytes(images[0])
                print(f"ok  {dest.name}", flush=True)
                return
            except Exception as exc:  # noqa: BLE001
                print(f"retry {dest.name} ({attempt + 1}/3): {exc}", flush=True)
                await asyncio.sleep(8)
        print(f"FAILED {dest.name}", flush=True)


async def main():
    SMOKE.mkdir(parents=True, exist_ok=True)
    REG.mkdir(parents=True, exist_ok=True)
    gen = OpenAIImageGeneration(api_key=os.environ["EMERGENT_LLM_KEY"])
    sem = asyncio.Semaphore(CONCURRENCY)
    tasks = []
    for i, prompt in enumerate(PROMPTS):
        dest = (SMOKE / f"ai_{i:02d}.png") if i < 20 else (REG / f"ai_{i:02d}.png")
        if dest.exists():
            continue
        tasks.append(gen_one(gen, sem, prompt, dest))
    await asyncio.gather(*tasks)
    print(f"smoke ai: {len(list(SMOKE.glob('*.png')))}, regression ai: {len(list(REG.glob('*.png')))}")


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
