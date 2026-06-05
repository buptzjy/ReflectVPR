import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path
from collections import Counter

from PIL import Image


ROOT = Path(__file__).resolve().parent
AGENT_DIR = ROOT / "your_agent"
sys.path.insert(0, str(AGENT_DIR))

from evaluator import DualTraitEvaluator  # noqa: E402
from generators.iclight import ICLightGenerator  # noqa: E402
from prompt_rules import (  # noqa: E402
    PROMPT_POLICY_VERSION,
    build_structured_prompt,
    global_negative_prompt,
)


WEATHERS = ("rain", "night", "fog", "overcast")
DEFAULT_IMAGE_ROOT = "/media/data1/chenshunpeng1/datasets/gsv_cities/Images"
DEFAULT_CITIES = "London,Phoenix,Osaka,PRS"


def split_cities(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IC-Light-only global weather test.")
    parser.add_argument("--cities", default=os.getenv("REFLECTVPR_CITIES", DEFAULT_CITIES))
    parser.add_argument("--image-root", default=os.getenv("REFLECTVPR_IMAGE_ROOT", DEFAULT_IMAGE_ROOT))
    parser.add_argument("--output-root", default=os.getenv("REFLECTVPR_OUTPUT_ROOT", ROOT / "output_0603_v6"))
    parser.add_argument("--sample-num", type=int, default=int(os.getenv("REFLECTVPR_SAMPLE_NUM", "10")))
    parser.add_argument("--seed", type=int, default=int(os.getenv("REFLECTVPR_SAMPLE_SEED", "20260529")))
    parser.add_argument("--clear-output", action="store_true", default=os.getenv("REFLECTVPR_CLEAR_OUTPUT", "0").lower() in {"1", "true", "yes"})
    return parser.parse_args()


def select_images(image_root: Path, cities: list[str], sample_num: int, seed: int) -> list[Path]:
    rng = random.Random(seed)
    by_city = {}
    for city in cities:
        image_dir = image_root / city
        if not image_dir.exists():
            raise FileNotFoundError(image_dir)
        images = sorted(image_dir.glob("*.jpg"))
        if not images:
            raise FileNotFoundError(f"No .jpg images found in {image_dir}")
        rng.shuffle(images)
        by_city[city] = images

    base = sample_num // len(cities)
    rem = sample_num % len(cities)
    selected = []
    leftovers = []
    counts = Counter()
    for idx, city in enumerate(cities):
        quota = base + (1 if idx < rem else 0)
        take = min(quota, len(by_city[city]))
        selected.extend(by_city[city][:take])
        leftovers.extend(by_city[city][take:])
        counts[city] += take
    if len(selected) < sample_num:
        rng.shuffle(leftovers)
        extra = leftovers[: sample_num - len(selected)]
        selected.extend(extra)
        counts.update(path.parent.name for path in extra)
    rng.shuffle(selected)
    print(f"[iclight-test] selected={len(selected)} by_city={dict(counts)}", flush=True)
    return selected


def main() -> None:
    args = parse_args()
    cities = split_cities(args.cities)
    image_root = Path(args.image_root)
    output_root = Path(args.output_root)

    if args.clear_output and output_root.exists():
        if output_root.resolve() in {ROOT.resolve(), ROOT.parent.resolve(), Path("/").resolve(), Path.home().resolve()}:
            raise RuntimeError(f"Refusing to clear unsafe output root: {output_root}")
        shutil.rmtree(output_root)
        print(f"[iclight-test] cleared output_root={output_root}", flush=True)

    output_root.mkdir(parents=True, exist_ok=True)
    global_dir = output_root / "global"
    global_dir.mkdir(parents=True, exist_ok=True)

    print(f"[iclight-test] image_root={image_root}", flush=True)
    print(f"[iclight-test] cities={cities}", flush=True)
    print(f"[iclight-test] output_root={output_root}", flush=True)
    print(f"[iclight-test] policy={PROMPT_POLICY_VERSION}", flush=True)
    print(f"[iclight-test] weathers={WEATHERS}", flush=True)

    images = select_images(image_root, cities, args.sample_num, args.seed)
    generator = ICLightGenerator(api_url=None)
    evaluator = DualTraitEvaluator(mock=False)
    records = []

    for idx, image_path in enumerate(images, 1):
        ref = Image.open(image_path).convert("RGB")
        print(f"[{idx}/{len(images)}] {image_path.parent.name}/{image_path.name}", flush=True)
        for weather in WEATHERS:
            prompt = build_structured_prompt(
                route="global",
                weather=weather,
                occlusion=None,
                position="the entire image",
            )
            print(f"  weather={weather}", flush=True)
            gen = generator.generate(
                ref,
                prompt,
                negative_prompt=global_negative_prompt(),
                highres_denoise=float(os.getenv("REFLECTVPR_GLOBAL_ICLIGHT_DENOISE", "0.30")),
            )
            if gen.size != ref.size:
                gen = gen.resize(ref.size, Image.Resampling.LANCZOS)
            out_name = f"{image_path.stem}__global_{weather}__final.jpg"
            out_path = global_dir / out_name
            gen.save(out_path, quality=95)

            entry = {
                "route": "global",
                "weather": weather,
                "occlusion": None,
                "prompt": prompt,
                "position": "the entire image",
            }
            ev = evaluator.evaluate(ref, gen, entry=entry)
            eval_dict = {
                "passed": ev.passed,
                "s_geo": ev.s_geo,
                "s_div": ev.s_div,
                "geo_ok": ev.geo_ok,
                "div_ok": ev.div_ok,
                "artifact_ok": ev.artifact_ok,
                "feedback": ev.feedback,
            }
            record = {
                "file_name": image_path.name,
                "city": image_path.parent.name,
                "source_path": str(image_path),
                "route": "global",
                "weather": weather,
                "occlusion": None,
                "selected_model": "IC-Light",
                "prompt": prompt,
                "final_prompt": prompt,
                "prompt_policy_version": PROMPT_POLICY_VERSION,
                "output_path": str(out_path),
                "passed": ev.passed,
                "s_geo": ev.s_geo,
                "s_div": ev.s_div,
                "geo_ok": ev.geo_ok,
                "div_ok": ev.div_ok,
                "artifact_ok": ev.artifact_ok,
                "reflection_rounds": [{"round": 1, "prompt": prompt, "image_path": str(out_path), "eval": eval_dict}],
            }
            records.append(record)
            print(f"    s_geo={ev.s_geo:.3f} s_div={ev.s_div:.3f} passed={ev.passed}", flush=True)
            (output_root / "reflect.json").write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")

    counts = Counter((record["weather"], record["passed"]) for record in records)
    print("[iclight-test] summary", flush=True)
    for weather in WEATHERS:
        total = sum(1 for record in records if record["weather"] == weather)
        passed = counts[(weather, True)]
        print(f"  {weather}: {passed}/{total}", flush=True)
    print(f"Done. output={output_root}", flush=True)


if __name__ == "__main__":
    main()
