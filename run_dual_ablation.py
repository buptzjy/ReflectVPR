import argparse
import json
import os
import random
import shutil
import sys
from pathlib import Path
from collections import Counter

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parent
AGENT_DIR = ROOT / "your_agent"
sys.path.insert(0, str(AGENT_DIR))

from evaluator import DualTraitEvaluator  # noqa: E402
from generators.lightx2v import Lightx2vGenerator  # noqa: E402
from prompt_rules import (  # noqa: E402
    PROMPT_POLICY_VERSION,
    build_structured_prompt,
    load_experience_bank,
    negative_prompt_from_experience,
)


WEATHERS = ("rain", "snow", "night")
DEFAULT_IMAGE_ROOT = "/media/data1/chenshunpeng1/datasets/gsv_cities/Images"
DEFAULT_CITIES = "London,Phoenix,Osaka,PRS"


def split_cities(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dual-only LightX2V ablation.")
    parser.add_argument("--cities", default=os.getenv("REFLECTVPR_CITIES", DEFAULT_CITIES))
    parser.add_argument("--image-root", default=os.getenv("REFLECTVPR_IMAGE_ROOT", DEFAULT_IMAGE_ROOT))
    parser.add_argument("--output-root", default=os.getenv("REFLECTVPR_OUTPUT_ROOT", ROOT / "output_0603_v7"))
    parser.add_argument("--sample-num", type=int, default=int(os.getenv("REFLECTVPR_SAMPLE_NUM", "10")))
    parser.add_argument("--seed", type=int, default=int(os.getenv("REFLECTVPR_SAMPLE_SEED", "20260529")))
    parser.add_argument("--clear-output", action="store_true", default=os.getenv("REFLECTVPR_CLEAR_OUTPUT", "0").lower() in {"1", "true", "yes"})
    parser.add_argument("--experience-json", default=os.getenv("REFLECTVPR_EXPERIENCE_JSON", ROOT / "experience_bank_v0.json"))
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
    print(f"[dual-ablation] selected={len(selected)} by_city={dict(counts)}", flush=True)
    return selected


def resized_arrays(ref_image: Image.Image, gen_image: Image.Image, size: int = 256) -> tuple[np.ndarray, np.ndarray]:
    ref = ref_image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    gen = gen_image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(ref).astype(np.float32), np.asarray(gen).astype(np.float32)


def luma(arr: np.ndarray) -> np.ndarray:
    return 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]


def saturation(arr: np.ndarray) -> np.ndarray:
    rgb = arr / 255.0
    maxc = rgb.max(axis=-1)
    minc = rgb.min(axis=-1)
    return (maxc - minc) / np.maximum(maxc, 1e-6)


def largest_component_ratio(mask: np.ndarray) -> float:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    largest = 0
    for y in range(h):
        for x in range(w):
            if not mask[y, x] or seen[y, x]:
                continue
            stack = [(y, x)]
            seen[y, x] = True
            count = 0
            while stack:
                cy, cx = stack.pop()
                count += 1
                for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            largest = max(largest, count)
    return largest / float(h * w)


def diagnose(ref_image: Image.Image, gen_image: Image.Image, weather: str, eval_feedback: dict) -> dict:
    ref, gen = resized_arrays(ref_image, gen_image)
    diff = np.abs(gen - ref).mean(axis=-1)
    ref_l = luma(ref)
    gen_l = luma(gen)
    ref_s = saturation(ref)
    gen_s = saturation(gen)
    h, _ = diff.shape
    yy = np.arange(h)[:, None] / h

    lower = np.broadcast_to(yy > 0.38, diff.shape)
    mid_lower = np.broadcast_to((yy > 0.38) & (yy < 0.88), diff.shape)
    changed = (diff > 38) & mid_lower
    changed_ratio = float(changed.mean())
    changed_component = largest_component_ratio(changed)
    occlusion_visible = changed_component >= 0.006 and changed_ratio >= 0.012

    luma_delta = float((gen_l - ref_l).mean())
    sat_delta = float((gen_s - ref_s).mean())
    diff_mean = float(diff.mean())
    lower_diff = float(diff[lower].mean()) if lower.any() else 0.0
    white_ref = (ref_l > 210) & (ref_s < 0.28)
    white_gen = (gen_l > 210) & (gen_s < 0.28)
    white_delta = float(white_gen.mean() - white_ref.mean())

    if weather == "night":
        weather_visible = luma_delta < -10.0 or diff_mean > 22.0
    elif weather == "snow":
        weather_visible = white_delta > 0.018 or luma_delta > 8.0 or diff_mean > 24.0
    elif weather == "rain":
        weather_visible = lower_diff > 18.0 or diff_mean > 20.0 or luma_delta < -4.0
    else:
        weather_visible = diff_mean > 18.0

    flags = []
    for key in ("artifact_issue", "artifact_warning", "quality_gate"):
        flags.extend((eval_feedback.get(key, {}) or {}).get("flags", []) or [])
    weak_dual = "weak_dual_weather_or_occlusion" in flags
    if not weak_dual:
        weak_source = "not_weak"
    elif not weather_visible and not occlusion_visible:
        weak_source = "both_weak"
    elif not weather_visible:
        weak_source = "weather_weak"
    elif not occlusion_visible:
        weak_source = "occlusion_weak"
    else:
        weak_source = "threshold_or_semantic_weak"

    return {
        "weather_visible": weather_visible,
        "occlusion_visible": occlusion_visible,
        "weak_source": weak_source,
        "changed_ratio": changed_ratio,
        "changed_component": changed_component,
        "diff_mean": diff_mean,
        "lower_diff": lower_diff,
        "luma_delta": luma_delta,
        "sat_delta": sat_delta,
        "white_delta": white_delta,
        "diagnostic_flags": sorted(set(flags)),
    }


def main() -> None:
    args = parse_args()
    cities = split_cities(args.cities)
    image_root = Path(args.image_root)
    output_root = Path(args.output_root)

    if args.clear_output and output_root.exists():
        if output_root.resolve() in {ROOT.resolve(), ROOT.parent.resolve(), Path("/").resolve(), Path.home().resolve()}:
            raise RuntimeError(f"Refusing to clear unsafe output root: {output_root}")
        shutil.rmtree(output_root)
        print(f"[dual-ablation] cleared output_root={output_root}", flush=True)

    output_root.mkdir(parents=True, exist_ok=True)
    dual_dir = output_root / "dual"
    dual_dir.mkdir(parents=True, exist_ok=True)

    print(f"[dual-ablation] image_root={image_root}", flush=True)
    print(f"[dual-ablation] cities={cities}", flush=True)
    print(f"[dual-ablation] output_root={output_root}", flush=True)
    print(f"[dual-ablation] policy={PROMPT_POLICY_VERSION}", flush=True)
    print(f"[dual-ablation] weathers={WEATHERS}", flush=True)

    images = select_images(image_root, cities, args.sample_num, args.seed)
    bank = load_experience_bank(args.experience_json)
    negative_prompt = negative_prompt_from_experience(bank)
    generator = Lightx2vGenerator(api_url=None)
    evaluator = DualTraitEvaluator(mock=False)
    records = []

    for weather in WEATHERS:
        print(f"[weather] {weather}", flush=True)
        for idx, image_path in enumerate(images, 1):
            ref = Image.open(image_path).convert("RGB")
            prompt = build_structured_prompt(
                route="dual",
                weather=weather,
                occlusion="vehicle",
                position="visible traffic lane, curbside lane, parking bay, or roadside parking area",
                experience_bank=bank,
            )
            print(f"  [{idx}/{len(images)}] {image_path.parent.name}/{image_path.name}", flush=True)
            gen = generator.generate_dual(ref, prompt, negative_prompt=negative_prompt)
            if gen.size != ref.size:
                gen = gen.resize(ref.size, Image.Resampling.LANCZOS)
            out_name = f"{image_path.stem}__dual_{weather}_vehicle__final.jpg"
            out_path = dual_dir / out_name
            gen.save(out_path, quality=95)

            entry = {
                "route": "dual",
                "weather": weather,
                "occlusion": "vehicle",
                "prompt": prompt,
                "position": "visible traffic lane, curbside lane, parking bay, or roadside parking area",
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
            diag = diagnose(ref, gen, weather, ev.feedback)
            record = {
                "file_name": image_path.name,
                "city": image_path.parent.name,
                "source_path": str(image_path),
                "route": "dual",
                "weather": weather,
                "occlusion": "vehicle",
                "selected_model": "LightX2V-Dual",
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
                **diag,
                "reflection_rounds": [{"round": 1, "prompt": prompt, "image_path": str(out_path), "eval": eval_dict}],
            }
            records.append(record)
            print(
                f"    s_geo={ev.s_geo:.3f} s_div={ev.s_div:.3f} passed={ev.passed} "
                f"weather_visible={diag['weather_visible']} occlusion_visible={diag['occlusion_visible']} "
                f"weak_source={diag['weak_source']}",
                flush=True,
            )
            (output_root / "reflect.json").write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[dual-ablation] summary", flush=True)
    for weather in WEATHERS:
        subset = [record for record in records if record["weather"] == weather]
        passed = sum(record["passed"] for record in subset)
        weather_visible = sum(record["weather_visible"] for record in subset)
        occlusion_visible = sum(record["occlusion_visible"] for record in subset)
        weak_sources = Counter(record["weak_source"] for record in subset)
        print(
            f"  {weather}: passed={passed}/{len(subset)} "
            f"weather_visible={weather_visible}/{len(subset)} "
            f"occlusion_visible={occlusion_visible}/{len(subset)} "
            f"weak_sources={dict(weak_sources)}",
            flush=True,
        )
    print(f"Done. output={output_root}", flush=True)


if __name__ == "__main__":
    main()
