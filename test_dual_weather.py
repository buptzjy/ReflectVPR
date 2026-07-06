"""
Quick dual-route weather test: 12 images, 4 rain / 4 snow / 4 rainy_night.
Uses Style-B prompts (Preserve → Apply → Add → Do not).
Output: /media/data/zhangjingyi/ReflectVPR/output_0605_v1/
"""

import json
import os
import random
import shutil
from pathlib import Path

import requests

LIGHTX2V_URL = "http://127.0.0.1:8001/generate"
OUTPUT_ROOT = Path("/media/data/zhangjingyi/ReflectVPR/output_0605_v1")
IMAGE_ROOT = Path("/media/data1/chenshunpeng1/datasets/gsv_cities/Images")
SEED = 42

# We use cities from the existing config
CITIES = ["London", "Phoenix", "Osaka", "PRS"]

WEATHER_INSTRUCTION_STYLE_B = {
    "rain": (
        "Apply heavy rainy weather only:\n"
        "replace the sky with dark overcast rain clouds,\n"
        "add clearly visible rain streaks across the image,\n"
        "make road surfaces wet and reflective,\n"
        "add puddles and subtle rain splashes,\n"
        "slightly reduce visibility and overall brightness."
    ),
    "snow": (
        "Apply snowy winter weather only:\n"
        "replace the sky with overcast winter clouds,\n"
        "add visible snowflakes falling across the image,\n"
        "make the ground, sidewalks and parked cars lightly snow-covered,\n"
        "coat rooftops with a thin layer of white snow,\n"
        "create a cold winter atmosphere with frosty surfaces."
    ),
    "rainy_night": (
        "Apply rainy night weather only:\n"
        "replace the sky with pitch-dark rain clouds,\n"
        "add clearly visible rain streaks under dim street lighting,\n"
        "make road surfaces wet with puddles reflecting street lamps,\n"
        "darken the overall scene to nocturnal atmosphere,\n"
        "keep building windows glowing with warm indoor lights."
    ),
}


def build_dual_prompt(weather: str, occlusion: str, position: str) -> str:
    preserve = "Preserve all scene geometry, buildings, storefronts, signs, existing vehicles, road layout and camera viewpoint."
    do_not = "Do not modify scene structure or object layout."

    if occlusion == "vehicle":
        occluder = (
            f"Add exactly one visible vehicle on {position}, "
            "occupying approximately 4-8% image area, partially blocking a visible lane, "
            "clearly visible vehicle body and wheels, noticeable but realistic occlusion."
        )
    else:
        occluder = (
            f"Add exactly one realistic full-body pedestrian on {position}, "
            "natural scale, normal clothing, feet on the visible ground plane, matched lighting."
        )

    return f"{preserve}\n\n{WEATHER_INSTRUCTION_STYLE_B[weather]}\n\n{occluder}\n\n{do_not}"


def pick_images(count: int = 12) -> list[Path]:
    """Pick random images across all cities."""
    rng = random.Random(SEED)
    all_images = []
    for city in CITIES:
        city_dir = IMAGE_ROOT / city
        if city_dir.exists():
            imgs = sorted(city_dir.glob("*.jpg"))
            rng.shuffle(imgs)
            all_images.extend(imgs)

    if len(all_images) < count:
        raise RuntimeError(f"Need {count} images, found {len(all_images)}")

    rng = random.Random(SEED + 1)
    selected = rng.sample(all_images, count)
    return selected


def main():
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    images = pick_images(12)
    # Assign weather: first 4 rain, next 4 snow, last 4 rainy_night
    assignments = []
    for i, img in enumerate(images):
        if i < 4:
            w = "rain"
        elif i < 8:
            w = "snow"
        else:
            w = "rainy_night"
        assignments.append((img, w, "vehicle", "the visible road lane"))

    rng = random.Random(SEED + 2)
    rng.shuffle(assignments)

    for idx, (img_path, weather, occlusion, position) in enumerate(assignments, 1):
        prompt = build_dual_prompt(weather, occlusion, position)
        stem = f"{img_path.parent.name}_{img_path.stem}_{weather}"
        out_path = OUTPUT_ROOT / f"{stem}.jpg"

        print(f"\n[{idx}/12] {img_path.name}")
        print(f"  weather={weather}")
        print(f"  prompt preview: {prompt[:80]}...")

        payload = {
            "image_path": str(img_path),
            "prompt": prompt,
            "negative_prompt": (
                "oil painting, cartoon, anime, stylized, warped buildings, changed road layout, "
                "warped geometry, distorted facade, changed camera viewpoint, extra vehicles, "
                "dense crowds, traffic jam, readable text, logo, license plate, "
                "solid black rectangle, black bar, black mask, edge shadow, "
                "floating object, pasted object, sticker-like object, low quality"
            ),
            "seed": SEED + idx,
            "infer_steps": 4,
            "guidance_scale": 1.0,
        }

        try:
            resp = requests.post(LIGHTX2V_URL, json=payload, timeout=300)
            resp.raise_for_status()
            data = resp.json()
            result_path = data.get("result_path")
            if not result_path or not os.path.exists(result_path):
                print(f"  ERROR: result_path not found: {result_path}")
                continue
            # Copy to output
            shutil.copy2(result_path, out_path)
            print(f"  OK -> {out_path.name}")
        except Exception as e:
            print(f"  ERROR: {e}")


if __name__ == "__main__":
    main()
