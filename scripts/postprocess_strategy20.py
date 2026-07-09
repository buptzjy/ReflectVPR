#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path("/media/data/zhangjingyi/ReflectVPR")
DEFAULT_PAIRED_MANIFEST = Path("/media/data/zhangjingyi/ImAge_v2/paired_triplets_0618.jsonl")
DEFAULT_DISTANCE_SCRIPT = Path("/media/data/zhangjingyi/ImAge_v3_LORA/tmp_distance.py")
DEFAULT_REFERENCE_SUMMARY = Path("/media/data/zhangjingyi/ImAge_v3_LORA/distance_analysis_500places/summary.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--image-output-dir", type=Path, required=True)
    parser.add_argument("--gpu-log", type=Path, required=True)
    parser.add_argument("--summary-path", type=Path, required=True)
    parser.add_argument("--contact-sheet", type=Path, required=True)
    parser.add_argument("--target-images", type=int, required=True)
    parser.add_argument("--elapsed-seconds", type=int, required=True)
    parser.add_argument("--paired-manifest", type=Path, default=DEFAULT_PAIRED_MANIFEST)
    parser.add_argument("--distance-script", type=Path, default=DEFAULT_DISTANCE_SCRIPT)
    parser.add_argument("--reference-summary", type=Path, default=DEFAULT_REFERENCE_SUMMARY)
    parser.add_argument(
        "--passed-only",
        action="store_true",
        default=os.getenv("REFLECTVPR_SELECT_PASSED_ONLY", "1").lower() in {"1", "true", "yes"},
        help="Only collect generated final images whose reflect record has passed=true.",
    )
    parser.add_argument("--effective-min-distance", type=float, default=float(os.getenv("REFLECTVPR_EFFECTIVE_MIN_DISTANCE", "1.08")))
    parser.add_argument("--preferred-min-distance", type=float, default=float(os.getenv("REFLECTVPR_PREFERRED_MIN_DISTANCE", "1.14")))
    parser.add_argument("--effective-max-distance", type=float, default=float(os.getenv("REFLECTVPR_EFFECTIVE_MAX_DISTANCE", "1.31")))
    parser.add_argument("--real-real-eps", type=float, default=float(os.getenv("REFLECTVPR_REAL_REAL_EPS", "0.02")))
    return parser.parse_args()


def load_records(output_root: Path) -> list[dict]:
    records: list[dict] = []
    for path in sorted(output_root.glob("*/reflect.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(data, list):
            for record in data:
                records.append(record)
                for candidate in record.get("training_candidates", []) or []:
                    candidate_record = dict(record)
                    candidate_record.update(candidate)
                    candidate_record["parent_output_path"] = record.get("output_path")
                    records.append(candidate_record)
    return records


def collect_finals(output_root: Path) -> list[Path]:
    finals: list[Path] = []
    for city_dir in sorted(output_root.iterdir()):
        if not city_dir.is_dir() or city_dir.name.startswith("_"):
            continue
        for route in ("global", "local", "dual"):
            finals.extend(sorted((city_dir / route).glob("*__final.jpg")))
    return finals


def build_anchor_index(paired_manifest: Path) -> dict[str, dict]:
    index: dict[str, dict] = {}
    with paired_manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            item = json.loads(line)
            anchor = item.get("anchor_real")
            if not anchor:
                continue
            index.setdefault(Path(anchor).name, item)
    return index


def copy_selected_images(
    finals: list[Path],
    image_output_dir: Path,
    target: int,
    record_by_final: dict[str, dict] | None = None,
    passed_only: bool = False,
) -> list[Path]:
    image_output_dir.mkdir(parents=True, exist_ok=True)
    for old in image_output_dir.glob("*.jpg"):
        old.unlink()
    if passed_only:
        record_by_final = record_by_final or {}
        finals = [
            final
            for final in finals
            if bool(record_by_final.get(str(final.resolve()), {}).get("passed"))
        ]
    selected = finals[:target]
    for index, source in enumerate(selected, 1):
        shutil.copy2(source, image_output_dir / f"{index:03d}__{source.name}")
    return selected


def replace_selected_images(selected: list[Path], image_output_dir: Path) -> None:
    image_output_dir.mkdir(parents=True, exist_ok=True)
    for old in image_output_dir.glob("*.jpg"):
        old.unlink()
    for index, source in enumerate(selected, 1):
        shutil.copy2(source, image_output_dir / f"{index:03d}__{source.name}")


def build_eval_manifest(
    selected: list[Path],
    record_by_final: dict[str, dict],
    anchor_index: dict[str, dict],
    output_root: Path,
) -> tuple[Path, list[dict]]:
    manifest_path = output_root.parent / "imagetest_manifest.jsonl"
    rows: list[dict] = []
    fallback_meta = next(iter(anchor_index.values()), None)
    for generated in selected:
        record = record_by_final.get(str(generated.resolve()))
        if not record:
            continue
        source = Path(record.get("source_path", ""))
        meta = anchor_index.get(source.name)
        if not source.is_file():
            continue
        if meta is None:
            if os.getenv("REFLECTVPR_IMAGETEST_GLOBAL_FALLBACK", "1").lower() in {"0", "false", "no"}:
                continue
            if fallback_meta is None:
                continue
            # Compute d(real, syn) even when the exact real-real positive is not in
            # the paired manifest. Later classification uses the 500-place global
            # real-real distribution instead of this placeholder d(real, real)=0.
            meta = {
                "positive_real": str(source),
                "negative_real": fallback_meta["negative_real"],
                "place_id": f"{source.parent.name}:{source.name.split('_')[1] if '_' in source.name else source.stem}",
                "negative_place_id": fallback_meta["negative_place_id"],
            }
        row = {
            "anchor_real": str(source),
            "positive_real": meta["positive_real"],
            "positive_syn": str(generated),
            "negative_real": meta["negative_real"],
            "place_id": meta["place_id"],
            "negative_place_id": meta["negative_place_id"],
        }
        rows.append(row)

    metadata = {
        "metadata": {
            "type": "paired_synthetic_triplet_manifest",
            "dataset_root": "mixed-runtime",
            "seed": 0,
            "triplets": len(rows),
        }
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(metadata, ensure_ascii=False) + "\n")
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return manifest_path, rows


def run_distance_analysis(distance_script: Path, manifest_path: Path, output_root: Path, count: int) -> Path:
    distance_output = output_root.parent / "imagetest_distance"
    distance_output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-reflectvpr")
    cmd = [
        sys.executable,
        str(distance_script),
        "--manifest",
        str(manifest_path),
        "--output-dir",
        str(distance_output),
        "--gallery-mode",
        "manifest",
        "--n-places",
        "0",
        "--max-pairs",
        str(count),
        "--batch-size",
        "32",
        "--num-workers",
        os.getenv("REFLECTVPR_IMAGETEST_NUM_WORKERS", "4"),
    ]
    subprocess.run(cmd, check=True, cwd=str(distance_script.parent), env=env)
    return distance_output / "pair_metrics.csv"


def load_reference_thresholds(reference_summary: Path) -> dict[str, float]:
    defaults = {
        "p50": 1.0785396099090576,
        "p75": 1.1421148180961609,
        "upper": 1.31,
    }
    if not reference_summary.is_file():
        return defaults
    try:
        data = json.loads(reference_summary.read_text(encoding="utf-8"))
        real_real = data.get("real_real", {})
        defaults["p50"] = float(real_real.get("p50", defaults["p50"]))
        defaults["p75"] = float(real_real.get("p75", defaults["p75"]))
    except Exception:
        return defaults
    return defaults


def classify_with_global_reference(distance: float, thresholds: dict[str, float]) -> str:
    if distance < thresholds["p50"]:
        return "too_easy"
    if distance < thresholds["p75"]:
        return "boundary"
    if distance <= thresholds["upper"]:
        return "hard"
    return "risky_noisy"


def load_categories(pair_metrics_csv: Path, reference_summary: Path | None = None) -> dict[str, str]:
    categories: dict[str, str] = {}
    if not pair_metrics_csv.is_file():
        return categories
    thresholds = load_reference_thresholds(reference_summary) if reference_summary else None
    with pair_metrics_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            category = row.get("category", "unknown")
            if thresholds and row.get("anchor_real") == row.get("positive_real"):
                distance_text = row.get("d_real_syn_l2") or row.get("d_pos_anchor_real_syn_l2")
                try:
                    category = classify_with_global_reference(float(distance_text), thresholds)
                except (TypeError, ValueError):
                    category = "unknown"
            categories[str(Path(row["positive_syn"]).resolve())] = category
    return categories


def load_image_metrics(
    pair_metrics_csv: Path,
    reference_summary: Path,
    effective_min: float,
    preferred_min: float,
    effective_max: float,
    real_real_eps: float,
) -> dict[str, dict]:
    metrics: dict[str, dict] = {}
    if not pair_metrics_csv.is_file():
        return metrics
    thresholds = load_reference_thresholds(reference_summary)
    with pair_metrics_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            generated = str(Path(row["positive_syn"]).resolve())
            distance_text = row.get("d_real_syn_l2") or row.get("d_pos_anchor_real_syn_l2")
            try:
                distance = float(distance_text)
            except (TypeError, ValueError):
                continue
            category = row.get("category") or classify_with_global_reference(distance, thresholds)
            if row.get("anchor_real") == row.get("positive_real"):
                category = classify_with_global_reference(distance, thresholds)

            real_real_distance = None
            for key in ("d_real_real_l2", "d_pos_anchor_real_real_l2", "d_anchor_positive_real_l2"):
                if row.get(key):
                    try:
                        real_real_distance = float(row[key])
                        break
                    except ValueError:
                        pass

            clears_real_real_gate = (
                real_real_distance is None
                or distance >= real_real_distance - real_real_eps
            )
            in_effective_band = effective_min <= distance <= effective_max
            in_preferred_band = preferred_min <= distance <= effective_max
            effective = clears_real_real_gate and in_effective_band
            metrics[generated] = {
                "distance": distance,
                "category": category,
                "real_real_distance": real_real_distance,
                "clears_real_real_gate": clears_real_real_gate,
                "in_effective_band": in_effective_band,
                "in_preferred_band": in_preferred_band,
                "effective": effective,
            }
    return metrics


def rank_effective_images(selected: list[Path], metrics: dict[str, dict], target: int) -> list[Path]:
    eligible = []
    for path in selected:
        item = metrics.get(str(path.resolve()))
        if not item or not item.get("effective"):
            continue
        eligible.append(path)
    eligible.sort(
        key=lambda path: (
            bool(metrics[str(path.resolve())].get("in_preferred_band")),
            float(metrics[str(path.resolve())].get("distance", 0.0)),
        ),
        reverse=True,
    )
    return eligible[:target]


def load_gpu_memory(gpu_log: Path) -> list[int]:
    memory: list[int] = []
    if not gpu_log.is_file():
        return memory
    with gpu_log.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                memory.append(int(row["memory_used_mib"].strip()))
            except Exception:
                continue
    return memory


def fit_image(image: Image.Image, width: int, height: int) -> Image.Image:
    canvas = Image.new("RGB", (width, height), "white")
    image.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas.paste(image, ((width - image.width) // 2, (height - image.height) // 2))
    return canvas


def color_for_category(category: str) -> str:
    return {
        "hard": "#b71c1c",
        "boundary": "#ef6c00",
        "easy": "#1565c0",
        "too_easy": "#1565c0",
        "risky_noisy": "#6a1b9a",
    }.get(category, "black")


def build_contact_sheet(
    selected: list[Path],
    record_by_final: dict[str, dict],
    categories: dict[str, str],
    contact_sheet: Path,
) -> int:
    cell_w, cell_h = 512, 288
    header_h = 52
    info_h = 34
    row_h = header_h + cell_h + info_h
    font = ImageFont.load_default()
    rows: list[Image.Image] = []

    for index, generated in enumerate(selected, 1):
        record = record_by_final.get(str(generated.resolve()))
        if not record:
            continue
        source = Path(record.get("source_path", ""))
        if not source.is_file():
            continue
        route = str(record.get("route", "unknown"))
        passed = bool(record.get("passed"))
        category = categories.get(str(generated.resolve()), "unknown")
        left = fit_image(Image.open(source).convert("RGB"), cell_w, cell_h)
        right = fit_image(Image.open(generated).convert("RGB"), cell_w, cell_h)
        row = Image.new("RGB", (cell_w * 2, row_h), "#f0f0f0")
        row.paste(left, (0, header_h))
        row.paste(right, (cell_w, header_h))
        draw = ImageDraw.Draw(row)
        draw.text((10, 10), f"{index:02d} Left: original", fill="black", font=font)
        draw.text((cell_w + 10, 10), f"{index:02d} Right: generated", fill="black", font=font)
        label = f"{route} | pass={str(passed).lower()} | ImAge={category}"
        draw.text((10, header_h + cell_h + 10), label, fill=color_for_category(category), font=font)
        rows.append(row)

    if not rows:
        return 0

    sheet = Image.new("RGB", (cell_w * 2, row_h * len(rows)), "#dddddd")
    for index, row in enumerate(rows):
        sheet.paste(row, (0, index * row_h))
    sheet.save(contact_sheet, quality=92)
    return len(rows)


def main() -> None:
    args = parse_args()
    records = load_records(args.output_root)
    finals = collect_finals(args.output_root)
    record_by_final = {
        str(Path(record.get("output_path", "")).resolve()): record
        for record in records
        if record.get("output_path")
    }
    selected = copy_selected_images(
        finals,
        args.image_output_dir,
        args.target_images,
        record_by_final=record_by_final,
        passed_only=args.passed_only,
    )
    anchor_index = build_anchor_index(args.paired_manifest)
    manifest_path, manifest_rows = build_eval_manifest(selected, record_by_final, anchor_index, args.output_root)
    categories: dict[str, str] = {}
    image_metrics: dict[str, dict] = {}
    pair_metrics_csv = None
    if manifest_rows:
        try:
            pair_metrics_csv = run_distance_analysis(
                args.distance_script,
                manifest_path,
                args.output_root,
                len(manifest_rows),
            )
            categories = load_categories(pair_metrics_csv, args.reference_summary)
            image_metrics = load_image_metrics(
                pair_metrics_csv,
                args.reference_summary,
                args.effective_min_distance,
                args.preferred_min_distance,
                args.effective_max_distance,
                args.real_real_eps,
            )
        except Exception as exc:
            fallback_csv = args.output_root.parent / "imagetest_distance" / "pair_metrics.csv"
            if fallback_csv.is_file():
                pair_metrics_csv = fallback_csv
                categories = load_categories(fallback_csv, args.reference_summary)
                image_metrics = load_image_metrics(
                    fallback_csv,
                    args.reference_summary,
                    args.effective_min_distance,
                    args.preferred_min_distance,
                    args.effective_max_distance,
                    args.real_real_eps,
                )
                print(
                    f"[postprocess] ImAge distance analysis ended non-zero, "
                    f"but loaded existing pair metrics: {fallback_csv}",
                    flush=True,
                )
            else:
                print(f"[postprocess] ImAge distance analysis failed; keep ImAge=unknown labels: {exc}", flush=True)
    effective_selected = rank_effective_images(selected, image_metrics, args.target_images) if image_metrics else selected
    if image_metrics:
        replace_selected_images(effective_selected, args.image_output_dir)
        selected = effective_selected
    comparison_pairs = build_contact_sheet(selected, record_by_final, categories, args.contact_sheet)
    memory = load_gpu_memory(args.gpu_log)
    routes = Counter(str(record.get("route", "unknown")) for record in records)
    category_counts = Counter(categories.values())
    candidate_count = len(finals)
    passed_candidate_count = sum(
        1
        for final in finals
        if bool(record_by_final.get(str(final.resolve()), {}).get("passed"))
    )
    effective_count = sum(1 for item in image_metrics.values() if item.get("effective")) if image_metrics else len(selected)
    preferred_count = sum(1 for item in image_metrics.values() if item.get("in_preferred_band")) if image_metrics else 0
    summary = {
        "input_records": len(records),
        "generated_final_images": candidate_count,
        "collected_images": len(selected),
        "target_images": args.target_images,
        "target_met": len(selected) >= args.target_images,
        "passed_only_selection": bool(args.passed_only),
        "effective_sample_definition": "passed=true and ImAge d(real,syn) clears real-real-eps gate and falls in effective distance band",
        "effective_distance_band": [args.effective_min_distance, args.effective_max_distance],
        "preferred_distance_band": [args.preferred_min_distance, args.effective_max_distance],
        "real_real_eps": args.real_real_eps,
        "passed_candidates": passed_candidate_count,
        "image_effective_candidates": effective_count,
        "image_preferred_candidates": preferred_count,
        "effective_utilization": (effective_count / candidate_count) if candidate_count else 0.0,
        "selected_effective_utilization": (len(selected) / candidate_count) if candidate_count else 0.0,
        "routes": dict(routes),
        "passed": sum(bool(record.get("passed")) for record in records),
        "generation_failed": sum(bool(record.get("generation_failed")) for record in records),
        "router_failed": sum(bool(record.get("router_failed")) for record in records),
        "elapsed_seconds": args.elapsed_seconds,
        "gpu_memory_min_mib": min(memory) if memory else None,
        "gpu_memory_max_mib": max(memory) if memory else None,
        "images_dir": str(args.image_output_dir),
        "contact_sheet": str(args.contact_sheet),
        "comparison_pairs": comparison_pairs,
        "imagetest_manifest": str(manifest_path),
        "imagetest_pair_metrics": str(pair_metrics_csv),
        "imagetest_category_counts": dict(category_counts),
    }
    args.summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if not summary["target_met"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
