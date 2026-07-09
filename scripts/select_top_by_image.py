#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


DEFAULT_REFERENCE_SUMMARY = Path("/media/data/zhangjingyi/ImAge_v3_LORA/distance_analysis_500places/summary.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--pair-metrics", type=Path, required=True)
    parser.add_argument("--image-output-dir", type=Path, required=True)
    parser.add_argument("--contact-sheet", type=Path, required=True)
    parser.add_argument("--summary-path", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--passed-only", action="store_true", default=True)
    parser.add_argument("--max-distance", type=float, default=1.31)
    parser.add_argument("--effective-min-distance", type=float, default=1.08)
    parser.add_argument("--preferred-min-distance", type=float, default=1.14)
    parser.add_argument("--real-real-eps", type=float, default=0.02)
    parser.add_argument("--reference-summary", type=Path, default=DEFAULT_REFERENCE_SUMMARY)
    return parser.parse_args()


def load_records(output_root: Path) -> dict[str, dict]:
    records: dict[str, dict] = {}
    for path in sorted(output_root.glob("*/reflect.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(data, list):
            continue
        for record in data:
            output_path = record.get("output_path")
            if output_path:
                records[str(Path(output_path).resolve())] = record
            for candidate in record.get("training_candidates", []) or []:
                candidate_record = dict(record)
                candidate_record.update(candidate)
                candidate_record["parent_output_path"] = record.get("output_path")
                candidate_path = candidate_record.get("output_path")
                if candidate_path:
                    records[str(Path(candidate_path).resolve())] = candidate_record
    return records


def load_thresholds(path: Path) -> dict[str, float]:
    thresholds = {"p50": 1.0785396099090576, "p75": 1.1421148180961609, "upper": 1.31}
    if not path.is_file():
        return thresholds
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        real_real = data.get("real_real", {})
        thresholds["p50"] = float(real_real.get("p50", thresholds["p50"]))
        thresholds["p75"] = float(real_real.get("p75", thresholds["p75"]))
    except Exception:
        return thresholds
    return thresholds


def classify(distance: float, thresholds: dict[str, float]) -> str:
    if distance < thresholds["p50"]:
        return "too_easy"
    if distance < thresholds["p75"]:
        return "boundary"
    if distance <= thresholds["upper"]:
        return "hard"
    return "risky_noisy"


def distance_from_row(row: dict) -> float:
    return float(row.get("d_real_syn_l2") or row.get("d_pos_anchor_real_syn_l2"))


def real_real_distance_from_row(row: dict) -> float | None:
    for key in ("d_real_real_l2", "d_pos_anchor_real_real_l2", "d_anchor_positive_real_l2"):
        if row.get(key):
            try:
                return float(row[key])
            except ValueError:
                pass
    return None


def fit_image(image: Image.Image, width: int, height: int) -> Image.Image:
    canvas = Image.new("RGB", (width, height), "white")
    image.thumbnail((width, height), Image.Resampling.LANCZOS)
    canvas.paste(image, ((width - image.width) // 2, (height - image.height) // 2))
    return canvas


def color_for_category(category: str) -> str:
    return {
        "hard": "#b71c1c",
        "boundary": "#ef6c00",
        "too_easy": "#1565c0",
        "risky_noisy": "#6a1b9a",
    }.get(category, "black")


def build_contact_sheet(rows: list[dict], record_by_final: dict[str, dict], contact_sheet: Path) -> int:
    cell_w, cell_h = 512, 288
    header_h = 58
    info_h = 42
    row_h = header_h + cell_h + info_h
    font = ImageFont.load_default()
    sheet = Image.new("RGB", (cell_w * 2, row_h * len(rows)), "#dddddd")

    for index, row_data in enumerate(rows, 1):
        generated = Path(row_data["positive_syn"])
        source = Path(row_data["anchor_real"])
        record = record_by_final.get(str(generated.resolve()), {})
        row = Image.new("RGB", (cell_w * 2, row_h), "#f0f0f0")
        row.paste(fit_image(Image.open(source).convert("RGB"), cell_w, cell_h), (0, header_h))
        row.paste(fit_image(Image.open(generated).convert("RGB"), cell_w, cell_h), (cell_w, header_h))
        draw = ImageDraw.Draw(row)
        draw.text((10, 10), f"{index:02d} Left: original", fill="black", font=font)
        draw.text((cell_w + 10, 10), f"{index:02d} Right: generated", fill="black", font=font)
        label = (
            f"dual | pass={str(bool(record.get('passed'))).lower()} | "
            f"ImAge={row_data['category']} | d={row_data['distance']:.4f}"
        )
        draw.text((10, header_h + cell_h + 10), label, fill=color_for_category(row_data["category"]), font=font)
        sheet.paste(row, (0, (index - 1) * row_h))
    if rows:
        sheet.save(contact_sheet, quality=92)
    return len(rows)


def main() -> None:
    args = parse_args()
    record_by_final = load_records(args.output_root)
    thresholds = load_thresholds(args.reference_summary)
    rows: list[dict] = []
    candidate_rows = 0
    passed_rows = 0
    with args.pair_metrics.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            candidate_rows += 1
            generated = str(Path(row["positive_syn"]).resolve())
            record = record_by_final.get(generated, {})
            if args.passed_only and not bool(record.get("passed")):
                continue
            passed_rows += 1
            distance = distance_from_row(row)
            category = classify(distance, thresholds)
            real_real_distance = real_real_distance_from_row(row)
            clears_real_real_gate = (
                real_real_distance is None
                or distance >= real_real_distance - args.real_real_eps
            )
            in_effective_band = args.effective_min_distance <= distance <= args.max_distance
            in_preferred_band = args.preferred_min_distance <= distance <= args.max_distance
            if not clears_real_real_gate or not in_effective_band:
                continue
            row["distance"] = distance
            row["real_real_distance"] = real_real_distance
            row["category"] = category
            row["clears_real_real_gate"] = clears_real_real_gate
            row["in_effective_band"] = in_effective_band
            row["in_preferred_band"] = in_preferred_band
            rows.append(row)

    rows = sorted(
        rows,
        key=lambda row: (bool(row["in_preferred_band"]), row["distance"]),
        reverse=True,
    )[: args.top_k]

    args.image_output_dir.mkdir(parents=True, exist_ok=True)
    for old in args.image_output_dir.glob("*.jpg"):
        old.unlink()
    for index, row in enumerate(rows, 1):
        generated = Path(row["positive_syn"])
        shutil.copy2(generated, args.image_output_dir / f"{index:03d}__{generated.name}")

    comparison_pairs = build_contact_sheet(rows, record_by_final, args.contact_sheet)
    summary_rows = []
    for index, row in enumerate(rows, 1):
        record = record_by_final.get(str(Path(row["positive_syn"]).resolve()), {})
        summary_rows.append(
            {
                "rank": index,
                "category": row["category"],
                "distance": row["distance"],
                "real_real_distance": row.get("real_real_distance"),
                "clears_real_real_gate": bool(row.get("clears_real_real_gate")),
                "in_preferred_band": bool(row.get("in_preferred_band")),
                "passed": bool(record.get("passed")),
                "s_geo": record.get("s_geo"),
                "s_div": record.get("s_div"),
                "condition": row.get("condition"),
                "generated": row["positive_syn"],
                "original": row["anchor_real"],
            }
        )
    summary = {
        "selected_top_k": len(rows),
        "requested_top_k": args.top_k,
        "passed_only": bool(args.passed_only),
        "effective_sample_definition": "passed=true and ImAge d(real,syn) clears real-real-eps gate and falls in effective distance band",
        "effective_distance_band": [args.effective_min_distance, args.max_distance],
        "preferred_distance_band": [args.preferred_min_distance, args.max_distance],
        "real_real_eps": args.real_real_eps,
        "max_distance": args.max_distance,
        "generated_candidates": candidate_rows,
        "passed_candidates": passed_rows,
        "effective_candidates": len(rows),
        "category_counts": dict(Counter(row["category"] for row in rows)),
        "effective_utilization": (len(rows) / candidate_rows) if candidate_rows else 0.0,
        "comparison_pairs": comparison_pairs,
        "contact_sheet": str(args.contact_sheet),
        "images_dir": str(args.image_output_dir),
        "rows": summary_rows,
    }
    args.summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
