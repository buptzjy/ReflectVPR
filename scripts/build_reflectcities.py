#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any


UNIFIED_ROOT = Path("/data_nvme/zhangjingyi/Unified_cities")
REFLECT_OUT = Path("/data_nvme/zhangjingyi/ReflectVPR/output_gsv23_100k")
DEST_ROOT = Path("/data_nvme/zhangjingyi/ReflectCities")

ORIGINAL_ROOT = DEST_ROOT / "original"
GENERATED_ROOT = DEST_ROOT / "generated"
MIXED_ROOT = DEST_ROOT / "mixed_unified"

TRAIN_COLUMNS = [
    "place_id",
    "year",
    "month",
    "northdeg",
    "city_id",
    "lat",
    "lon",
    "panoid",
    "similarity",
]

GEN_METADATA_COLUMNS = [
    "candidate_id",
    "source_id",
    "city",
    "source_path",
    "generated_path",
    "route",
    "weather",
    "occlusion",
    "generator",
    "seed_instruction",
    "generation_seed",
    "final_prompt",
    "s_geo",
    "s_div",
    "geo_pass",
    "div_pass",
    "artifact_pass",
    "final_pass",
    "failure_reason",
    "source_place_id",
    "source_year",
    "source_month",
    "source_northdeg",
    "source_lat",
    "source_lon",
    "source_panoid",
    "mixed_filename",
    "reflect_record_path",
]

ROUTE_TO_GENERATOR = {
    "global": "IC-Light",
    "local": "LightX2V-Local",
    "dual": "LightX2V-Dual",
}


def safe_symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink():
        if os.readlink(dst) == str(src):
            return
        dst.unlink()
    elif dst.exists():
        return
    dst.symlink_to(src)


def parse_source_image_name(path: str) -> dict[str, Any] | None:
    name = Path(path).name
    if not name.endswith(".jpg"):
        return None
    parts = name[:-4].split("_")
    if len(parts) < 8:
        return None
    try:
        return {
            "city": parts[0],
            "place_id": int(parts[1]),
            "year": int(parts[2]),
            "month": int(parts[3]),
            "northdeg": int(parts[4]),
            "lat": parts[5],
            "lon": parts[6],
            "panoid": "_".join(parts[7:]),
        }
    except ValueError:
        return None


def canonical_filename(city: str, row: dict[str, Any]) -> str:
    return (
        f"{city}_{int(row['place_id']):07d}_{int(row['year']):04d}_"
        f"{int(row['month']):02d}_{int(row['northdeg']):03d}_"
        f"{row['lat']}_{row['lon']}_{row['panoid']}.jpg"
    )


def read_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return list(reader.fieldnames or []), rows


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def failure_reason(record: dict[str, Any]) -> str:
    if record.get("passed") is True:
        return ""
    reasons: list[str] = []
    if record.get("generation_failed"):
        reasons.append(str(record.get("generation_error_type") or "generation_failed"))
    if record.get("geo_ok") is False:
        reasons.append("geo_failed")
    if record.get("div_ok") is False:
        reasons.append("div_failed")
    if record.get("artifact_ok") is False:
        reasons.append("artifact_failed")
    for key in ("quality_gate_flags", "risk_flags"):
        value = record.get(key)
        if isinstance(value, list):
            reasons.extend(str(item) for item in value)
    return ";".join(dict.fromkeys(item for item in reasons if item))


def build_original_links() -> tuple[int, int]:
    image_count = 0
    csv_count = 0
    for city_dir in sorted((UNIFIED_ROOT / "Images").iterdir()):
        if city_dir.is_dir():
            safe_symlink(city_dir, ORIGINAL_ROOT / "Images" / city_dir.name)
            image_count += 1
    for csv_file in sorted((UNIFIED_ROOT / "Dataframes").glob("*.csv")):
        safe_symlink(csv_file, ORIGINAL_ROOT / "Dataframes" / csv_file.name)
        csv_count += 1
    return image_count, csv_count


def collect_passed_records() -> list[dict[str, Any]]:
    reflects = REFLECT_OUT / "_records" / "reflects"
    records: list[dict[str, Any]] = []
    for record_path in sorted(reflects.glob("*/*.json")):
        try:
            record = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if record.get("passed") is not True:
            continue
        route = str(record.get("route") or "")
        if route not in {"global", "local", "dual"}:
            continue
        source_path = str(record.get("source_path") or "")
        output_path = str(record.get("output_path") or "")
        if not source_path or not output_path:
            continue
        if not Path(output_path).exists():
            continue
        parsed = parse_source_image_name(source_path)
        if parsed is None:
            continue
        record["_parsed_source"] = parsed
        record["_record_path"] = str(record_path)
        records.append(record)
    return records


def generated_panoid(record: dict[str, Any]) -> str:
    parsed = record["_parsed_source"]
    route = record.get("route") or "unknown"
    weather = record.get("weather") or "none"
    occlusion = record.get("occlusion") or "none"
    return f"{parsed['panoid']}__reflectvpr_{route}_{weather}_{occlusion}"


def build_generated(records: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    by_city: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_names: set[tuple[str, str, str]] = set()

    for index, record in enumerate(records, 1):
        parsed = record["_parsed_source"]
        city = parsed["city"]
        route = str(record.get("route"))
        row = {
            "place_id": parsed["place_id"],
            "year": parsed["year"],
            "month": parsed["month"],
            "northdeg": parsed["northdeg"],
            "city_id": city,
            "lat": parsed["lat"],
            "lon": parsed["lon"],
            "panoid": generated_panoid(record),
            "similarity": "1.0",
        }
        filename = canonical_filename(city, row)
        key = (city, route, filename)
        if key in seen_names:
            continue
        seen_names.add(key)

        src = Path(str(record["output_path"]))
        generated_dst = GENERATED_ROOT / "images" / city / route / filename
        generated_dst.parent.mkdir(parents=True, exist_ok=True)
        if not generated_dst.exists():
            shutil.copy2(src, generated_dst)

        candidate_id = f"{city}_{parsed['place_id']:07d}_{index:06d}_{route}"
        meta = {
            "candidate_id": candidate_id,
            "source_id": Path(str(record.get("source_path"))).stem,
            "city": city,
            "source_path": record.get("source_path", ""),
            "generated_path": str(generated_dst),
            "route": route,
            "weather": record.get("weather", ""),
            "occlusion": record.get("occlusion", ""),
            "generator": record.get("selected_model") or ROUTE_TO_GENERATOR.get(route, ""),
            "seed_instruction": " ".join(
                str(record.get(key, "") or "")
                for key in ("weather", "occlusion", "position")
            ).strip(),
            "generation_seed": record.get("generation_seed", ""),
            "final_prompt": record.get("final_prompt") or record.get("prompt", ""),
            "s_geo": record.get("s_geo", ""),
            "s_div": record.get("s_div", ""),
            "geo_pass": record.get("geo_ok", ""),
            "div_pass": record.get("div_ok", ""),
            "artifact_pass": record.get("artifact_ok", ""),
            "final_pass": record.get("passed", ""),
            "failure_reason": failure_reason(record),
            "source_place_id": parsed["place_id"],
            "source_year": parsed["year"],
            "source_month": parsed["month"],
            "source_northdeg": parsed["northdeg"],
            "source_lat": parsed["lat"],
            "source_lon": parsed["lon"],
            "source_panoid": parsed["panoid"],
            "mixed_filename": filename,
            "reflect_record_path": record.get("_record_path", ""),
            "_train_row": row,
            "_generated_dst": generated_dst,
        }
        by_city[city].append(meta)

    for city, rows in by_city.items():
        write_csv(
            GENERATED_ROOT / "metadata" / f"{city}.csv",
            GEN_METADATA_COLUMNS,
            rows,
        )
    return by_city


def build_mixed(by_city: dict[str, list[dict[str, Any]]]) -> tuple[int, int, int]:
    mixed_image_dirs = 0
    mixed_csvs = 0
    generated_links = 0
    affected_cities = set(by_city)

    for csv_file in sorted((UNIFIED_ROOT / "Dataframes").glob("*.csv")):
        city = csv_file.stem
        source_img_dir = UNIFIED_ROOT / "Images" / city
        mixed_img_dir = MIXED_ROOT / "Images" / city
        mixed_csv = MIXED_ROOT / "Dataframes" / csv_file.name

        if city not in affected_cities:
            safe_symlink(source_img_dir, mixed_img_dir)
            safe_symlink(csv_file, mixed_csv)
            continue

        mixed_img_dir.mkdir(parents=True, exist_ok=True)
        if source_img_dir.exists():
            for item in source_img_dir.iterdir():
                if item.is_file():
                    safe_symlink(item, mixed_img_dir / item.name)
        mixed_image_dirs += 1

        fieldnames, original_rows = read_csv_rows(csv_file)
        output_fields = list(fieldnames)
        if "similarity" not in output_fields:
            output_fields.append("similarity")
        for required in TRAIN_COLUMNS:
            if required not in output_fields:
                output_fields.append(required)

        normalized_original_rows: list[dict[str, Any]] = []
        for row in original_rows:
            copied = dict(row)
            copied.setdefault("similarity", "1.0")
            normalized_original_rows.append(copied)

        generated_train_rows = []
        for meta in by_city[city]:
            gen_dst: Path = meta["_generated_dst"]
            train_row = dict(meta["_train_row"])
            safe_symlink(gen_dst, mixed_img_dir / meta["mixed_filename"])
            generated_links += 1
            generated_train_rows.append(train_row)

        write_csv(mixed_csv, output_fields, normalized_original_rows + generated_train_rows)
        mixed_csvs += 1

    return mixed_image_dirs, mixed_csvs, generated_links


def main() -> None:
    for path in [
        ORIGINAL_ROOT / "Images",
        ORIGINAL_ROOT / "Dataframes",
        GENERATED_ROOT / "images",
        GENERATED_ROOT / "metadata",
        MIXED_ROOT / "Images",
        MIXED_ROOT / "Dataframes",
        DEST_ROOT / "logs" / "generation",
        DEST_ROOT / "logs" / "verification",
    ]:
        path.mkdir(parents=True, exist_ok=True)

    original_image_dirs, original_csvs = build_original_links()
    records = collect_passed_records()
    by_city = build_generated(records)
    mixed_image_dirs, mixed_csvs, generated_links = build_mixed(by_city)

    summary = {
        "dest_root": str(DEST_ROOT),
        "original_image_dir_links": original_image_dirs,
        "original_csv_links": original_csvs,
        "passed_generated_records": len(records),
        "generated_cities": len(by_city),
        "generated_images": sum(len(rows) for rows in by_city.values()),
        "mixed_cities_with_real_dirs": mixed_image_dirs,
        "mixed_csvs_rebuilt": mixed_csvs,
        "mixed_generated_image_links": generated_links,
    }
    (DEST_ROOT / "logs" / "verification" / "build_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
