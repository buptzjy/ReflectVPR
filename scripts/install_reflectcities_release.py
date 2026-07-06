#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Install a ReflectCities generated release and rebuild the mixed dataset "
            "from the base dataset plus generated metadata."
        )
    )
    parser.add_argument(
        "--reflectcities-root",
        default="/data_nvme/zhangjingyi/ReflectCities",
        help="Root containing generated/, gsv/, mixed_GSV/, unified/, mixed_Unified/.",
    )
    parser.add_argument(
        "--base-root",
        default="/data_nvme/zhangjingyi/GSV-Cities",
        help="Base dataset root containing Images/ and Dataframes/.",
    )
    parser.add_argument(
        "--unified-root",
        default=None,
        help="Deprecated alias for --base-root.",
    )
    parser.add_argument(
        "--generated-dir",
        default=None,
        help="Generated release dir. Defaults to <reflectcities-root>/generated.",
    )
    parser.add_argument(
        "--mixed-name",
        default="mixed_GSV",
        help="Output mixed dataset directory name under reflectcities-root.",
    )
    parser.add_argument(
        "--original-name",
        default="gsv",
        help="Output original dataset link directory name under reflectcities-root.",
    )
    parser.add_argument(
        "--copy-images",
        action="store_true",
        help="Copy generated images into the mixed dataset instead of symlinking.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any metadata row cannot be installed.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), list(reader)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def replace_path(src: Path, dst: Path) -> None:
    if dst.is_symlink() or dst.is_file():
        dst.unlink()
    elif dst.exists():
        shutil.rmtree(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.symlink_to(src)


def safe_link(src: Path, dst: Path, *, copy: bool = False) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        if dst.is_symlink() and os.readlink(dst) == str(src):
            return "exists"
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    if copy:
        shutil.copy2(src, dst)
        return "copy"
    dst.symlink_to(src)
    return "symlink"


def metadata_rows(generated_dir: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for csv_path in sorted((generated_dir / "metadata").glob("*.csv")):
        _, city_rows = read_csv(csv_path)
        rows.extend(city_rows)
    return rows


def generated_image_path(generated_dir: Path, row: dict[str, str]) -> Path:
    city = row["city"]
    route = row["route"]
    filename = row.get("dedup_filename") or row.get("mixed_filename")
    if not filename:
        filename = Path(row["generated_path"]).name
    local_path = generated_dir / "images" / city / route / filename
    if local_path.exists():
        return local_path
    return Path(row.get("generated_path", ""))


def generated_train_row(row: dict[str, str]) -> dict[str, Any]:
    filename = row.get("dedup_filename") or row.get("mixed_filename")
    if not filename:
        filename = Path(row["generated_path"]).name
    panoid = filename[:-4].split("_", 7)[7] if filename.endswith(".jpg") else filename
    return {
        "place_id": row["source_place_id"],
        "year": row["source_year"],
        "month": row["source_month"],
        "northdeg": row["source_northdeg"],
        "city_id": row["city"],
        "lat": row["source_lat"],
        "lon": row["source_lon"],
        "panoid": panoid,
        "similarity": "1.0",
    }


def install_original_links(
    base_root: Path, reflectcities_root: Path, original_name: str
) -> dict[str, int]:
    original_root = reflectcities_root / original_name
    images = base_root / "Images"
    dataframes = base_root / "Dataframes"
    image_dirs = 0
    csvs = 0

    (original_root / "Images").mkdir(parents=True, exist_ok=True)
    (original_root / "Dataframes").mkdir(parents=True, exist_ok=True)

    for city_dir in sorted(images.iterdir()):
        if city_dir.is_dir():
            replace_path(city_dir, original_root / "Images" / city_dir.name)
            image_dirs += 1
    for csv_path in sorted(dataframes.glob("*.csv")):
        replace_path(csv_path, original_root / "Dataframes" / csv_path.name)
        csvs += 1

    return {"original_image_dir_links": image_dirs, "original_csv_links": csvs}


def rebuild_mixed(
    base_root: Path,
    reflectcities_root: Path,
    generated_dir: Path,
    mixed_name: str,
    rows: list[dict[str, str]],
    *,
    copy_images: bool,
    strict: bool,
) -> dict[str, Any]:
    mixed_root = reflectcities_root / mixed_name
    mixed_images = mixed_root / "Images"
    mixed_csvs = mixed_root / "Dataframes"
    mixed_images.mkdir(parents=True, exist_ok=True)
    mixed_csvs.mkdir(parents=True, exist_ok=True)

    by_city: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_city[row["city"]].append(row)

    stats: Counter[str] = Counter()
    route_counts: Counter[str] = Counter()
    link_modes: Counter[str] = Counter()
    bad_rows: list[dict[str, str]] = []

    for csv_path in sorted((base_root / "Dataframes").glob("*.csv")):
        city = csv_path.stem
        source_img_dir = base_root / "Images" / city
        mixed_img_dir = mixed_images / city
        mixed_csv = mixed_csvs / csv_path.name
        city_generated = by_city.get(city, [])

        if not city_generated:
            replace_path(source_img_dir, mixed_img_dir)
            replace_path(csv_path, mixed_csv)
            stats["cities_symlinked_without_generated"] += 1
            continue

        if mixed_img_dir.is_symlink() or mixed_img_dir.is_file():
            mixed_img_dir.unlink()
        mixed_img_dir.mkdir(parents=True, exist_ok=True)

        for src_img in sorted(source_img_dir.iterdir()):
            if src_img.is_file():
                link_modes[safe_link(src_img, mixed_img_dir / src_img.name)] += 1

        fieldnames, original_rows = read_csv(csv_path)
        output_fields = list(fieldnames)
        for column in TRAIN_COLUMNS:
            if column not in output_fields:
                output_fields.append(column)

        normalized_rows: list[dict[str, Any]] = []
        for original_row in original_rows:
            copied = dict(original_row)
            copied.setdefault("similarity", "1.0")
            normalized_rows.append(copied)

        generated_rows: list[dict[str, Any]] = []
        for row in city_generated:
            if row.get("passed") not in {"True", "true", "1"} and row.get("final_pass") not in {
                "True",
                "true",
                "1",
            }:
                stats["metadata_not_pass_true"] += 1
                continue
            if row.get("route") not in {"global", "local", "dual"}:
                stats["metadata_non_training_route"] += 1
                continue

            image_path = generated_image_path(generated_dir, row)
            if not image_path.exists():
                stats["missing_generated_images"] += 1
                bad_rows.append(row)
                continue

            filename = row.get("dedup_filename") or row.get("mixed_filename") or image_path.name
            link_modes[safe_link(image_path, mixed_img_dir / filename, copy=copy_images)] += 1
            generated_rows.append(generated_train_row(row))
            route_counts[row["route"]] += 1

        write_csv(mixed_csv, output_fields, normalized_rows + generated_rows)
        stats["cities_rebuilt_with_generated"] += 1
        stats["generated_rows_written"] += len(generated_rows)

    if strict and bad_rows:
        raise RuntimeError(f"Missing generated images: {len(bad_rows)}")

    return {
        **dict(stats),
        "route_counts": dict(route_counts),
        "link_modes": dict(link_modes),
        "bad_rows_sample": bad_rows[:20],
    }


def verify_generated(rows: list[dict[str, str]], generated_dir: Path) -> dict[str, Any]:
    reflect_paths = [row.get("reflect_record_path", "") for row in rows]
    generated_paths = [str(generated_image_path(generated_dir, row)) for row in rows]
    routes = Counter(row.get("route", "") for row in rows)
    passed = Counter(row.get("passed") or row.get("final_pass", "") for row in rows)
    missing = sum(1 for path in generated_paths if not Path(path).exists())
    return {
        "generated_metadata_rows": len(rows),
        "generated_unique_reflect_records": len(set(reflect_paths)),
        "generated_duplicate_reflect_records": len(rows) - len(set(reflect_paths)),
        "generated_unique_image_paths": len(set(generated_paths)),
        "generated_duplicate_image_paths": len(rows) - len(set(generated_paths)),
        "generated_missing_images": missing,
        "generated_routes": dict(routes),
        "generated_pass_values": dict(passed),
    }


def main() -> None:
    args = parse_args()
    reflectcities_root = Path(args.reflectcities_root)
    base_root = Path(args.unified_root or args.base_root)
    generated_dir = Path(args.generated_dir) if args.generated_dir else reflectcities_root / "generated"

    if not (base_root / "Images").is_dir() or not (base_root / "Dataframes").is_dir():
        raise SystemExit(f"Invalid base root: {base_root}")
    if not (generated_dir / "images").is_dir() or not (generated_dir / "metadata").is_dir():
        raise SystemExit(f"Invalid generated dir: {generated_dir}")

    rows = metadata_rows(generated_dir)
    original_summary = install_original_links(base_root, reflectcities_root, args.original_name)
    mixed_summary = rebuild_mixed(
        base_root,
        reflectcities_root,
        generated_dir,
        args.mixed_name,
        rows,
        copy_images=args.copy_images,
        strict=args.strict,
    )
    summary = {
        "reflectcities_root": str(reflectcities_root),
        "base_root": str(base_root),
        "generated_dir": str(generated_dir),
        "original_root": str(reflectcities_root / args.original_name),
        "mixed_root": str(reflectcities_root / args.mixed_name),
        **original_summary,
        **verify_generated(rows, generated_dir),
        **mixed_summary,
    }
    summary_path = reflectcities_root / "logs" / "verification" / "install_reflectcities_release_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
