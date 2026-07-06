#!/usr/bin/env python3
"""Backfill missing decision records from existing reflect records.

Some completed generations may have a reflect JSON and r1 image but no matching
single decision JSON, usually after interrupted early runs. This script restores
those decision records without overwriting existing planner outputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DECISION_KEYS = {
    "file_name",
    "city",
    "route",
    "weather",
    "occlusion",
    "weather_score",
    "occlusion_score",
    "selected_model",
    "position",
    "prompt",
    "reason",
    "skip_reason",
    "street_scene_quality",
    "occlusion_feasibility",
    "weather_feasibility",
    "balance_reason",
    "risk_score",
    "risk_flags",
    "skip_recommendation",
    "source_path",
    "original_route",
    "downgrade_reason",
    "bad_image",
    "quota_eligible_routes",
    "quota_deficits",
    "quota_route_scores",
    "quota_reason",
    "experience_bank_version",
    "prompt_policy_version",
    "router_failed",
    "router_status",
    "router_error_type",
    "router_error",
    "original_vehicle_crowded",
    "original_vehicle_crowding_reasons",
    "local_vehicle_policy",
    "selected_weather",
    "weather_selection_reason",
    "scene_policy_restricted_weather",
    "scene_policy_restriction_reasons",
    "scene_weather_original",
    "scene_weather_restricted_to",
    "global_weather_counts_before",
    "global_weather_pass_counts_before",
    "global_weather_pass_counts_projected_after",
    "global_weather_target_ratios",
    "global_weather_candidate_pool",
    "global_weather_deficits",
    "global_weather_quota_basis",
    "global_iclight_highres_denoise",
    "global_scene_policy",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return data


def decision_from_reflect(record: dict[str, Any], reflect_path: Path) -> dict[str, Any]:
    decision = {key: record[key] for key in DECISION_KEYS if key in record}
    if "prompt" not in decision and record.get("final_prompt"):
        decision["prompt"] = record["final_prompt"]
    if "source_path" not in decision:
        raise ValueError(f"Reflect record has no source_path: {reflect_path}")
    if "route" not in decision:
        raise ValueError(f"Reflect record has no route: {reflect_path}")
    decision.setdefault("city", reflect_path.parent.name)
    decision.setdefault("file_name", Path(str(decision["source_path"])).name)
    decision.setdefault("router_failed", False)
    decision["backfilled_from_reflect"] = True
    decision["backfilled_reflect_record_path"] = str(reflect_path)
    return decision


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        default="/data_nvme/zhangjingyi/ReflectVPR/output_gsv23_100k",
        help="ReflectVPR output root.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    record_root = output_root / "_records"
    decision_root = record_root / "decisions"
    reflect_root = record_root / "reflects"

    r1_paths = list(output_root.glob("*/rounds/*/r1.jpg"))
    checked = 0
    backfilled = 0
    missing_reflect = 0
    skipped_existing = 0
    errors = 0

    for r1_path in r1_paths:
        city = r1_path.parts[-4]
        stem = r1_path.parent.name
        decision_path = decision_root / city / f"{stem}.json"
        if decision_path.exists():
            skipped_existing += 1
            continue
        checked += 1
        reflect_path = reflect_root / city / f"{stem}.json"
        if not reflect_path.exists():
            missing_reflect += 1
            continue
        try:
            decision = decision_from_reflect(load_json(reflect_path), reflect_path)
            if not args.dry_run:
                atomic_write_json(decision_path, decision)
            backfilled += 1
        except Exception as exc:  # noqa: BLE001
            errors += 1
            print(f"[error] {reflect_path}: {type(exc).__name__}: {exc}", flush=True)

    print(
        " ".join(
            [
                f"r1_total={len(r1_paths)}",
                f"missing_decision_checked={checked}",
                f"backfilled={backfilled}",
                f"missing_reflect={missing_reflect}",
                f"skipped_existing={skipped_existing}",
                f"errors={errors}",
                f"dry_run={args.dry_run}",
            ]
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
