import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path("/media/data/zhangjingyi/ReflectVPR")
AGENT_DIR = ROOT / "your_agent"
sys.path.insert(0, str(AGENT_DIR))

from prompt_rules import (  # noqa: E402
    EXPERIENCE_RULES,
    REFLECTION_LESSONS,
    normalize_experience_bank,
)


def route_key(record: dict) -> str:
    route = record.get("route", "unknown")
    weather = record.get("weather")
    occlusion = record.get("occlusion")
    if route == "global":
        return f"global_{weather}"
    if route == "local":
        return f"local_{occlusion}"
    if route == "dual":
        return f"dual_{weather}_{occlusion}"
    return str(route)


def eval_from_record(record: dict) -> dict:
    if "s_geo" in record or "s_div" in record:
        return {
            "passed": bool(record.get("passed")),
            "s_geo": record.get("s_geo"),
            "s_div": record.get("s_div"),
        }
    # Backward compatibility with older VLM-judge JSON.
    return {
        "passed": bool(record.get("passed")),
        "geometry_score": record.get("geometry_score"),
        "edit_score": record.get("edit_score"),
        "artifact_score": record.get("artifact_score"),
    }


def feedback_texts(record: dict) -> list[str]:
    texts = []
    for item in record.get("reflection_rounds", []):
        ev = item.get("eval", {})
        feedback = ev.get("feedback", "")
        if isinstance(feedback, dict):
            texts.append(json.dumps(feedback, ensure_ascii=False))
        elif feedback:
            texts.append(str(feedback))
        summary = ev.get("quality_summary")
        if summary:
            texts.append(str(summary))
    if record.get("quality_summary"):
        texts.append(str(record["quality_summary"]))
    return texts


def mine(records: list[dict]) -> dict:
    grouped = defaultdict(list)
    for record in records:
        grouped[route_key(record)].append(record)

    bank = normalize_experience_bank({
        "base_rules": EXPERIENCE_RULES,
        "reflection_lessons": REFLECTION_LESSONS,
        "learned": {},
        "global_summary": {
            "num_records": len(records),
            "passed": sum(1 for r in records if r.get("passed")),
            "failed": sum(1 for r in records if not r.get("passed")),
            "routes": Counter(r.get("route") for r in records),
            "risk_flags": Counter(flag for r in records for flag in r.get("risk_flags", [])),
        },
    })

    for key, items in grouped.items():
        passed = [r for r in items if r.get("passed")]
        failed = [r for r in items if not r.get("passed")]
        failure_terms = Counter()
        for record in failed:
            text = " ".join(feedback_texts(record)).lower()
            for term in [
                "hallucinated", "new vehicles", "traffic sign", "changed viewpoint",
                "warped", "pasted", "too far", "not significantly obscure",
                "over-dark", "text", "license plate", "building geometry",
            ]:
                if term in text:
                    failure_terms[term] += 1

        bank["learned"][key] = {
            "num_records": len(items),
            "passed": len(passed),
            "failed": len(failed),
            "pass_rate": round(len(passed) / len(items), 4) if items else 0.0,
            "avg_s_geo": avg([r.get("s_geo") for r in items]),
            "avg_s_div": avg([r.get("s_div") for r in items]),
            "good_prompts": [r.get("final_prompt") or r.get("prompt") for r in passed[:5]],
            "bad_case_paths": [r.get("final_reflect_path") or r.get("output_path") for r in failed[:20]],
            "failure_terms": dict(failure_terms.most_common()),
        }
    return bank


def avg(values: list) -> float | None:
    nums = [float(v) for v in values if isinstance(v, (int, float))]
    if not nums:
        return None
    return round(sum(nums) / len(nums), 4)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(ROOT / "all_4city_decision_reflect.json"))
    parser.add_argument("--output", default=str(ROOT / "experience_bank.json"))
    args = parser.parse_args()

    records = json.loads(Path(args.input).read_text(encoding="utf-8"))
    bank = mine(records)
    Path(args.output).write_text(json.dumps(bank, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Done. experience_bank={args.output}")


if __name__ == "__main__":
    main()
