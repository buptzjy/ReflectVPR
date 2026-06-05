import argparse
import json
import os
import random
import shutil
import socket
import subprocess
import sys
import time
import traceback
from collections import Counter
from urllib.error import URLError
from urllib.request import ProxyHandler, build_opener, urlopen
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent
AGENT_DIR = ROOT / "your_agent"
SCRIPTS_DIR = ROOT / "scripts"
sys.path.insert(0, str(AGENT_DIR))
sys.path.insert(0, str(SCRIPTS_DIR))

from agent import SceneAugmentAgent  # noqa: E402
from mine_experience import mine  # noqa: E402
from prompt_rules import PROMPT_POLICY_VERSION as DEFAULT_PROMPT_POLICY_VERSION  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ReflectVPR generation.")
    parser.add_argument(
        "--image-root",
        default=None,
        help="Root directory containing city subdirectories. Overrides REFLECTVPR_IMAGE_ROOT.",
    )
    parser.add_argument(
        "--city",
        default=None,
        help="Single city subdirectory under image root, e.g. Boston.",
    )
    parser.add_argument(
        "--cities",
        default=None,
        help="Comma-separated city subdirectories under image root, e.g. Boston,Chicago,Rome.",
    )
    args, _ = parser.parse_known_args()
    return args


def split_cities(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def clear_output_root() -> None:
    output = OUTPUT_ROOT.resolve()
    protected = {
        ROOT.resolve(),
        ROOT.parent.resolve(),
        Path("/").resolve(),
        Path.home().resolve(),
    }
    if output in protected:
        raise RuntimeError(f"Refusing to clear unsafe output root: {output}")
    if output.exists():
        shutil.rmtree(output)
        print(f"[config] cleared output_root={output}", flush=True)


ARGS = parse_args()

# Change this number for each experiment.
SAMPLE_NUM = int(os.getenv("REFLECTVPR_SAMPLE_NUM", "50"))

# Stable defaults. Override with env vars only when needed.
DEFAULT_IMAGE_ROOT = "/media/data1/chenshunpeng1/datasets/gsv_cities/Images"
IMAGE_ROOT = Path(ARGS.image_root or os.getenv("REFLECTVPR_IMAGE_ROOT", DEFAULT_IMAGE_ROOT))
CITY_NAMES = split_cities(ARGS.cities) or split_cities(os.getenv("REFLECTVPR_CITIES"))
if not CITY_NAMES:
    CITY_NAMES = split_cities(ARGS.city) or split_cities(os.getenv("REFLECTVPR_CITY")) or ["London"]
IMAGE_DIRS = [IMAGE_ROOT / city for city in CITY_NAMES]
OUTPUT_ROOT = Path(os.getenv("REFLECTVPR_OUTPUT_ROOT", ROOT / "output"))
DECISION_JSON_OVERRIDE = os.getenv("REFLECTVPR_DECISION_JSON")
REFLECT_JSON_OVERRIDE = os.getenv("REFLECTVPR_REFLECT_JSON")
DEFAULT_EXPERIENCE_INPUT_JSON = ROOT / "experience_bank_v0.json"
EXPERIENCE_INPUT_JSON = Path(os.getenv("REFLECTVPR_EXPERIENCE_JSON", DEFAULT_EXPERIENCE_INPUT_JSON))
EXPERIENCE_OUTPUT_JSON_OVERRIDE = os.getenv("REFLECTVPR_EXPERIENCE_OUTPUT_JSON")
SEED = int(os.getenv("REFLECTVPR_SAMPLE_SEED", "20260529"))
RESUME = os.getenv("REFLECTVPR_RESUME", "1").lower() not in {"0", "false", "no"}
CLEAR_OUTPUT = os.getenv("REFLECTVPR_CLEAR_OUTPUT", "0").lower() in {"1", "true", "yes"}
MOCK = os.getenv("REFLECTVPR_MOCK", "0").lower() in {"1", "true", "yes"}
START_SERVICES = os.getenv("REFLECTVPR_START_SERVICES", "1").lower() not in {"0", "false", "no"}
MINE_EXPERIENCE = os.getenv("REFLECTVPR_MINE_EXPERIENCE", "1").lower() not in {"0", "false", "no"}
PROMPT_POLICY_VERSION = os.getenv("REFLECTVPR_PROMPT_POLICY_VERSION", DEFAULT_PROMPT_POLICY_VERSION)

# Real experiments should fail loudly if a generation service is not loaded.
# This prevents early startup/mock images with black rectangles from entering output.
os.environ.setdefault("REFLECTVPR_DISABLE_SERVICE_MOCK", "1")
os.environ.setdefault("REFLECTVPR_DISABLE_MOCK", "1")
os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
os.environ.setdefault("no_proxy", "127.0.0.1,localhost")

SERVICE_TIMEOUT = int(os.getenv("REFLECTVPR_SERVICE_TIMEOUT", "1800"))
SERVICE_PORTS = {
    "iclight": int(os.getenv("ICLIGHT_PORT", "8002")),
    "lightx2v": int(os.getenv("LIGHTX2V_PORT", "8001")),
}
LOCAL_OPENER = build_opener(ProxyHandler({}))

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
}


def port_open(port: int) -> bool:
    with socket.socket() as sock:
        sock.settimeout(1)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def service_loaded(port: int) -> tuple[bool, str]:
    try:
        with LOCAL_OPENER.open(f"http://127.0.0.1:{port}/health", timeout=3) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except (OSError, URLError, json.JSONDecodeError) as exc:
        return False, str(exc)
    if data.get("model_loaded") is True:
        return True, "model_loaded=true"
    status = data.get("load_status") or data.get("status") or "unknown"
    error = data.get("load_error") or ""
    return False, f"status={status} error={error}".strip()


def wait_for_services() -> None:
    deadline = time.time() + SERVICE_TIMEOUT
    pending = set(SERVICE_PORTS)
    last_messages = {}
    while pending and time.time() < deadline:
        for name in list(pending):
            port = SERVICE_PORTS[name]
            if not port_open(port):
                message = "port not open"
            else:
                loaded, message = service_loaded(port)
                if loaded:
                    print(f"[services] {name} ready on port {port}: {message}", flush=True)
                    pending.remove(name)
                    continue
            if last_messages.get(name) != message:
                print(f"[services] waiting for {name} on port {port}: {message}", flush=True)
                last_messages[name] = message
        if pending:
            time.sleep(2)
    if pending:
        raise TimeoutError(f"Services not ready before timeout: {sorted(pending)}")


def start_services() -> None:
    if MOCK:
        print("[services] mock mode, skip service startup", flush=True)
        return
    if not START_SERVICES:
        print("[services] startup disabled by REFLECTVPR_START_SERVICES=0", flush=True)
        return

    print("[services] start or reuse IC-Light and LightX2V services", flush=True)
    subprocess.run(["bash", str(ROOT / "start_generation_services.sh")], cwd=ROOT, check=True)
    wait_for_services()


def city_output_root(city: str) -> Path:
    return OUTPUT_ROOT / city


def decision_json_for_city(city: str) -> Path:
    if DECISION_JSON_OVERRIDE:
        return Path(DECISION_JSON_OVERRIDE)
    return city_output_root(city) / "decision.json"


def reflect_json_for_city(city: str) -> Path:
    if REFLECT_JSON_OVERRIDE:
        return Path(REFLECT_JSON_OVERRIDE)
    return city_output_root(city) / "reflect.json"


def experience_output_json_for_city(city: str) -> Path:
    if EXPERIENCE_OUTPUT_JSON_OVERRIDE:
        return Path(EXPERIENCE_OUTPUT_JSON_OVERRIDE)
    return city_output_root(city) / "experience_bank.json"


def backup_file(path: Path) -> None:
    if not path.exists():
        return
    now = datetime.now()
    date_tag = now.strftime("%Y%m%d")
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    backup_dir = path.parent / "backups" / date_tag
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / f"{path.stem}.{timestamp}.bak{path.suffix}"
    shutil.copy2(path, backup_path)
    print(f"[backup] {path} -> {backup_path}", flush=True)


def load_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError(f"{path} must contain a JSON list")
    print(f"[resume] loaded {len(records)} records from {path}", flush=True)
    return records


def write_records(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp_path.replace(path)


def to_decision_record(record: dict) -> dict:
    return {key: record[key] for key in DECISION_KEYS if key in record}


def is_current_policy(record: dict) -> bool:
    return record.get("prompt_policy_version") == PROMPT_POLICY_VERSION


def select_images() -> list[Path]:
    if not IMAGE_ROOT.exists():
        raise FileNotFoundError(IMAGE_ROOT)
    rng = random.Random(SEED)
    images_by_city: dict[str, list[Path]] = {}
    for city, image_dir in zip(CITY_NAMES, IMAGE_DIRS):
        if not image_dir.exists():
            raise FileNotFoundError(image_dir)
        images = sorted(image_dir.glob("*.jpg"))
        if not images:
            raise FileNotFoundError(f"No .jpg images found in {image_dir}")
        rng.shuffle(images)
        images_by_city[city] = images

    city_count = len(CITY_NAMES)
    base_quota = SAMPLE_NUM // city_count
    remainder = SAMPLE_NUM % city_count
    selected: list[Path] = []
    leftovers: list[Path] = []
    selected_counts: Counter[str] = Counter()

    for index, city in enumerate(CITY_NAMES):
        quota = base_quota + (1 if index < remainder else 0)
        images = images_by_city[city]
        take = min(quota, len(images))
        selected.extend(images[:take])
        selected_counts[city] += take
        leftovers.extend(images[take:])

    if len(selected) < SAMPLE_NUM and leftovers:
        rng.shuffle(leftovers)
        needed = SAMPLE_NUM - len(selected)
        extra = leftovers[:needed]
        selected.extend(extra)
        selected_counts.update(path.parent.name for path in extra)

    rng.shuffle(selected)
    print(f"[batch] cities={CITY_NAMES} selected_by_city={dict(selected_counts)}", flush=True)
    return selected


def prepare_dirs() -> None:
    for city in CITY_NAMES:
        root = city_output_root(city)
        for route_dir in ["dual", "local", "global", "skip", "rounds", "failed_generation"]:
            (root / route_dir).mkdir(parents=True, exist_ok=True)


def generation_failure_record(image_path: Path, output_root: Path, decision: dict, exc: Exception) -> dict:
    failed_dir = output_root / "failed_generation"
    failed_dir.mkdir(parents=True, exist_ok=True)
    failed_path = failed_dir / f"{image_path.stem}__generation_failed.json"
    record = dict(decision)
    record.update({
        "file_name": image_path.name,
        "city": image_path.parent.name,
        "source_path": str(image_path),
        "output_path": str(failed_path),
        "final_reflect_path": str(failed_path),
        "final_prompt": decision.get("prompt", ""),
        "reflection_rounds": [],
        "passed": False,
        "s_geo": 0.0,
        "s_div": 0.0,
        "geo_ok": False,
        "div_ok": False,
        "artifact_ok": False,
        "rounds_used": 0,
        "generation_failed": True,
        "generation_error_type": type(exc).__name__,
        "generation_error": str(exc),
        "generation_traceback": traceback.format_exc(limit=8),
    })
    failed_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return record


def run_generation() -> None:
    selected = select_images()
    prepare_dirs()

    decision_records_by_city: dict[str, list[dict]] = {}
    reflect_records_by_city: dict[str, list[dict]] = {}
    for city in CITY_NAMES:
        decision_json = decision_json_for_city(city)
        reflect_json = reflect_json_for_city(city)
        backup_file(decision_json)
        backup_file(reflect_json)
        decision_records_by_city[city] = (
            [to_decision_record(record) for record in load_records(decision_json)] if RESUME else []
        )
        reflect_records_by_city[city] = load_records(reflect_json) if RESUME else []

    all_decision_records = [record for records in decision_records_by_city.values() for record in records]
    all_reflect_records = [record for records in reflect_records_by_city.values() for record in records]
    stale_decisions = sum(1 for record in all_decision_records if record.get("source_path") and not is_current_policy(record))
    stale_reflects = sum(1 for record in all_reflect_records if record.get("source_path") and not is_current_policy(record))
    if stale_decisions or stale_reflects:
        print(
            f"[resume] ignore stale records: decisions={stale_decisions} reflects={stale_reflects}",
            flush=True,
        )
    decision_by_source = {
        str(Path(record["source_path"])): record
        for record in all_decision_records
        if record.get("source_path") and is_current_policy(record)
    }
    completed = {
        str(Path(record["source_path"]))
        for record in all_reflect_records
        if record.get("source_path") and is_current_policy(record)
    }

    pending = [image_path for image_path in selected if str(image_path) not in completed]
    print(
        f"[batch] selected={len(selected)} completed={len(completed)} pending={len(pending)} "
        f"output={OUTPUT_ROOT}",
        flush=True,
    )
    if not pending:
        return

    agent = SceneAugmentAgent(mock=MOCK, experience_path=EXPERIENCE_INPUT_JSON)
    for record in decision_by_source.values():
        route = record.get("route")
        if route in agent.route_counts:
            agent.route_counts[route] += 1
        weather = record.get("weather")
        if route in {"global", "dual"} and weather in agent.weather_counts:
            agent.weather_counts[weather] += 1
        if route == "global" and hasattr(agent, "global_weather_counts") and weather in agent.global_weather_counts:
            agent.global_weather_counts[weather] += 1
        occlusion = record.get("occlusion")
        if route in {"local", "dual"} and occlusion in agent.occlusion_counts:
            agent.occlusion_counts[occlusion] += 1
    for idx, image_path in enumerate(selected, 1):
        source_key = str(image_path)
        if source_key in completed:
            print(f"[{idx}/{len(selected)}] skip completed: {image_path.name}", flush=True)
            continue

        print(f"[{idx}/{len(selected)}] run: {image_path.name}", flush=True)
        decision = decision_by_source.get(source_key)
        city = image_path.parent.name
        decision_json = decision_json_for_city(city)
        reflect_json = reflect_json_for_city(city)
        city_root = city_output_root(city)
        if decision is None:
            decision = agent.plan_image(image_path)
            decision["prompt_policy_version"] = PROMPT_POLICY_VERSION
            decision_records_by_city.setdefault(city, []).append(to_decision_record(decision))
            decision_by_source[source_key] = decision
            write_records(decision_json, decision_records_by_city[city])
        else:
            print(f"[resume] reuse decision: {image_path.name}", flush=True)

        try:
            record = agent.run_path(image_path, output_root=city_root, entry=decision)
        except Exception as exc:
            print(
                f"[generation_failed] file={image_path.name} route={decision.get('route')} "
                f"weather={decision.get('weather')} occlusion={decision.get('occlusion')} "
                f"error={type(exc).__name__}: {exc}",
                flush=True,
            )
            record = generation_failure_record(image_path, city_root, decision, exc)
        record["prompt_policy_version"] = PROMPT_POLICY_VERSION
        reflect_records_by_city.setdefault(city, []).append(record)
        completed.add(source_key)
        write_records(reflect_json, reflect_records_by_city[city])

def print_summary(records: list[dict]) -> None:
    current_records = [record for record in records if is_current_policy(record)]
    if not current_records:
        print("[summary] no current-policy records", flush=True)
        return

    route_counts = Counter(record.get("route", "unknown") for record in current_records)
    lightx2v_count = route_counts.get("local", 0) + route_counts.get("dual", 0)
    weather_counts = Counter(
        record.get("weather")
        for record in current_records
        if record.get("route") in {"global", "dual"} and record.get("weather")
    )
    occlusion_counts = Counter(
        record.get("occlusion")
        for record in current_records
        if record.get("route") in {"local", "dual"} and record.get("occlusion")
    )
    skip_reasons = Counter(
        (record.get("skip_reason") or record.get("reason") or "unspecified").strip()
        for record in current_records
        if record.get("route") == "skip"
    )
    downgrade_counts = Counter(
        record.get("downgrade_reason") or record.get("balance_reason") or "none"
        for record in current_records
        if record.get("original_route") and record.get("original_route") != record.get("route")
    )
    vehicle = occlusion_counts.get("vehicle", 0)
    person = occlusion_counts.get("person", 0)

    print("[summary] route counts:", flush=True)
    for route in ["skip", "global", "local", "dual"]:
        print(f"  {route}: {route_counts.get(route, 0)}", flush=True)
    print(f"[summary] lightx2v count = local + dual: {lightx2v_count}", flush=True)
    print(f"[summary] weather counts for global + dual: {dict(weather_counts)}", flush=True)
    print(f"[summary] occlusion counts for local + dual: {dict(occlusion_counts)}", flush=True)
    print(f"[summary] vehicle/person ratio: {vehicle}/{person}", flush=True)
    print(f"[summary] skip reasons top-k: {skip_reasons.most_common(5)}", flush=True)
    print(f"[summary] fallback/downgrade counts: {dict(downgrade_counts)}", flush=True)
    if lightx2v_count / len(current_records) < 0.25:
        print(
            "[summary] lightx2v below target: legal vehicle/person placement surfaces were insufficient "
            "or were rejected by risk gates; skip decisions were not upgraded.",
            flush=True,
        )


def load_all_reflect_records() -> list[dict]:
    records: list[dict] = []
    for city in CITY_NAMES:
        records.extend(load_records(reflect_json_for_city(city)))
    return records


def write_experience_bank() -> None:
    if not MINE_EXPERIENCE:
        return
    for city in CITY_NAMES:
        reflect_json = reflect_json_for_city(city)
        if not reflect_json.exists():
            continue
        records = json.loads(reflect_json.read_text(encoding="utf-8"))
        bank = mine(records)
        output_json = experience_output_json_for_city(city)
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(bank, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[mine] {city} experience_bank={output_json}", flush=True)


def main() -> None:
    print(f"[config] sample_num={SAMPLE_NUM} seed={SEED}", flush=True)
    print(f"[config] image_root={IMAGE_ROOT}", flush=True)
    print(f"[config] cities={CITY_NAMES}", flush=True)
    print(f"[config] output_root={OUTPUT_ROOT}", flush=True)
    print("[config] per-city outputs: decision.json reflect.json experience_bank.json under output_root/<city>/", flush=True)
    print(f"[config] experience_input_json={EXPERIENCE_INPUT_JSON}", flush=True)
    print(f"[config] prompt_policy_version={PROMPT_POLICY_VERSION}", flush=True)
    if CLEAR_OUTPUT:
        clear_output_root()
    start_services()
    run_generation()
    all_reflect_records = load_all_reflect_records()
    if all_reflect_records:
        print_summary(all_reflect_records)
    write_experience_bank()
    print(f"Done. output={OUTPUT_ROOT}", flush=True)


if __name__ == "__main__":
    main()
