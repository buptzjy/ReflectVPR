import base64
import json
import os
import random
import re
import shutil
from io import BytesIO
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

from router import Route, normalize_decision
from evaluator import DualTraitEvaluator
from generators.iclight import ICLightGenerator
from generators.lightx2v import Lightx2vGenerator
from llm.planner import ScenePlanner
from llm.refiner import PromptRefiner
from llm.base import build_llm_client
from prompt_rules import (
    WEATHERS,
    NEGATIVE_PROMPT,
    build_structured_prompt,
    choose_occlusion_from_context,
    choose_occlusion_from_experience,
    choose_weather_from_experience,
    ensure_global_iclight_constraints,
    global_negative_prompt,
    has_person_surface,
    has_vehicle_surface,
    load_experience_bank,
    negative_prompt_from_experience,
    predict_bad_image,
)
from schemas import AgentResult


MAX_ROUNDS = int(os.getenv("REFLECTVPR_MAX_ROUNDS", "3"))
VLM_ROUTER_RETRIES = int(os.getenv("REFLECTVPR_VLM_ROUTER_RETRIES", "2"))
VLM_MODEL = os.getenv("OPENAI_VLM_MODEL", "qwen3.5-35b-a3b")
TARGET_ROUTE_RATIOS = {
    "skip": 0.25,
    "global": 0.25,
    "local": 0.25,
    "dual": 0.25,
}
TARGET_LIGHTX2V_RATIO = TARGET_ROUTE_RATIOS["dual"] + TARGET_ROUTE_RATIOS["local"]
WEATHER_THRESHOLD = float(os.getenv("REFLECTVPR_WEATHER_THRESHOLD", "0.50"))
OCCLUSION_THRESHOLD = float(os.getenv("REFLECTVPR_OCCLUSION_THRESHOLD", "0.50"))
DEFICIT_WEIGHT = float(os.getenv("REFLECTVPR_DEFICIT_WEIGHT", "1.0"))
CAPABILITY_WEIGHT = float(os.getenv("REFLECTVPR_CAPABILITY_WEIGHT", "0.25"))
MIN_RATIO = float(os.getenv("REFLECTVPR_MIN_ROUTE_RATIO", "0.20"))
MAX_RATIO = float(os.getenv("REFLECTVPR_MAX_ROUTE_RATIO", "0.30"))
MIN_RATIO_DEFICIT_MULTIPLIER = float(os.getenv("REFLECTVPR_MIN_ROUTE_DEFICIT_MULTIPLIER", "2.0"))
GLOBAL_WEATHER_TARGET_RATIOS = {
    "overcast": 0.30,
    "fog": 0.30,
    "rain": 0.20,
    "snow": 0.10,
    "night": 0.10,
}
GLOBAL_WEATHER_ORDER = ("overcast", "fog", "rain", "snow", "night")
GLOBAL_SAFE_WEATHERS = ("overcast", "fog")
GLOBAL_MAX_SINGLE_WEATHER_RATIO = float(os.getenv("REFLECTVPR_GLOBAL_MAX_WEATHER_RATIO", "0.50"))
OCCLUSION_STRENGTH_DEBUG = os.getenv("REFLECTVPR_OCCLUSION_STRENGTH_DEBUG", "")
DEBUG_ROUTE = os.getenv("REFLECTVPR_DEBUG_ROUTE", "dual").strip().lower()
GLOBAL_ICLIGHT_HIGHRES_DENOISE = float(os.getenv("REFLECTVPR_GLOBAL_ICLIGHT_DENOISE", "0.30"))


SYSTEM_PROMPT = """You are a strict VPR image augmentation capability scorer.
Return only one JSON object. Do not use markdown.
Do not choose or output the final route. The final route is selected by a quota
scheduler outside the VLM.

Score only the independent capabilities:
- weather_score in [0,1]: whether the image can safely receive a global weather
  or lighting edit while preserving VPR identity.
- occlusion_score in [0,1]: whether the image can safely receive exactly one
  realistic vehicle/person occluder on a legal ground-plane surface.

weather_score should be high when road geometry, building boundaries, sky/lighting,
and place-defining structures remain clear under weather/time edits. Lower it for
close facades, dense repetitive facades, vegetation-dominated scenes, weak road
visibility, sky fragments, or weather edits likely to damage facade texture.

occlusion_score should be high only when there is a clear, real,
perspective-consistent legal vehicle/person placement surface. Vehicle is preferred:
visible traffic lane, road lane, curbside lane, parking bay, parking lane, or
roadside parking area. Person is fallback: sidewalk, curb, crosswalk, roadside
pavement, or road-edge pavement. Lower it when an occluder would sit on the image
border, wall, sky, building facade, or would cover the main facade, storefront,
key sign, or road layout.

Set bad_image=true for close-up building/detail fragments, walls, doors, windows,
sign/storefront crops, sky fragments, scenes with no clear street space, views too
close or narrow, or images where neither weather editing nor occlusion can be done
realistically.

Allowed weather values are exactly: "rain", "snow", "night", "overcast", "fog".
Allowed occlusion values are exactly: "person", "vehicle".

The generated prompt must be English, realistic, and preserve road layout, camera viewpoint,
building geometry, lane markings, traffic signs, and place identity. Do not add dense crowds,
traffic jams, text, logos, black rectangles, black masks, edge shadows, or unrealistic objects."""


def image_to_b64(path: Path, max_side: int = 1024) -> str:
    image = Image.open(path).convert("RGB")
    image.thumbnail((max_side, max_side))
    buf = BytesIO()
    image.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def extract_json(text: str) -> dict:
    text = (text or "").strip()
    if not text:
        raise json.JSONDecodeError("empty VLM response", text, 0)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


class SceneAugmentAgent:

    def __init__(
        self,
        llm_client=None,
        mock: bool = False,
        planning_only: bool = False,
        experience_path: str | Path | None = None,
    ):
        print("[Agent] 初始化所有模块...")

        if llm_client is None:
            llm_client = build_llm_client()

        self.llm_client = llm_client
        self.planner = ScenePlanner(llm_client)
        self.refiner = PromptRefiner(llm_client)
        self.route_counts = {"skip": 0, "global": 0, "local": 0, "dual": 0}
        self.weather_counts = {weather: 0 for weather in sorted(WEATHERS)}
        self.global_weather_counts = {weather: 0 for weather in GLOBAL_WEATHER_ORDER}
        self.occlusion_counts = {"vehicle": 0, "person": 0}
        self.experience_bank = load_experience_bank(experience_path)
        self.negative_prompt = negative_prompt_from_experience(self.experience_bank)
        print(f"[Agent] experience_bank_version={self.experience_bank.get('version')}")
        self.iclight = None
        self.lightx2v = None
        self.evaluator = None
        if not planning_only:
            self.iclight = ICLightGenerator(api_url="" if mock else None)
            self.lightx2v = Lightx2vGenerator(api_url="" if mock else None)
            self.evaluator = DualTraitEvaluator(mock=mock)

        print("[Agent] 初始化完成")

    def plan_image(self, image_path: str | Path, entry: dict = None) -> dict:
        image_path = Path(image_path)
        city = image_path.parent.name
        if entry:
            decision = normalize_decision(entry, image_path=image_path)
        else:
            user_prompt = f"""Analyze this VPR street-view image and score augmentation capabilities.
Return this JSON schema:
{{
  "file_name": "{image_path.name}",
  "city": "{city}",
  "weather_score": 0.0-1.0,
  "occlusion_score": 0.0-1.0,
  "bad_image": true|false,
  "weather": "rain|snow|night|overcast|fog|null",
  "occlusion": "person|vehicle|null",
  "position": "precise plausible edit position",
  "prompt": "English generation prompt",
  "reason": "short reason",
  "skip_reason": "why skip is needed, or empty string",
  "street_scene_quality": "good|partial|bad",
  "occlusion_feasibility": "high|medium|low|none",
  "weather_feasibility": "high|medium|low|none",
  "road_visibility": "clear|partial|none",
  "sky_visibility": "clear|partial|none",
  "vegetation_level": "low|medium|high",
  "facade_density": "low|medium|high",
  "close_building": "yes|no",
  "distant_landmarks_readable": "high|medium|low",
  "global_weather_risk": "low|medium|high",
  "safe_global_weathers": ["rain|snow|night|overcast|fog"]
}}"""
            raw_decision = None
            last_error = None
            for attempt in range(1, VLM_ROUTER_RETRIES + 2):
                response = self.llm_client.chat_with_images(
                    system=SYSTEM_PROMPT,
                    user=user_prompt,
                    images=[image_to_b64(image_path)],
                    json_mode=True,
                    temperature=0.2,
                    model=VLM_MODEL,
                )
                try:
                    raw_decision = extract_json(response)
                    break
                except json.JSONDecodeError as exc:
                    last_error = exc
                    preview = (response or "").replace("\n", " ")[:160]
                    print(
                        f"[Agent] VLM route JSON parse failed attempt={attempt}/{VLM_ROUTER_RETRIES + 1} "
                        f"file={image_path.name} error={exc} response_preview={preview!r}",
                        flush=True,
                    )

            if raw_decision is None:
                decision = self._fallback_skip_decision(image_path, last_error)
            else:
                scheduled = self._schedule_route_from_capabilities(raw_decision)
                decision = normalize_decision(scheduled, image_path=image_path)
                for key in (
                    "bad_image",
                    "quota_eligible_routes",
                    "quota_deficits",
                    "quota_route_scores",
                    "quota_reason",
                ):
                    if key in scheduled:
                        decision[key] = scheduled[key]

        decision["city"] = city
        decision["source_path"] = str(image_path)
        decision = self._apply_experience_bank(decision)
        decision = self._apply_occlusion_strength_debug(decision)
        decision = self._apply_bad_image_policy(decision)
        decision = self._apply_scene_aware_global_policy(decision)

        # Dual 路由：以图片路径为种子，1:1:1 随机分配 rain / rainy_night / snow
        if decision.get("route") == Route.DUAL.value:
            weather_rng = random.Random(str(image_path))
            weather_choices = ["rain", "rainy_night", "snow"]
            decision["weather"] = weather_rng.choice(weather_choices)
            decision["prompt"] = build_structured_prompt(
                route="dual",
                weather=decision["weather"],
                occlusion=decision.get("occlusion"),
                position=decision.get("position", ""),
                base_prompt=decision.get("reason", ""),
                experience_bank=self.experience_bank,
            )

        if not entry:
            self._record_decision_counts(decision)
        return decision

    def _fallback_skip_decision(self, image_path: Path, error: Exception | None = None) -> dict:
        reason = "vlm_empty_or_invalid_json"
        if error is not None:
            reason = f"{reason}: {error}"
        raw = {
            "file_name": image_path.name,
            "city": image_path.parent.name,
            "route": Route.SKIP.value,
            "weather": None,
            "occlusion": None,
            "weather_score": 0,
            "occlusion_score": 0,
            "position": "none",
            "prompt": reason,
            "reason": reason,
            "skip_reason": reason,
            "street_scene_quality": "bad",
            "occlusion_feasibility": "none",
            "weather_feasibility": "none",
            "router_error": reason,
        }
        return normalize_decision(raw, image_path=image_path)

    def run_path(
        self,
        image_path: str | Path,
        output_root: str | Path,
        entry: dict = None,
        collect_bad: bool = True,
    ) -> dict:
        image_path = Path(image_path)
        output_root = Path(output_root)
        decision = self.plan_image(image_path, entry=entry)
        ref = Image.open(image_path).convert("RGB")

        route = decision["route"]
        prompt = decision["prompt"]

        # ── skip 路由：不生成，只存 JSON 记录 ──
        if route in ("skip", "pass"):
            stem = image_path.stem
            skip_dir = output_root / "skip"
            skip_dir.mkdir(parents=True, exist_ok=True)
            skip_path = skip_dir / f"{stem}__skip.json"
            record = dict(decision)
            record.update({
                "output_path": str(skip_path),
                "final_reflect_path": str(skip_path),
                "final_prompt": prompt,
                "reflection_rounds": [],
                "passed": True,
                "s_geo": 1.0,
                "s_div": 0.0,
                "rounds_used": 0,
            })
            skip_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
            return record

        # ── 正常路由：生图 + 自反思 ──
        rounds = []
        final_image = None
        final_eval = None
        final_path = None

        for round_num in range(1, MAX_ROUNDS + 1):
            print(f"[Agent] {image_path.name} round={round_num}/{MAX_ROUNDS} route={route}", flush=True)
            gen_image = self._generate(ref, prompt, route)
            out_path = self._output_path(output_root, image_path, decision, round_num)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            if gen_image.size != ref.size:
                gen_image = gen_image.resize(ref.size, Image.Resampling.LANCZOS)
            gen_image.save(out_path, quality=95)

            eval_result = self.evaluator.evaluate(ref, gen_image, entry=decision)
            eval_dict = self._eval_to_dict(eval_result)
            rounds.append({
                "round": round_num,
                "prompt": prompt,
                "image_path": str(out_path),
                "eval": eval_dict,
            })
            print(
                f"  s_geo={eval_result.s_geo:.3f} s_div={eval_result.s_div:.3f} passed={eval_result.passed}",
                flush=True,
            )

            final_image = gen_image
            final_eval = eval_result
            final_path = out_path
            if eval_result.passed or round_num == MAX_ROUNDS:
                break

            refined = self.refiner.refine(prompt, eval_result.feedback, round_num)
            prompt = build_structured_prompt(
                route=route,
                weather=decision.get("weather"),
                occlusion=decision.get("occlusion"),
                position=decision.get("position", ""),
                base_prompt=refined,
                experience_bank=self.experience_bank,
            )

        # ── 保存最终图到 {route}/ ──
        stem = image_path.stem
        suffix_parts = [route]
        if decision.get("weather"):
            suffix_parts.append(decision["weather"])
        if decision.get("occlusion"):
            suffix_parts.append(decision["occlusion"])
        final_name = f"{stem}__{'_'.join(suffix_parts)}__final.jpg"
        final_save_dir = output_root / route
        final_save_dir.mkdir(parents=True, exist_ok=True)
        final_save_path = final_save_dir / final_name
        if final_image.size != ref.size:
            final_image = final_image.resize(ref.size, Image.Resampling.LANCZOS)
        final_image.save(final_save_path, quality=95)

        record = dict(decision)
        record.update({
            "output_path": str(final_save_path),
            "final_reflect_path": str(final_path),
            "final_prompt": prompt,
            "reflection_rounds": rounds,
            "passed": final_eval.passed,
            "s_geo": final_eval.s_geo,
            "s_div": final_eval.s_div,
            "geo_ok": final_eval.geo_ok,
            "div_ok": final_eval.div_ok,
            "artifact_ok": getattr(final_eval, "artifact_ok", True),
            "rounds_used": rounds[-1]["round"],
        })

        if collect_bad and not final_eval.passed:
            self._collect_bad_case(output_root, image_path, final_path, record)

        return record

    def run(self, input_image: Image.Image, save_dir: str = None, entry: dict = None) -> AgentResult:
        print("\n" + "=" * 50)
        print("[Agent] ===== 开始处理 =====")

        print("[Agent] Step 1: 场景理解与路由...")
        if entry:
            decision = normalize_decision(entry)
        else:
            scene_info = self.planner.understand_scene(input_image)
            print(f"[Agent] 场景信息: {scene_info.get('scene_summary', '未知')}")
            raw_decision = {
                "file_name": "input.jpg",
                "city": "unknown",
                "route": "dual" if scene_info.get("occlusion") else "global",
                "weather": "rain",
                "occlusion": "vehicle" if scene_info.get("occlusion") else None,
                "weather_score": 7,
                "occlusion_score": 7 if scene_info.get("occlusion") else 0,
                "prompt": self.planner.generate_initial_prompt(scene_info, "dual"),
                "reason": scene_info.get("difficulty_hints", ""),
            }
            decision = normalize_decision(raw_decision)

        decision = self._apply_bad_image_policy(decision)
        route = decision["route"]
        prompt = decision["prompt"]

        gen_image = None
        eval_result = None
        round_num = 1

        for round_num in range(1, MAX_ROUNDS + 1):
            print(f"\n[Agent] ----- Round {round_num}/{MAX_ROUNDS} -----")
            print(f"[Agent] Prompt: {prompt[:80]}...")

            gen_image = self._generate(input_image, prompt, route)

            if save_dir:
                os.makedirs(save_dir, exist_ok=True)
                gen_image.save(f"{save_dir}/round_{round_num}.jpg")
                print(f"[Agent] 中间结果保存: {save_dir}/round_{round_num}.jpg")

            eval_entry = dict(entry or {})
            eval_entry.setdefault("route", route)
            eval_result = self.evaluator.evaluate(input_image, gen_image, entry=eval_entry)
            print(
                f"[Agent] 评估 -> s_geo={eval_result.s_geo:.3f}  "
                f"s_div={eval_result.s_div:.3f}  "
                f"passed={eval_result.passed}  skipped={eval_result.skipped}"
            )

            if eval_result.passed:
                print(f"[Agent] 质量通过，提前退出（第{round_num}轮）")
                break

            if round_num == MAX_ROUNDS:
                print(f"[Agent] 达到最大轮次({MAX_ROUNDS})，强制输出")
                break

            print("[Agent] 质量未通过，重写prompt...")
            refined = self.refiner.refine(
                old_prompt=prompt,
                feedback=eval_result.feedback,
                round_num=round_num,
            )
            prompt = build_structured_prompt(
                route=route,
                weather=decision.get("weather"),
                occlusion=decision.get("occlusion"),
                base_prompt=refined,
                experience_bank=self.experience_bank,
            )

        result = AgentResult(
            final_image=gen_image,
            route=route,
            rounds_used=round_num,
            final_prompt=prompt,
            final_score_geo=eval_result.s_geo,
            final_score_div=eval_result.s_div,
            passed=eval_result.passed,
        )

        print(f"\n[Agent] ===== 处理完成 =====")
        print(f"[Agent] 路由={result.route} | 轮次={result.rounds_used} | "
              f"s_geo={result.final_score_geo:.3f} | s_div={result.final_score_div:.3f}")

        return result

    def _generate(self, ref_image: Image.Image, prompt: str, route: str) -> Image.Image:
        if route == Route.GLOBAL.value:
            prompt = ensure_global_iclight_constraints(prompt)
            return self.iclight.generate(
                ref_image,
                prompt,
                negative_prompt=global_negative_prompt(),
                highres_denoise=GLOBAL_ICLIGHT_HIGHRES_DENOISE,
            )
        elif route == Route.LOCAL.value:
            return self.lightx2v.generate_local(ref_image, prompt, negative_prompt=self.negative_prompt)
        elif route == Route.DUAL.value:
            return self.lightx2v.generate_dual(ref_image, prompt, negative_prompt=self.negative_prompt)
        else:
            raise ValueError(f"未知路由: {route}")

    def _apply_bad_image_policy(self, decision: dict) -> dict:
        route = decision["route"]
        risk = predict_bad_image(
            route=route,
            prompt=decision.get("prompt", ""),
            reason=decision.get("reason", ""),
            weather=decision.get("weather"),
            occlusion=decision.get("occlusion"),
        )
        decision["risk_score"] = risk["risk_score"]
        decision["risk_flags"] = risk["risk_flags"]
        decision["skip_recommendation"] = risk["skip_recommendation"]
        print(
            f"[Agent] 路由: {route} | 模型: {decision.get('selected_model')} | "
            f"风险: {risk['risk_score']} {risk['risk_flags']}"
        )
        if risk["skip_recommendation"] and route in {Route.LOCAL.value, Route.DUAL.value}:
            hard_skip_flags = {
                "occlusion_implausible",
                "foreground_foliage_may_confuse_occluder",
            }
            weather_feasibility = str(decision.get("weather_feasibility", "")).lower()
            should_skip = bool(hard_skip_flags.intersection(risk["risk_flags"]))
            if should_skip or weather_feasibility not in {"high", "medium"}:
                print("[Agent] 坏图预判风险较高，降级为 skip")
                decision["original_route"] = route
                decision["route"] = Route.SKIP.value
                decision["weather"] = None
                decision["occlusion"] = None
                decision["selected_model"] = "None"
                decision["skip_reason"] = self._skip_reason_from_risk(risk["risk_flags"])
                decision["prompt"] = build_structured_prompt(
                    route=decision["route"],
                    weather=None,
                    occlusion=None,
                    base_prompt=decision["skip_reason"],
                    experience_bank=self.experience_bank,
                )
                return decision

            print("[Agent] LightX2V 风险较高，但天气仍适合，降级为 global")
            decision["original_route"] = route
            decision["downgrade_reason"] = "lightx2v_risk_high_weather_feasible"
            decision["route"] = Route.GLOBAL.value
            decision["weather"] = self._choose_weather(decision.get("weather"))
            decision["occlusion"] = None
            decision["selected_model"] = "IC-Light"
            decision["prompt"] = build_structured_prompt(
                route=decision["route"],
                weather=decision["weather"],
                occlusion=None,
                base_prompt=decision.get("prompt", ""),
                experience_bank=self.experience_bank,
            )
        return decision

    def _record_route_count(self, route: str) -> None:
        if route in self.route_counts:
            self.route_counts[route] += 1

    def _record_decision_counts(self, decision: dict) -> None:
        route = decision.get("route")
        self._record_route_count(route)
        weather = decision.get("weather")
        if route in {Route.GLOBAL.value, Route.DUAL.value} and weather in self.weather_counts:
            self.weather_counts[weather] += 1
        if route == Route.GLOBAL.value and weather in self.global_weather_counts:
            self.global_weather_counts[weather] += 1
        occlusion = decision.get("occlusion")
        if route in {Route.LOCAL.value, Route.DUAL.value} and occlusion in self.occlusion_counts:
            self.occlusion_counts[occlusion] += 1

    def _route_deficits(self) -> dict:
        planned = sum(self.route_counts.values()) + 1
        return {
            "skip": TARGET_ROUTE_RATIOS["skip"] * planned - self.route_counts["skip"],
            "global": TARGET_ROUTE_RATIOS["global"] * planned - self.route_counts["global"],
            "dual": TARGET_ROUTE_RATIOS["dual"] * planned - self.route_counts["dual"],
            "local": TARGET_ROUTE_RATIOS["local"] * planned - self.route_counts["local"],
            "lightx2v": TARGET_LIGHTX2V_RATIO * planned - (self.route_counts["local"] + self.route_counts["dual"]),
        }

    def _score01(self, value) -> float:
        try:
            score = float(value or 0.0)
        except (TypeError, ValueError):
            return 0.0
        if score > 1.0:
            score = score / 10.0
        return max(0.0, min(1.0, score))

    def _boolish(self, value) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "y"}
        return bool(value)

    def _route_capability(self, route: str, weather_score: float, occlusion_score: float) -> float:
        if route == Route.GLOBAL.value:
            return weather_score
        if route == Route.LOCAL.value:
            return occlusion_score
        if route == Route.DUAL.value:
            return min(weather_score, occlusion_score)
        if route == Route.SKIP.value:
            return 1.0 - max(weather_score, occlusion_score)
        return 0.0

    def _schedule_route_from_capabilities(self, raw: dict) -> dict:
        weather_score = self._score01(raw.get("weather_score"))
        occlusion_score = self._score01(raw.get("occlusion_score"))
        bad_image = self._boolish(raw.get("bad_image"))
        street_quality = str(raw.get("street_scene_quality", "") or "").lower()
        if street_quality == "bad":
            bad_image = True

        eligible: list[str] = []
        if weather_score >= WEATHER_THRESHOLD:
            eligible.append(Route.GLOBAL.value)
        if occlusion_score >= OCCLUSION_THRESHOLD:
            eligible.append(Route.LOCAL.value)
        if weather_score >= WEATHER_THRESHOLD and occlusion_score >= OCCLUSION_THRESHOLD:
            eligible.append(Route.DUAL.value)
        if bad_image or not eligible:
            eligible.append(Route.SKIP.value)

        planned = sum(self.route_counts.values()) + 1
        deficits = self._route_deficits()
        filtered = [
            route
            for route in eligible
            if self.route_counts.get(route, 0) / planned <= MAX_RATIO
        ]
        candidates = filtered or eligible

        def route_score(route: str) -> float:
            current_ratio = self.route_counts.get(route, 0) / planned
            deficit_weight = DEFICIT_WEIGHT
            if current_ratio < MIN_RATIO:
                deficit_weight *= MIN_RATIO_DEFICIT_MULTIPLIER
            return (
                deficit_weight * deficits.get(route, 0.0)
                + CAPABILITY_WEIGHT * self._route_capability(route, weather_score, occlusion_score)
            )

        route = max(candidates, key=route_score)
        scheduled = dict(raw)
        scheduled["route"] = route
        scheduled["weather_score"] = weather_score
        scheduled["occlusion_score"] = occlusion_score
        scheduled["quota_eligible_routes"] = eligible
        scheduled["quota_deficits"] = {key: round(value, 4) for key, value in deficits.items() if key in TARGET_ROUTE_RATIOS}
        scheduled["quota_route_scores"] = {key: round(route_score(key), 4) for key in candidates}
        scheduled["quota_reason"] = (
            f"quota_scheduler route={route} eligible={eligible} "
            f"weather_score={weather_score:.3f} occlusion_score={occlusion_score:.3f}"
        )

        if route == Route.SKIP.value:
            scheduled["weather"] = None
            scheduled["occlusion"] = None
            scheduled["skip_reason"] = scheduled.get("skip_reason") or scheduled.get("reason") or "quota_scheduler_skip"
        elif route == Route.GLOBAL.value:
            scheduled["weather"] = self._choose_weather(scheduled.get("weather"))
            scheduled["occlusion"] = None
        elif route == Route.LOCAL.value:
            scheduled["weather"] = None
            scheduled["occlusion"] = self._choose_occlusion(scheduled)
        elif route == Route.DUAL.value:
            scheduled["weather"] = self._choose_weather(scheduled.get("weather"))
            scheduled["occlusion"] = self._choose_occlusion(scheduled)

        scheduled["prompt"] = build_structured_prompt(
            route=route,
            weather=scheduled.get("weather"),
            occlusion=scheduled.get("occlusion"),
            position=str(scheduled.get("position", "") or scheduled.get("target_region", "")).strip(),
            base_prompt=scheduled.get("skip_reason") if route == Route.SKIP.value else scheduled.get("reason", ""),
            experience_bank=self.experience_bank,
        )
        print(
            f"[Agent] quota scheduler: route={route} eligible={eligible} "
            f"scores={{'weather': {weather_score:.3f}, 'occlusion': {occlusion_score:.3f}}} "
            f"counts={self.route_counts}",
            flush=True,
        )
        return scheduled

    def _apply_experience_bank(self, decision: dict) -> dict:
        route = decision.get("route")
        if route == Route.SKIP.value:
            decision["weather"] = None
            decision["occlusion"] = None
            decision["prompt"] = build_structured_prompt(
                route=route,
                weather=None,
                occlusion=None,
                base_prompt=decision.get("skip_reason") or decision.get("reason", ""),
                experience_bank=self.experience_bank,
            )
            decision["experience_bank_version"] = self.experience_bank.get("version")
            return decision

        context = (
            decision.get("position"),
            decision.get("prompt"),
            decision.get("reason"),
            decision.get("skip_reason"),
        )
        vehicle_ok = has_vehicle_surface(*context)
        person_ok = has_person_surface(*context)

        if route in {Route.LOCAL.value, Route.DUAL.value}:
            chosen = choose_occlusion_from_experience(
                current=decision.get("occlusion"),
                vehicle_ok=vehicle_ok,
                person_ok=person_ok,
                bank=self.experience_bank,
            )
            if chosen:
                decision["occlusion"] = chosen

        if route == Route.DUAL.value:
            decision["weather"] = choose_weather_from_experience(
                route=route,
                requested=decision.get("weather"),
                occlusion=decision.get("occlusion"),
                bank=self.experience_bank,
            )
        elif route == Route.GLOBAL.value:
            requested_weather = decision.get("weather")
            decision["weather"] = requested_weather if requested_weather in GLOBAL_WEATHER_TARGET_RATIOS else None
        elif route == Route.LOCAL.value:
            decision["weather"] = None

        decision["prompt"] = build_structured_prompt(
            route=decision["route"],
            weather=decision.get("weather"),
            occlusion=decision.get("occlusion"),
            position=decision.get("position", ""),
            base_prompt=decision.get("reason", ""),
            experience_bank=self.experience_bank,
        )
        decision["experience_bank_version"] = self.experience_bank.get("version")
        return decision

    def _apply_scene_aware_global_policy(self, decision: dict) -> dict:
        if decision.get("route") != Route.GLOBAL.value:
            return decision

        road_visibility = str(decision.get("road_visibility", "") or "").lower()
        sky_visibility = str(decision.get("sky_visibility", "") or "").lower()
        vegetation_level = str(decision.get("vegetation_level", "") or "").lower()
        facade_density = str(decision.get("facade_density", "") or "").lower()
        close_building = str(decision.get("close_building", "") or "").lower()
        landmarks = str(decision.get("distant_landmarks_readable", "") or "").lower()
        vlm_risk = str(decision.get("global_weather_risk", "") or "").lower()
        safe_weathers = [
            str(item).lower().strip()
            for item in decision.get("safe_global_weathers", [])
            if str(item).lower().strip() in GLOBAL_WEATHER_TARGET_RATIOS
        ]

        text = " ".join(
            str(decision.get(key, "") or "").lower()
            for key in (
                "city",
                "position",
                "prompt",
                "reason",
                "skip_reason",
                "street_scene_quality",
                "weather_feasibility",
            )
        )
        vegetation_terms = (
            "tree", "trees", "vegetation", "foliage", "bush", "bushes", "park",
            "greenery", "roadside scene dominated by vegetation", "dense green",
        )
        billboard_terms = ("billboard", "advertisement", "advertising", "signboard", "roadside sign")
        dense_facade_terms = (
            "dense facade", "repetitive facade", "high-density facade", "storefront row",
            "narrow building facade", "close facade", "building fills", "facade-dominated",
        )
        weak_road_terms = (
            "weak road", "limited road", "peripheral road", "partial road", "no clear road",
            "road barely visible", "road is partly visible",
        )

        def has_scene_term(terms: tuple[str, ...]) -> bool:
            for term in terms:
                if " " in term or "-" in term:
                    if term in text:
                        return True
                elif re.search(rf"\b{re.escape(term)}\b", text):
                    return True
            return False

        vegetation_risk = vegetation_level == "high" or has_scene_term(vegetation_terms + billboard_terms)
        dense_facade_risk = (
            facade_density == "high"
            or close_building in {"yes", "true", "1"}
            or has_scene_term(dense_facade_terms)
        )
        weak_road_risk = road_visibility in {"partial", "none", "low"} or has_scene_term(weak_road_terms)
        weak_landmark_risk = landmarks == "low" and weak_road_risk
        scene_high_risk = (
            vlm_risk == "high"
            or (vegetation_risk and weak_road_risk)
            or (dense_facade_risk and weak_road_risk)
            or weak_landmark_risk
        )
        scene_medium_risk = (
            vlm_risk == "medium"
            or vegetation_risk
            or dense_facade_risk
            or weak_road_risk
            or sky_visibility == "none"
        )
        restriction_reasons = []
        if vlm_risk in {"medium", "high"}:
            restriction_reasons.append(f"global_weather_risk={vlm_risk}")
        if vegetation_risk:
            restriction_reasons.append("vegetation_or_billboard_risk")
        if dense_facade_risk:
            restriction_reasons.append("dense_or_close_facade_risk")
        if weak_road_risk:
            restriction_reasons.append("weak_road_visibility")
        if weak_landmark_risk:
            restriction_reasons.append("weak_landmark_with_weak_road")
        if sky_visibility == "none":
            restriction_reasons.append("no_sky_visibility")

        if scene_high_risk and road_visibility == "none":
            original_route = decision.get("route")
            decision["original_route"] = decision.get("original_route", original_route)
            decision["route"] = Route.SKIP.value
            decision["weather"] = None
            decision["occlusion"] = None
            decision["selected_model"] = "None"
            decision["skip_reason"] = "scene_aware_global_skip: high-risk global scene with no clear road geometry"
            decision["global_scene_policy"] = {
                "action": "skip",
                "risk": "high",
                "reason": "no_clear_road_geometry_for_safe_global_weather",
            }
            decision["prompt"] = build_structured_prompt(
                route=decision["route"],
                weather=None,
                occlusion=None,
                base_prompt=decision["skip_reason"],
                experience_bank=self.experience_bank,
            )
            return decision

        scene_restricted = scene_high_risk or scene_medium_risk
        pool = list(GLOBAL_SAFE_WEATHERS if scene_restricted else GLOBAL_WEATHER_ORDER)
        if safe_weathers:
            filtered = [weather for weather in pool if weather in safe_weathers]
            if filtered:
                pool = filtered

        before = {weather: int(self.global_weather_counts.get(weather, 0)) for weather in GLOBAL_WEATHER_ORDER}
        planned = sum(before.values()) + 1
        capped_pool = [
            weather
            for weather in pool
            if planned <= 1 or self.global_weather_counts.get(weather, 0) / max(planned - 1, 1) <= GLOBAL_MAX_SINGLE_WEATHER_RATIO
        ]
        if capped_pool:
            pool = capped_pool

        deficits = {
            weather: GLOBAL_WEATHER_TARGET_RATIOS[weather] * planned - self.global_weather_counts.get(weather, 0)
            for weather in pool
        }
        max_deficit = max(deficits.values())
        candidates = [weather for weather, deficit in deficits.items() if deficit == max_deficit]
        selected_weather = random.choice(candidates)
        after = dict(before)
        after[selected_weather] = after.get(selected_weather, 0) + 1

        original_weather = decision.get("weather")
        decision["weather"] = selected_weather
        decision["occlusion"] = None
        decision["selected_model"] = "IC-Light"
        decision["selected_weather"] = selected_weather
        decision["weather_selection_reason"] = (
            "scene_restricted_safe_pool_quota" if scene_restricted else "target_ratio_quota"
        )
        decision["scene_policy_restricted_weather"] = scene_restricted
        decision["scene_policy_restriction_reasons"] = restriction_reasons
        decision["global_weather_counts_before"] = before
        decision["global_weather_counts_after"] = after
        decision["global_weather_target_ratios"] = GLOBAL_WEATHER_TARGET_RATIOS
        decision["global_weather_candidate_pool"] = pool
        decision["global_weather_deficits"] = {key: round(value, 4) for key, value in deficits.items()}
        decision["global_scene_policy"] = {
            "action": "restrict_weather" if scene_restricted else "quota_select_weather",
            "risk": "high" if scene_high_risk else "medium" if scene_medium_risk else "low",
            "original_weather": original_weather,
            "selected_weather": selected_weather,
            "allowed_weathers": pool,
            "road_visibility": road_visibility,
            "sky_visibility": sky_visibility,
            "vegetation_level": vegetation_level,
            "facade_density": facade_density,
            "close_building": close_building,
            "distant_landmarks_readable": landmarks,
        }
        if original_weather != selected_weather:
            decision["scene_weather_original"] = original_weather
            decision["scene_weather_restricted_to"] = selected_weather if scene_restricted else None

        decision["prompt"] = build_structured_prompt(
            route=decision["route"],
            weather=decision["weather"],
            occlusion=None,
            position=decision.get("position", ""),
            base_prompt=decision.get("reason", ""),
            experience_bank=self.experience_bank,
        )
        print(
            f"[Agent] global weather quota: {original_weather} -> {selected_weather} "
            f"reason={decision['weather_selection_reason']} counts={before}->{after} "
            f"file={decision.get('file_name')}",
            flush=True,
        )

        return decision

    def _apply_occlusion_strength_debug(self, decision: dict) -> dict:
        if OCCLUSION_STRENGTH_DEBUG != "occlusion_strength_debug_v1":
            return decision
        route = DEBUG_ROUTE if DEBUG_ROUTE in {Route.LOCAL.value, Route.DUAL.value} else Route.DUAL.value
        decision["original_route"] = decision.get("route")
        decision["route"] = route
        decision["weather"] = "rain" if route == Route.DUAL.value else None
        decision["occlusion"] = "vehicle"
        decision["selected_model"] = "LightX2V-Dual" if route == Route.DUAL.value else "LightX2V-Local"
        decision["balance_reason"] = "occlusion_strength_debug_v1_fixed_vehicle"
        if not decision.get("position") or str(decision.get("position")).strip().upper() == "N/A":
            decision["position"] = "visible traffic lane, curbside lane, parking bay, or roadside parking area"
        decision["prompt"] = build_structured_prompt(
            route=decision["route"],
            weather=decision.get("weather"),
            occlusion=decision.get("occlusion"),
            position=decision.get("position", ""),
            base_prompt=decision.get("reason", ""),
            experience_bank=self.experience_bank,
        )
        return decision

    def _choose_occlusion(self, decision: dict) -> str | None:
        return choose_occlusion_from_context(
            decision.get("position"),
            decision.get("prompt"),
            decision.get("reason"),
            decision.get("skip_reason"),
            decision.get("street_scene_quality"),
            decision.get("occlusion_feasibility"),
            requested=decision.get("occlusion"),
        )

    def _choose_weather(self, requested: str | None) -> str:
        return choose_weather_from_experience(
            route=Route.GLOBAL.value,
            requested=requested,
            bank=self.experience_bank,
        ) or "rain"

    def _skip_reason_from_risk(self, flags: list[str]) -> str:
        if "occlusion_implausible" in flags:
            return "no_valid_street_surface_for_occlusion"
        if "foreground_foliage_may_confuse_occluder" in flags:
            return "occlusion_would_be_unreliable_or_edge_artifact"
        if "night_vehicle_high_hallucination_risk" in flags:
            return "vehicle_occlusion_high_hallucination_risk_and_weather_not_feasible"
        return "local_or_dual_risk_too_high"

    def _eval_to_dict(self, eval_result) -> dict:
        return {
            "passed": eval_result.passed,
            "s_geo": eval_result.s_geo,
            "s_div": eval_result.s_div,
            "geo_ok": eval_result.geo_ok,
            "div_ok": eval_result.div_ok,
            "artifact_ok": getattr(eval_result, "artifact_ok", True),
            "feedback": eval_result.feedback,
        }

    def _output_path(self, output_root: Path, image_path: Path, decision: dict, round_num: int) -> Path:
        return output_root / "rounds" / image_path.stem / f"r{round_num}.jpg"

    def _collect_bad_case(self, output_root: Path, image_path: Path, final_path: Path, record: dict) -> None:
        bad_dir = output_root / "bad_cases"
        bad_img_dir = bad_dir / "images"
        bad_img_dir.mkdir(parents=True, exist_ok=True)
        bad_path = bad_img_dir / f"{image_path.stem}__{final_path.name}"
        shutil.copy2(final_path, bad_path)
        bad_record = dict(record)
        bad_record["bad_case_image"] = str(bad_path)
        with (bad_dir / "bad_cases.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps(bad_record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    agent = SceneAugmentAgent(mock=True)
    test_image = Image.new("RGB", (512, 384), color=(100, 110, 90))
    result = agent.run(test_image, save_dir="/tmp/agent_test")
    result.final_image.save("/tmp/agent_final.jpg")
    print(f"\n[Main] 最终图像保存: /tmp/agent_final.jpg")
    print(f"[Main] 路由: {result.route}")
    print(f"[Main] 使用轮次: {result.rounds_used}")
    print(f"[Main] 最终Prompt: {result.final_prompt}")
