# evaluator.py - Dual-Trait Verification
# 计算 s_geo (几何一致性) 和 s_div (多样性得分)

import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image

from prompt_rules import has_person_surface, has_vehicle_surface


VISMATCH_ROOT = Path("/media/data/zhangjingyi/vismatch")
if VISMATCH_ROOT.exists() and str(VISMATCH_ROOT) not in sys.path:
    sys.path.insert(0, str(VISMATCH_ROOT))


LOCAL_MIN_GEO = 0.82
LOCAL_MIN_DIV = 0.09
DUAL_MIN_GEO = 0.72
DUAL_MIN_DIV = 0.20
DUAL_RAIN_RELAXED_MIN_DIV = 0.12

ROUTE_THRESHOLDS = {
    "local": {"TAU_GEO": LOCAL_MIN_GEO, "TAU_DIV": LOCAL_MIN_DIV},
    "global": {"TAU_GEO": 0.78, "TAU_DIV": 0.15},
    "dual": {"TAU_GEO": DUAL_MIN_GEO, "TAU_DIV": DUAL_MIN_DIV},
}

MIN_LOCAL_DUAL_S_DIV = LOCAL_MIN_DIV
MIN_DUAL_S_DIV = DUAL_MIN_DIV
MIN_GLOBAL_S_DIV = 0.15

LOCAL_VISIBILITY_FLAGS = {
    "weak_or_ineffective_local_occlusion",
    "weak_or_ineffective_local_dual_occlusion",
    "tiny_or_ineffective_person_occluder",
}

LOCAL_LEGALITY_FLAGS = {
    "person_surface_not_explicit",
    "vehicle_surface_not_explicit",
    "vehicle_not_on_road_or_parking_area",
    "person_on_wall",
    "vehicle_on_wall",
    "vehicle_on_sky",
}

MAJOR_ARTIFACTS = {
    "road_geometry_change",
    "facade_geometry_deformation",
    "floating_vehicle",
    "vehicle_on_wall",
    "vehicle_on_sky",
    "building_structure_change",
}

WARNING_ARTIFACTS = {
    "black_vehicle_block",
    "sign_or_storefront_color_deformation",
}

ROUTE_ALIASES = {
    "occlusion": "local",
    "occlusion_only": "local",
    "weather": "global",
    "weather_only": "global",
    "both": "dual",
    "weather_and_occlusion": "dual",
    "skip": "pass",
}


@dataclass
class EvalResult:
    s_geo: float = 1.0
    s_div: float = 1.0
    geo_ok: bool = True
    div_ok: bool = True
    artifact_ok: bool = True
    passed: bool = True
    feedback: dict = field(default_factory=dict)
    skipped: bool = False


class DualTraitEvaluator:
    def __init__(
        self,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        matcher_name: str = "superpoint-lightglue",
        img_size: int = 512,
        n_kpts: int = 2048,
        mock: bool = False,
    ):
        """
        s_geo: 使用 SuperPoint/LightGlue 或 LoFTR 等 vismatch matcher 的 RANSAC 内点率。
        s_div: 使用 CLIP 图像语义嵌入余弦距离，和 s_geo 完全独立。

        matcher_name 可设为 "superpoint-lightglue" 或 "loftr"。
        mock=True 时跳过模型加载，仅用于流程测试。
        """
        self.mock = mock
        self.matcher_name = os.getenv("REFLECTVPR_MATCHER_NAME", matcher_name)
        self.img_size = img_size
        self.n_kpts = n_kpts
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.matcher = None
        clip_model_name = os.getenv("REFLECTVPR_CLIP_MODEL_NAME", clip_model_name)

        if mock:
            print("[Evaluator] Mock模式，跳过CLIP和vismatch模型加载")
            self.model = None
            self.processor = None
            return

        from transformers import CLIPModel, CLIPProcessor

        print(f"[Evaluator] Loading CLIP: {clip_model_name}")
        local_files_only = os.getenv("REFLECTVPR_CLIP_LOCAL_FILES_ONLY", "1").lower() in {"1", "true", "yes"}
        self.model = CLIPModel.from_pretrained(
            clip_model_name,
            local_files_only=local_files_only,
        ).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(
            clip_model_name,
            local_files_only=local_files_only,
        )
        self.model.eval()
        print(f"[Evaluator] CLIP loaded on {self.device}")

    def _normalize_route(self, entry: Optional[dict[str, Any]] = None, route: Optional[str] = None) -> str:
        raw_route = route
        if raw_route is None and entry is not None:
            raw_route = entry.get("route")
        if hasattr(raw_route, "value"):
            raw_route = raw_route.value
        route_name = str(raw_route or "dual").lower().strip()
        return ROUTE_ALIASES.get(route_name, route_name)

    def _thresholds_for(self, route: str) -> tuple[float, float]:
        thresholds = ROUTE_THRESHOLDS.get(route, ROUTE_THRESHOLDS["dual"])
        return thresholds["TAU_GEO"], thresholds["TAU_DIV"]

    def _load_matcher(self):
        if self.matcher is None:
            try:
                from vismatch import get_matcher
            except ImportError as exc:
                raise RuntimeError(
                    "无法导入 vismatch。请在 /media/data/zhangjingyi/vismatch 中执行 "
                    "`source .venv/bin/activate` 后运行 agent。"
                ) from exc

            print(f"[Evaluator] Loading matcher: {self.matcher_name} on {self.device}")
            try:
                self.matcher = get_matcher(
                    self.matcher_name,
                    device=self.device,
                    max_num_keypoints=self.n_kpts,
                )
            except Exception as exc:
                fallback_matcher = "sift-nn"
                if self.matcher_name == fallback_matcher:
                    raise
                print(
                    f"[Evaluator] Matcher '{self.matcher_name}' failed: {exc!r}. "
                    f"Falling back to '{fallback_matcher}'."
                )
                self.matcher_name = fallback_matcher
                self.matcher = get_matcher(
                    fallback_matcher,
                    device=self.device,
                    max_num_keypoints=self.n_kpts,
                )
        return self.matcher

    def _save_temp_image(self, image: Image.Image, path: Path) -> None:
        image.convert("RGB").save(path, format="JPEG", quality=95)

    def _compute_s_geo(self, ref_image: Image.Image, gen_image: Image.Image) -> float:
        matcher = self._load_matcher()
        with tempfile.TemporaryDirectory(prefix="reflectvpr_eval_") as tmp_dir:
            tmp_path = Path(tmp_dir)
            ref_path = tmp_path / "ref.jpg"
            gen_path = tmp_path / "gen.jpg"
            self._save_temp_image(ref_image, ref_path)
            self._save_temp_image(gen_image, gen_path)

            image0 = matcher.load_image(ref_path, resize=self.img_size)
            image1 = matcher.load_image(gen_path, resize=self.img_size)
            result = matcher(image0, image1)

        matched_kpts = result.get("matched_kpts0", [])
        num_matched = len(matched_kpts)
        num_inliers = int(result.get("num_inliers", 0))
        if num_matched <= 0:
            return 0.0
        return max(0.0, min(1.0, num_inliers / num_matched))

    def _extract_clip_feature(self, image: Image.Image) -> torch.Tensor:
        inputs = self.processor(images=image.convert("RGB"), return_tensors="pt").to(self.device)
        with torch.no_grad():
            feat = self.model.get_image_features(**inputs)
        return F.normalize(feat, dim=-1)

    def _compute_s_div(self, ref_image: Image.Image, gen_image: Image.Image) -> float:
        feat_ref = self._extract_clip_feature(ref_image)
        feat_gen = self._extract_clip_feature(gen_image)
        cosine_sim = F.cosine_similarity(feat_ref, feat_gen).item()
        return max(0.0, min(1.0, 1.0 - cosine_sim))

    def _resized_arrays(
        self,
        ref_image: Image.Image,
        gen_image: Image.Image,
        size: int = 256,
    ) -> tuple[np.ndarray, np.ndarray]:
        ref = ref_image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
        gen = gen_image.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
        return np.asarray(ref).astype(np.float32), np.asarray(gen).astype(np.float32)

    def _luma(self, arr: np.ndarray) -> np.ndarray:
        return 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]

    def _saturation(self, arr: np.ndarray) -> np.ndarray:
        rgb = arr / 255.0
        maxc = rgb.max(axis=-1)
        minc = rgb.min(axis=-1)
        return (maxc - minc) / np.maximum(maxc, 1e-6)

    def _largest_component_ratio(self, mask: np.ndarray) -> float:
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

    def _dual_visibility_diagnostics(
        self,
        ref_image: Image.Image,
        gen_image: Image.Image,
        weather: str,
    ) -> dict[str, float | bool]:
        ref, gen = self._resized_arrays(ref_image, gen_image)
        diff = np.abs(gen - ref).mean(axis=-1)
        ref_luma = self._luma(ref)
        gen_luma = self._luma(gen)
        ref_sat = self._saturation(ref)
        gen_sat = self._saturation(gen)
        h, _ = diff.shape
        yy = np.arange(h)[:, None] / h

        lower = np.broadcast_to(yy > 0.38, diff.shape)
        mid_lower = np.broadcast_to((yy > 0.38) & (yy < 0.88), diff.shape)
        changed = (diff > 38) & mid_lower
        changed_ratio = float(changed.mean())
        changed_component = self._largest_component_ratio(changed)
        occlusion_visible = changed_component >= 0.006 and changed_ratio >= 0.012

        luma_delta = float((gen_luma - ref_luma).mean())
        diff_mean = float(diff.mean())
        lower_diff = float(diff[lower].mean()) if lower.any() else 0.0
        white_ref = (ref_luma > 210) & (ref_sat < 0.28)
        white_gen = (gen_luma > 210) & (gen_sat < 0.28)
        white_delta = float(white_gen.mean() - white_ref.mean())

        if weather == "night":
            weather_visible = luma_delta < -10.0 or diff_mean > 22.0
        elif weather == "snow":
            weather_visible = white_delta > 0.018 or luma_delta > 8.0 or diff_mean > 24.0
        elif weather == "rain":
            weather_visible = lower_diff > 18.0 or diff_mean > 20.0 or luma_delta < -4.0
        else:
            weather_visible = diff_mean > 18.0

        return {
            "weather_visible": weather_visible,
            "occlusion_visible": occlusion_visible,
            "changed_ratio": changed_ratio,
            "changed_component": changed_component,
            "diff_mean": diff_mean,
            "lower_diff": lower_diff,
            "luma_delta": luma_delta,
            "white_delta": white_delta,
        }

    def _artifact_flags(
        self,
        ref_image: Image.Image,
        gen_image: Image.Image,
        route: str,
        entry: Optional[dict[str, Any]] = None,
    ) -> list[str]:
        ref, gen = self._resized_arrays(ref_image, gen_image)
        diff = np.abs(gen - ref).mean(axis=-1)
        ref_luma = self._luma(ref)
        gen_luma = self._luma(gen)
        ref_sat = self._saturation(ref)
        gen_sat = self._saturation(gen)
        h, w = diff.shape
        yy = np.arange(h)[:, None] / h
        xx = np.arange(w)[None, :] / w

        flags: list[str] = []
        occlusion = str((entry or {}).get("occlusion", "") or "").lower()
        entry_text = " ".join(
            str((entry or {}).get(key, "") or "")
            for key in ("position", "prompt", "reason", "skip_reason")
        )

        if route in {"local", "dual"}:
            valid_y = yy > 0.18
            changed = (diff > 38) & valid_y
            changed_ratio = float(changed.mean())
            edge_band = (yy < 0.08) | (yy > 0.92) | (xx < 0.06) | (xx > 0.94)
            compact_patch = changed_ratio < 0.18 and changed_ratio > 0.006
            if compact_patch:
                edge_share = float((changed & edge_band).sum() / max(changed.sum(), 1))
                if edge_share > 0.24:
                    flags.append("border_or_edge_pasted_occluder")

            if occlusion == "person":
                if not has_person_surface(entry_text):
                    flags.append("person_surface_not_explicit")
                if changed_ratio < 0.018:
                    flags.append("tiny_or_ineffective_person_occluder")
                new_dark = (gen_luma < 48) & ((ref_luma - gen_luma) > 24) & valid_y
                dark_ratio = float(new_dark.mean())
                dark_component = self._largest_component_ratio(new_dark)
                dark_sat = float(gen_sat[new_dark].mean()) if new_dark.any() else 1.0
                if dark_ratio > 0.010 and dark_component > 0.006 and dark_sat < 0.35:
                    flags.append("black_shadow_or_silhouette_person")
                very_dark_changed = new_dark & changed
                if very_dark_changed.any():
                    dark_changed_share = float(very_dark_changed.sum() / max(changed.sum(), 1))
                    if dark_changed_share > 0.35:
                        flags.append("shadow_like_person_dominates_occluder")

            if occlusion == "vehicle":
                if not has_vehicle_surface(entry_text):
                    flags.append("vehicle_surface_not_explicit")
                new_dark_vehicle = (gen_luma < 42) & ((ref_luma - gen_luma) > 22) & valid_y
                vehicle_dark_ratio = float(new_dark_vehicle.mean())
                vehicle_dark_component = self._largest_component_ratio(new_dark_vehicle)
                if vehicle_dark_ratio > 0.014 and vehicle_dark_component > 0.008:
                    flags.append("black_vehicle_block")
                upper_changed = changed & (yy < 0.42)
                if changed.any() and float(upper_changed.sum() / max(changed.sum(), 1)) > 0.55:
                    flags.append("vehicle_not_on_road_or_parking_area")

            if compact_patch:
                neighbors = (
                    np.roll(changed, 1, axis=0)
                    & np.roll(changed, -1, axis=0)
                    & np.roll(changed, 1, axis=1)
                    & np.roll(changed, -1, axis=1)
                )
                boundary = changed & ~neighbors
                boundary_contrast = float(diff[boundary].mean()) if boundary.any() else 0.0
                if boundary_contrast > 72:
                    flags.append("pasted_cutout_or_sticker_occluder")

        if route in {"global", "dual"}:
            upper_scene = (yy < 0.72) & (ref_luma < 238)
            upper_diff = diff[upper_scene]
            upper_sat_delta = (gen_sat - ref_sat)[upper_scene]
            if upper_diff.size:
                neon_like = (
                    upper_scene
                    & (gen_sat > 0.58)
                    & ((gen_sat - ref_sat) > 0.16)
                    & (diff > 34)
                    & (gen_luma > 45)
                )
                if float(neon_like.mean()) > 0.022:
                    flags.append("facade_or_fixed_structure_color_shift")
                if float(upper_diff.mean()) > 44 and float(upper_sat_delta.mean()) > 0.075:
                    flags.append("large_facade_material_or_color_shift")

            storefront_band = (yy > 0.22) & (yy < 0.78) & (ref_luma < 235)
            sign_like_change = (
                storefront_band
                & (diff > 55)
                & ((ref_sat > 0.30) | (gen_sat > 0.48))
                & (np.abs(gen_sat - ref_sat) > 0.12)
            )
            if float(sign_like_change.mean()) > 0.030 and self._largest_component_ratio(sign_like_change) > 0.004:
                flags.append("sign_or_storefront_color_deformation")

        return flags

    def evaluate(
        self,
        ref_image: Image.Image,
        gen_image: Image.Image,
        entry: Optional[dict[str, Any]] = None,
        route: Optional[str] = None,
    ) -> EvalResult:
        route_name = self._normalize_route(entry=entry, route=route)
        if route_name == "pass":
            return EvalResult(passed=True, skipped=True, feedback={"status": "pass route，跳过评估。"})

        tau_geo, tau_div = self._thresholds_for(route_name)

        if self.mock:
            import random

            s_geo = random.uniform(0.6, 0.9)
            s_div = random.uniform(0.05, 0.35)
        else:
            s_geo = self._compute_s_geo(ref_image, gen_image)
            s_div = self._compute_s_div(ref_image, gen_image)

        artifact_flags = self._artifact_flags(ref_image, gen_image, route_name, entry=entry)
        weather = str((entry or {}).get("weather", "") or "").lower()
        occlusion = str((entry or {}).get("occlusion", "") or "").lower()
        dual_visibility = None
        dual_rain_relaxed_ok = False
        if route_name == "dual":
            dual_visibility = self._dual_visibility_diagnostics(ref_image, gen_image, weather)
            dual_rain_relaxed_ok = (
                weather == "rain"
                and occlusion == "vehicle"
                and s_div >= DUAL_RAIN_RELAXED_MIN_DIV
                and bool(dual_visibility["weather_visible"])
                and bool(dual_visibility["occlusion_visible"])
            )
        quality_gate_flags = []
        if route_name == "local" and s_div < LOCAL_MIN_DIV:
            quality_gate_flags.append("weak_or_ineffective_local_occlusion")
        if route_name == "dual" and s_div < MIN_LOCAL_DUAL_S_DIV:
            quality_gate_flags.append("weak_or_ineffective_local_dual_occlusion")
        if route_name == "dual" and s_div < MIN_DUAL_S_DIV and not dual_rain_relaxed_ok:
            quality_gate_flags.append("weak_dual_weather_or_occlusion")
        if route_name == "global" and s_div < MIN_GLOBAL_S_DIV:
            quality_gate_flags.append("weak_global")
        artifact_flags.extend(quality_gate_flags)
        major_artifact_flags = [flag for flag in artifact_flags if flag in MAJOR_ARTIFACTS]
        warning_artifact_flags = [flag for flag in artifact_flags if flag not in MAJOR_ARTIFACTS]
        artifact_ok = not major_artifact_flags
        geo_ok = s_geo >= tau_geo
        div_ok = s_div >= tau_div or dual_rain_relaxed_ok
        local_visibility_ok = True
        local_legality_ok = True
        if route_name == "local":
            local_visibility_ok = not any(flag in LOCAL_VISIBILITY_FLAGS for flag in artifact_flags)
            local_legality_ok = not any(flag in LOCAL_LEGALITY_FLAGS for flag in artifact_flags)
            passed = geo_ok and local_visibility_ok and local_legality_ok and artifact_ok
        else:
            passed = geo_ok and div_ok and artifact_ok

        feedback = {}
        if not geo_ok:
            feedback["geo_issue"] = {
                "prompt_instruction": (
                    "严格保留原始道路布局、车道线、建筑轮廓和透视关系，不得改变任何结构性元素。"
                ),
                "param_adjustment": {"denoising_strength": "降低 0.05–0.10"},
            }

        if not div_ok:
            feedback["div_issue"] = {
                "prompt_instruction": (
                    "加强真实天气效果（更明显但自然的雨、雾、湿路面或柔和光照变化），"
                    "或提升遮挡物的视觉显著性（自然尺度、清晰纹理、合理落点），不要使用黑影或贴片式遮挡。"
                ),
                "param_adjustment": {"style_strength": "提升 0.05–0.10"},
            }

        if not artifact_ok:
            feedback["artifact_issue"] = {
                "flags": artifact_flags,
                "major_flags": major_artifact_flags,
                "warning_flags": warning_artifact_flags,
                "prompt_instruction": (
                    "拒绝黑影人、剪影人、无效小人物、贴片感、边缘遮挡、黑色车辆块、车辆离开道路/停车区域、"
                    "悬浮遮挡、建筑立面变色、招牌变形和店面颜色漂移。保持固定建筑材料、立面颜色、门窗、招牌和店面身份不变；"
                    "遮挡物必须自然尺度、真实纹理、脚或车轮落在有效地面。"
                ),
                "param_adjustment": {"artifact_filter": "regenerate_or_discard"},
            }
        elif warning_artifact_flags:
            feedback["artifact_warning"] = {
                "flags": warning_artifact_flags,
                "blocking_policy": "warning_only_non_major_artifacts",
            }

        if route_name == "local" and (not local_visibility_ok or not local_legality_ok):
            feedback["local_gate"] = {
                "visibility_ok": local_visibility_ok,
                "legality_ok": local_legality_ok,
                "policy": "LOCAL_MIN_DIV=0.09; local_passed requires s_geo, visible legal occlusion, and no major artifact",
            }

        if quality_gate_flags:
            feedback["quality_gate"] = {
                "flags": quality_gate_flags,
                "policy": "local_min_div_0.09_dual_min_div_0.20_dual_rain_vehicle_visible_min_div_0.09_geo_min_0.72",
            }

        if route_name == "dual" and dual_visibility is not None:
            feedback["dual_gate"] = {
                **dual_visibility,
                "rain_vehicle_relaxed_ok": dual_rain_relaxed_ok,
                "policy": (
                    "dual keeps DUAL_MIN_DIV=0.20; dual rain vehicle may pass at "
                    "s_div>=0.12 only when weather and occlusion are visibly present"
                ),
            }

        if passed:
            feedback["status"] = "通过，无需修改。"

        return EvalResult(
            s_geo=s_geo,
            s_div=s_div,
            geo_ok=geo_ok,
            div_ok=div_ok,
            artifact_ok=artifact_ok,
            passed=passed,
            feedback=feedback,
        )


if __name__ == "__main__":
    evaluator = DualTraitEvaluator(mock=True)
    img = Image.new("RGB", (224, 224), color=(100, 100, 100))
    result = evaluator.evaluate(img, img, entry={"route": "global"})
    print(result)
