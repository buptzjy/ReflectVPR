from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from dataclasses import dataclass


WEATHERS = {"rain", "snow", "night", "overcast", "fog", "rainy_night"}
OCCLUSIONS = {"person", "vehicle"}
OCCLUSION_STRENGTH_DEBUG = os.getenv("REFLECTVPR_OCCLUSION_STRENGTH_DEBUG", "")
DUAL_PROMPT_STRATEGY = os.getenv("REFLECTVPR_DUAL_PROMPT_STRATEGY", "dual_hard_v4").strip().lower()
PROMPT_POLICY_VERSION = f"20260703_dual_strategy_{DUAL_PROMPT_STRATEGY or 'default'}"

VEHICLE_SURFACE_TERMS = (
    "traffic lane", "road lane", "visible lane", "curbside lane", "curb lane",
    "parking bay", "parking lane", "roadside parking", "roadside parking area",
    "parking area", "parked car", "parked cars", "roadway", "carriageway",
    "street lane", "right lane", "left lane", "near lane", "foreground lane",
)
PERSON_SURFACE_TERMS = (
    "sidewalk", "paved sidewalk", "curb", "kerb", "crosswalk", "zebra crossing",
    "roadside pavement", "road-edge pavement", "visible pavement", "pedestrian area",
)


PRESERVE_CLAUSE = (
    "preserve exact road layout, camera viewpoint, building geometry, lane markings, "
    "traffic signs, existing vehicles, and place identity"
)
GLOBAL_ICLIGHT_FACADE_CONSTRAINT = (
    "Preserve the original building facade colors, wall materials, architectural textures, "
    "doors, windows, signs, storefronts, rooflines, and structural geometry. Apply weather "
    "or time-of-day changes through realistic illumination, sky, road surface, shadows, "
    "and atmosphere only. Do not repaint, restyle, relight, replace, or stylize fixed "
    "architectural surfaces. Keep facade color shifts extremely subtle and physically plausible"
)
LIGHTX2V_LOCAL_VEHICLE_CONSTRAINT = (
    "exactly one clearly visible vehicle, occupying approximately 6-10% image area, "
    "partially blocking the nearest visible lane, clearly visible vehicle body, clearly visible wheels, "
    "noticeable but realistic local occlusion"
)
if DUAL_PROMPT_STRATEGY == "dual_hard_v5":
    LIGHTX2V_DUAL_VEHICLE_CONSTRAINT = (
        "one realistic box truck, delivery van, shuttle bus, city bus, or tall service vehicle with visible wheels, "
        "realistic side panels, perspective-aligned tires, and contact shadows. The dominant vehicle should occupy "
        "approximately 20-28% image area, placed in the lower-middle or near-midground on an existing legal traffic "
        "lane, curbside lane, parking bay, roadside parking area, or intersection approach. It must only use existing "
        "road, curb, lane, crosswalk, parking-strip, sidewalk-road, or parked-vehicle boundary cues already visible in "
        "the original image. Do not invent storefronts, facade edges, windows, walls, buildings, skylines, or new "
        "architectural structures to justify the occlusion. Preserve all existing background structures exactly"
    )
elif DUAL_PROMPT_STRATEGY == "dual_hard_v4":
    LIGHTX2V_DUAL_VEHICLE_CONSTRAINT = (
        "one large realistic box truck, delivery van, city bus, shuttle bus, or tall service vehicle, preferably white "
        "or light-colored with visible wheels and realistic side panels, plus optional small natural street clutter such "
        "as cones, bollards, a scooter, a partial car hood, or a small roadwork barrier. The dominant vehicle should "
        "occupy approximately 24-32% image area, placed in the lower-middle or near-midground on a legal traffic lane, "
        "curbside lane, parking bay, or intersection approach. It must cut across a continuous lower-scene recognition "
        "band: lane markings, crosswalk bars, curb line, parking-lane boundary, sidewalk-road transition, storefront "
        "lower edge, or sign-adjacent lower facade margin. Prefer scenes where the truck or bus blocks part of an "
        "intersection, parking strip, or long curb/lane line, while preserving landmark facades, building silhouettes, "
        "road geometry, perspective, main sign identity, and place-defining structures"
    )
elif DUAL_PROMPT_STRATEGY == "dual_hard_v3":
    LIGHTX2V_DUAL_VEHICLE_CONSTRAINT = (
        "one dominant realistic mid-ground street vehicle such as a delivery van, bus, truck, taxi, or parked-car "
        "cluster plus one smaller natural foreground-side cue such as a partial car hood, scooter, umbrella pedestrian, "
        "bollards, sign pole, or roadwork barrier only if it fits the visible geometry. The combined new occluder area "
        "should occupy approximately 22-30% image area, with the dominant vehicle occupying 16-22% and the secondary "
        "cue occupying 5-8%. Place the dominant vehicle on a legal lane, curbside lane, parking bay, or roadside parking "
        "area so it crosses the lower facade, curb, lane, parking-strip, crosswalk, storefront-lower-margin, or "
        "sidewalk-road boundary region. The occlusion must interrupt continuous VPR place cues rather than only empty "
        "road surface, while preserving landmark facades, building silhouettes, road geometry, camera viewpoint, main "
        "sign identity, and all place-defining structures"
    )
elif DUAL_PROMPT_STRATEGY == "dual_hard_v2":
    LIGHTX2V_DUAL_VEHICLE_CONSTRAINT = (
        "one dominant realistic foreground-to-midground vehicle plus one subtle secondary street-level occlusion cue "
        "such as a partial parked vehicle edge, scooter, cyclist, roadwork cone cluster, or curbside pole only if it "
        "fits the visible road geometry. The combined new occluder area should occupy approximately 18-25% image area, "
        "with the dominant vehicle on a legal lane, curbside lane, parking bay, or roadside parking area. The occlusion "
        "must cover a continuous lower-scene place cue band such as lane markings, curb transitions, parking-strip "
        "boundaries, crosswalk edges, lower road-storefront boundary, or parked-car edges, while preserving landmark "
        "facades, building silhouettes, road geometry, camera viewpoint, and all place-defining structures"
    )
elif DUAL_PROMPT_STRATEGY == "test20_v1":
    LIGHTX2V_DUAL_VEHICLE_CONSTRAINT = (
        "exactly one dominant realistic vehicle, occupying approximately 16-22% image area, "
        "placed in the lower-middle or near-midground of the scene, occluding one continuous and meaningful "
        "street-level cue region such as a full lane-marking segment, a curbside parking strip, a crosswalk edge, "
        "or the lower boundary between roadway and storefront or parked-car area, with clearly visible vehicle body, "
        "clearly visible wheels, readable silhouette, realistic body details, and strong but legal place-level occlusion"
    )
else:
    LIGHTX2V_DUAL_VEHICLE_CONSTRAINT = (
        "exactly one dominant visible vehicle, occupying approximately 12-16% image area, "
        "placed in the lower-middle or near-midground of the scene, partially blocking one full visible lane segment "
        "or one curbside parking segment, with clearly visible vehicle body, clearly visible wheels, "
        "clearly readable silhouette, and strong but realistic occlusion"
    )
LIGHTX2V_VEHICLE_CONSTRAINT = LIGHTX2V_LOCAL_VEHICLE_CONSTRAINT
LIGHTX2V_VEHICLE_DEBUG_CONSTRAINT = LIGHTX2V_LOCAL_VEHICLE_CONSTRAINT
LIGHTX2V_PERSON_CONSTRAINT = (
    "exactly one realistic full-body pedestrian, natural scale, normal clothing, feet on the "
    "visible ground plane, matched lighting, realistic shadow"
)
LIGHTX2V_LOCAL_MULTI_NEGATIVE_TERMS = (
    "tiny occluder, weak occlusion, barely visible object, sticker-like object, "
    "floating object, black silhouette, border-attached object, tiny vehicle, tiny pedestrian, "
    "invisible cyclist, weak local edit"
)
FORBID_CLAUSE = (
    "no new background vehicles, no dense crowds, no traffic jams, no new traffic signs, "
    "no readable hallucinated text, no logo or license plate changes, no warped buildings, "
    "no changed viewpoint, no solid black rectangle, no black frame, no black mask, "
    "no black shadow person, no dark blob, no border occluder, no faceless silhouette, "
    "no object on wall, sky, or building facade, do not cover main building structure, "
    "do not change road geometry"
)
GLOBAL_ICLIGHT_FORBID_CLAUSE = (
    "do not recolor walls, facades, doors, windows, signs, storefronts, roofs, or fixed "
    "architectural structures into neon, rainbow, multicolored, oversaturated, glowing, "
    "or large-area unnatural color-pollution effects; do not change shop-sign colors, sign shapes, "
    "facade materials, window frames, awnings, storefront identity, or roofline colors"
)
LIGHTX2V_FORBID_CLAUSE = (
    "no black silhouette person, no shadow-like person, no tiny pedestrian, no cutout person, "
    "no sticker-like person, no border pedestrian, no person floating, no person on wall, sky, or "
    "building facade, no black vehicle block, no floating vehicle, no vehicle on sidewalk, wall, sky, "
    "or building facade, no oversized foreground vehicle, no flat pasted cutout, no sticker-like "
    "occluder, no border-attached occluder, no object covering place-defining facades, signs, windows, "
    "storefronts, road layout, or lane markings"
)
QUALITY_CLAUSE = (
    "photorealistic street-view image, natural exposure, matched lighting, realistic shadows, "
    "no pasted-object appearance"
)
NEGATIVE_PROMPT = (
    "warped buildings, changed road layout, changed camera viewpoint, new signs, new text, "
    "readable license plates, extra vehicles, dense crowds, traffic jam, unrealistic objects, "
    "floating person, pasted object, pasted cutout, sticker-like object, broken pavement, distorted facade, cartoon, low quality, "
    "solid black rectangle, black bar, black frame, black mask, pure black occluder, "
    "edge shadow person, black shadow person, shadow-like person, faceless silhouette, dark blob, ghost person, "
    "tiny pedestrian, cutout person, sticker-like person, border pedestrian, person floating, "
    "black vehicle block, floating vehicle, vehicle on sidewalk, vehicle on wall, "
    "border occluder, object on wall, object in sky, object on building facade, "
    "close-up person, close-up car, oversized foreground vehicle, large foreground object, blocked main building, over-occlusion, "
    "neon facade, rainbow facade, multicolored building, oversaturated storefront, changed shop sign, "
    "changed facade color, changed wall material, distorted sign, deformed storefront, "
    "tiny occluder, weak occlusion, barely visible object, invisible cyclist, weak local edit"
)
GLOBAL_NEGATIVE_PROMPT = (
    "oil painting, painterly, illustration, watercolor, brush strokes, artistic style, stylized image, "
    "cartoon, anime, cinematic color grading, fantasy, surreal, cyberpunk, sci-fi, colorful lighting, neon facade, "
    "rainbow facade, oversaturated storefront, changed shop sign, sign deformation, storefront color shift, "
    "warped buildings, changed road layout, changed camera viewpoint"
)

GLOBAL_SNOW_CONSERVATIVE_PROMPT = """Preserve scene geometry, buildings, signs, vehicles and viewpoint.

Apply light snowy winter weather only:
overcast sky, visible snowflakes,
light snow accumulation on horizontal surfaces,


Natural street-view photo, conservative edit.
Avoid structural changes, deformation, blur or detail loss."""

GLOBAL_NIGHT_CONSERVATIVE_PROMPT = """Preserve scene geometry, buildings, signs, vehicles and viewpoint.

Apply urban night conditions only:
dim street lighting,
reduced ambient brightness,


Natural street-view photo, conservative edit.
Avoid structural changes, blur or deformation."""

GLOBAL_RAIN_CONSERVATIVE_PROMPT = """Preserve scene geometry, buildings, signs, vehicles, lane markings, road layout, and viewpoint.

Apply conservative light rainy weather only:
subtle wet pavement, soft overcast sky,
very light visible rain streaks,
mild reflections near the road surface.

Keep building facades, storefronts, signs, windows, trees, cars, and road boundaries unchanged.
Do not add heavy rain, flooding, haze, darkness, strong blur, large reflections, new vehicles, or structural changes.

Natural documentary street-view photo, conservative edit."""

REFLECTION_LESSONS = [
    "Occluders must be real street participants, not abstract masks or edge shadows.",
    "Prefer vehicle occlusion whenever a clear lane, curbside lane, parking bay, or roadside parking area is available.",
    "A person occluder should be a full-body pedestrian walking or standing on visible sidewalk, curb, crosswalk, or road-edge pavement, with feet touching the ground plane.",
    "A vehicle occluder should be a normal car, van, bus, or taxi placed on a visible road lane, curb lane, or parking lane with tires aligned to the road perspective.",
    "Avoid placing people at the image border as black silhouettes; this often produces unnatural shadow-like artifacts.",
    "Skip close-up facades, wall/window/sign/storefront fragments, sky fragments, and scenes without a valid street surface.",
    "Do not ask for close-up, large, dense, or significantly obscuring occluders unless the experiment explicitly wants severe occlusion.",
]


EXPERIENCE_RULES = {
    "global_rain": {
        "target": "wet reflective pavement, overcast sky, subtle visible falling rain",
        "avoid": "rain only shown as wet ground without atmosphere",
    },
    "global_snow": {
        "target": "light snow, realistic winter atmosphere, thin snow on sidewalks and parked cars",
        "avoid": "heavy snow that hides place-defining facades or road geometry",
    },
    "global_night": {
        "target": "dim artificial street lighting, reduced saturation, controlled low-light exposure",
        "avoid": "over-dark image or changed facade/window/fence/tree structure",
    },
    "global_overcast": {
        "target": "uniform cloudy sky, soft diffuse daylight, muted contrast, no strong shadows",
        "avoid": "turning the scene into rain, night, or heavy fog",
    },
    "global_fog": {
        "target": "realistic dense fog with reduced long-range visibility while keeping nearby road and facade geometry readable",
        "avoid": "fog so heavy that it hides place-defining facades, signs, lane markings, or road layout",
    },
    "local_person": {
        "target": "exactly one normal full-body pedestrian, clearly visible, natural scale, realistic clothing color, not a black silhouette, not a dark blob, not tiny, not at the image border, feet firmly on visible sidewalk, curb, crosswalk, or roadside pavement",
        "avoid": "edge shadow person, black shadow person, shadow-like person, faceless black silhouette, dark blob, tiny pedestrian, border pedestrian, floating person, cutout person, sticker-like person, pasted sharp person, crowd",
    },
    "local_vehicle": {
        "target": "exactly one normal street vehicle such as car, van, bus, or taxi, wheels aligned with road perspective, natural scale, not a black block, placed only on visible traffic lane, curbside lane, parking bay, or roadside parking area",
        "avoid": "pure black vehicle blob, black vehicle block, floating vehicle, vehicle on sidewalk, vehicle on wall, vehicle on sky, vehicle on building facade, oversized foreground vehicle, border occluder, traffic jam, extra parked cars, vehicle that changes road layout",
    },
    "dual_rain_person": {
        "target": "wet reflective pavement plus one natural pedestrian on visible sidewalk, crosswalk, curb, or road-edge pavement with matched rainy lighting",
        "avoid": "hallucinated vehicles, edge shadow person, wrong ground-plane position, pasted person",
    },
    "dual_rain_vehicle": {
        "target": "wet reflective pavement plus one normal vehicle on a visible road lane, curb lane, or parking lane with matched rainy lighting",
        "avoid": "pure black vehicle blob, wrong lane position, traffic jam, changed road layout",
    },
    "dual_snow_person": {
        "target": "light snow plus one natural pedestrian on visible sidewalk, crosswalk, curb, or road-edge pavement with matched winter lighting",
        "avoid": "edge shadow person, snow covering place-defining geometry, pasted person",
    },
    "dual_snow_vehicle": {
        "target": "light snow plus one normal vehicle on a visible lane or parking lane with tires aligned to the road perspective",
        "avoid": "pure black vehicle blob, traffic jam, heavy snow hiding landmarks",
    },
    "dual_overcast_person": {
        "target": "soft overcast daylight plus one natural pedestrian on visible sidewalk, crosswalk, curb, or road-edge pavement",
        "avoid": "edge shadow person, rain artifacts, night lighting, pasted person",
    },
    "dual_overcast_vehicle": {
        "target": "heavy low-cloud overcast daylight plus one dominant vehicle that blocks a meaningful lane or curbside segment under flat cool grey lighting",
        "avoid": "rain artifacts, sunny contrast, decorative tiny car, traffic jam, changed road layout",
    },
    "dual_fog_person": {
        "target": "realistic fog plus one natural pedestrian on visible nearby pavement, with background visibility reduced but local geometry preserved",
        "avoid": "faceless silhouette, edge shadow person, fog hiding the ground plane, pasted person",
    },
    "dual_fog_vehicle": {
        "target": "realistic fog plus one normal vehicle on a visible nearby lane or parking lane, with tires aligned to road perspective",
        "avoid": "pure black vehicle blob, fog hiding road layout, traffic jam, changed road geometry",
    },
    "dual_night_person": {
        "target": "dim night lighting plus one natural full-body pedestrian on visible pavement, with readable body shape and matched exposure",
        "avoid": "black silhouette, edge shadow person, over-bright person, changed building details",
    },
    "dual_night_vehicle": {
        "target": "night lighting plus one normal vehicle on a visible lane or curb lane with subtle realistic headlight spill",
        "avoid": "pure black vehicle blob, multiple new cars, traffic jam, changed lane geometry",
    },
}

EXPERIENCE_BANK_V0 = {
    "version": "experience_bank_v0",
    "source": ["output_0602_v1", "output_0602_v2"],
    "route_policy": {
        "target_ratio": {"skip": 0.40, "global": 0.30, "lightx2v": 0.30},
        "hard_skip": [
            "building_closeup",
            "wall_closeup",
            "window_closeup",
            "door_closeup",
            "storefront_closeup",
            "sign_closeup",
            "no_street_space",
            "no_valid_occlusion_surface",
        ],
    },
    "weather_statistics": {
        "rain": {"priority": 1, "status": "recommended"},
        "overcast": {"priority": 2, "status": "recommended"},
        "fog": {"priority": 3, "status": "experimental"},
        "snow": {"priority": 4, "status": "experimental"},
        "night": {
            "priority": 5,
            "status": "high_risk",
            "reason": "dual_night_vehicle pass rate 0%",
        },
    },
    "occlusion_statistics": {
        "vehicle": {"priority": 1, "status": "preferred"},
        "person": {"priority": 2, "status": "fallback"},
    },
    "dual_vehicle_failures": {
        "black_vehicle_block": 0.917,
        "storefront_deformation": 0.667,
        "pasted_vehicle": 0.583,
        "facade_deformation": 0.167,
    },
    "prompt_rules": {
        "vehicle": {
            "preferred_size": "small_to_medium",
            "max_area_ratio": 0.08,
            "required": [
                "realistic vehicle",
                "visible windows",
                "visible wheels",
                "visible body details",
                "natural paint color",
                "road-perspective aligned",
                "lane aligned",
                "natural shadow",
            ],
            "forbidden": [
                "black vehicle",
                "dark blob vehicle",
                "silhouette vehicle",
                "sticker vehicle",
                "flat pasted vehicle",
                "featureless vehicle",
                "storefront occlusion",
                "sign occlusion",
                "facade occlusion",
            ],
        },
        "preservation": [
            "preserve building facade",
            "preserve storefront",
            "preserve sign",
            "preserve windows",
            "preserve doors",
            "preserve road layout",
            "preserve traffic signs",
            "preserve street geometry",
        ],
    },
    "recommended_templates": {
        "global_rain": {"weight": 1.0},
        "global_overcast": {"weight": 0.8},
        "dual_rain_vehicle": {"weight": 0.5},
        "dual_overcast_vehicle": {"weight": 0.4},
        "dual_night_vehicle": {"weight": 0.0},
    },
}

WEATHER_SCENE_PHRASES = {
    "rain": "rainy street scene",
    "snow": "snowy street scene",
    "night": "night street scene",
    "overcast": "overcast street scene",
    "fog": "foggy street scene",
    "rainy_night": "rainy night street scene",
}

WEATHER_INSTRUCTION_PHRASES = {
    "rain": (
        "Apply strong rainy weather only:\n"
        "replace the sky with dark overcast rain clouds,\n"
        "add clearly visible rain streaks across the image,\n"
        "make the full drivable road surface wet with continuous reflective sheen,\n"
        "add several realistic puddles and elongated reflections on lanes and curbside asphalt,\n"
        "reduce visibility and local contrast slightly, darken the scene modestly, and keep all geometry sharp without blur."
    ),
    "snow": (
        "Apply snowy winter weather only:\n"
        "replace the sky with overcast winter clouds,\n"
        "add visible snowflakes falling across the image,\n"
        "make the ground, sidewalks and parked cars lightly snow-covered,\n"
        "coat rooftops with a thin layer of white snow,\n"
        "create a cold winter atmosphere with frosty surfaces."
    ),
    "night": (
        "Apply stronger urban night time only:\n"
        "darken the sky to nighttime black,\n"
        "add clear pools of artificial street lighting on the road surface,\n"
        "make building windows glow with warm indoor lights,\n"
        "darken the roadway and sidewalk while keeping lane geometry readable,\n"
        "reduce overall saturation and create a distinctly nocturnal atmosphere with stronger foreground-background contrast."
    ),
    "overcast": (
        "Apply heavy overcast weather only:\n"
        "replace the sky with a low dense blanket of thick grey clouds,\n"
        "flatten the entire scene lighting into strongly diffused cool daylight,\n"
        "remove nearly all direct-sun contrast and suppress crisp cast shadows,\n"
        "desaturate the full street scene moderately, cool the white balance slightly, and make distant/background regions look duller and more compressed,\n"
        "make some asphalt and curbside surfaces look faintly damp and less contrasty without adding rain streaks, puddles, or night lighting,\n"
        "preserve facade identity and geometry while making the whole scene feel distinctly gloomier and heavier than normal daylight."
    ),
    "fog": (
        "Apply foggy weather only:\n"
        "add thick realistic fog across the entire scene,\n"
        "reduce long-range visibility significantly,\n"
        "mute colors and soften contrast,\n"
        "keep nearby road and facade geometry readable."
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


def _deep_merge(base: dict, override: dict) -> dict:
    merged = deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def normalize_experience_bank(bank: dict | None = None) -> dict:
    """Return a complete frozen-v0-compatible bank, preserving learned stats if present."""
    merged = _deep_merge(EXPERIENCE_BANK_V0, bank or {})
    if DUAL_PROMPT_STRATEGY == "dual_hard_v5":
        templates = merged.setdefault("recommended_templates", {})
        templates["dual_rain_vehicle"] = {"weight": 1.0}
        templates["dual_rainy_night_vehicle"] = {"weight": 0.55}
        templates["dual_snow_vehicle"] = {"weight": 0.25}
        templates["dual_night_vehicle"] = {"weight": 0.0}
        templates["dual_overcast_vehicle"] = {"weight": 0.0}
        templates["dual_fog_vehicle"] = {"weight": 0.0}
        for key in list(templates):
            if key.startswith("dual_") and key.endswith("_person"):
                templates[key] = {"weight": 0.0}
        occlusion_stats = merged.setdefault("occlusion_statistics", {})
        occlusion_stats["vehicle"] = {"priority": 1, "status": "required"}
        occlusion_stats["person"] = {"priority": 99, "status": "disabled_for_v5"}
        prompt_rules = merged.setdefault("prompt_rules", {})
        vehicle_rules = prompt_rules.setdefault("vehicle", {})
        vehicle_rules["preferred_size"] = "medium_large_vehicle_existing_geometry_only"
        vehicle_rules["max_area_ratio"] = 0.28
        required = list(vehicle_rules.get("required", []))
        for item in [
            "existing road and parking geometry only",
            "no new architecture",
            "no new windows",
            "preserve empty skyline and background",
            "wet road reflection without scene reconstruction",
        ]:
            if item not in required:
                required.append(item)
        vehicle_rules["required"] = required
    elif DUAL_PROMPT_STRATEGY == "dual_hard_v4":
        templates = merged.setdefault("recommended_templates", {})
        templates["dual_rain_vehicle"] = {"weight": 1.0}
        templates["dual_rainy_night_vehicle"] = {"weight": 0.95}
        templates["dual_snow_vehicle"] = {"weight": 0.35}
        templates["dual_night_vehicle"] = {"weight": 0.0}
        templates["dual_overcast_vehicle"] = {"weight": 0.0}
        templates["dual_fog_vehicle"] = {"weight": 0.0}
        for key in list(templates):
            if key.startswith("dual_") and key.endswith("_person"):
                templates[key] = {"weight": 0.0}
        occlusion_stats = merged.setdefault("occlusion_statistics", {})
        occlusion_stats["vehicle"] = {"priority": 1, "status": "required"}
        occlusion_stats["person"] = {"priority": 99, "status": "disabled_for_v4"}
        prompt_rules = merged.setdefault("prompt_rules", {})
        vehicle_rules = prompt_rules.setdefault("vehicle", {})
        vehicle_rules["preferred_size"] = "large_box_truck_or_bus_wet_reflection_band"
        vehicle_rules["max_area_ratio"] = 0.32
        required = list(vehicle_rules.get("required", []))
        for item in [
            "large box truck or bus",
            "continuous wet reflective road band",
            "intersection or parking-strip cue interruption",
            "lane curb crosswalk lower-facade boundary occlusion",
            "visible wheels and perspective-aligned contact shadows",
        ]:
            if item not in required:
                required.append(item)
        vehicle_rules["required"] = required
    elif DUAL_PROMPT_STRATEGY == "dual_hard_v3":
        templates = merged.setdefault("recommended_templates", {})
        templates["dual_rainy_night_vehicle"] = {"weight": 1.0}
        templates["dual_rain_vehicle"] = {"weight": 0.82}
        templates["dual_snow_vehicle"] = {"weight": 0.72}
        templates["dual_night_vehicle"] = {"weight": 0.25}
        templates["dual_overcast_vehicle"] = {"weight": 0.0}
        templates["dual_fog_vehicle"] = {"weight": 0.0}
        prompt_rules = merged.setdefault("prompt_rules", {})
        vehicle_rules = prompt_rules.setdefault("vehicle", {})
        vehicle_rules["preferred_size"] = "large_midground_plus_small_foreground"
        vehicle_rules["max_area_ratio"] = 0.30
        required = list(vehicle_rules.get("required", []))
        for item in [
            "continuous lower-facade curb lane cue interruption",
            "hard weather visibility reduction",
            "wet reflection or slush around occluder",
            "secondary foreground-side cue",
        ]:
            if item not in required:
                required.append(item)
        vehicle_rules["required"] = required
    elif DUAL_PROMPT_STRATEGY == "test20_v1":
        templates = merged.setdefault("recommended_templates", {})
        templates["dual_rain_vehicle"] = {"weight": 0.9}
        templates["dual_overcast_vehicle"] = {"weight": 0.85}
        templates["dual_night_vehicle"] = {"weight": 0.65}
        templates["global_rain"] = {"weight": 0.9}
        templates["global_overcast"] = {"weight": 0.5}
        prompt_rules = merged.setdefault("prompt_rules", {})
        vehicle_rules = prompt_rules.setdefault("vehicle", {})
        vehicle_rules["preferred_size"] = "medium_to_large"
        vehicle_rules["max_area_ratio"] = 0.22
        required = list(vehicle_rules.get("required", []))
        for item in [
            "continuous lane-level occlusion",
            "meaningful curbside occlusion",
            "clear road-cue interruption",
        ]:
            if item not in required:
                required.append(item)
        vehicle_rules["required"] = required
    return merged


def load_experience_bank(path: str | Path | None = None) -> dict:
    if not path:
        return normalize_experience_bank()
    path = Path(path)
    if not path.exists():
        return normalize_experience_bank()
    try:
        return normalize_experience_bank(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError):
        return normalize_experience_bank()


def _template_weight(bank: dict, route: str, weather: str | None, occlusion: str | None) -> float:
    if route == "global":
        key = f"global_{weather}"
    elif route == "dual":
        key = f"dual_{weather}_{occlusion}"
    elif route == "local":
        key = f"local_{occlusion}"
    else:
        return 1.0
    templates = bank.get("recommended_templates", {})
    if key not in templates and route in {"global", "dual"}:
        return 0.0
    info = templates.get(key, {})
    try:
        return float(info.get("weight", 1.0))
    except (TypeError, ValueError):
        return 1.0


def choose_weather_from_experience(
    route: str,
    requested: str | None,
    occlusion: str | None = None,
    bank: dict | None = None,
) -> str | None:
    if route not in {"global", "dual"}:
        return None
    bank = normalize_experience_bank(bank)
    candidates = [weather for weather in WEATHERS if _template_weight(bank, route, weather, occlusion) > 0]
    if not candidates:
        candidates = list(WEATHERS)
    requested = requested if requested in candidates else None
    if requested and _template_weight(bank, route, requested, occlusion) > 0:
        return requested

    weather_stats = bank.get("weather_statistics", {})

    def sort_key(weather: str) -> tuple[float, int, str]:
        priority = weather_stats.get(weather, {}).get("priority", 99)
        try:
            priority = int(priority)
        except (TypeError, ValueError):
            priority = 99
        return (-_template_weight(bank, route, weather, occlusion), priority, weather)

    return sorted(candidates, key=sort_key)[0]


def choose_occlusion_from_experience(
    current: str | None,
    vehicle_ok: bool,
    person_ok: bool,
    bank: dict | None = None,
) -> str | None:
    bank = normalize_experience_bank(bank)
    stats = bank.get("occlusion_statistics", {})
    vehicle_priority = stats.get("vehicle", {}).get("priority", 1)
    person_priority = stats.get("person", {}).get("priority", 2)
    try:
        vehicle_priority = int(vehicle_priority)
    except (TypeError, ValueError):
        vehicle_priority = 1
    try:
        person_priority = int(person_priority)
    except (TypeError, ValueError):
        person_priority = 2
    if vehicle_ok and (vehicle_priority <= person_priority or current != "person"):
        return "vehicle"
    if current == "vehicle" and vehicle_ok:
        return "vehicle"
    if person_ok:
        return "person"
    return None


def negative_prompt_from_experience(bank: dict | None = None) -> str:
    bank = normalize_experience_bank(bank)
    rules = bank.get("prompt_rules", {})
    forbidden = []
    for key in ("global", "vehicle", "person"):
        forbidden.extend(rules.get(key, {}).get("forbidden", []))
    extra = ", ".join(str(item) for item in forbidden)
    if not extra:
        return f"{NEGATIVE_PROMPT}, {GLOBAL_NEGATIVE_PROMPT}, {LIGHTX2V_LOCAL_MULTI_NEGATIVE_TERMS}"
    return f"{NEGATIVE_PROMPT}, {GLOBAL_NEGATIVE_PROMPT}, {LIGHTX2V_LOCAL_MULTI_NEGATIVE_TERMS}, {extra}"


def global_negative_prompt() -> str:
    return (
        "oil painting, painterly, illustration, watercolor, brush strokes, stylized image, "
        "cartoon, anime, cinematic color grading, colorful lighting, neon facade, rainbow facade, "
        "oversaturated storefront, changed facade color, changed wall material, changed shop sign, "
        "sign deformation, storefront color shift, warped buildings, changed road layout, "
        "changed camera viewpoint"
    )


@dataclass(frozen=True)
class PromptPackage:
    prompt: str
    negative_prompt: str = NEGATIVE_PROMPT


def _rule_key(route: str, weather: str | None, occlusion: str | None) -> str:
    if route == "skip":
        return "skip"
    if route == "global":
        return f"global_{weather}"
    if route == "local":
        return f"local_{occlusion}"
    if route == "dual":
        return f"dual_{weather}_{occlusion}"
    return "skip"


def default_position(route: str, occlusion: str | None) -> str:
    if route == "skip":
        return "none"
    if route == "global":
        return "the entire image"
    if occlusion == "person":
        return "visible sidewalk, curb, crosswalk, roadside pavement, or road-edge ground"
    if occlusion == "vehicle":
        return "visible traffic lane, curbside lane, parking bay, or roadside parking area"
    return "a visible street surface with clear ground-plane support"


def _context_text(*parts: object) -> str:
    return " ".join(str(part or "") for part in parts).lower()


def has_vehicle_surface(*parts: object) -> bool:
    text = _context_text(*parts)
    return any(term in text for term in VEHICLE_SURFACE_TERMS)


def has_person_surface(*parts: object) -> bool:
    text = _context_text(*parts)
    return any(term in text for term in PERSON_SURFACE_TERMS)


def choose_occlusion_from_context(*parts: object, requested: str | None = None) -> str | None:
    """Vehicle-first occlusion choice from planner text; never fabricates a person fallback."""
    requested = requested if requested in OCCLUSIONS else None
    vehicle_ok = has_vehicle_surface(*parts)
    person_ok = has_person_surface(*parts)
    if vehicle_ok:
        return "vehicle"
    if requested == "person" and person_ok:
        return "person"
    if requested == "vehicle" and vehicle_ok:
        return "vehicle"
    if person_ok:
        return "person"
    return None


def ensure_global_iclight_constraints(prompt: str) -> str:
    """Append global IC-Light facade constraints when a structured prompt predates them."""
    if "Natural documentary street-view photo" in prompt or "natural documentary street-view photo" in prompt:
        return prompt
    return (
        f"{prompt} Natural documentary street-view photo, realistic lighting, conservative edit. "
        "Preserve road geometry, building facades, storefronts, signs, doors, windows, "
        "wall materials, original facade colors, and storefront identity."
    ).strip()


def _short_weather_phrase(weather: str | None, rule: dict) -> str:
    scene_phrase = WEATHER_SCENE_PHRASES.get(weather or "", f"{weather} street scene")
    target = str(rule.get("target", "")).strip()
    if not target:
        return scene_phrase
    target = target.split(".")[0]
    return f"{scene_phrase}, {target}"


def _short_position(position: str, occlusion: str | None) -> str:
    if not position:
        return default_position("local", occlusion)
    text = " ".join(str(position).replace("\n", " ").split())
    if len(text) > 140:
        text = text[:140].rsplit(" ", 1)[0]
    return text


def _local_vehicle_crowded_remove_prompt(position: str) -> str:
    pos = _short_position(position, "vehicle")
    return (
        "Conservative street-view edit for a vehicle-heavy original scene. The original image already "
        "contains several vehicles or visually crowded curb/road areas. "
        "If more than three vehicles are visible in the target road, curbside, or parking area, reduce "
        "vehicle clutter instead of adding cars: remove 1-2 least important background or curbside "
        "vehicles only, and naturally fill those regions with matching road, curb, sidewalk, parking-lane "
        "texture, shadows, or background continuation. Keep at most 1-2 clearly visible vehicles in the "
        "edited target area. If a local occluder is still needed, prefer one cyclist or one pedestrian near "
        "the road edge rather than adding another car. "
        f"Place any remaining local edit only on or immediately beside: {pos}. "
        "Preserve road geometry, lane markings, camera viewpoint, building facades, storefronts, signs, "
        "windows, doors, existing place identity, and global lighting/weather. Do not remove landmark "
        "objects, traffic signs, storefront identity, lane markings, or vehicles that define the place. "
        "Avoid traffic jams, dense vehicle clusters, new background cars, oversized foreground cars, black "
        "vehicle blocks, pasted cutouts, floating objects, and over-occlusion. Natural photorealistic "
        "street-view image."
    )


def _local_multi_occlusion_prompt(occlusion: str | None, position: str, original_vehicle_crowded: bool = False) -> str:
    if occlusion == "vehicle" and original_vehicle_crowded:
        return _local_vehicle_crowded_remove_prompt(position)

    pos = _short_position(position, occlusion)
    if occlusion == "person":
        occluders = (
            "Add controlled multi-occlusion with 2-4 realistic full-body pedestrians, "
            "optionally one cyclist or scooter rider if a road-edge or crosswalk is visible, "
            "and optionally one small roadside tree or pole-side shrub only near the curb."
        )
    elif occlusion == "vehicle":
        occluders = (
            "Add controlled multi-occlusion with 2-3 realistic street vehicles such as cars, "
            "vans, taxis, or a small bus, optionally one cyclist or scooter rider near the road edge, "
            "and optionally one small roadside tree near the curb."
        )
    else:
        occluders = (
            "Add controlled multi-occlusion with 1-3 realistic local street occluders, including "
            "vehicles, pedestrians, cyclists or scooter riders, and optionally one small roadside tree."
        )
    return (
        f"{occluders} Place them only on or immediately beside: {pos}. "
        "Total new occluder area should occupy approximately 12-20% of the image. "
        "Occluders should be clearly visible at mid-ground or near-ground scale, with realistic texture, "
        "matched lighting, contact shadows, wheels or feet aligned to the ground plane, and noticeable but "
        "natural local occlusion. Prioritize partially occluding traffic lanes, curbside lanes, roadside areas, "
        "sidewalks, and lower street-level foreground. Mildly occluding the lower edge of buildings, shopfront "
        "corners, parked-car edges, or non-core facade margins is allowed. Do not cover main sign text, landmark "
        "structures, central building facade areas, storefront identity, lane geometry, or place-defining details. "
        "Preserve road geometry, camera viewpoint, building structure, road layout, lane markings, signs, windows, "
        "doors, existing place identity, and global lighting/weather. Do not change weather, time of day, color "
        "style, viewpoint, architecture, or road layout. Natural photorealistic street-view image. Avoid tiny "
        "occluders, weak occlusion, barely visible objects, sticker-like cutouts, floating objects, black silhouettes, "
        "edge-attached objects, dense crowds, traffic jams, or over-occlusion."
    )


def build_structured_prompt(
    route: str,
    weather: str | None,
    occlusion: str | None,
    position: str = "",
    base_prompt: str = "",
    experience_bank: dict | None = None,
    original_vehicle_crowded: bool = False,
) -> str:
    """Build compact model prompts. Positive prompts only say what to generate."""
    bank = normalize_experience_bank(experience_bank)
    key = _rule_key(route, weather, occlusion)
    rule = EXPERIENCE_RULES.get(key, {})
    position = position or default_position(route, occlusion)

    if route == "skip":
        reason = base_prompt or "image is not suitable for realistic weather/time or street-participant occlusion."
        return (
            "Task: skip generation. "
            f"Reason: {reason} "
            "Do not call generation service."
        )
    if route == "global":
        if weather == "rain":
            return GLOBAL_RAIN_CONSERVATIVE_PROMPT
        if weather == "snow":
            return GLOBAL_SNOW_CONSERVATIVE_PROMPT
        if weather == "night":
            return GLOBAL_NIGHT_CONSERVATIVE_PROMPT
        weather_text = _short_weather_phrase(weather, rule)
        return (
            f"{weather_text}. Natural documentary street-view photo, realistic lighting, conservative edit. "
            "Preserve road geometry, camera viewpoint, building facades, storefronts, signs, doors, "
            "windows, wall materials, rooflines, original facade colors, and storefront identity."
        )

    if route == "local":
        return _local_multi_occlusion_prompt(occlusion, position, original_vehicle_crowded=original_vehicle_crowded)

    pos = _short_position(position, occlusion)
    preserve = (
        "Preserve road layout, camera viewpoint, building facades, storefronts, signs, "
        "windows, doors, traffic signs, and place identity."
    )

    if occlusion == "vehicle":
        vehicle_constraint = (
            LIGHTX2V_DUAL_VEHICLE_CONSTRAINT
            if route == "dual"
            else LIGHTX2V_LOCAL_VEHICLE_CONSTRAINT
        )
        occluder_text = f"Add {vehicle_constraint} on {pos}."
    elif occlusion == "person":
        occluder_text = f"Add {LIGHTX2V_PERSON_CONSTRAINT} on {pos}."
    else:
        occluder_text = f"Add exactly one realistic street participant on {pos}."

    if route == "dual":
        weather_instr = WEATHER_INSTRUCTION_PHRASES.get(weather or "", f"Apply {weather} weather only.")
        if DUAL_PROMPT_STRATEGY == "dual_hard_v5" and weather == "rainy_night":
            weather_instr = (
                "Apply rainy night weather only:\n"
                "darken the existing sky and scene into realistic rainy night or late-evening lighting,\n"
                "add visible rain streaks, wet pavement, puddles, and reflections from already visible lamps, headlights, signs, or vehicles,\n"
                "only existing windows may become slightly lit if they are already present in the original image,\n"
                "do not add any new buildings, windows, walls, storefronts, skyline blocks, or architectural structures."
            )
        if DUAL_PROMPT_STRATEGY == "dual_hard_v5":
            dual_vehicle_focus = (
                "Make this a hard but structure-preserving VPR-positive edit. The vehicle and weather should make the "
                "lower street harder through existing road cues only: lane markings, curb lines, parking-strip edges, "
                "crosswalk bars, roadside parking, puddles, wet asphalt, and reflections. Do not use or mention "
                "storefront lower margins, facade edges, building windows, skyline, or architectural features unless "
                "they are clearly already present in the original image; even then, preserve their exact shape and "
                "layout. No new architecture is allowed: no new buildings, no added facades, no added windows, no new "
                "walls, no new storefronts, no new skyline blocks, no new background structures. If the original scene "
                "is an open road, parking lot, highway ramp, industrial yard, or sparse street, keep it open and sparse. "
                "Use rain, puddles, tire spray, headlight reflection, and a perspective-aligned truck/bus to interrupt "
                "existing road/parking geometry, while preserving exact camera viewpoint, road topology, existing "
                "background, signs, trees, poles, barriers, parked vehicles, and place identity."
            )
        elif DUAL_PROMPT_STRATEGY == "dual_hard_v4":
            dual_vehicle_focus = (
                "Make this a hard VPR-positive edit by copying the observed successful pattern: a large realistic "
                "box truck, delivery van, bus, or service vehicle creates the main occlusion while heavy rain or rainy "
                "night lighting creates a broad wet reflection band across the lower half of the street. The truck/bus "
                "must interrupt one continuous structural cue used for place recognition: crosswalk stripes, lane "
                "markings, curb line, parking-lane boundary, sidewalk-road transition, storefront lower margin, or "
                "sign-adjacent lower facade edge. Add headlight reflections, puddles, rain streaks, tire spray, and "
                "glossy asphalt around the vehicle so the vehicle plus reflection becomes one difficult lower-scene "
                "region. Keep the edit photorealistic and VPR-valid: preserve exact camera viewpoint, road topology, "
                "building silhouettes, landmark structure, facade layout, main sign identity, and skyline. Do not "
                "replace buildings, warp facades, create traffic jams, use black masks, or cover the full landmark facade."
            )
        elif DUAL_PROMPT_STRATEGY == "dual_hard_v3":
            dual_vehicle_focus = (
                "Make this a deliberately hard but still VPR-valid dual edit. The weather should reduce street-level "
                "visibility through rain streaks, wet reflective glare, headlight spill, puddles, slush, tire tracks, "
                "or low-light contrast loss. The dominant vehicle must cut across a continuous lower-scene recognition "
                "band: curb line, lane marking, crosswalk edge, parking strip, storefront lower margin, sidewalk-road "
                "transition, or sign-adjacent lower facade area. Add the smaller foreground-side cue only as natural "
                "partial clutter, never as a black border block. Preserve the exact building silhouettes, landmark "
                "structure, facade layout, main sign identity, road geometry, perspective, and camera viewpoint. Do not "
                "replace buildings, repaint facades, cover the full landmark facade, or destroy place identity."
            )
        elif DUAL_PROMPT_STRATEGY == "dual_hard_v2":
            dual_vehicle_focus = (
                "Make the dual edit substantially harder than a normal positive while still VPR-valid. Combine a "
                "strong weather/time shift with a contiguous lower-scene occlusion band. The dominant vehicle should "
                "interrupt lane, curb, parking, or road-edge geometry, and the secondary cue should add local clutter "
                "without becoming a traffic jam or crowd. Keep all landmark structure, building outline, facade layout, "
                "main signs, road geometry, and camera viewpoint unchanged. Do not cover the central landmark facade or "
                "destroy place identity."
            )
        elif DUAL_PROMPT_STRATEGY == "test20_v1":
            dual_vehicle_focus = (
                "Make the added vehicle the main place-recognition difficulty source. Do not place a decorative small "
                "car. Place one larger realistic vehicle on the legal road surface or curbside parking area so that it "
                "hides one continuous and meaningful street-level cue region, such as lane markings, curb transitions, "
                "crosswalk edges, curbside parking-strip boundaries, parked-car edges, or lower storefront margins. "
                "Prefer a closer, heavier, mid-ground occluder that interrupts contiguous road or curb geometry rather "
                "than a tiny object. Keep landmark structure, main sign readability, overall facade identity, road "
                "geometry, and viewpoint unchanged."
            )
        else:
            dual_vehicle_focus = (
                "Make the added vehicle the main local difficulty source: place it on the legal road surface or curbside "
                "parking area so that it occludes a meaningful contiguous road region, such as lane markings, curbside "
                "space, parked-car edges, or the lower street-level facade margin. Prefer a closer and slightly larger "
                "vehicle placement that occupies one meaningful road chunk rather than a small decorative car. Keep "
                "landmark structure and main sign text readable."
            )
        if weather == "overcast" and occlusion == "vehicle":
            dual_vehicle_focus = (
                f"{dual_vehicle_focus} Under the heavy overcast lighting, make the vehicle read clearly as a darker "
                "mid-ground mass against the flatter street background, and let it block a slightly longer continuous "
                "lane or curbside strip than in the rain/night variants, while still preserving the landmark facade."
            )
        elif DUAL_PROMPT_STRATEGY in {"test20_v1", "dual_hard_v2", "dual_hard_v3", "dual_hard_v4"} and weather == "rain" and occlusion == "vehicle":
            dual_vehicle_focus = (
                f"{dual_vehicle_focus} Under rain, extend the wet reflective region around the vehicle so the vehicle "
                "plus adjacent lane, curb, and storefront-lower-boundary cues form one continuous difficult area, but do not blur geometry."
            )
        elif DUAL_PROMPT_STRATEGY in {"test20_v1", "dual_hard_v2", "dual_hard_v3", "dual_hard_v4"} and weather in {"night", "rainy_night"} and occlusion == "vehicle":
            dual_vehicle_focus = (
                f"{dual_vehicle_focus} Under night lighting, make the vehicle read clearly against the darker road and "
                "use headlight glare and wet reflections to make the lower curb/lane/facade-boundary band harder without turning the whole image into unreadable darkness."
            )
        elif DUAL_PROMPT_STRATEGY in {"dual_hard_v2", "dual_hard_v3"} and weather == "snow" and occlusion == "vehicle":
            dual_vehicle_focus = (
                f"{dual_vehicle_focus} Under snow, add visible slush and compacted tire-track contrast around the vehicle "
                "so the lower road, curb, and parking-strip region becomes harder, while keeping building geometry recoverable."
            )
        return (
            f"Preserve all scene geometry, buildings, storefronts, signs, existing vehicles, "
            f"road layout and camera viewpoint.\n\n"
            f"{weather_instr}\n\n"
            f"{occluder_text}\n\n"
            f"{dual_vehicle_focus}\n\n"
            f"Do not modify scene structure or object layout."
        )

    return base_prompt or "Skip augmentation."


def package_prompt(decision: dict, prompt: str | None = None) -> PromptPackage:
    structured = build_structured_prompt(
        route=decision.get("route", "dual"),
        weather=decision.get("weather"),
        occlusion=decision.get("occlusion"),
        position=decision.get("position", ""),
        base_prompt=prompt or decision.get("prompt", ""),
    )
    return PromptPackage(prompt=structured)


def predict_bad_image(
    route: str,
    prompt: str,
    reason: str = "",
    weather: str | None = None,
    occlusion: str | None = None,
) -> dict:
    text = f"{prompt} {reason}".lower()
    flags = []

    if route in {"local", "dual"} and not occlusion:
        flags.append("missing_occlusion")
    if route in {"global", "dual"} and not weather:
        flags.append("missing_weather")
    if route in {"local", "dual"} and any(x in text for x in ["taxi rear", "rear of the", "specific sign"]):
        flags.append("over_specific_occlusion_target")
    if any(x in text for x in ["facade close-up", "no visible sidewalk", "no visible road", "no plausible sidewalk"]):
        flags.append("occlusion_implausible")
    if route in {"local", "dual"} and occlusion == "person" and has_vehicle_surface(text):
        flags.append("person_selected_despite_vehicle_surface")
    if route in {"local", "dual"} and occlusion == "vehicle" and not has_vehicle_surface(text):
        flags.append("vehicle_surface_not_explicit")
    if route in {"local", "dual"} and occlusion == "person" and not has_person_surface(text):
        flags.append("person_surface_not_explicit")
    if "dense green tree" in text and route in {"local", "dual"}:
        flags.append("foreground_foliage_may_confuse_occluder")
    if route == "dual" and occlusion == "vehicle" and weather == "night":
        flags.append("night_vehicle_high_hallucination_risk")

    risk_score = min(10, len(flags) * 3)
    return {
        "risk_score": risk_score,
        "risk_flags": flags,
        "skip_recommendation": risk_score >= 6,
    }
