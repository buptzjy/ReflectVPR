import argparse
import json
import random
import sys
from pathlib import Path


ROOT = Path("/media/data/zhangjingyi/ReflectVPR")
AGENT_DIR = ROOT / "your_agent"
sys.path.insert(0, str(AGENT_DIR))

from agent import SceneAugmentAgent  # noqa: E402


def select_images(image_dir: Path, limit: int, seed: int) -> list[Path]:
    images = sorted(image_dir.glob("*.jpg"))
    if limit <= 0 or limit >= len(images):
        return images
    rng = random.Random(seed)
    return rng.sample(images, limit)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image-dir",
        default="/media/data1/chenshunpeng1/datasets/gsv_cities/Images/London",
    )
    parser.add_argument("--output", default=str(ROOT / "batch_plan.json"))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260529)
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    if not image_dir.exists():
        raise FileNotFoundError(image_dir)

    agent = SceneAugmentAgent(mock=args.mock, planning_only=True)
    plans = []
    images = select_images(image_dir, args.limit, args.seed)
    for idx, image_path in enumerate(images, 1):
        print(f"[{idx}/{len(images)}] plan: {image_path.name}", flush=True)
        plan = agent.plan_image(image_path)
        plans.append(plan)
        Path(args.output).write_text(json.dumps(plans, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Done. batch_plan={args.output}")


if __name__ == "__main__":
    main()
