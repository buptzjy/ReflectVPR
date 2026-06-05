#!/usr/bin/env python3
import os, random
from pathlib import Path
from collections import defaultdict

GEN_BASE = Path("/media/data/zhangjingyi/datasets/additional_cities/Images")
ORIG_BASE = Path("/media/data1/chenshunpeng1/datasets/gsv_cities/Images")

GEN_DIRS = {
    "LondonRainy": "London",
    "LondonSnowy": "London",
    "OsakaRainy":  "Osaka",
    "OsakaSnowy":  "Osaka",
    "PhoenixRainy":"Phoenix",
    "PhoenixSnowy":"Phoenix",
    # "LA_rainy":    "LosAngeles",
    # "LA_snowy":    "LosAngeles",
}

N_SAMPLE = 5000

# 1. 扫描原图：按子目录遍历
print("扫描原图...")
orig_index = defaultdict(list)
total_orig = 0
for city_dir in ORIG_BASE.iterdir():
    if not city_dir.is_dir():
        continue
    city = city_dir.name  # London, Osaka, Phoenix, LosAngeles ...
    count = 0
    for f in city_dir.iterdir():
        if f.suffix.lower() == ".jpg":
            name = f.stem
            parts = name.split("_", 2)
            if len(parts) >= 2:
                orig_index[(city, parts[1])].append(str(f))
            count += 1
    total_orig += count
    print(f"  {city}: {count} 张")
print(f"  总计: {total_orig} 张, {len(orig_index)} 个场景")

# 2. 扫描生成图，配对
print("\n扫描生成图并配对...")
all_pairs = []
for gen_dir, orig_city in GEN_DIRS.items():
    gen_path = GEN_BASE / gen_dir
    if not gen_path.exists():
        print(f"  [WARN] 不存在: {gen_path}")
        continue
    matched = 0
    for f in gen_path.iterdir():
        if f.suffix.lower() != ".jpg":
            continue
        name = f.stem
        parts = name.split("_", 2)
        if len(parts) < 2:
            continue
        scene_id = parts[1]
        key = (orig_city, scene_id)
        if key in orig_index and orig_index[key]:
            all_pairs.append((str(f), orig_index[key][0]))
            matched += 1
    print(f"  {gen_dir} → {orig_city}: {matched} 对")

print(f"\n总配对数: {len(all_pairs)}")

# 3. 采样
random.seed(42)
if len(all_pairs) > N_SAMPLE:
    sampled = random.sample(all_pairs, N_SAMPLE)
    print(f"采样: {N_SAMPLE} 对")
else:
    sampled = all_pairs
    print(f"全部使用: {len(sampled)} 对")

# 4. 写 pairs.txt
out_file = Path("/media/data/zhangjingyi/ReflectVPR/pairs.txt")
out_file.parent.mkdir(parents=True, exist_ok=True)
with open(out_file, "w") as f:
    for gen, orig in sampled:
        f.write(f"{gen}\t{orig}\n")
print(f"已保存: {out_file}")