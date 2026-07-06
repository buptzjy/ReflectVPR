import os
import json
import base64
import concurrent.futures
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# --- 配置区 ---
load_dotenv(Path(__file__).parent / ".env")
load_dotenv(Path(__file__).parent / "your_agent" / ".env")

API_KEY = os.getenv("OPENAI_API_KEY", "")
BASE_URL = os.getenv("OPENAI_API_BASE", "https://api.yunwu.ai/v1")
MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "qwen3-vl-flash")

IMAGE_BASE_DIR = "/media/data1/chenshunpeng1/datasets/gsv_cities/Images"
OUTPUT_DIR = "/media/data/zhangjingyi/ReflectVPR/"
CITIES = ["London", "Osaka", "Phoenix", "LosAngeles"] 

MAX_FILES_PER_CITY = 5  
TARGET_QWEN_PERCENT = 30

if not API_KEY or API_KEY == "sk-xxx":
    raise RuntimeError("OPENAI_API_KEY 未配置，请在 ReflectVPR/.env 或 your_agent/.env 中设置。")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

SYSTEM_PROMPT = """你是一个街景图像专家。请观察图像并给出评分（0-10）：
1. weather_score: 晴天、强日光、干噪程度。强日光阴影给8-10，多云给6-7，阴天/弱光给0-5。
2. occlusion_score: 画面中已有物体的密集程度。极其空旷给0-2，有少量车马给3-5，拥挤给6-10。

请严格输出JSON:
{"weather_score":int, "occlusion_score":int, "reason":str}"""

def encode_image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def process_image_file(city, image_path):
    try:
        base64_image = encode_image_to_base64(image_path)
        user_content = [
            {"type": "text", "text": f"城市: {city}"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
        ]

        response = client.chat.completions.create(
            model=MODEL_NAME, 
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_content}]
        )
        
        raw_content = response.choices[0].message.content.strip()
        if "```json" in raw_content:
            raw_content = raw_content.split("```json")[1].split("```")[0].strip()
        
        res = json.loads(raw_content)
        w = res.get("weather_score", 0)
        o = res.get("occlusion_score", 0)
        
        # --- Step 1: 语义路由判定 ---
        route = "skip"
        candidate_model = "NONE"

        if w < 6 and o >= 3:
            # 阴天 + 拥挤
            route = "skip"
            candidate_model = "NONE"
        elif w < 6 and o < 3:
            # 阴天 + 空旷
            route = "occlusion_only"
            candidate_model = "Qwen"
        elif w >= 6 and o >= 3:
            # 晴天 + 拥挤
            route = "weather_only"
            candidate_model = "IC-Light"
        elif w >= 6 and o < 3:
            # 晴天 + 空旷
            route = "weather_and_occlusion"
            candidate_model = "Qwen"

        return {
            "file_name": image_path.name, 
            "city": city,
            "route": route,
            "weather_score": w,
            "occlusion_score": o,
            "total_score": w + o,
            "candidate_model": candidate_model,
            "selected_model": "PENDING", # 等待 Step 2 刷新
            "reason": res.get("reason", "")
        }
    except Exception as e:
        print(f"Error: {e}")
        return None

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = []
    
    # 扫描与并发
    tasks = []
    for city in CITIES:
        city_path = Path(IMAGE_BASE_DIR) / city
        if not city_path.exists(): continue
        files = []
        for ext in ["*.jpg", "*.jpeg", "*.png"]: files.extend(list(city_path.glob(ext)))
        if MAX_FILES_PER_CITY: files = files[:MAX_FILES_PER_CITY]
        for f in files: tasks.append((city, f))

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_to_file = {executor.submit(process_image_file, city, f): f for city, f in tasks}
        for future in concurrent.futures.as_completed(future_to_file):
            res = future.result()
            if res: results.append(res)

    # --- Step 2: Qwen 配额控制 ---
    valid_tasks = [r for r in results if r["route"] != "skip"]
    total_valid_count = len(valid_tasks)
    qwen_quota = int(total_valid_count * (TARGET_QWEN_PERCENT / 100))

    # 提取候选 Qwen 并按总分降序
    qwen_pool = [r for r in valid_tasks if r["candidate_model"] == "Qwen"]
    qwen_pool.sort(key=lambda x: x["total_score"], reverse=True)

    # 确定最终入选 Qwen 的名单
    qwen_selected_names = set([r["file_name"] for r in qwen_pool[:qwen_quota]])

    for item in results:
        if item["route"] == "skip":
            item["selected_model"] = "SKIP"
        elif item["candidate_model"] == "IC-Light":
            item["selected_model"] = "IC-Light"
        elif item["candidate_model"] == "Qwen":
            if item["file_name"] in qwen_selected_names:
                item["selected_model"] = "Qwen"
            else:
                item["selected_model"] = "IC-Light" # 降级

    # 保存文件
    with open(os.path.join(OUTPUT_DIR, "all_4city_decision.json"), 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    qwen_final = [r for r in results if r["selected_model"] == "Qwen"]
    with open(os.path.join(OUTPUT_DIR, "qwen.json"), 'w', encoding='utf-8') as f:
        json.dump(qwen_final, f, indent=2, ensure_ascii=False)

    print(f"📊 统计报告:")
    print(f"   - 生图任务总数: {total_valid_count}")
    print(f"   - Qwen 硬上限 (30%): {qwen_quota}")
    print(f"   - Qwen 实际分配: {len(qwen_final)}")

if __name__ == "__main__":
    main()
