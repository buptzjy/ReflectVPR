import os
import json
import numpy as np

# ==========================================
#               全局配置区
# ==========================================
# 输入文件与目录路径
JSON_PATH = "/media/data/zhangjingyi/vismatch/outputs/superpoint-lightglue/results.json"
EMBEDDINGS_DIR = "/media/data/zhangjingyi/datasets/additional_cities/ImageEmbeddings"

# 输出文件保存路径
OUTPUT_DIR = "/media/data/zhangjingyi/ReflectVPR"
OUTPUT_FILENAME = "thresholds.json"

# 百分位数配置 (c=1, c=2, c=3)
GEO_PERCENTILES = [40, 25, 15]  # s_geo 对应的百分位
DIV_PERCENTILES = [30, 20, 10]  # s_div 对应的百分位
# ==========================================


def load_all_embeddings(embeddings_dir):
    """
    加载指定目录下所有的 npz 嵌入文件，并建立从 key 到 embedding 的映射字典。
    """
    embeddings_dict = {}
    print("正在加载 NPZ 嵌入文件...")
    for filename in os.listdir(embeddings_dir):
        if filename.endswith('.npz'):
            filepath = os.path.join(embeddings_dir, filename)
            try:
                data = np.load(filepath)
                for key in data.files:
                    pure_key = os.path.basename(key).split('.')[0]
                    embeddings_dict[pure_key] = data[key]
            except Exception as e:
                print(f"加载文件 {filename} 失败: {e}")
    print(f"所有嵌入加载完成，共计 {len(embeddings_dict)} 个唯一 Key。")
    return embeddings_dict


def extract_key_from_path(img_path):
    """
    从图像路径中提取唯一的 key ID。
    例如: /.../PhoenixSnowy_..._ZOXLftAwvAo9qJo_iJ4UzA.jpg -> ZOXLftAwvAo9qJo_iJ4UzA
    """
    if not img_path:
        return None
    basename = os.path.basename(img_path)
    name_without_ext = os.path.splitext(basename)[0]
    parts = name_without_ext.split('_')
    if len(parts) > 0:
        return parts[-1]
    return name_without_ext


def compute_cosine_similarity(v1, v2):
    """
    计算两个向量的余弦相似度
    """
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


def main():
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. 加载所有 npz 向量
    embeddings_dict = load_all_embeddings(EMBEDDINGS_DIR)

    # 2. 读取匹配结果的 json 文件
    print(f"正在读取匹配结果: {JSON_PATH}")
    pairs_list = []
    with open(JSON_PATH, 'r', encoding='utf-8') as f:
        try:
            data_json = json.load(f)
            # 【核心修复】：如果外层是字典且包含 "results"，获取其对应的图对列表
            if isinstance(data_json, dict) and "results" in data_json:
                pairs_list = data_json["results"]
            elif isinstance(data_json, list):
                pairs_list = data_json
            else:
                pairs_list = [data_json]
        except json.JSONDecodeError:
            # 每行一个 JSON 的兜底处理
            f.seek(0)
            pairs_list = [json.loads(line.strip()) for line in f if line.strip()]

    s_geo_list = []
    s_div_list = []
    missing_keys_count = 0

    # 3. 遍历每一对图像，提取 s_geo 和计算 s_div
    print(f"开始处理图像对，图对总数: {len(pairs_list)} 对...")
    for item in pairs_list:
        if not isinstance(item, dict):
            continue

        inlier_ratio = item.get("inlier_ratio", None)
        if inlier_ratio is None:
            continue
            
        img0_path = item.get("img0", "")
        img1_path = item.get("img1", "")

        # 提取两张图的 key
        key0 = extract_key_from_path(img0_path)
        key1 = extract_key_from_path(img1_path)

        # 检查两个 key 是否都在 npz 数据集中
        if key0 in embeddings_dict and key1 in embeddings_dict:
            v0 = embeddings_dict[key0]
            v1 = embeddings_dict[key1]
            
            # 计算 s_div：余弦相似度
            cos_sim = compute_cosine_similarity(v0, v1)
            
            s_geo_list.append(float(inlier_ratio))
            s_div_list.append(cos_sim)
        else:
            missing_keys_count += 1

    print(f"处理完成。成功匹配并计算图对: {len(s_geo_list)} 组，缺失特征 Key 的图对: {missing_keys_count} 组。")

    if not s_geo_list or not s_div_list:
        print("错误：未能成功计算任何图对的得分，请检查图像路径中的 Key 提取规则是否与 NPZ 中的 Key 一致。")
        return

    # 4. 根据配置区的分位数计算对应的阈值
    tau_geo_c1 = float(np.percentile(s_geo_list, GEO_PERCENTILES[0]))
    tau_geo_c2 = float(np.percentile(s_geo_list, GEO_PERCENTILES[1]))
    tau_geo_c3 = float(np.percentile(s_geo_list, GEO_PERCENTILES[2]))

    tau_div_c1 = float(np.percentile(s_div_list, DIV_PERCENTILES[0]))
    tau_div_c2 = float(np.percentile(s_div_list, DIV_PERCENTILES[1]))
    tau_div_c3 = float(np.percentile(s_div_list, DIV_PERCENTILES[2]))

    # 5. 组装输出 JSON 结构
    thresholds_output = {
        "tau_geo_c1": tau_geo_c1,
        "tau_geo_c2": tau_geo_c2,
        "tau_geo_c3": tau_geo_c3,
        "tau_div_c1": tau_div_c1,
        "tau_div_c2": tau_div_c2,
        "tau_div_c3": tau_div_c3
    }

    # 6. 保存到指定位置
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(thresholds_output, f, indent=4, ensure_ascii=False)

    print(f"\n阈值文件预校准成功！已保存至: {output_path}")
    print("计算出的阈值结果如下:")
    print(json.dumps(thresholds_output, indent=4))


if __name__ == "__main__":
    main()