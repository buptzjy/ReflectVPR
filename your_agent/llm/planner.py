import base64
import json
from io import BytesIO
from PIL import Image


def _image_to_base64(image: Image.Image) -> str:
    buffer = BytesIO()
    image.save(buffer, format="JPEG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def build_scene_understanding_prompt() -> str:
    return """你是一个自动驾驶场景分析专家。
请分析输入的道路场景图像，返回JSON格式的场景描述：
{
  "scene_summary": "场景简要描述",
  "weather": "当前天气（晴/阴/雨/雪/雾）",
  "time_of_day": "白天/夜晚/黄昏",
  "road_type": "城市道路/高速/乡村",
  "key_objects": ["主要对象列表"],
  "occlusion": "是否存在遮挡（true/false）",
  "difficulty_hints": "可能的增强难点"
}
只返回JSON，不要其他文字。"""


def build_generation_prompt_request(scene_info: dict, route: str) -> str:
    route_instruction = {
        "global": "只修改天气和光照条件，保持场景结构和所有对象位置不变。",
        "local":  "在场景中添加真实的动态遮挡物（如卡车、行人、障碍物），不改变天气。",
        "dual":   "同时修改天气条件并添加动态遮挡物，两种变化需自然融合。",
    }
    return f"""基于以下场景信息，生成一个用于图像生成模型的英文prompt。

场景信息：
{json.dumps(scene_info, ensure_ascii=False, indent=2)}

生成任务：{route_instruction[route]}

要求：
1. prompt用英文编写
2. 明确描述目标天气/遮挡效果
3. 强调"preserve road structure, lane markings, and spatial layout"
4. 长度控制在50-80词
5. 只返回prompt文字，不要其他内容。"""


class ScenePlanner:
    def __init__(self, llm_client):
        self.client = llm_client

    def understand_scene(self, image: Image.Image) -> dict:
        b64 = _image_to_base64(image)
        system_prompt = build_scene_understanding_prompt()

        raw = self.client.chat_with_images(
            system=system_prompt,
            user="请分析这张道路场景图像。",
            images=[b64],
            json_mode=False,
            temperature=0.3,
            model="qwen-vl-max",
        )

        raw = raw.strip().replace("```json", "").replace("```", "").strip()
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            print(f"[Planner] JSON解析失败，返回原始文本: {raw}")
            return {"scene_summary": raw, "parse_error": True}

    def generate_initial_prompt(self, scene_info: dict, route: str) -> str:
        request_text = build_generation_prompt_request(scene_info, route)

        prompt = self.client.chat(
            user=request_text,
            temperature=0.7,
            model="qwen-max",
        )

        print(f"[Planner] 初始Prompt: {prompt}")
        return prompt


if __name__ == "__main__":
    print("[Planner] 模块加载成功，需传入llm_client才能运行。")
