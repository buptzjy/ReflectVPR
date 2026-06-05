import os
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def build_llm_client():
    api_key = os.getenv("OPENAI_API_KEY", "")
    api_base = os.getenv("OPENAI_API_BASE", "https://api.yunwu.ai/v1")
    model_name = os.getenv("OPENAI_MODEL_NAME", "qwen3.5-35b-a3b")

    if api_key and api_key != "sk-xxx":
        try:
            from openai import OpenAI
        except ImportError:
            print("[LLM] 检测到 API_KEY 但未安装 openai 包，使用 Mock 客户端")
            return _MockLLMClient()
        client = OpenAI(api_key=api_key, base_url=api_base)
        print(f"[LLM] 真实客户端: base_url={api_base}, model={model_name}")
        return _RealLLMClient(client, model_name)

    print("[LLM] 未检测到有效 API_KEY，使用 Mock 客户端")
    return _MockLLMClient()


class _RealLLMClient:
    def __init__(self, client, model_name: str):
        self._client = client
        self._model = model_name

    def chat(self, system="", user="", json_mode=False, temperature=0.7, model=None):
        kwargs = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        resp = self._client.chat.completions.create(
            model=model or self._model,
            messages=messages,
            temperature=temperature,
            **kwargs,
        )
        return resp.choices[0].message.content

    def chat_with_images(self, system="", user="", images=None, json_mode=False, temperature=0.3, model=None):
        kwargs = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        content = [{"type": "text", "text": user}]
        if images:
            for b64 in images:
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}})
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": content})
        resp = self._client.chat.completions.create(
            model=model or self._model,
            messages=messages,
            temperature=temperature,
            **kwargs,
        )
        return resp.choices[0].message.content


class _MockLLMClient:
    def chat(self, system="", user="", json_mode=False, temperature=0.7, model=None):
        if json_mode or model == "qwen-vl-max":
            return json.dumps({
                "scene_summary": "rainy night road with truck occlusion",
                "weather": "rain",
                "time_of_day": "night",
                "occlusion": "true",
            })
        return "Heavy rain at night, wet reflective road surface, preserve lane markings and road structure, realistic atmospheric lighting, high detail"

    def chat_with_images(self, system="", user="", images=None, json_mode=False, temperature=0.3, model=None):
        return self.chat(system=system, user=user, json_mode=json_mode, temperature=temperature, model=model)
