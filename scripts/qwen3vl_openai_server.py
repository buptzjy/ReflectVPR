#!/usr/bin/env python3
"""Minimal OpenAI-compatible server for a local Qwen3-VL checkpoint."""

import argparse
import base64
import io
import time
import uuid

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


class ChatRequest(BaseModel):
    model: str
    messages: list[dict]
    temperature: float = 0.2
    max_tokens: int | None = None
    response_format: dict | None = None


def decode_image(url: str) -> Image.Image:
    if not url.startswith("data:image/") or "," not in url:
        raise ValueError("Only data:image/...;base64 URLs are supported")
    encoded = url.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGB")


def normalized_messages(messages: list[dict]) -> list[dict]:
    result = []
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            result.append({"role": message["role"], "content": content})
            continue
        parts = []
        for part in content:
            kind = part.get("type")
            if kind == "text":
                parts.append({"type": "text", "text": part.get("text", "")})
            elif kind == "image_url":
                image_url = part.get("image_url", {})
                url = image_url.get("url", "") if isinstance(image_url, dict) else str(image_url)
                parts.append({"type": "image", "image": decode_image(url)})
        result.append({"role": message["role"], "content": parts})
    return result


def create_app(model_path: str, served_model_name: str, device: str) -> FastAPI:
    app = FastAPI()
    print(f"Loading Qwen3-VL from {model_path} on {device}", flush=True)
    processor = AutoProcessor.from_pretrained(model_path)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map=device,
    ).eval()
    print("Qwen3-VL ready", flush=True)

    @app.get("/health")
    def health():
        return {"status": "ok", "model_loaded": True}

    @app.get("/v1/models")
    def models():
        return {
            "object": "list",
            "data": [{"id": served_model_name, "object": "model", "owned_by": "local"}],
        }

    @app.post("/v1/chat/completions")
    def chat(request: ChatRequest):
        if request.model != served_model_name:
            raise HTTPException(status_code=404, detail=f"Unknown model: {request.model}")
        try:
            messages = normalized_messages(request.messages)
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(model.device)
            with torch.inference_mode():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=request.max_tokens or 768,
                    do_sample=request.temperature > 0,
                    temperature=max(request.temperature, 1e-5),
                )
            trimmed = output_ids[:, inputs.input_ids.shape[1]:]
            text = processor.batch_decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"{type(exc).__name__}: {exc}") from exc

        created = int(time.time())
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": created,
            "model": served_model_name,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }

    return app


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--served-model-name", default="qwen3-vl-4b-instruct-remote")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=23002)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    uvicorn.run(
        create_app(args.model, args.served_model_name, args.device),
        host=args.host,
        port=args.port,
        log_level="info",
    )


if __name__ == "__main__":
    main()
