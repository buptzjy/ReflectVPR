#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import io
import json
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


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


class QwenServer:
    def __init__(self, model_path: str, served_model_name: str, device: str):
        self.served_model_name = served_model_name
        print(f"Loading Qwen3-VL from {model_path} on {device}", flush=True)
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            device_map=device,
        ).eval()
        self.lock = threading.Lock()
        print("Qwen3-VL ready", flush=True)

    def health(self) -> dict:
        return {"status": "ok", "model_loaded": True}

    def models(self) -> dict:
        return {
            "object": "list",
            "data": [{"id": self.served_model_name, "object": "model", "owned_by": "local"}],
        }

    def chat(self, payload: dict) -> dict:
        model_name = payload.get("model")
        if model_name != self.served_model_name:
            raise ValueError(f"Unknown model: {model_name}")
        messages = normalized_messages(payload.get("messages", []))
        temperature = float(payload.get("temperature", 0.2) or 0.2)
        max_tokens = int(payload.get("max_tokens") or 768)
        with self.lock:
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device)
            with torch.inference_mode():
                output_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=temperature > 0,
                    temperature=max(temperature, 1e-5),
                )
            trimmed = output_ids[:, inputs.input_ids.shape[1]:]
            text = self.processor.batch_decode(
                trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
        created = int(time.time())
        return {
            "id": f"chatcmpl-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": created,
            "model": self.served_model_name,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        }


def build_handler(server_state: QwenServer):
    class Handler(BaseHTTPRequestHandler):
        def _send_json(self, payload: dict, status: int = 200) -> None:
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, fmt: str, *args) -> None:
            print(f"[{self.log_date_time_string()}] {fmt % args}", flush=True)

        def do_GET(self) -> None:
            if self.path == "/health":
                self._send_json(server_state.health())
                return
            if self.path == "/v1/models":
                self._send_json(server_state.models())
                return
            self._send_json({"error": f"unknown path: {self.path}"}, status=HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            if self.path != "/v1/chat/completions":
                self._send_json({"error": f"unknown path: {self.path}"}, status=HTTPStatus.NOT_FOUND)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                response = server_state.chat(payload)
                self._send_json(response)
            except Exception as exc:
                self._send_json(
                    {"error": {"message": f"{type(exc).__name__}: {exc}", "type": "server_error"}},
                    status=HTTPStatus.INTERNAL_SERVER_ERROR,
                )

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--served-model-name", default="qwen3-vl-4b-instruct-remote")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=23002)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    state = QwenServer(args.model, args.served_model_name, args.device)
    server = ThreadingHTTPServer((args.host, args.port), build_handler(state))
    print(f"Serving on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
