import os
import io
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
from PIL import Image, ImageDraw

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


DEFAULT_START_SCRIPT = "/media/data/zhangjingyi/LightX2V/start_server.sh"
SERVICE_LOG_DIR = Path("/tmp/reflectvpr_services")


def _is_local_url(url: str) -> bool:
    host = urlparse(url).hostname
    return host in {"127.0.0.1", "localhost", "0.0.0.0"}


def _health_url(api_url: str) -> str:
    parsed = urlparse(api_url)
    return f"{parsed.scheme}://{parsed.netloc}/health"


def _service_ready(session: requests.Session, api_url: str) -> bool:
    try:
        resp = session.get(_health_url(api_url), timeout=2)
        if resp.status_code != 200:
            return False
        data = resp.json()
        return data.get("status") == "ok" and data.get("model_loaded") is True
    except requests.RequestException:
        return False
    except ValueError:
        return False


def _start_local_service(session: requests.Session, api_url: str) -> None:
    if not api_url or not _is_local_url(api_url):
        return
    if os.getenv("LIGHTX2V_AUTO_START", "1").lower() in {"0", "false", "no"}:
        return
    if _service_ready(session, api_url):
        return

    script = os.getenv("LIGHTX2V_START_SCRIPT", DEFAULT_START_SCRIPT)
    SERVICE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = SERVICE_LOG_DIR / "lightx2v.log"
    print(f"[Lightx2v] 未检测到服务，自动启动: {script}")
    log_file = open(log_path, "ab", buffering=0)
    subprocess.Popen(
        ["/bin/bash", script],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )

    deadline = time.time() + int(os.getenv("LIGHTX2V_START_TIMEOUT", "180"))
    while time.time() < deadline:
        if _service_ready(session, api_url):
            print(f"[Lightx2v] 服务已就绪: {api_url}")
            return
        time.sleep(2)
    print(f"[Lightx2v] 服务启动超时，日志: {log_path}")


def _mock_disabled() -> bool:
    return os.getenv("REFLECTVPR_DISABLE_MOCK", "0").lower() in {"1", "true", "yes"}


class Lightx2vGenerator:

    def __init__(self, api_url: str = None):
        self.api_url = os.getenv("LIGHTX2V_API_URL", "") if api_url is None else api_url
        self._session = requests.Session()
        self._session.trust_env = False
        if self.api_url:
            print(f"[Lightx2v] API模式: {self.api_url}")
            _start_local_service(self._session, self.api_url)
        else:
            if _mock_disabled():
                raise RuntimeError("[Lightx2v] REFLECTVPR_DISABLE_MOCK=1，但未配置 LIGHTX2V_API_URL")
            print("[Lightx2v] 无API配置，使用Mock模式")

    def _save_temp_image(self, image: Image.Image) -> str:
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=95)
        buf.seek(0)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
        tmp.write(buf.read())
        tmp.close()
        return tmp.name

    def _call_api(self, ref_image: Image.Image, prompt: str, **kwargs) -> Image.Image:
        if not self.api_url:
            if _mock_disabled():
                raise RuntimeError("[Lightx2v] REFLECTVPR_DISABLE_MOCK=1，禁止无API时回退Mock")
            return None

        temp_path = None
        try:
            temp_path = self._save_temp_image(ref_image)
            payload = {
                "image_path": temp_path,
                "prompt": prompt,
                "negative_prompt": kwargs.get("negative_prompt", ""),
                "seed": kwargs.get("seed", 42),
                "infer_steps": kwargs.get("infer_steps", 4),
                "guidance_scale": kwargs.get("guidance_scale", 1.0),
            }
            last_error = None
            max_attempts = int(os.getenv("LIGHTX2V_API_RETRIES", "2"))
            for attempt in range(1, max_attempts + 1):
                try:
                    resp = self._session.post(self.api_url, json=payload, timeout=300)
                    resp.raise_for_status()
                    break
                except requests.RequestException as exc:
                    last_error = exc
                    status = getattr(getattr(exc, "response", None), "status_code", None)
                    body = ""
                    response = getattr(exc, "response", None)
                    if response is not None:
                        body = (response.text or "")[:500]
                    if attempt >= max_attempts:
                        raise RuntimeError(
                            f"[Lightx2v] API调用失败 attempts={max_attempts} "
                            f"status={status} body={body!r}: {exc}"
                        ) from exc
                    print(
                        f"[Lightx2v] API调用失败，重试 {attempt}/{max_attempts}: "
                        f"status={status} error={exc}",
                        flush=True,
                    )
                    time.sleep(2)
            else:
                raise RuntimeError(f"[Lightx2v] API调用失败: {last_error}")
            data = resp.json()
            result_path = data.get("result_path")
            if not result_path or not os.path.exists(result_path):
                raise RuntimeError(f"API返回的result_path不存在: {result_path}")
            result = Image.open(result_path).convert("RGB")
            if result.size != ref_image.size:
                result = result.resize(ref_image.size, Image.Resampling.LANCZOS)
            return result
        except (requests.RequestException, RuntimeError) as e:
            if _mock_disabled():
                raise RuntimeError(f"[Lightx2v] API调用失败，严格模式禁止回退Mock: {e}") from e
            print(f"[Lightx2v] API调用失败 ({e})")
            return None
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)

    def generate_local(
        self,
        ref_image: Image.Image,
        prompt: str,
        mask: Image.Image = None,
        seed: int = 42,
        negative_prompt: str = "",
        **kwargs,
    ) -> Image.Image:
        result = self._call_api(ref_image, prompt, seed=seed, negative_prompt=negative_prompt, **kwargs)
        if result is not None:
            return result
        if _mock_disabled():
            raise RuntimeError("[Lightx2v] 严格模式下 API 未返回结果，禁止回退Mock Local")
        return self._mock_local(ref_image, prompt, mask)

    def generate_dual(
        self,
        ref_image: Image.Image,
        prompt: str,
        seed: int = 42,
        negative_prompt: str = "",
        **kwargs,
    ) -> Image.Image:
        result = self._call_api(ref_image, prompt, seed=seed, negative_prompt=negative_prompt, **kwargs)
        if result is not None:
            return result
        if _mock_disabled():
            raise RuntimeError("[Lightx2v] 严格模式下 API 未返回结果，禁止回退Mock Dual")
        return self._mock_dual(ref_image, prompt)

    def _mock_local(
        self,
        ref_image: Image.Image,
        prompt: str,
        mask: Image.Image = None,
    ) -> Image.Image:
        print(f"[Lightx2v][Mock Local] prompt='{prompt[:50]}...'")
        result = ref_image.copy()
        draw = ImageDraw.Draw(result)
        w, h = result.size
        draw.rectangle(
            [w // 3, h // 3, w * 2 // 3, h * 2 // 3],
            fill=(30, 30, 30)
        )
        return result

    def _mock_dual(self, ref_image: Image.Image, prompt: str) -> Image.Image:
        print(f"[Lightx2v][Mock Dual] prompt='{prompt[:50]}...'")
        import numpy as np
        img_array = np.array(ref_image).astype(np.float32)
        img_array[:, :, 2] = np.clip(img_array[:, :, 2] * 1.1, 0, 255)
        img_array = np.clip(img_array * 0.8, 0, 255).astype(np.uint8)
        result = Image.fromarray(img_array)
        draw = ImageDraw.Draw(result)
        w, h = result.size
        draw.rectangle([w // 4, h // 4, w * 3 // 4, h // 2], fill=(20, 20, 20))
        return result


if __name__ == "__main__":
    gen = Lightx2vGenerator()
    img = Image.new("RGB", (512, 512), color=(150, 150, 150))

    local_out = gen.generate_local(img, "A large truck partially blocking the road")
    local_out.save("/tmp/lightx2v_local_test.jpg")
    print("[Lightx2v] Local输出保存至 /tmp/lightx2v_local_test.jpg")

    dual_out = gen.generate_dual(img, "Heavy rain with a truck blocking the road, wet road surface")
    dual_out.save("/tmp/lightx2v_dual_test.jpg")
    print("[Lightx2v] Dual输出保存至 /tmp/lightx2v_dual_test.jpg")
