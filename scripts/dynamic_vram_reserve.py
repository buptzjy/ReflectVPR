#!/usr/bin/env python3
"""Keep selected GPUs above a low VRAM watermark and below a high watermark.

The controller itself does not import torch. It launches one lightweight worker
per reserved chunk, so releasing placeholder memory is just terminating workers.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from collections import defaultdict


STOP = False


def request_stop(_signum: int, _frame: object) -> None:
    global STOP
    STOP = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="1,2,3", help="Physical GPU ids, comma-separated.")
    parser.add_argument("--low-gb", type=float, default=31.0, help="Fill up to this total used VRAM.")
    parser.add_argument("--high-gb", type=float, default=35.0, help="Release placeholder above this total used VRAM.")
    parser.add_argument("--chunk-mb", type=int, default=256, help="Allocation/release granularity.")
    parser.add_argument("--max-new-chunks", type=int, default=2, help="Max workers to start per GPU per loop.")
    parser.add_argument("--max-own-gb", type=float, default=8.0, help="Hard cap for this script's own placeholder per GPU.")
    parser.add_argument("--interval", type=float, default=5.0, help="Polling interval in seconds.")
    parser.add_argument("--i-understand-oom-risk", action="store_true", help="Required to actually reserve VRAM.")
    return parser.parse_args()


def query_used_mib() -> dict[int, int]:
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    used: dict[int, int] = {}
    for line in out.splitlines():
        if not line.strip():
            continue
        idx_s, used_s = [part.strip() for part in line.split(",", 1)]
        used[int(idx_s)] = int(used_s)
    return used


def allocate_chunk(gpu: int, chunk_mb: int) -> subprocess.Popen:
    code = (
        "import time, torch\n"
        f"gpu={gpu}\n"
        f"chunk_mb={chunk_mb}\n"
        "torch.cuda.set_device(gpu)\n"
        "x=torch.empty((chunk_mb*1024*1024,), dtype=torch.uint8, device=f'cuda:{gpu}')\n"
        "x.fill_(1)\n"
        "torch.cuda.synchronize(gpu)\n"
        "print(f'[worker] gpu={gpu} reserved={chunk_mb}MiB', flush=True)\n"
        "time.sleep(10**9)\n"
    )
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in torch_device_count_hint())
    return subprocess.Popen(
        [sys.executable, "-u", "-c", code],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
        start_new_session=True,
    )


def torch_device_count_hint() -> list[int]:
    try:
        return sorted(query_used_mib())
    except Exception:
        return list(range(8))


def terminate_worker(proc: subprocess.Popen) -> None:
    if proc.poll() is not None:
        return
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return


def release_chunks(chunks: list[subprocess.Popen], count: int) -> None:
    for _ in range(min(count, len(chunks))):
        terminate_worker(chunks.pop())


def reap_dead(chunks: dict[int, list[subprocess.Popen]]) -> None:
    for gpu, procs in chunks.items():
        chunks[gpu] = [proc for proc in procs if proc.poll() is None]


def main() -> int:
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGHUP, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    args = parse_args()
    if not args.i_understand_oom_risk:
        raise SystemExit(
            "Refusing to reserve VRAM without --i-understand-oom-risk. "
            "This placeholder strategy has caused OOM before."
        )
    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    low_mib = int(args.low_gb * 1024)
    high_mib = int(args.high_gb * 1024)
    max_own_mib = int(args.max_own_gb * 1024)
    chunk_mb = args.chunk_mb

    if high_mib <= low_mib:
        raise SystemExit("--high-gb must be larger than --low-gb")

    chunks: dict[int, list[subprocess.Popen]] = defaultdict(list)
    print(
        f"[reserve] dynamic mode gpus={gpus} low={args.low_gb:.1f}G "
        f"high={args.high_gb:.1f}G chunk={chunk_mb}MiB interval={args.interval}s",
        flush=True,
    )

    while not STOP:
        try:
            reap_dead(chunks)
            used_by_gpu = query_used_mib()
            for gpu in gpus:
                used = used_by_gpu.get(gpu)
                if used is None:
                    print(f"[reserve] gpu={gpu} not found", flush=True)
                    continue

                own_mib = len(chunks[gpu]) * chunk_mb
                action = "hold"

                effective_used = used + own_mib

                if own_mib > max_own_mib:
                    release_count = max(1, (own_mib - max_own_mib + chunk_mb - 1) // chunk_mb)
                    release_chunks(chunks[gpu], release_count)
                    action = f"safety_release {release_count * chunk_mb}MiB"
                if used > high_mib and chunks[gpu]:
                    release_mib = min(own_mib, used - low_mib)
                    release_count = max(1, (release_mib + chunk_mb - 1) // chunk_mb)
                    release_chunks(chunks[gpu], release_count)
                    action = f"release {release_count * chunk_mb}MiB"
                elif effective_used < low_mib and own_mib < max_own_mib:
                    need_mib = low_mib - effective_used
                    need_mib = min(need_mib, max_own_mib - own_mib)
                    alloc_count = max(1, (need_mib + chunk_mb - 1) // chunk_mb)
                    alloc_count = min(alloc_count, args.max_new_chunks)
                    ok = 0
                    for _ in range(alloc_count):
                        chunks[gpu].append(allocate_chunk(gpu, chunk_mb))
                        ok += 1
                    action = f"alloc {ok * chunk_mb}MiB"

                print(
                    f"[reserve] gpu={gpu} used={used}MiB own={own_mib}MiB "
                    f"effective={effective_used}MiB chunks={len(chunks[gpu])} action={action}",
                    flush=True,
                )

            time.sleep(args.interval)
        except Exception as exc:
            print(f"[reserve] error: {exc}", file=sys.stderr, flush=True)
            time.sleep(args.interval)

    for procs in chunks.values():
        release_chunks(procs, len(procs))
    chunks.clear()
    print("[reserve] stopped", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
