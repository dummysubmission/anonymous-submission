#!/usr/bin/env python3
"""Launch a simple vLLM server with at most one image and no video input."""

import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path

import hydra
from omegaconf import DictConfig


def _wait_for_health(
    host: str,
    port: int,
    *,
    max_wait_s: int,
    proc: subprocess.Popen,
    log_path: Path,
) -> None:
    health_url = f"http://{host}:{port}/health"
    poll_interval = 2.0
    waited = 0.0

    while waited < float(max_wait_s):
        if proc.poll() is not None:
            raise RuntimeError(
                f"vllm serve exited with code {proc.returncode} while waiting for health. "
                f"Check {log_path} for details."
            )
        try:
            urllib.request.urlopen(health_url, timeout=3)
            return
        except (urllib.error.URLError, ConnectionError, OSError):
            time.sleep(poll_interval)
            waited += poll_interval

    raise TimeoutError(
        f"vLLM health check did not pass within {max_wait_s}s: {health_url}"
    )


def _build_command(cfg: DictConfig) -> list[str]:
    cmd = [
        "vllm",
        "serve",
        str(cfg.model.model_name_or_path),
        "--host",
        str(cfg.server.host),
        "--port",
        str(int(cfg.server.port)),
        "--tensor-parallel-size",
        str(int(cfg.vllm.tensor_parallel_size)),
        "--trust-remote-code",
        "--limit-mm-per-prompt.image",
        str(int(cfg.vllm.limit_mm_per_prompt_image)),
        "--limit-mm-per-prompt.video",
        str(int(cfg.vllm.limit_mm_per_prompt_video)),
        "--gdn-prefill-backend",
        "triton",
    ]

    if cfg.model.short_name == "Qwen3.5-9B-Instruct":
        cmd += [
            "--enable-prefix-caching",
        ]

    max_model_len = cfg.vllm.max_model_len
    if max_model_len not in (None, "auto", "null", "None"):
        cmd += ["--max-model-len", str(int(max_model_len))]

    lora_path = getattr(cfg, "lora_path", None)
    if lora_path not in (None, "null", "None", ""):
        cmd += [
            "--enable-lora",
            "--lora-modules",
            f"ft_adapter={lora_path}",
            "--max-lora-rank",
            str(int(cfg.vllm.max_lora_rank)),
        ]

    return cmd


@hydra.main(
    config_path="configs", config_name="simple_vllm_server.yaml", version_base=None
)
def main(cfg: DictConfig) -> None:
    log_path = Path(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    ) / Path(str(cfg.log_path))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = _build_command(cfg)

    print(f"Starting vllm serve on {cfg.server.host}:{cfg.server.port} ...")
    print(f"  cmd: {' '.join(cmd)}")
    print(f"  log: {log_path}")
    if getattr(cfg, "lora_path", None) not in (None, "null", "None", ""):
        print("  served model name for API requests: ft_adapter")

    log_file = open(log_path, "w")
    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )

    def _shutdown(signum: int, _frame) -> None:
        print(f"Received signal {signum}; shutting down vllm serve ...")
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        _wait_for_health(
            str(cfg.server.host),
            int(cfg.server.port),
            max_wait_s=int(cfg.max_wait_seconds),
            proc=proc,
            log_path=log_path,
        )
        print("vllm serve is healthy. Blocking until exit ...")
        while True:
            code = proc.poll()
            if code is not None:
                raise RuntimeError(
                    f"vllm serve exited with code {code}. Check {log_path} for details."
                )
            time.sleep(5)
    finally:
        try:
            if proc.poll() is None:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                proc.wait(timeout=30)
        except ProcessLookupError:
            pass
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                proc.wait(timeout=10)
            except ProcessLookupError:
                pass
        log_file.close()
        print("vllm serve stopped.")


if __name__ == "__main__":
    main()
