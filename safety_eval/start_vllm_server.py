import os
import signal
import subprocess
import time
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf


def _wait_for_health(
    host: str,
    port: int,
    *,
    max_wait_s: int,
    proc: subprocess.Popen,
    log_path: Path,
) -> None:
    import urllib.error
    import urllib.request

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


def _stage_model_path(cfg: DictConfig) -> str:
    stage = str(cfg.stage)
    if stage == "model":
        m = OmegaConf.select(cfg, "model")
        if m is None:
            raise ValueError("stage=model requires cfg.model in the Hydra config.")
        return str(m.model_name_or_path)
    if stage == "guard":
        return str(cfg.guard_model.model_name_or_path)
    if stage == "judge":
        j = cfg.judge
        if str(getattr(j, "provider", "") or "").lower() == "portkey":
            raise ValueError(
                "Judge provider is portkey (API-only). No local vLLM server is needed for this stage."
            )
        mpath = getattr(j, "model_name_or_path", None)
        if mpath in (None, "", "null", "None"):
            raise ValueError("judge.model_name_or_path must be set for local vLLM judge stage.")
        return str(mpath)
    raise ValueError(f"Invalid stage: {stage} (expected: model|guard|judge)")


def _stage_max_model_len(cfg: DictConfig) -> int | None:
    stage = str(cfg.stage)
    if stage == "model":
        v = cfg.vllm.max_model_len
        if v in (None, "auto", "null", "None"):
            return None
        return int(v)
    if stage == "guard":
        v = getattr(cfg.guard_model, "max_model_len", None)
        if v in (None, "auto", "null", "None"):
            return None
        return int(v)
    if stage == "judge":
        v = getattr(cfg.vllm, "judge_max_model_len", 16384)
        if v in (None, "auto", "null", "None"):
            return None
        return int(v)
    return None


def _stage_extra_args(cfg: DictConfig) -> list[str]:
    common = [
        "--limit-mm-per-prompt.image",
        str(int(cfg.vllm.limit_mm_per_prompt_image)),
        "--limit-mm-per-prompt.video",
        str(int(cfg.vllm.limit_mm_per_prompt_video)),
    ]
    stage = str(cfg.stage)
    if stage == "model":
        lora_path = getattr(cfg, "lora_path", None)
        if lora_path is not None and str(lora_path) not in ("null", "None", ""):
            common += [
                "--enable-lora",
                "--lora-modules",
                f"ft_adapter={lora_path}",
                "--max-lora-rank",
                "64",
            ]
    return common


@hydra.main(
    config_path="../configs", config_name="start_vllm_server.yaml", version_base=None
)
def main(cfg: DictConfig) -> None:
    model_path = _stage_model_path(cfg)
    host = str(cfg.server.host)
    port = int(cfg.server.port)
    tp = int(cfg.vllm.tensor_parallel_size)
    max_model_len = _stage_max_model_len(cfg)
    extra_args = _stage_extra_args(cfg)

    log_path = Path(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    ) / Path(str(cfg.log_path))
    log_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "vllm",
        "serve",
        model_path,
        "--host",
        host,
        "--port",
        str(port),
        "--tensor-parallel-size",
        str(tp),
        "--trust-remote-code",
    ]
    if max_model_len is not None:
        cmd += ["--max-model-len", str(max_model_len)]
    cmd += extra_args

    print(f"Starting vllm serve stage={cfg.stage} on {host}:{port} …")
    print(f"  cmd: {' '.join(cmd)}")
    print(f"  log: {log_path}")

    log_file = open(log_path, "w")
    proc = subprocess.Popen(
        cmd,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
    )

    def _shutdown(signum: int, _frame) -> None:
        print(f"Received signal {signum}; shutting down vllm serve …")
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except ProcessLookupError:
                pass

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        # If the process crashes during startup, fail fast with a helpful hint.
        _wait_for_health(
            host,
            port,
            max_wait_s=int(cfg.max_wait_seconds),
            proc=proc,
            log_path=log_path,
        )
        print("vllm serve is healthy. Blocking until exit …")
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
