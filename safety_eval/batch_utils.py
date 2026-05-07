"""Shared path helpers for safety_eval offline inference runners."""

from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from safety_eval.utils import (
    judge_key_prefix,
    normalize_judge_rubric,
    output_dir_model_key,
    sanitize_model_key,
)


# ---------------------------------------------------------------------------
# Save-path resolution
# ---------------------------------------------------------------------------

def _dataset_use_adversarial_images(cfg: DictConfig) -> bool:
    d = OmegaConf.select(cfg, "dataset")
    if d is None:
        return False
    v = OmegaConf.select(d, "use_adversarial_images")
    return bool(v) if v is not None else False


def _output_suffix_for_adv_images(cfg: DictConfig) -> str:
    return "_adversarial" if _dataset_use_adversarial_images(cfg) else ""


def get_responses_save_path(cfg: DictConfig, repo_root: Path) -> Path:
    """Compute the canonical path for the generated-responses JSON file."""
    raw = getattr(cfg, "save_path", None)
    if raw not in (None, "", "null", "None"):
        return Path(str(raw))
    dataset_split = str(cfg.dataset.dataset_split)
    model_key = str(output_dir_model_key(cfg))
    fname = (
        "responses_adversarial.json"
        if _dataset_use_adversarial_images(cfg)
        else "responses.json"
    )
    return repo_root / "output" / "safety_eval" / dataset_split / model_key / fname


def get_guard_save_path(cfg: DictConfig, repo_root: Path) -> Path:
    """Compute the canonical path for the guard-label JSON file."""
    raw = getattr(cfg, "save_path", None)
    if raw not in (None, "", "null", "None"):
        return Path(str(raw))
    dataset_split = str(cfg.dataset.dataset_split)
    model_key = str(output_dir_model_key(cfg))
    guard_key = sanitize_model_key(str(cfg.guard_model.short_name))
    suf = _output_suffix_for_adv_images(cfg)
    return (
        repo_root
        / "output"
        / "safety_eval"
        / dataset_split
        / model_key
        / f"guard_{guard_key}{suf}.json"
    )


def get_judge_save_path(cfg: DictConfig, repo_root: Path) -> Path:
    """Compute the canonical path for the judge-label JSON file."""
    raw = getattr(cfg, "save_path", None)
    if raw not in (None, "", "null", "None"):
        return Path(str(raw))
    dataset_split = str(cfg.dataset.dataset_split)
    model_key = str(output_dir_model_key(cfg))
    judge_key = judge_key_prefix(cfg.judge)
    rubric = normalize_judge_rubric(getattr(cfg.judge, "rubric", "strongreject"))
    suf = _output_suffix_for_adv_images(cfg)
    return (
        repo_root
        / "output"
        / "safety_eval"
        / dataset_split
        / model_key
        / f"judge_{judge_key}_{rubric}{suf}.json"
    )


def get_response_path(cfg: DictConfig, repo_root: Path) -> Path:
    """Resolve the path to previously generated responses.

    Falls back to the canonical responses save path derived from
    ``target_model_name`` / model ``short_name`` and ``dataset`` if
    ``response_path`` is not explicitly set.
    """
    raw = getattr(cfg, "response_path", None)
    if raw not in (None, "", "null", "None"):
        return Path(str(raw))
    return get_responses_save_path(cfg, repo_root)


