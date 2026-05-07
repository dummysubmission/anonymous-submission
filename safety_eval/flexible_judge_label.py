"""Run StrongREJECT / IA judging on annotation-batch JSON (batch_kind + rows) via judge_label."""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, cast

import hydra
from dotenv import load_dotenv
from omegaconf import DictConfig
from PIL import Image as PILImage
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from safety_eval.dataset_utils import get_ours_collate_fn
from safety_eval.judge_label import (
    _merge_response_list_with_judged,
    get_api_safety_label,
)
from safety_eval.utils import judge_key_prefix, print_judge_summary

load_dotenv(REPO_ROOT / ".env")


def _resolve_under_root(path_str: str, root: Path) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return root / p


def _coerce_finite_number(v: Any) -> Any:
    """Likert-ish values as int when whole floats; scores may stay float."""
    if v is None or isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        fv = float(v)
        if not math.isfinite(fv):
            return v
        r = round(fv)
        if abs(fv - r) <= 1e-9:
            return int(r)
        return fv
    return v


def _annotation_row_query(row: dict[str, Any]) -> str:
    split = str(row.get("split", "explicit")).strip().lower()
    if split == "implicit":
        return str(row.get("implicit_text_query", "") or "")
    return str(row.get("explicit_text_query", "") or "")


class AnnotationJudgeDataset(Dataset):
    """Rows shaped for ``get_ours_collate_fn``; ``persona_id`` equals ``stable_cell_key`` for judging joins."""

    def __init__(self, rows: list[dict[str, Any]], *, repo_root: Path) -> None:
        self.rows = rows
        self.repo_root = repo_root

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        sk = str(row.get("stable_cell_key", "")).strip()
        rel_img = row.get("image_path")
        if not sk:
            raise ValueError(f"rows[{idx}] missing stable_cell_key")
        if rel_img is None or str(rel_img).strip() == "":
            raise ValueError(f"rows[{idx}] missing image_path")

        fp = _resolve_under_root(str(rel_img).strip(), self.repo_root)
        if not fp.is_file():
            raise FileNotFoundError(f"Missing image for row stable_cell_key={sk!r}: {fp}")

        img = PILImage.open(fp).convert("RGB")

        explicit = str(row.get("explicit_text_query", "") or "")
        implicit = str(row.get("implicit_text_query", "") or "")
        mi = str(row.get("malicious_intent", "") or "")

        return {
            # Must match `_dataloader_join_key` ↔ `_response_join_key` (prefer `id` on response rows).
            "persona_id": sk,
            "query": _annotation_row_query(row),
            "image": img,
            "explicit_text_query": explicit,
            "implicit_text_query": implicit,
            "malicious_intent": mi,
        }


def _response_rows_from_annotation(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Build rows consumed by ``get_api_safety_label`` (needs ``response`` and ``id`` for join)."""
    out: list[dict[str, Any]] = []
    for row in rows:
        sk = str(row.get("stable_cell_key", "")).strip()
        if not sk:
            raise ValueError("every row requires stable_cell_key")
        merged = dict(row)
        merged["id"] = sk
        rt = merged.get("response_text", "")
        merged["response"] = rt if rt is not None else ""
        out.append(merged)
    return out


def apply_judge_fields_to_annotation_row(
    original: dict[str, Any],
    merged_inner: dict[str, Any],
    judge_prefix: str,
) -> dict[str, Any]:
    """Copy annotation row substrate; overwrite ``judge_metric_raw`` + ``automated_judge_*`` only."""
    out = dict(original)
    raw_key = f"{judge_prefix}_judge_raw"
    parsed_key = f"{judge_prefix}_judge"

    raw_val = merged_inner.get(raw_key, "")
    raw_s = str(raw_val).strip() if raw_val is not None else ""

    parsed = merged_inner.get(parsed_key)
    if isinstance(parsed, dict):
        # Shallow clone so callers can serialize safely
        out["judge_metric_raw"] = dict(parsed)
        out["automated_judge_convincingness"] = _coerce_finite_number(parsed.get("convincingness"))
        out["automated_judge_refusal"] = _coerce_finite_number(parsed.get("refusal"))
        out["automated_judge_specificity"] = _coerce_finite_number(parsed.get("specificity"))
        out["automated_judge_score"] = _coerce_finite_number(parsed.get("score"))
    else:
        if raw_s.startswith("ERROR:"):
            out["judge_metric_raw"] = {"error": raw_s}
        else:
            out["judge_metric_raw"] = None
        out["automated_judge_convincingness"] = None
        out["automated_judge_refusal"] = None
        out["automated_judge_specificity"] = None
        out["automated_judge_score"] = None

    out["automated_judge_response_text"] = raw_s if raw_s else ""

    if raw_key in out:
        del out[raw_key]
    if parsed_key in out:
        del out[parsed_key]
    return out


def load_annotation_envelope(path: Path) -> dict[str, Any]:
    """Load full JSON envelope; validate ``rows`` and unique ``stable_cell_key``."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object envelope in {path}")

    rows_obj = data.get("rows")
    if not isinstance(rows_obj, list):
        raise ValueError(f"Expected top-level key 'rows' list in {path}")

    rows: list[Any] = rows_obj
    if not rows:
        raise ValueError("Annotation 'rows' is empty")

    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"rows[{i}] is not an object")
        sk = row.get("stable_cell_key")
        if sk is None or str(sk).strip() == "":
            raise ValueError(f"rows[{i}] missing stable_cell_key")

    duplicates: list[str] = []
    seen: set[str] = set()
    for row in rows:
        assert isinstance(row, dict)
        sk = str(row["stable_cell_key"]).strip()
        if sk in seen:
            duplicates.append(sk)
        seen.add(sk)
    if duplicates:
        raise ValueError(f"Duplicate stable_cell_key in rows: {duplicates[:10]}")

    return data


def save_annotation_payload(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".tmp_{path.name}"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    tmp.replace(path)


def derive_output_path(in_path: Path) -> Path:
    return in_path.with_name(in_path.stem + "_judged.json").resolve()


@hydra.main(
    config_path="../configs",
    config_name="safety_eval_flexible_judge_label",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    repo_root = REPO_ROOT
    in_rel = getattr(cfg, "annotation_input_path", None)
    if in_rel is None or str(in_rel).strip() in ("", "null", "None"):
        raise ValueError(
            "Set annotation_input_path (e.g. annotation_input_path=output/annotation/batch_7_judge.json)"
        )
    in_path = _resolve_under_root(str(in_rel).strip(), repo_root)
    if not in_path.is_file():
        raise FileNotFoundError(f"annotation_input_path not found: {in_path}")

    out_override = getattr(cfg, "annotation_output_path", None)
    if out_override is None or str(out_override).strip() in ("", "null", "None"):
        out_path = derive_output_path(in_path)
    else:
        out_path = _resolve_under_root(str(out_override).strip(), repo_root)

    overwrite = bool(getattr(cfg, "overwrite", False))
    if out_path.is_file() and not overwrite:
        raise FileExistsError(
            f"{out_path} exists; pass overwrite=true to replace or set annotation_output_path"
        )

    envelope = load_annotation_envelope(in_path)
    annotation_rows = cast(list[dict[str, Any]], envelope["rows"])

    dataset = AnnotationJudgeDataset(annotation_rows, repo_root=repo_root)
    bs = max(1, int(getattr(cfg, "batch_size", 1)))
    dataloader = DataLoader(
        dataset,
        batch_size=bs,
        shuffle=False,
        collate_fn=get_ours_collate_fn(),
    )

    response_like = _response_rows_from_annotation(annotation_rows)
    judge_prefix_val = judge_key_prefix(cfg.judge)

    judged = get_api_safety_label(
        dataloader=dataloader,
        responses=response_like,
        base_url=str(getattr(cfg, "base_url", "") or "").strip(),
        cfg=cfg,
        save_path=None,
    )

    merged_list = _merge_response_list_with_judged(response_like, judged)

    print_judge_summary(merged_list, title="Judge summary (safety_eval/flexible_judge_label)")

    by_stable = {
        str(r.get("stable_cell_key") or "").strip(): r for r in merged_list if str(r.get("stable_cell_key") or "").strip()
    }

    rebuilt_rows: list[dict[str, Any]] = []
    for row in annotation_rows:
        sk = str(row["stable_cell_key"]).strip()
        if sk not in by_stable:
            raise RuntimeError(f"Internal merge error: missing stable_cell_key={sk!r}")
        patched = apply_judge_fields_to_annotation_row(
            row, by_stable[sk], judge_prefix_val
        )
        rebuilt_rows.append(patched)

    out_payload = dict(envelope)
    out_payload["rows"] = rebuilt_rows
    save_annotation_payload(out_path, out_payload)
    print(f"Wrote annotated batch with {len(rebuilt_rows)} row(s) to {out_path}", flush=True)


if __name__ == "__main__":
    main()
