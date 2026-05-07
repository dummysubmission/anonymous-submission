"""GCG-style multimodal attack: optimize image pixels to elicit ``response_prefix``.

**Gemma4:** Leaf tensor matches ``Gemma4ImageProcessor`` after resize + rescale (~``[0, 1]``). Forwards use
``images_kwargs`` that skip resize/rescale/normalize before patchify.

**Qwen 3.5 VL:** Leaf tensor matches ``Qwen2VLImageProcessor`` after ``smart_resize`` + resize + rescale only
(~``[0, 1]``). Each forward passes the leaf through ``Qwen3VLProcessor`` with resize/rescale skipped and
CLIP normalization enabled inside the library implementation, then patchify.
"""

from __future__ import annotations

import inspect
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from tqdm import tqdm

import hydra
import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Number of assistant tokens at the end of ``completion_len`` excluded from CE (masked with -100 in ``labels``).
DEFAULT_COMPLETION_LOSS_TAIL_IGNORE = 2

from adversarial.dataset_utils import (  # noqa: E402
    get_ours_collate_fn,
    load_ours_dataset,
)
from adversarial.gemma4_vision_pipeline import (  # noqa: E402
    GEMMA4_PRECOMPUTED_VISION_KWARGS,
    chw_preprocessed_to_uint8_chw,
    gemma4_chw_preprocessed_from_pil,
)
from adversarial.qwen3_vl_pixel_pipeline import (  # noqa: E402
    QWEN3_VL_ATTACK_MERGE_IMAGES_KWARGS,
    qwen3_vl_rescaled_spatial_from_pil,
    qwen_slice_visual_for_batch_row,
    rescaled_spatial_to_uint8_chw,
)
from safety_eval.utils import sanitize_model_dir_name  # noqa: E402

PixelAttackFamily = Literal["gemma4", "qwen3_vl"]


def _prepare_user_messages(query: str, image: Image.Image) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": query},
            ],
        }
    ]


def _attack_optimizer_kind(attack_cfg: Any) -> Literal["adam", "sgd"]:
    """Resolve ``attack.optimizer``: YAML ``null`` becomes Python ``None``, which still means **manual SGD** here.

    Only ``adam`` uses ``torch.optim.Adam``; everything else (including ``null``, ``sgd``, omitted default) runs
    ``image_rgb.add_(-lr * image_rgb.grad)`` each step.
    """

    raw = getattr(attack_cfg, "optimizer", "sgd")
    if raw is None:
        return "sgd"
    s = str(raw).strip().lower()
    if s in ("", "none", "null", "sgd", "manual"):
        return "sgd"
    if s == "adam":
        return "adam"
    raise ValueError(
        f"attack.optimizer={raw!r}: use 'adam', 'sgd' (or null / omit for manual SGD)."
    )


def _dtype_from_cfg(v: str) -> Any:
    v = (v or "auto").lower()
    if v == "auto":
        return "auto"
    if v in ("bf16", "bfloat16"):
        return torch.bfloat16
    if v in ("fp16", "float16"):
        return torch.float16
    if v in ("fp32", "float32"):
        return torch.float32
    raise ValueError(f"Unsupported torch_dtype: {v}")


def _load_vlm_model(model_name_or_path: str, torch_dtype: Any, device: torch.device):
    try:
        from transformers import AutoModelForImageTextToText  # type: ignore

        return AutoModelForImageTextToText.from_pretrained(
            model_name_or_path,
            dtype=torch_dtype,
            device_map="auto",
        )
    except Exception:
        from transformers import AutoModelForCausalLM  # type: ignore

        return AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            dtype=torch_dtype,
            device_map="auto",
        )


def _build_prompt_messages(
    *,
    query: str,
    image: Image.Image,
    system_prompt: str | None,
) -> list[dict[str, Any]]:
    """User (+ optional system) only; used with ``add_generation_prompt=True`` for attack prompts."""
    user_msgs = _prepare_user_messages(query, image)
    messages: list[dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(user_msgs)
    return messages


def _build_messages_with_assistant_prefix(
    *,
    query: str,
    image: Image.Image,
    system_prompt: str | None,
    response_prefix: str,
) -> list[dict[str, Any]]:
    """Same as `_build_prompt_messages` plus an assistant turn so the target prefix is chat-formatted."""
    messages = _build_prompt_messages(
        query=query, image=image, system_prompt=system_prompt
    )
    messages.append({"role": "assistant", "content": response_prefix})
    return messages


def _pixel_attack_processor_family(processor: Any) -> PixelAttackFamily:
    cls = processor.__class__.__name__
    if cls == "Gemma4Processor":
        return "gemma4"
    if cls == "Qwen3VLProcessor":
        return "qwen3_vl"
    raise RuntimeError(
        "Pixel-space attack supports Gemma4Processor or Qwen3VLProcessor; "
        f"got {cls!r}."
    )


def _apply_chat_template_for_attack(
    processor: Any,
    family: PixelAttackFamily,
    messages: list[dict[str, Any]],
    *,
    tokenize: bool,
    add_generation_prompt: bool,
    enable_thinking: bool,
) -> Any:
    kwargs: dict[str, Any] = {
        "conversation": messages,
        "tokenize": tokenize,
        "add_generation_prompt": add_generation_prompt,
    }
    kwargs["enable_thinking"] = enable_thinking
    return processor.apply_chat_template(**kwargs)


def _rgb_leaf_from_pil_gemma4(
    processor: Any,
    pil: Image.Image,
    device: torch.device,
) -> torch.Tensor:
    """Initialize optimizable CHW stack matching post-``rescale_and_normalize`` Gemma4 tensors.

    Chains ``process_image`` → ``aspect_ratio_preserving_resize`` → ``rescale_and_normalize`` on
    ``processor.image_processor`` (same as ``Gemma4ImageProcessor._preprocess`` before patchify).

    Returns float tensor ``[1, 3, H, W]`` (usually ~``[0, 1]`` when ``do_rescale`` is ``1/255``).
    """
    ip = processor.image_processor
    t = gemma4_chw_preprocessed_from_pil(ip, pil)
    t = t.unsqueeze(0).to(device=device, dtype=torch.float32).contiguous()
    return t.clone().requires_grad_(True)


def _rgb_leaf_from_pil_qwen3_vl(
    processor: Any,
    pil: Image.Image,
    device: torch.device,
) -> torch.Tensor:
    """~``[0, 1]`` CHW after official Qwen resize + rescale only (no CLIP normalize on leaf)."""
    ip = processor.image_processor
    t = qwen3_vl_rescaled_spatial_from_pil(ip, pil)
    t = t.to(device=device, dtype=torch.float32).contiguous()
    return t.clone().requires_grad_(True)


def _rgb_leaf_from_pil(
    processor: Any,
    pil: Image.Image,
    device: torch.device,
    family: PixelAttackFamily,
) -> torch.Tensor:
    if family == "gemma4":
        return _rgb_leaf_from_pil_gemma4(processor, pil, device)
    return _rgb_leaf_from_pil_qwen3_vl(processor, pil, device)


@dataclass
class ActiveSample:
    persona_id: str
    query: str
    response_prefix: str
    prompt_text: str
    full_text: str
    image_pil: Image.Image
    completion_start: int
    completion_len: int
    supervised_completion_len: int  # CE applies to first this many assistant tokens (tail excluded from loss).
    seq_len: int
    # Spatial CHW batch ``[B, 3, H, W]``: Gemma4 ≈ post-rescale ``[0,1]``; Qwen3 VL ≈ post-rescale ``[0,1]``
    # (CLIP normalize runs inside the processor on each forward).
    image_rgb: torch.Tensor
    image_rgb_ref: torch.Tensor
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    extras: dict[str, torch.Tensor] = field(default_factory=dict)
    iter_count: int = 0
    optimizer: torch.optim.Optimizer | None = None


def _processor_encode(
    processor: Any,
    *,
    image: Image.Image,
    family: PixelAttackFamily,
    messages: list[dict[str, Any]] | None = None,
    text: str | None = None,
    add_generation_prompt: bool = True,
    enable_thinking: bool = False,
) -> dict[str, torch.Tensor]:
    if text is None:
        if messages is None:
            raise TypeError("_processor_encode requires `messages` or `text`")
        text = _apply_chat_template_for_attack(
            processor,
            family,
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=enable_thinking,
        )
    out = processor(
        images=[image],
        text=text,
        return_tensors="pt",
        add_special_tokens=False,
    )
    return {k: v for k, v in out.items()}


def _build_active_sample(
    *,
    processor: Any,
    family: PixelAttackFamily,
    persona_id: str,
    query: str,
    response_prefix: str,
    image: Image.Image,
    system_prompt: str | None,
    enable_thinking: bool,
    device: torch.device,
    attack_cfg: Any,
) -> ActiveSample:
    image_pil = image.convert("RGB")

    messages_prompt = _build_prompt_messages(
        query=query,
        image=image_pil,
        system_prompt=system_prompt,
    )
    messages_full = _build_messages_with_assistant_prefix(
        query=query,
        image=image_pil,
        system_prompt=system_prompt,
        response_prefix=response_prefix,
    )
    prompt_text = _apply_chat_template_for_attack(
        processor,
        family,
        messages_prompt,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    full_text = _apply_chat_template_for_attack(
        processor,
        family,
        messages_full,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=enable_thinking,
    )
    # Prompt-only encode defines where supervised assistant-token span begins;
    # full sequence uses one multimodal encode so prefix tokens match the template.
    enc_prompt = _processor_encode(
        processor,
        family=family,
        image=image_pil,
        text=prompt_text,
    )
    enc_full = _processor_encode(
        processor,
        family=family,
        image=image_pil,
        text=full_text,
    )
    completion_start = int(enc_prompt["input_ids"].shape[1])
    completion_len = int(enc_full["input_ids"].shape[1]) - completion_start
    p_ids = enc_prompt["input_ids"]
    f_ids = enc_full["input_ids"]
    if f_ids.shape[1] < p_ids.shape[1] or not torch.equal(
        f_ids[:, :completion_start], p_ids
    ):
        raise ValueError(
            "Prompt prefix token mismatch: assistant-turn encoding must extend "
            "the generation-prompt encode (check chat template / multimodal text)."
        )
    if completion_len <= 0:
        raise ValueError(
            f"Non-positive completion_len={completion_len} for persona_id={persona_id!r}"
        )

    input_ids = enc_full["input_ids"].to(device)
    attention_mask = enc_full["attention_mask"].to(device)
    seq_len = int(attention_mask.sum().item())
    expected = completion_start + completion_len
    if seq_len != expected:
        raise ValueError(
            f"seq_len mismatch for persona_id={persona_id!r}: got {seq_len}, expected {expected}"
        )

    tail_ignore = max(
        0, int(getattr(attack_cfg, "completion_loss_tail_ignore", DEFAULT_COMPLETION_LOSS_TAIL_IGNORE))
    )
    labels = input_ids.clone()
    labels[:, :completion_start] = -100
    # Exclude last ``tail_ignore`` assistant tokens from CE (requires completion_len > tail_ignore to avoid
    # slicing past the assistant span or into the prompt).
    if completion_len > tail_ignore > 0:
        labels[:, completion_start + completion_len - tail_ignore : completion_start + completion_len] = (
            -100
        )
        supervised_completion_len = completion_len - tail_ignore
    else:
        supervised_completion_len = completion_len
    labels[attention_mask == 0] = -100

    if supervised_completion_len <= 0:
        raise ValueError(
            f"No supervised assistant tokens after tail ignore (completion_len={completion_len}, "
            f"completion_loss_tail_ignore={tail_ignore}) for persona_id={persona_id!r}"
        )

    image_rgb = _rgb_leaf_from_pil(processor, image_pil, device, family)
    image_rgb_ref = image_rgb.detach().clone()

    extras: dict[str, torch.Tensor] = {}
    for key in enc_full:
        if key in (
            "input_ids",
            "attention_mask",
            "pixel_values",
            "labels",
            "image_position_ids",
            "image_grid_thw",
        ):
            continue
        t = enc_full[key]
        if isinstance(t, torch.Tensor):
            extras[key] = t.to(device)

    if _attack_optimizer_kind(attack_cfg) == "adam":
        optimizer = torch.optim.Adam(
            [image_rgb],
            lr=float(attack_cfg.lr),
            betas=(
                float(getattr(attack_cfg, "adam_beta1", 0.9)),
                float(getattr(attack_cfg, "adam_beta2", 0.999)),
            ),
            eps=float(getattr(attack_cfg, "adam_eps", 1e-8)),
        )
    else:
        optimizer = None

    return ActiveSample(
        persona_id=persona_id,
        query=query,
        response_prefix=response_prefix,
        prompt_text=prompt_text,
        full_text=full_text,
        image_pil=image_pil,
        completion_start=completion_start,
        completion_len=completion_len,
        supervised_completion_len=supervised_completion_len,
        seq_len=seq_len,
        image_rgb=image_rgb,
        image_rgb_ref=image_rgb_ref,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        extras=extras,
        optimizer=optimizer,
    )


def _pad_batch(
    actives: list[ActiveSample],
    processor: Any,
    pad_token_id: int,
    device: torch.device,
) -> dict[str, Any]:
    input_ids_list = [s.input_ids.squeeze(0) for s in actives]
    attn_list = [s.attention_mask.squeeze(0) for s in actives]
    labels_list = [s.labels.squeeze(0) for s in actives]

    input_ids = pad_sequence(
        input_ids_list, batch_first=True, padding_value=pad_token_id
    )
    attention_mask = pad_sequence(attn_list, batch_first=True, padding_value=0)
    labels = pad_sequence(labels_list, batch_first=True, padding_value=-100)

    image_rgb = torch.cat([s.image_rgb for s in actives], dim=0)

    batch: dict[str, Any] = {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "labels": labels.to(device),
        "image_rgb": image_rgb,
    }
    if actives and actives[0].extras:
        if "token_type_ids" in actives[0].extras:
            tt_list = [s.extras["token_type_ids"].squeeze(0) for s in actives]
            batch["token_type_ids"] = pad_sequence(
                tt_list, batch_first=True, padding_value=0
            ).to(device)
        if "mm_token_type_ids" in actives[0].extras:
            mtt_list = [s.extras["mm_token_type_ids"].squeeze(0) for s in actives]
            batch["mm_token_type_ids"] = pad_sequence(
                mtt_list, batch_first=True, padding_value=0
            ).to(device)
    if actives and actives[0].extras:
        common_keys = set(actives[0].extras)
        for s in actives[1:]:
            common_keys &= set(s.extras)
        for skip_k in ("token_type_ids", "mm_token_type_ids"):
            common_keys.discard(skip_k)
        for k in sorted(common_keys):
            batch[k] = torch.cat([s.extras[k] for s in actives], dim=0).to(device)
    return batch


def _merge_processor_vision_into_batch(
    processor: Any,
    batch: dict[str, Any],
    processor_texts: list[str],
    device: torch.device,
) -> dict[str, Any]:
    """
    Refresh vision tensors from optimizable spatial CHW leaves (see module docstring).
    Only visual is filled into the batch, text is not.
    """
    family = _pixel_attack_processor_family(processor)
    image_rgb = batch.pop("image_rgb")
    chw_list = [image_rgb[i].squeeze(0) for i in range(image_rgb.shape[0])]
    images_per_prompt = [[t] for t in chw_list]
    if family == "gemma4":
        # the default padding is True for gemma4
        proc_out = processor(
            images=images_per_prompt,
            text=processor_texts,
            return_tensors="pt",
            add_special_tokens=False,
            images_kwargs=GEMMA4_PRECOMPUTED_VISION_KWARGS,
        )
    else:
        # the default padding is False for qwen3_vl
        # therefore, we need to add padding=True to the processor call
        proc_out = processor(
            images=images_per_prompt,
            text=processor_texts,
            return_tensors="pt",
            add_special_tokens=False,
            padding=True,
            images_kwargs=QWEN3_VL_ATTACK_MERGE_IMAGES_KWARGS,
        )
    out = dict(batch)
    pv = proc_out["pixel_values"]
    if pv.dtype != torch.float32:
        pv = pv.float()
    out["pixel_values"] = pv.to(device)
    if "image_position_ids" in proc_out:
        out["image_position_ids"] = proc_out["image_position_ids"].to(
            device=device, dtype=torch.long
        )
    if "image_grid_thw" in proc_out:
        out["image_grid_thw"] = proc_out["image_grid_thw"].to(
            device=device, dtype=torch.long
        )
    return out


def _trunc_for_log(s: str, max_chars: int = 6000) -> str:
    if len(s) <= max_chars:
        return s
    half = max_chars // 2
    return s[:half] + "\n... [truncated] ...\n" + s[-half:]


def _sanity_check_first_item_decode(
    processor: Any,
    first: ActiveSample,
    input_ids_row: torch.Tensor,
    labels_row: torch.Tensor,
    attention_mask_row: torch.Tensor,
) -> None:
    """Log decoded prompt vs completion span for batch index 0 and basic label/id checks."""
    tok = getattr(processor, "tokenizer", None)
    if tok is None:
        print("[sanity] skip: processor has no tokenizer", flush=True)
        return

    ids = input_ids_row.detach().cpu().long()
    lab = labels_row.detach().cpu().long()
    attn = attention_mask_row.detach().cpu().bool()

    cs = first.completion_start
    cl = first.completion_len
    sl = first.supervised_completion_len
    valid = int(attn.sum().item())
    if valid != cs + cl:
        print(
            f"[sanity] WARN persona_id={first.persona_id!r}: sum(attention_mask)={valid} "
            f"!= completion_start+completion_len={cs + cl}",
            flush=True,
        )

    prompt_ids = ids[:cs]
    comp_ids = ids[cs : cs + cl]
    lbl_prompt = lab[:cs]
    lbl_comp = lab[cs : cs + cl]

    if not torch.all(lbl_prompt == -100):
        n_bad = int((lbl_prompt != -100).sum().item())
        print(
            f"[sanity] WARN persona_id={first.persona_id!r}: {n_bad} label(s) in prompt span "
            f"are not -100 (expected all masked)",
            flush=True,
        )
    sup_mask = lbl_comp != -100
    if sup_mask.any() and not torch.equal(comp_ids[sup_mask], lbl_comp[sup_mask]):
        print(
            f"[sanity] WARN persona_id={first.persona_id!r}: completion input_ids != labels "
            f"where CE is applied",
            flush=True,
        )

    n_sup = int((lab != -100).sum().item())
    if n_sup != sl:
        print(
            f"[sanity] WARN persona_id={first.persona_id!r}: (labels != -100).sum()={n_sup}, "
            f"expected supervised_completion_len={sl}",
            flush=True,
        )

    prompt_dec = tok.decode(prompt_ids.tolist(), skip_special_tokens=False)
    comp_dec = tok.decode(comp_ids.tolist(), skip_special_tokens=False)
    comp_dec_skip = tok.decode(comp_ids.tolist(), skip_special_tokens=True)

    print(
        f"[sanity] first batch item persona_id={first.persona_id!r} "
        f"completion_start={cs} completion_len={cl} supervised_completion_len={sl} seq_tokens={valid}",
        flush=True,
    )
    print(
        "[sanity] decoded prompt (input_ids[0:completion_start], skip_special_tokens=False):",
        flush=True,
    )
    print(_trunc_for_log(prompt_dec), flush=True)
    print(
        "[sanity] decoded full assistant span (input_ids[completion_start:completion_start+completion_len]); "
        "CE applies only to first supervised_completion_len tokens:",
        flush=True,
    )
    print(f"  raw decode: {comp_dec!r}", flush=True)
    print(f"  skip_special_tokens=True: {comp_dec_skip!r}", flush=True)
    print(f"  dataset response_prefix: {first.response_prefix!r}", flush=True)


def _l2_to_ref_loss(actives: list[ActiveSample], weight: float) -> torch.Tensor:
    if weight <= 0.0:
        return torch.zeros((), device=actives[0].image_rgb.device)
    total = torch.zeros((), device=actives[0].image_rgb.device)
    for s in actives:
        total = total + weight * F.mse_loss(s.image_rgb, s.image_rgb_ref)
    return total / max(len(actives), 1)


def _teacher_forced_greedy_completion_ids(
    logits: torch.Tensor,
    completion_start: int,
    completion_span_len: int,
) -> list[int]:
    """Collect ``argmax(logits[pos])`` for ``pos = completion_start-1 ..`` from **one** forward pass.

    .. note::

        This is **not** ``generate(do_sample=False)``. One forward embeds the **full** sequence including **gold**
        assistant tokens from ``input_ids``. At ``logits[completion_start-1+j]`` the model predicts the token at
        ``completion_start+j`` while attending (via causal LM) to **gold** tokens at earlier assistant positions.

        ``generate()`` is **autoregressive greedy**: each step feeds **model predictions**, not gold tokens. Only the
        **first** new token matches ``argmax(logits[completion_start-1])`` (same prompt-only prefix). After that,
        stitched teacher-step argmax IDs generally **differ** from ``generate()`` unless greedy equals gold at every
        step — which is exactly what ``_teacher_forced_prefix_match`` checks.

    ``logits[t]`` predicts ``input_ids[t+1]`` (HF causal layout); first supervised assistant token is predicted from
    ``logits[completion_start - 1]``.

    ``completion_span_len`` should match ``supervised_completion_len`` (prefix CE span), not necessarily full
    ``completion_len``.
    """
    out: list[int] = []
    for j in range(completion_span_len):
        pos = completion_start - 1 + j
        if pos < 0 or pos >= logits.shape[1]:
            return out
        out.append(int(logits[0, pos].argmax().item()))
    return out


def _teacher_forced_prefix_match(
    logits: torch.Tensor,
    input_ids_1row: torch.Tensor,
    completion_start: int,
    supervised_completion_len: int,
) -> tuple[bool, list[int]]:
    """Whether teacher-forced greedy preds match ground-truth on the CE-supervised assistant prefix."""
    pred_ids = _teacher_forced_greedy_completion_ids(
        logits, completion_start, supervised_completion_len
    )
    if len(pred_ids) != supervised_completion_len:
        return False, pred_ids
    for j, pid in enumerate(pred_ids):
        if pid != int(input_ids_1row[0, completion_start + j].item()):
            return False, pred_ids
    return True, pred_ids


def _batch_row_for_generate(model_batch: dict[str, Any], row: int) -> dict[str, Any]:
    """Single-batch-row slice for ``generate`` (drops ``labels``)."""
    out: dict[str, Any] = {}
    for k, v in model_batch.items():
        if k == "labels":
            continue
        if isinstance(v, torch.Tensor):
            out[k] = v[row : row + 1]
        else:
            out[k] = v
    return out


@torch.no_grad()
def _generate_greedy_completion_ids(
    model: Any,
    model_batch: dict[str, Any],
    *,
    pixel_family: PixelAttackFamily,
    completion_start: int,
    completion_len: int,
    pad_token_id: int,
    eos_token_id: int | None,
) -> list[int]:
    """Autoregressive greedy continuation like ``model.generate(..., do_sample=False)`` (not teacher forcing)."""
    pl = completion_start
    if pixel_family == "gemma4":
        gen_kw: dict[str, Any] = dict(
            input_ids=model_batch["input_ids"][:1, :pl],
            attention_mask=model_batch["attention_mask"][:1, :pl],
            pixel_values=model_batch["pixel_values"][:1],
            image_position_ids=model_batch["image_position_ids"][:1],
            max_new_tokens=completion_len,
            do_sample=False,
            pad_token_id=pad_token_id,
        )
    else:
        image_nums, _ = model._get_image_nums_and_video_nums(model_batch["input_ids"])
        pv0, ig0 = qwen_slice_visual_for_batch_row(
            model_batch["pixel_values"],
            model_batch["image_grid_thw"],
            image_nums,
            row=0,
        )
        gen_kw = dict(
            input_ids=model_batch["input_ids"][:1, :pl],
            attention_mask=model_batch["attention_mask"][:1, :pl],
            pixel_values=pv0,
            image_grid_thw=ig0,
            max_new_tokens=completion_len,
            do_sample=False,
            pad_token_id=pad_token_id,
        )
    mmt = model_batch.get("mm_token_type_ids")
    if mmt is not None:
        gen_kw["mm_token_type_ids"] = mmt[:1, :pl]
    if eos_token_id is not None:
        gen_kw["eos_token_id"] = eos_token_id
    out_ids = model.generate(**gen_kw)
    row = out_ids[0]
    span = row[pl : pl + completion_len]
    return [int(x.item()) for x in span]


@torch.no_grad()
def _sanity_first_token_generate_vs_forward(
    model: Any,
    logits: torch.Tensor,
    model_batch: dict[str, Any],
    *,
    pixel_family: PixelAttackFamily,
    completion_start: int,
    pad_token_id: int,
    eos_token_id: int | None,
) -> None:
    """Log whether ``generate`` (greedy, one token) agrees with forward argmax at the prompt boundary."""
    pl = completion_start
    if pixel_family == "gemma4":
        gen_kw: dict[str, Any] = dict(
            input_ids=model_batch["input_ids"][:1, :pl],
            attention_mask=model_batch["attention_mask"][:1, :pl],
            pixel_values=model_batch["pixel_values"][:1],
            image_position_ids=model_batch["image_position_ids"][:1],
            max_new_tokens=1,
            do_sample=False,
            pad_token_id=pad_token_id,
        )
    else:
        image_nums, _ = model._get_image_nums_and_video_nums(model_batch["input_ids"])
        pv0, ig0 = qwen_slice_visual_for_batch_row(
            model_batch["pixel_values"],
            model_batch["image_grid_thw"],
            image_nums,
            row=0,
        )
        gen_kw = dict(
            input_ids=model_batch["input_ids"][:1, :pl],
            attention_mask=model_batch["attention_mask"][:1, :pl],
            pixel_values=pv0,
            image_grid_thw=ig0,
            max_new_tokens=1,
            do_sample=False,
            pad_token_id=pad_token_id,
        )
    mmt = model_batch.get("mm_token_type_ids")
    if mmt is not None:
        gen_kw["mm_token_type_ids"] = mmt[:1, :pl]
    if eos_token_id is not None:
        gen_kw["eos_token_id"] = eos_token_id
    g1 = model.generate(**gen_kw)
    gen_first = int(g1[0, pl].item())
    fwd_first = int(logits[0, pl - 1].argmax().item())
    tgt = int(model_batch["input_ids"][0, pl].item())
    print(
        f"[sanity] first new token: forward_argmax={fwd_first} generate_greedy={gen_first} "
        f"ground_truth={tgt} (forward vs generate should match for step-1 greedy)",
        flush=True,
    )


def _save_adversarial_png(
    image_rgb_row: torch.Tensor,
    path: Path,
    *,
    processor: Any,
) -> None:
    """Save optimized spatial CHW (shape ``[1, 3, H, W]``) as an 8-bit PNG."""
    path.parent.mkdir(parents=True, exist_ok=True)
    row = image_rgb_row.detach().float().cpu().squeeze(0)
    family = _pixel_attack_processor_family(processor)
    ip = processor.image_processor
    if family == "gemma4":
        u8 = chw_preprocessed_to_uint8_chw(row, ip)
    else:
        rf = float(getattr(ip, "rescale_factor", 1.0 / 255.0))
        u8 = rescaled_spatial_to_uint8_chw(row, rf)
    arr = u8.permute(1, 2, 0).numpy()
    Image.fromarray(arr, mode="RGB").save(path)


def _optimizer_step(
    actives: list[ActiveSample],
    attack_cfg: Any,
    *,
    lr: float,
    clamp_min: float,
    clamp_max: float,
) -> None:
    for s in actives:
        if s.image_rgb.grad is None:
            continue
        if (
            _attack_optimizer_kind(attack_cfg) == "adam"
            and s.optimizer is not None
        ):
            s.optimizer.step()
            s.optimizer.zero_grad(set_to_none=True)
        else:
            with torch.no_grad():
                s.image_rgb.add_(-lr * s.image_rgb.grad)
            s.image_rgb.grad = None
        # clamp
        with torch.no_grad():
            s.image_rgb.clamp_(clamp_min, clamp_max)


@hydra.main(
    config_path="../configs",
    config_name="adversarial_run_attack.yaml",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    torch.manual_seed(int(cfg.seed))
    device_s = str(cfg.device).lower()
    if device_s == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    device = torch.device(cfg.device)

    attack_cfg = cfg.attack
    out_root = Path(str(attack_cfg.output_dir))
    if not out_root.is_absolute():
        out_root = REPO_ROOT / out_root
    model_dir = sanitize_model_dir_name(cfg)
    save_dir = out_root / model_dir
    save_dir.mkdir(parents=True, exist_ok=True)

    from transformers import AutoProcessor

    model_path = str(cfg.model.model_name_or_path)
    processor = AutoProcessor.from_pretrained(model_path)
    pixel_family = _pixel_attack_processor_family(processor)
    print(f"[attack] pixel_family={pixel_family}", flush=True)
    pad_token_id = int(
        getattr(processor.tokenizer, "pad_token_id", None)
        or processor.tokenizer.eos_token_id
    )
    eos_tok_id = getattr(processor.tokenizer, "eos_token_id", None)

    dtype = _dtype_from_cfg(str(cfg.torch_dtype))
    model = _load_vlm_model(model_path, dtype, device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()

    system_prompt = getattr(attack_cfg, "system_prompt", None)
    if system_prompt is not None:
        system_prompt = str(system_prompt).strip() or None
    enable_thinking = bool(getattr(attack_cfg, "enable_thinking", False))

    ds = load_ours_dataset(cfg)
    loader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        collate_fn=get_ours_collate_fn(),
    )
    data_iter = iter(loader)

    max_batch = int(attack_cfg.batch_size)
    max_iters = int(attack_cfg.max_iters_per_sample)
    lr = float(attack_cfg.lr)
    opt_kind = _attack_optimizer_kind(attack_cfg)
    print(
        f"[attack] optimizer={opt_kind}"
        + (" (torch.optim.Adam(image_rgb))" if opt_kind == "adam" else " (manual SGD: image_rgb -= lr * grad)")
        + f" lr={lr}",
        flush=True,
    )
    l2_w = float(getattr(attack_cfg, "l2_weight", 0.0))
    clamp_min = float(
        getattr(
            attack_cfg,
            "image_clamp_min",
            getattr(attack_cfg, "pixel_clamp_min", 0.0),
        )
    )
    clamp_max = float(
        getattr(
            attack_cfg,
            "image_clamp_max",
            getattr(attack_cfg, "pixel_clamp_max", 1.0),
        )
    )
    skip_existing = bool(getattr(attack_cfg, "skip_existing", False))
    save_failed = bool(getattr(attack_cfg, "save_failed", False))
    failed_sub = str(getattr(attack_cfg, "failed_subdir", "failed"))

    def output_path_for(pid: str, *, failed: bool = False) -> Path:
        if failed and save_failed:
            return save_dir / failed_sub / f"{pid}.png"
        return save_dir / f"{pid}.png"

    def should_skip(pid: str) -> bool:
        if not skip_existing:
            return False
        p = output_path_for(pid)
        return p.is_file()

    actives: list[ActiveSample] = []

    def refill() -> None:
        nonlocal data_iter
        while len(actives) < max_batch:
            try:
                batch = next(data_iter)
            except StopIteration:
                break
            pid = str(batch["persona_id"][0]).strip()
            if not pid or should_skip(pid):
                continue
            img = batch["image"][0]
            if img is None:
                print(f"[skip] persona_id={pid!r}: no image", flush=True)
                continue
            q = str(batch["query"][0])
            rp = str(batch["response_prefix"][0])
            try:
                sample = _build_active_sample(
                    processor=processor,
                    family=pixel_family,
                    persona_id=pid,
                    query=q,
                    response_prefix=rp,
                    image=img,
                    system_prompt=system_prompt,
                    enable_thinking=enable_thinking,
                    device=device,
                    attack_cfg=attack_cfg,
                )
            except Exception as e:  # pragma: no cover
                print(f"[skip] persona_id={pid!r}: {e}", flush=True)
                continue
            actives.append(sample)

    refill()
    global_step = 0
    use_amp = device.type == "cuda" and dtype in (torch.bfloat16, torch.float16)
    amp_dtype = torch.bfloat16 if dtype == torch.bfloat16 else torch.float16

    pbar = tqdm(total=len(ds), desc="Attack progress")

    while actives:
        global_step += 1
        for s in actives:
            if (
                _attack_optimizer_kind(attack_cfg) == "adam"
                and s.optimizer is not None
            ):
                s.optimizer.zero_grad(set_to_none=True)
            elif s.image_rgb.grad is not None:
                s.image_rgb.grad.zero_()
        batch = _pad_batch(actives, processor, pad_token_id, device)
        labels = batch.pop("labels")
        processor_texts = [s.full_text for s in actives]
        model_batch = _merge_processor_vision_into_batch(
            processor, batch, processor_texts, device
        )
        model_batch["labels"] = labels
        padded_ids = model_batch["input_ids"]
        if global_step == 1 and actives:
            _sanity_check_first_item_decode(
                processor,
                actives[0],
                padded_ids[0],
                labels[0],
                model_batch["attention_mask"][0],
            )

        model.zero_grad(set_to_none=True)
        if use_amp:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                out = model(**model_batch)
                logits = out.logits
                if out.loss is None:
                    raise RuntimeError(
                        "Model returned no loss despite labels=...; check multimodal forward (Gemma4 or Qwen3 VL)."
                    )
                loss = out.loss + _l2_to_ref_loss(actives, l2_w)
        else:
            out = model(**model_batch)
            logits = out.logits
            if out.loss is None:
                raise RuntimeError(
                    "Model returned no loss despite labels=...; check multimodal forward (Gemma4 or Qwen3 VL)."
                )
            loss = out.loss + _l2_to_ref_loss(actives, l2_w)

        if global_step == 1 and actives:
            _sanity_first_token_generate_vs_forward(
                model,
                logits,
                model_batch,
                pixel_family=pixel_family,
                completion_start=actives[0].completion_start,
                pad_token_id=pad_token_id,
                eos_token_id=eos_tok_id,
            )

        loss.backward()
        print(f"loss: {loss.item()}", flush=True)
        if global_step == 1:
            if not any(s.image_rgb.grad is not None for s in actives):
                raise RuntimeError(
                    "No gradient on image_rgb after backward; this transformers "
                    "build may not propagate grads through the processor vision path "
                    "(Gemma4Processor / Qwen3VLProcessor)."
                )

        # Decode before optimizer so logits / pixel_values match this forward (generate runs under no_grad).
        tf_matches: list[bool] = []
        for i, s in enumerate(actives):
            row_logits = logits[i : i + 1]
            row_ids = padded_ids[i : i + 1]
            ok, pred_ids = _teacher_forced_prefix_match(
                row_logits,
                row_ids,
                s.completion_start,
                s.supervised_completion_len,
            )
            tf_matches.append(ok)
            # gen_ids: list[int] | None = None
            # if tok is not None:
            #     mb_row = _batch_row_for_generate(model_batch, i)
            #     gen_ids = _generate_greedy_completion_ids(
            #         model,
            #         mb_row,
            #         completion_start=s.completion_start,
            #         completion_len=s.completion_len,
            #         pad_token_id=pad_token_id,
            #         eos_token_id=eos_tok_id,
            #     )
            #     cs, _, sl = s.completion_start, s.completion_len, s.supervised_completion_len
            #     target_ids = row_ids[0, cs : cs + sl].long().tolist()
            #     pred_dec = tok.decode(pred_ids, skip_special_tokens=False)
            #     gen_dec = tok.decode(gen_ids, skip_special_tokens=False) if gen_ids else ""
            #     targ_dec = tok.decode(target_ids, skip_special_tokens=False)
            #     print(
            #         f"[decode] persona_id={s.persona_id} iter={s.iter_count + 1} "
            #         f"teacher_forced_prefix_match={ok}\n"
            #         "  single-pass stitched decode (NOT generate): argmax at each step while assistant slots STILL "
            #         "contain GOLD tokens in this forward — decode text need not match generate(); only first token "
            #         "must match sanity check.\n"
            #         f"  stitched gold-conditioned argmax (skip_special_tokens=False): {pred_dec!r}\n"
            #         f"  generate() greedy autoregressive (skip_special_tokens=False): {gen_dec!r}\n"
            #         f"  target span (gold assistant ids; skip_special_tokens=False): {targ_dec!r}\n",
            #         flush=True,
            #     )

        _optimizer_step(
            actives, attack_cfg, lr=lr, clamp_min=clamp_min, clamp_max=clamp_max
        )

        finished_indices: list[int] = []
        for i, s in enumerate(actives):
            ok = tf_matches[i]
            s.iter_count += 1
            if ok:
                path = output_path_for(s.persona_id)
                _save_adversarial_png(s.image_rgb, path, processor=processor)
                print(
                    f"[success] persona_id={s.persona_id} iters={s.iter_count} -> {path}",
                    flush=True,
                )
                pbar.update(1)
                finished_indices.append(i)
            elif s.iter_count >= max_iters:
                print(
                    f"[max_iter] persona_id={s.persona_id} iters={s.iter_count}",
                    flush=True,
                )
                if save_failed:
                    fpath = output_path_for(s.persona_id, failed=True)
                    _save_adversarial_png(s.image_rgb, fpath, processor=processor)
                    print(f"  saved failed image -> {fpath}", flush=True)
                pbar.update(1)
                finished_indices.append(i)

        for j in reversed(finished_indices):
            actives.pop(j)

        refill()

    print("Attack run finished.", flush=True)


if __name__ == "__main__":
    main()
