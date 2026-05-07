"""Qwen3 VL pixel attack helpers: ~[0,1] rescaled spatial leaf; normalize inside HF processor."""

from __future__ import annotations

from typing import Any

import torch
from PIL import Image

from transformers.image_utils import SizeDict
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

# Passed to ``Qwen3VLProcessor.__call__(..., images_kwargs=...)`` when ``images`` are already
# resized + rescaled to ~[0,1]; processor applies CLIP normalize + patchify via ``Qwen2VLImageProcessor``.
QWEN3_VL_ATTACK_MERGE_IMAGES_KWARGS: dict[str, Any] = {
    "do_resize": False,
    "do_rescale": False,
    "do_normalize": True,
    "do_convert_rgb": False,
}


def assert_qwen3_vl_torchvision_image_processor(image_processor: Any) -> None:
    backend = getattr(image_processor, "backend", None)
    if backend != "torchvision":
        raise RuntimeError(
            "Differentiable Qwen3 VL pixel attack requires a torchvision-backed "
            f"Qwen2VLImageProcessor; got backend={backend!r}. "
            "Use the torchvision image processor variant from AutoProcessor."
        )


def qwen3_vl_rescaled_spatial_from_pil(image_processor: Any, pil: Image.Image) -> torch.Tensor:
    """RGB PIL → ``[1, 3, H, W]`` float after official ``smart_resize`` + ``resize`` + rescale only.

    Matches ``Qwen2VLImageProcessor._preprocess`` up to (but not including) CLIP normalization.
    """
    assert_qwen3_vl_torchvision_image_processor(image_processor)

    stacked = image_processor.process_image(pil.convert("RGB"), do_convert_rgb=True).unsqueeze(0)

    height, width = stacked.shape[-2], stacked.shape[-1]
    patch_size = int(image_processor.patch_size)
    merge_size = int(image_processor.merge_size)
    size = image_processor.size
    if isinstance(size, dict):
        min_px = int(size["shortest_edge"])
        max_px = int(size["longest_edge"])
    else:
        min_px = int(size.shortest_edge)
        max_px = int(size.longest_edge)

    resized_height, resized_width = smart_resize(
        height,
        width,
        factor=patch_size * merge_size,
        min_pixels=min_px,
        max_pixels=max_px,
    )
    stacked = image_processor.resize(
        stacked,
        SizeDict(height=resized_height, width=resized_width),
        resample=image_processor.resample,
    )

    rf = float(getattr(image_processor, "rescale_factor", 1.0 / 255.0))
    stacked = image_processor.rescale_and_normalize(
        stacked,
        do_rescale=True,
        rescale_factor=rf,
        do_normalize=False,
        image_mean=image_processor.image_mean,
        image_std=image_processor.image_std,
    )
    return stacked.contiguous()


def rescaled_spatial_to_uint8_chw(chw: torch.Tensor, rescale_factor: float) -> torch.Tensor:
    """Invert rescale-only leaf (~[0,1] float CHW) to uint8 CHW."""
    x = chw.float()
    # Forward: x = uint8_scale * rescale_factor → invert: uint8_scale = x / rescale_factor
    u = (x / float(rescale_factor)).clamp(0, 255).round().to(torch.uint8)
    return u


def qwen_pixel_lengths_per_batch_row(image_grid_thw: torch.Tensor, image_nums: torch.Tensor) -> list[int]:
    """First dim of ``pixel_values`` is concat over batch images; these are split lengths per row."""
    nums_list = [int(x) for x in image_nums.tolist()]
    grid_chunks = torch.split(image_grid_thw, nums_list, dim=0)
    return [int(torch.prod(chunk, dim=1).sum().item()) for chunk in grid_chunks]


def qwen_slice_visual_for_batch_row(
    pixel_values: torch.Tensor,
    image_grid_thw: torch.Tensor,
    image_nums: torch.Tensor,
    row: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Isolate vision tensors for batch index ``row`` (for ``generate`` on one prompt)."""
    lengths = qwen_pixel_lengths_per_batch_row(image_grid_thw, image_nums)
    nums_list = [int(x) for x in image_nums.tolist()]
    pv_chunks = torch.split(pixel_values, lengths, dim=0)
    grid_chunks = torch.split(image_grid_thw, nums_list, dim=0)
    return pv_chunks[row], grid_chunks[row]
