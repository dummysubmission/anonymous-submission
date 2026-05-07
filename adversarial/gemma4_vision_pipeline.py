"""Gemma4 vision helpers: match ``Gemma4ImageProcessor._preprocess`` before patchify."""

from __future__ import annotations

from typing import Any

import torch
from PIL import Image

# Passed to ``Gemma4Processor.__call__(..., images_kwargs=...)`` when ``images`` are already
# resized + ``rescale_and_normalize``''d CHW tensors (see ``gemma4_chw_preprocessed_from_pil``).
GEMMA4_PRECOMPUTED_VISION_KWARGS: dict[str, Any] = {
    "do_resize": False,
    "do_rescale": False,
    "do_normalize": False,
    "do_convert_rgb": False,
}


def _max_patches(image_processor: Any) -> int:
    return int(image_processor.max_soft_tokens * image_processor.pooling_kernel_size**2)


def gemma4_chw_preprocessed_from_pil(image_processor: Any, pil: Image.Image) -> torch.Tensor:
    """RGB PIL → CHW float after official resize and ``rescale_and_normalize`` (no patchify).

    Uses the same ``process_image`` → ``aspect_ratio_preserving_resize`` → ``rescale_and_normalize``
    chain as ``Gemma4ImageProcessor._preprocess``, delegated to the given processor instance.
    """
    t = image_processor.process_image(pil.convert("RGB"), do_convert_rgb=True)
    t = image_processor.aspect_ratio_preserving_resize(
        image=t,
        patch_size=image_processor.patch_size,
        max_patches=_max_patches(image_processor),
        pooling_kernel_size=image_processor.pooling_kernel_size,
        resample=image_processor.resample,
    )
    return image_processor.rescale_and_normalize(
        t,
        image_processor.do_rescale,
        image_processor.rescale_factor,
        image_processor.do_normalize,
        image_processor.image_mean,
        image_processor.image_std,
    )


def chw_preprocessed_to_uint8_chw(chw: torch.Tensor, image_processor: Any) -> torch.Tensor:
    """Invert ``rescale_and_normalize`` for saving 8-bit PNG (CHW uint8)."""
    mean = torch.tensor(
        image_processor.image_mean, device=chw.device, dtype=torch.float32
    ).view(3, 1, 1)
    std = torch.tensor(
        image_processor.image_std, device=chw.device, dtype=torch.float32
    ).view(3, 1, 1)
    x = chw.float()
    if image_processor.do_normalize:
        if image_processor.do_rescale:
            mean = mean * (1.0 / float(image_processor.rescale_factor))
            std = std * (1.0 / float(image_processor.rescale_factor))
        x = x * std + mean
    elif image_processor.do_rescale:
        x = x / float(image_processor.rescale_factor)
    return x.clamp(0, 255).round().to(torch.uint8)
