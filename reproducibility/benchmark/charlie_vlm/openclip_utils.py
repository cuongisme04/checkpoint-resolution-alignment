from __future__ import annotations

import math
from typing import Any

from .registry import ModelSpec


def require_openclip():
    try:
        import open_clip
    except ImportError as exc:
        raise SystemExit("Install OpenCLIP first: python3 -m pip install -r requirements.txt") from exc
    return open_clip


def _interpolate_vit_pos_embed(pe, target_grid: int):
    """Bicubic-resize a (grid^2 + 1, D) ViT positional embedding to (target_grid^2 + 1, D).

    Used to work around open_clip 2.26.1's load_openai_model code path silently
    ignoring force_image_size for OpenAI CLIP checkpoints. See factory.py:216-223:
    the OpenAI branch loads via load_openai_model() which has no force_image_size
    argument, so the PE stays at the native 224 grid and the model runs at 224
    regardless of the requested resolution. We interpolate explicitly here,
    matching the bicubic+antialias procedure used by open_clip.model.resize_pos_embed.
    """
    import torch
    import torch.nn.functional as F

    n = pe.shape[0]
    current_grid = int(round((n - 1) ** 0.5))
    if current_grid * current_grid != n - 1:
        raise RuntimeError(
            f"Unexpected positional_embedding layout: shape[0]={n}; "
            "expected (grid^2 + 1) for a ViT with CLS token. "
            "Cannot safely interpolate positional embedding."
        )
    if current_grid == target_grid:
        return pe
    cls, grid_tok = pe[:1], pe[1:]
    dim = grid_tok.shape[1]
    grid_tok = grid_tok.reshape(1, current_grid, current_grid, dim).permute(0, 3, 1, 2)
    grid_tok = F.interpolate(
        grid_tok.float(),
        size=(target_grid, target_grid),
        mode="bicubic",
        antialias=True,
        align_corners=False,
    ).to(pe.dtype)
    grid_tok = grid_tok.permute(0, 2, 3, 1).reshape(target_grid * target_grid, dim)
    return torch.cat([cls, grid_tok], dim=0)


def _interpolate_timm_pos_embed(pe, target_grid: int, num_prefix: int = 0):
    """Bicubic-resize a timm-style positional embedding to a new grid.

    Handles two timm layouts:
      - (1, num_prefix + grid^2, D) where num_prefix is 0 (SigLIP) or 1 (CLS).
    Returns a tensor of shape (1, num_prefix + target_grid^2, D).
    """
    import torch
    import torch.nn.functional as F

    if pe.dim() != 3 or pe.shape[0] != 1:
        raise RuntimeError(
            f"Unexpected timm pos_embed layout: shape={tuple(pe.shape)}; expected (1, N, D)."
        )
    n = pe.shape[1]
    grid_tokens = n - num_prefix
    current_grid = int(round(grid_tokens ** 0.5))
    if current_grid * current_grid != grid_tokens:
        raise RuntimeError(
            f"Unexpected timm pos_embed grid: N={n}, num_prefix={num_prefix}, "
            f"so grid tokens={grid_tokens} is not a perfect square."
        )
    if current_grid == target_grid:
        return pe
    dim = pe.shape[2]
    prefix = pe[:, :num_prefix, :] if num_prefix > 0 else None
    grid = pe[:, num_prefix:, :].reshape(1, current_grid, current_grid, dim).permute(0, 3, 1, 2)
    grid = F.interpolate(
        grid.float(),
        size=(target_grid, target_grid),
        mode="bicubic",
        antialias=True,
        align_corners=False,
    ).to(pe.dtype)
    grid = grid.permute(0, 2, 3, 1).reshape(1, target_grid * target_grid, dim)
    if prefix is not None:
        return torch.cat([prefix, grid], dim=1)
    return grid


def _patch_timm_visual(visual, resolution: int):
    """Resize a timm-backed open_clip TimmModel visual to a new input resolution.

    Used for SigLIP, which open_clip wraps via TimmModel(trunk=timm.VisionTransformer)
    rather than the native open_clip CLIP visual. The trunk's pos_embed has shape
    (1, num_prefix + grid^2, D) and the patch_embed has a hard-coded img_size check
    that must be relaxed when we change resolution.
    """
    import torch

    trunk = getattr(visual, "trunk", None)
    if trunk is None:
        return False
    pe = getattr(trunk, "pos_embed", None)
    patch_embed = getattr(trunk, "patch_embed", None)
    if pe is None or patch_embed is None:
        return False
    patch_size = getattr(patch_embed, "patch_size", None)
    if isinstance(patch_size, (tuple, list)):
        ps = int(patch_size[0])
    elif isinstance(patch_size, int):
        ps = int(patch_size)
    else:
        return False
    if resolution % ps != 0:
        return False
    target_grid = resolution // ps
    n = pe.shape[1]
    grid_n = int(round(n ** 0.5))
    num_prefix = 1 if grid_n * grid_n != n else 0
    new_pe = _interpolate_timm_pos_embed(pe.detach(), target_grid, num_prefix=num_prefix)
    trunk.pos_embed = torch.nn.Parameter(
        new_pe.to(pe.device, dtype=pe.dtype), requires_grad=False
    )
    # Relax the patch_embed's input-size check so the forward pass accepts the
    # new resolution. timm PatchEmbed checks img_size in forward(); some versions
    # also gate on a strict flag we disable here when available.
    if hasattr(patch_embed, "img_size"):
        patch_embed.img_size = (resolution, resolution)
    if hasattr(patch_embed, "grid_size"):
        patch_embed.grid_size = (target_grid, target_grid)
    if hasattr(patch_embed, "num_patches"):
        patch_embed.num_patches = target_grid * target_grid
    if hasattr(patch_embed, "strict_img_size"):
        patch_embed.strict_img_size = False
    if hasattr(visual, "image_size"):
        visual.image_size = (resolution, resolution)
    return True


def load_model(open_clip_module, spec: ModelSpec, resolution: int, device: str):
    kwargs = {
        "model_name": spec.model_name,
        "pretrained": spec.pretrained,
        "device": device,
        "force_image_size": resolution,
    }
    try:
        model, _, preprocess = open_clip_module.create_model_and_transforms(**kwargs)
    except TypeError as exc:
        raise RuntimeError(
            "This OpenCLIP build does not support force_image_size. "
            "Upgrade open_clip_torch so resolution sweeps resize positional embeddings correctly."
        ) from exc

    # Open_clip 2.26.1 silently ignores force_image_size for pretrained='openai'
    # (factory.py routes to load_openai_model which does not accept it). The PE
    # comes back sized for the native 224 grid and the returned preprocess also
    # produces 224x224 tensors, so the model effectively runs at 224 regardless
    # of the resolution arg. Detect when the PE grid does not match the target
    # resolution and explicitly: (1) interpolate the PE, (2) update image_size /
    # grid_size, (3) rebuild the preprocess transform at the target resolution.
    visual = getattr(model, "visual", None)
    pe = getattr(visual, "positional_embedding", None)
    ps = patch_size(model)
    patched = False
    if pe is not None and pe.dim() == 2 and ps and resolution % ps == 0:
        import torch

        target_grid = resolution // ps
        current_grid = int(round((pe.shape[0] - 1) ** 0.5))
        if current_grid > 0 and current_grid != target_grid:
            new_pe = _interpolate_vit_pos_embed(pe.detach(), target_grid)
            visual.positional_embedding = torch.nn.Parameter(
                new_pe.to(pe.device, dtype=pe.dtype), requires_grad=False
            )
            visual.image_size = (resolution, resolution)
            if hasattr(visual, "grid_size"):
                visual.grid_size = (target_grid, target_grid)
            patched = True
    elif visual is not None and hasattr(visual, "trunk"):
        # SigLIP and other open_clip TimmModel-wrapped ViTs have a different
        # PE layout (timm.VisionTransformer.pos_embed) that the OpenAI/CLIP
        # branch above does not handle. Try the timm-specific resize path.
        patched = _patch_timm_visual(visual, resolution)
    elif (
        visual is not None
        and getattr(visual, "attnpool", None) is not None
        and getattr(visual.attnpool, "positional_embedding", None) is not None
    ):
        # CLIP RN50 (ModifiedResNet) has its positional embedding inside the
        # final AttentionPool2d rather than at the top of `visual`. The layout
        # is still (grid^2 + 1, D) so the existing ViT interpolation helper
        # works on it directly. The "patch size" is the ResNet effective stride
        # from input pixels to the spatial output (e.g. 32 for CLIP RN50:
        # 224 -> 7x7 spatial). We derive that stride from the native image_size
        # and the PE grid shape rather than hard-coding 32.
        import torch
        attnpool = visual.attnpool
        ap_pe = attnpool.positional_embedding
        n = ap_pe.shape[0]
        current_grid = int(round((n - 1) ** 0.5))
        if current_grid > 0 and current_grid * current_grid + 1 == n:
            native_size = getattr(visual, "image_size", None)
            if isinstance(native_size, (tuple, list)):
                native_size = int(native_size[0])
            if isinstance(native_size, int) and native_size > 0 and native_size % current_grid == 0:
                stride = native_size // current_grid
                if resolution % stride == 0:
                    target_grid = resolution // stride
                    if target_grid != current_grid:
                        new_pe = _interpolate_vit_pos_embed(ap_pe.detach(), target_grid)
                        attnpool.positional_embedding = torch.nn.Parameter(
                            new_pe.to(ap_pe.device, dtype=ap_pe.dtype),
                            requires_grad=False,
                        )
                        visual.image_size = (resolution, resolution)
                        patched = True
                    elif resolution != native_size:
                        # current_grid already matches; still rebuild preprocess
                        # if the requested resolution differs from native_size
                        # (handled implicitly below if patched stays False).
                        visual.image_size = (resolution, resolution)
                        patched = True

    if patched:
        normalize_tf = preprocess.transforms[-1]
        preprocess = open_clip_module.image_transform(
            image_size=resolution,
            is_train=False,
            mean=tuple(normalize_tf.mean),
            std=tuple(normalize_tf.std),
        )

    model.eval()
    tokenizer = open_clip_module.get_tokenizer(spec.model_name)
    return model, tokenizer, preprocess


def normalize(features):
    return features / features.norm(dim=-1, keepdim=True)


def count_parameters(model) -> int:
    return sum(parameter.numel() for parameter in model.parameters())


def patch_size(model) -> int | None:
    visual = getattr(model, "visual", None)
    direct_patch_size = getattr(visual, "patch_size", None)
    if isinstance(direct_patch_size, tuple):
        return int(direct_patch_size[0])
    if isinstance(direct_patch_size, int):
        return direct_patch_size

    # timm-wrapped trunks (SigLIP via TimmModel) expose patch_size on patch_embed
    trunk = getattr(visual, "trunk", None)
    if trunk is not None:
        patch_embed = getattr(trunk, "patch_embed", None)
        embed_ps = getattr(patch_embed, "patch_size", None)
        if isinstance(embed_ps, (tuple, list)):
            return int(embed_ps[0])
        if isinstance(embed_ps, int):
            return embed_ps

    # CLIP RN50 (ModifiedResNet): no ViT-style patch_size; derive the effective
    # ResNet stride from native image_size / attnpool PE grid (e.g. 224 / 7 = 32).
    attnpool = getattr(visual, "attnpool", None)
    if attnpool is not None and getattr(attnpool, "positional_embedding", None) is not None:
        ap_pe = attnpool.positional_embedding
        n = ap_pe.shape[0]
        grid = int(round((n - 1) ** 0.5))
        native_size = getattr(visual, "image_size", None)
        if isinstance(native_size, (tuple, list)):
            native_size = int(native_size[0])
        if (
            isinstance(native_size, int)
            and grid > 0
            and grid * grid + 1 == n
            and native_size % grid == 0
        ):
            return native_size // grid

    if not hasattr(visual, "positional_embedding"):
        return None
    conv1 = getattr(visual, "conv1", None)
    kernel_size = getattr(conv1, "kernel_size", None)
    if isinstance(kernel_size, tuple):
        return int(kernel_size[0])
    return None


def visual_token_count(model, resolution: int) -> int | None:
    size = patch_size(model)
    if not size or resolution % size:
        return None
    grid = resolution // size
    return grid * grid + 1


def model_metadata(model, spec: ModelSpec, resolution: int) -> dict[str, Any]:
    params = count_parameters(model)
    tokens = visual_token_count(model, resolution)
    proxy = params * tokens if tokens else ""
    return {
        "model_id": spec.model_id,
        "model_name": spec.model_name,
        "pretrained": spec.pretrained,
        "architecture_family": spec.architecture_family,
        "visual_architecture": spec.visual_architecture,
        "scale": spec.scale,
        "native_resolution": spec.native_resolution,
        "resolution": resolution,
        "pixel_count": resolution * resolution,
        "parameter_count": params,
        "parameter_count_m": params / 1e6,
        "patch_size": patch_size(model) or "",
        "visual_tokens": tokens or "",
        "parameter_token_proxy": proxy,
    }


def finite_or_blank(value: float) -> float | str:
    return value if math.isfinite(value) else ""
