"""
从普通位图（PNG/JPEG 等）离线生成与推理接口兼容的双尺度 H5（features + coords）。

说明：
- 训练数据通常为病理切片经专用编码器（如 CONCH）得到的 512 维特征；此处使用 ImageNet 预训练 ResNet50
  提取 2048 维后截取前 512 维作为近似，仅用于「上传图像 → 可走通预测链路」的演示。
- 双尺度：对原图与 0.5× 线分辨率版本分别划窗取块，保证两路 patch 数量一致供 MIL 拼接。
"""

from __future__ import annotations

import os
from typing import Any

import h5py
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFile
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights

ImageFile.LOAD_TRUNCATED_IMAGES = True

_RESNET: nn.Module | None = None


def _get_resnet(device: torch.device) -> tuple[nn.Module, transforms.Compose]:
    global _RESNET
    weights = ResNet50_Weights.IMAGENET1K_V1
    if _RESNET is None:
        m = models.resnet50(weights=weights)
        m.fc = nn.Identity()
        m.eval()
        _RESNET = m
    tfm = weights.transforms()
    _RESNET = _RESNET.to(device)
    return _RESNET, tfm


def _prepare_base_image(path: str, max_side: int = 4096) -> Image.Image:
    im = Image.open(path).convert("RGB")
    w, h = im.size
    m = max(w, h)
    if m > max_side:
        scale = max_side / float(m)
        im = im.resize((int(w * scale), int(h * scale)), Image.Resampling.BILINEAR)
    return im


def _iter_grid_patches(im: Image.Image, patch_size: int, stride: int) -> list[tuple[int, int, Image.Image]]:
    w, h = im.size
    out: list[tuple[int, int, Image.Image]] = []
    if w < patch_size or h < patch_size:
        # 过小则整体缩放到 patch_size
        im2 = im.resize((patch_size, patch_size), Image.Resampling.BILINEAR)
        return [(0, 0, im2)]
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            out.append((x, y, im.crop((x, y, x + patch_size, y + patch_size))))
    if not out:
        im2 = im.resize((patch_size, patch_size), Image.Resampling.BILINEAR)
        return [(0, 0, im2)]
    return out


def build_dual_scale_h5_from_image(
    image_path: str,
    out_h5_20: str,
    out_h5_10: str,
    *,
    patch_size: int = 256,
    stride: int = 256,
    max_patches: int = 192,
    max_side: int = 4096,
    device: str | None = None,
) -> dict[str, Any]:
    """
    生成两个 H5：「20×」「10×」 surrogate —— 分别来自全分辨率与半分辨率整图划窗。
    返回元信息（patch 数、尺寸等）供 API 写入响应。
    """
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, tfm = _get_resnet(dev)

    base = _prepare_base_image(image_path, max_side=max_side)
    w0, h0 = base.size
    low = base.resize((max(1, w0 // 2), max(1, h0 // 2)), Image.Resampling.BILINEAR)

    # 低分辨率分支使用一半大小的 patch，使感受野与尺度差异更合理
    ps_h = patch_size
    ps_l = max(64, patch_size // 2)
    st_h = stride
    st_l = max(32, stride // 2)

    patches_h = _iter_grid_patches(base, ps_h, st_h)
    patches_l = _iter_grid_patches(low, ps_l, st_l)
    k = min(len(patches_h), len(patches_l), max_patches)
    patches_h = patches_h[:k]
    patches_l = patches_l[:k]

    def run_batch(patches: list[tuple[int, int, Image.Image]]) -> tuple[np.ndarray, np.ndarray]:
        feats: list[np.ndarray] = []
        coords: list[list[float]] = []
        batch_imgs: list[Any] = []
        batch_coords: list[tuple[int, int]] = []

        def flush():
            nonlocal batch_imgs, batch_coords, feats, coords
            if not batch_imgs:
                return
            batch = torch.stack(batch_imgs, dim=0).to(dev)
            with torch.no_grad():
                z = model(batch)
            z = z[:, :512].detach().cpu().numpy().astype(np.float32)
            feats.append(z)
            for (xc, yc) in batch_coords:
                coords.append([float(xc), float(yc)])
            batch_imgs = []
            batch_coords = []

        bs = 16
        for (xc, yc, crop) in patches:
            t = tfm(crop)
            batch_imgs.append(t)
            batch_coords.append((xc, yc))
            if len(batch_imgs) >= bs:
                flush()
        flush()

        f_arr = np.vstack(feats) if feats else np.zeros((0, 512), dtype=np.float32)
        c_arr = np.array(coords, dtype=np.float32) if coords else np.zeros((0, 2), dtype=np.float32)
        return f_arr, c_arr

    f20, c20 = run_batch(patches_h)
    f10, c10 = run_batch(patches_l)

    for _p in (out_h5_20, out_h5_10):
        _d = os.path.dirname(os.path.abspath(_p))
        if _d:
            os.makedirs(_d, exist_ok=True)

    with h5py.File(out_h5_20, "w") as f:
        f["features"] = f20
        f["coords"] = c20
        f.attrs["source"] = "raster_resnet50_slice512"
        f.attrs["scale"] = "high"

    with h5py.File(out_h5_10, "w") as f:
        f["features"] = f10
        f["coords"] = c10
        f.attrs["source"] = "raster_resnet50_slice512"
        f.attrs["scale"] = "low"

    return {
        "patchCount": int(f20.shape[0]),
        "imageSize": [w0, h0],
        "encoder": "torchvision.resnet50_imagenet_slice512",
    }


def raster_image_to_tiff(src_path: str, dst_tiff_path: str) -> None:
    """将位图保存为未金字塔的单层 TIFF（内部 manifest 登记用；可被 OpenSlide 部分环境读取）。"""
    os.makedirs(os.path.dirname(dst_tiff_path) or ".", exist_ok=True)
    im = Image.open(src_path).convert("RGB")
    im.save(dst_tiff_path, format="TIFF", compression="tiff_lzw")
