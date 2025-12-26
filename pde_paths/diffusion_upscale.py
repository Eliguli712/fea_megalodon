#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Stable Diffusion x4 超分辨率（CPU 友好版 + 自动压缩输入尺寸）

特性：
- 自动把输入图片的最长边压到 max_side（默认 384），只缩不放大
- 使用 stabilityai/stable-diffusion-x4-upscaler 做 4 倍超分
- 没有 CUDA 时自动走 CPU，刻意跳过 MPS，避免 Apple GPU OOM 问题
"""

import argparse
from pathlib import Path

import torch
from PIL import Image
from diffusers import StableDiffusionUpscalePipeline


def load_and_maybe_resize(path: str, max_side: int | None = None) -> Image.Image:
    """
    读取图片，并按需要把最长边等比例压到 max_side（只缩小，不放大）。
    """
    img = Image.open(path).convert("RGB")
    if max_side is None:
        return img

    w, h = img.size
    longest = max(w, h)
    if longest <= max_side:
        # 已经不大，就不动
        print(f"[+] Input size {w}x{h} <= max_side {max_side}, no resize.")
        return img

    scale = max_side / float(longest)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    print(f"[+] Resizing input from {w}x{h} → {new_w}x{new_h} (max_side={max_side})")
    img = img.resize((new_w, new_h), Image.LANCZOS)
    return img


def pick_device(use_cpu: bool = False):
    """
    设备选择：
    - 有 CUDA 且不强制 CPU：用 CUDA（float16）
    - 否则一律 CPU（float32）
    故意不走 MPS，避免 Apple GPU 报 OOM。
    """
    if not use_cpu and torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.float16
        print("[+] Using device: cuda (GPU, float16)")
    else:
        device = torch.device("cpu")
        dtype = torch.float32
        print("[+] Using device: cpu (float32)")
    return device, dtype


def upscale_image(
    input_path: str,
    output_path: str,
    prompt: str = "",
    model_id: str = "stabilityai/stable-diffusion-x4-upscaler",
    steps: int = 20,
    guidance_scale: float = 7.5,
    seed: int | None = None,
    use_cpu: bool = False,
    max_side: int | None = 384,
):
    """
    用扩散超分模型把低分辨率图片放大并增强清晰度。

    参数：
        input_path:  输入图片路径
        output_path: 输出图片路径
        prompt:      文本提示，留空则偏向“忠实增强 + 稍微锐化”
        steps:       采样步数（CPU 上 10~30 比较友好）
        max_side:    输入图像最长边上限，只缩不放大（默认 384）
    """
    device, dtype = pick_device(use_cpu=use_cpu)

    print(f"[+] Loading model: {model_id}")
    pipe = StableDiffusionUpscalePipeline.from_pretrained(
        model_id,
        dtype=dtype,
    ).to(device)

    try:
        pipe.enable_attention_slicing()
    except Exception:
        pass

    # 载入 & 按需缩放
    low_res = load_and_maybe_resize(input_path, max_side=max_side)

    # 随机种子（可选）
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
        print(f"[+] Using seed = {seed}")

    # 默认提示词
    if not prompt:
        prompt = "high quality, sharp focus, detailed, retain original composition"

    print(f"[+] Prompt: {prompt}")
    print(f"[+] Steps: {steps}, guidance_scale: {guidance_scale}")
    print("[+] Running diffusion upscaling...")

    with torch.no_grad():
        result = pipe(
            prompt=prompt,
            image=low_res,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )

    upscaled = result.images[0]

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    upscaled.save(out_path)
    print(f"[+] Saved upscaled image to: {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Diffusion-based image upscaling (with auto-resize)")

    parser.add_argument("--input", type=str, required=True, help="输入的低清图片路径")
    parser.add_argument("--output", type=str, default="upscaled.png", help="输出图片路径")
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="可选：引导增强风格的文字描述（不填则偏向忠实增强）",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="stabilityai/stable-diffusion-x4-upscaler",
        help="HuggingFace diffusers 上的超分模型 ID",
    )
    parser.add_argument("--steps", type=int, default=20, help="推理步数（CPU 测试建议 10~30）")
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="CFG scale，越高越贴近 prompt，过高易失真",
    )
    parser.add_argument("--seed", type=int, default=None, help="随机种子（方便复现）")
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="强制用 CPU（否则自动优先 CUDA→CPU）",
    )
    parser.add_argument(
        "--max_side",
        type=int,
        default=384,
        help="输入图片最长边上限，只缩不放大（例如 256 或 384）",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    upscale_image(
        input_path=args.input,
        output_path=args.output,
        prompt=args.prompt,
        model_id=args.model_id,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        use_cpu=args.cpu,
        max_side=args.max_side,
    )