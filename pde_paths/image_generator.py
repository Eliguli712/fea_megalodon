# #!/usr/bin/env python
# # -*- coding: utf-8 -*-

# """
# 通用图片生成脚本（文生图 + 图生图）

# 示例用法：

# 1. 文生图（仅文字）：
#    python image_generator.py \
#        --prompt "a gothic glass tower floating over a mirrored sea, dark fantasy, high detail" \
#        --output out_text2img.png

# 2. 图生图（根据图片+文字修改）：
#    python image_generator.py \
#        --prompt "turn this character into a stained-glass witch saint with broken wings" \
#        --init_image input.png \
#        --strength 0.6 \
#        --output out_img2img.png
# """

# import argparse
# import torch
# from PIL import Image
# from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline


# def load_pipelines(model_id: str, device: torch.device, dtype):
#     """
#     加载 text2img 和 img2img 两个 pipeline（按需要再用）。
#     为了省显存，你也可以只加载其中一个。
#     """
#     print(f"Loading model: {model_id}")
#     text2img = StableDiffusionPipeline.from_pretrained(
#         model_id,
#         torch_dtype=dtype,
#         safety_checker=None,   # 如需开启 NSFW 过滤，可改为默认或自定义
#     ).to(device)

#     img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
#         model_id,
#         torch_dtype=dtype,
#         safety_checker=None,
#     ).to(device)

#     return text2img, img2img


# def load_image(path: str, target_width: int = None, target_height: int = None) -> Image.Image:
#     """
#     读取一张图片，并可选缩放到指定大小（宽高最好是 64 的倍数，如 512x512）。
#     """
#     img = Image.open(path).convert("RGB")
#     if target_width is not None and target_height is not None:
#         img = img.resize((target_width, target_height), Image.LANCZOS)
#     return img


# def generate(
#     prompt: str,
#     output_path: str,
#     init_image_path: str = None,
#     model_id: str = "runwayml/stable-diffusion-v1-5",
#     steps: int = 30,
#     guidance_scale: float = 7.5,
#     strength: float = 0.6,
#     width: int = 512,
#     height: int = 512,
#     seed: int = None,
#     use_cpu: bool = False,
# ):
#     """
#     核心生成函数：
#     - 若 init_image_path 为 None：文生图
#     - 若有 init_image_path：图生图（在原图基础上按 prompt 改）
#     """
#     # 设备与精度
#     if torch.cuda.is_available() and not use_cpu:
#         device = torch.device("cuda")
#         dtype = torch.float16
#     else:
#         device = torch.device("cpu")
#         dtype = torch.float32

#     # 随机种子（可选）
#     generator = None
#     if seed is not None:
#         generator = torch.Generator(device=device).manual_seed(seed)

#     # 加载 pipelines
#     text2img_pipe, img2img_pipe = load_pipelines(model_id, device, dtype)

#     # 真正生成
#     if init_image_path is None:
#         # 文生图
#         print("Mode: text → image")
#         image = text2img_pipe(
#             prompt=prompt,
#             num_inference_steps=steps,
#             guidance_scale=guidance_scale,
#             width=width,
#             height=height,
#             generator=generator,
#         ).images[0]
#     else:
#         # 图生图
#         print("Mode: image + text → image")
#         init_img = load_image(init_image_path, width, height)
#         image = img2img_pipe(
#             prompt=prompt,
#             image=init_img,
#             strength=strength,           # 0.3 保留原图多，0.7 变化更大
#             num_inference_steps=steps,
#             guidance_scale=guidance_scale,
#             generator=generator,
#         ).images[0]

#     # 保存
#     image.save(output_path)
#     print(f"Saved image to: {output_path}")


# def parse_args():
#     parser = argparse.ArgumentParser(description="通用文字/图片 条件生成图像脚本")

#     parser.add_argument(
#         "--prompt",
#         type=str,
#         required=True,
#         help="文字描述（提示词）",
#     )
#     parser.add_argument(
#         "--output",
#         type=str,
#         default="output.png",
#         help="输出图像路径（例如 output.png）",
#     )
#     parser.add_argument(
#         "--init_image",
#         type=str,
#         default=None,
#         help="可选：作为图生图输入的原始图片路径（不填则进行文生图）",
#     )
#     parser.add_argument(
#         "--model_id",
#         type=str,
#         default="runwayml/stable-diffusion-v1-5",
#         help="HuggingFace 上的 Diffusers 文生图模型 ID",
#     )
#     parser.add_argument(
#         "--steps",
#         type=int,
#         default=30,
#         help="采样步数（越高越精细，但越慢）",
#     )
#     parser.add_argument(
#         "--guidance_scale",
#         type=float,
#         default=7.5,
#         help="CFG scale，越高越贴合 prompt，但容易过拟合或出现伪影",
#     )
#     parser.add_argument(
#         "--strength",
#         type=float,
#         default=0.6,
#         help="图生图强度：0.0~1.0，越高改动越大",
#     )
#     parser.add_argument(
#         "--width",
#         type=int,
#         default=512,
#         help="生成图宽度（建议为 64 的倍数）",
#     )
#     parser.add_argument(
#         "--height",
#         type=int,
#         default=512,
#         help="生成图高度（建议为 64 的倍数）",
#     )
#     parser.add_argument(
#         "--seed",
#         type=int,
#         default=None,
#         help="随机种子（相同参数+种子 → 可复现结果）",
#     )
#     parser.add_argument(
#         "--cpu",
#         action="store_true",
#         help="强制使用 CPU（没有 GPU 时可以加上）",
#     )

#     return parser.parse_args()


# if __name__ == "__main__":
#     args = parse_args()

#     generate(
#         prompt=args.prompt,
#         output_path=args.output,
#         init_image_path=args.init_image,
#         model_id=args.model_id,
#         steps=args.steps,
#         guidance_scale=args.guidance_scale,
#         strength=args.strength,
#         width=args.width,
#         height=args.height,
#         seed=args.seed,
#         use_cpu=args.cpu,
#     )

#!/usr/bin/env python
# -*- coding: utf-8 -*-

#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import torch
from PIL import Image
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline


def load_pipelines(model_id: str, device: torch.device, dtype):
    """
    加载 text2img 和 img2img 两个 pipeline。
    """
    print(f"[+] Loading model: {model_id} on {device} (dtype={dtype})")

    text2img = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,   # 关闭 safety checker
    ).to(device)
    text2img.safety_checker = None

    img2img = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,
    ).to(device)
    img2img.safety_checker = None

    # 可选：省一点显存
    try:
        text2img.enable_attention_slicing()
        img2img.enable_attention_slicing()
    except Exception:
        pass

    return text2img, img2img


def load_image(path: str, target_width: int = None, target_height: int = None) -> Image.Image:
    """
    读取一张图片，并可选缩放到指定大小（宽高最好是 64 的倍数，如 512x512）。
    """
    img = Image.open(path).convert("RGB")
    if target_width is not None and target_height is not None:
        img = img.resize((target_width, target_height), Image.LANCZOS)
    return img


def pick_device(use_cpu: bool = False):
    """
    优先 CUDA → MPS → CPU
    注意：MPS 用 float32 更稳，配合 autocast。
    """
    if not use_cpu and torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.float16
        print("[+] Using CUDA GPU")
    elif (
        not use_cpu
        and getattr(torch.backends, "mps", None) is not None
        and torch.backends.mps.is_available()
    ):
        device = torch.device("mps")
        dtype = torch.float32  # 关键：MPS 上用 float32
        print("[+] Using Apple MPS GPU (float32)")
    else:
        device = torch.device("cpu")
        dtype = torch.float32
        print("[+] Using CPU")
    return device, dtype


def generate(
    prompt: str,
    output_path: str,
    init_image_path: str = None,
    model_id: str = "runwayml/stable-diffusion-v1-5",
    steps: int = 30,
    guidance_scale: float = 7.5,
    strength: float = 0.6,
    width: int = 512,
    height: int = 512,
    seed: int = None,
    use_cpu: bool = False,
):
    """
    核心生成函数：
    - 若 init_image_path 为 None：文生图
    - 若有 init_image_path：图生图（在原图基础上按 prompt 改）
    """
    device, dtype = pick_device(use_cpu=use_cpu)

    # 随机种子（可选）
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)
        print(f"[+] Using seed = {seed}")

    # 加载 pipelines
    text2img_pipe, img2img_pipe = load_pipelines(model_id, device, dtype)

    def run_pipe(pipe_call):
        # 只在 CUDA 上用 autocast，MPS/CPU 直接算，避免 NaN
        if device.type == "cuda":
            with torch.autocast("cuda"):
                return pipe_call()
        else:
            return pipe_call()

    if init_image_path is None:
        # 文生图
        print("[+] Mode: text → image")

        def _call():
            return text2img_pipe(
                prompt=prompt,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                width=width,
                height=height,
                generator=generator,
            )

        result = run_pipe(_call)
        image = result.images[0]

    else:
        # 图生图
        print("[+] Mode: image + text → image")
        init_img = load_image(init_image_path, width, height)

        def _call():
            return img2img_pipe(
                prompt=prompt,
                image=init_img,
                strength=strength,           # 0.3 保留原图多，0.7 变化更大
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )

        result = run_pipe(_call)
        image = result.images[0]

    image.save(output_path)
    print(f"[+] Saved image to: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="通用文字/图片 条件生成图像脚本（优先 GPU）")

    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="文字描述（提示词）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output.png",
        help="输出图像路径（例如 output.png）",
    )
    parser.add_argument(
        "--init_image",
        type=str,
        default=None,
        help="可选：作为图生图输入的原始图片路径（不填则进行文生图）",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="runwayml/stable-diffusion-v1-5",
        help="HuggingFace 上的 Diffusers 文生图模型 ID",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=30,
        help="采样步数（越高越精细，但越慢）",
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="CFG scale，越高越贴合 prompt，但容易过拟合或出现伪影",
    )
    parser.add_argument(
        "--strength",
        type=float,
        default=0.6,
        help="图生图强度：0.0~1.0，越高改动越大",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=512,
        help="生成图宽度（建议为 64 的倍数）",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=512,
        help="生成图高度（建议为 64 的倍数）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子（相同参数+种子 → 可复现结果）",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="强制使用 CPU（否则自动优先 CUDA→MPS→CPU）",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    generate(
        prompt=args.prompt,
        output_path=args.output,
        init_image_path=args.init_image,
        model_id=args.model_id,
        steps=args.steps,
        guidance_scale=args.guidance_scale,
        strength=args.strength,
        width=args.width,
        height=args.height,
        seed=args.seed,
        use_cpu=args.cpu,
    )