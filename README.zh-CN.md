![蓝色颗粒渐变上的 H3 Image MLX 液体玻璃字](assets/hero.jpg)

<p align="center"><em>这张头图通过 <a href="https://github.com/Anionex/h3-image-mlx">H3 Image MLX</a> 项目调用 MiniMax H3 Image 端到端生成。</em></p>

# H3 Image MLX

[English](README.md) · **简体中文**

[![在 X 上关注 @anion_ex](https://img.shields.io/static/v1?label=&message=%40anion_ex&color=171717&style=flat-square&logo=x&logoColor=white)](https://x.com/anion_ex)
[![PyPI 版本](https://img.shields.io/pypi/v/h3-image-mlx?include_prereleases&style=flat-square)](https://pypi.org/project/h3-image-mlx/)
[![需要 Python 3.11 或更高版本](https://img.shields.io/static/v1?label=Python&message=3.11%2B&color=3776AB&style=flat-square&logo=python&logoColor=white)](pyproject.toml)
[![软件包许可：Apache-2.0 AND MIT](https://img.shields.io/static/v1?label=License&message=Apache-2.0%20AND%20MIT&color=326790&style=flat-square)](pyproject.toml)

[快速开始](#生成第一张图) · [参考图编辑](#用参考图编辑) · [推理原理](#推理原理) · [实测数据](#实测数据)

在 Apple Silicon Mac 上用 MiniMax H3 Image 生图、编辑图片。以尽可能快的速度推理；能够在最低7G峰值内存下生成4K超高清图像；提供命令行和 Python API。开箱即用。

如果这个仓库对你有用，欢迎 🌟 & fork~

## 亮点

- **Apple Silicon 原生推理。** 基于 MLX，充分利用 Mac 算力，在 Mac 本地运行 H3 Image，支持文生图和多图编辑。
- **W8A8 量化与 Metal 加速。** 降低权重内存占用，提供通用 MLX 推理路径，并在支持的芯片上使用 W8A8 Metal TensorOps 加速。M1–M4 兼容。
- **分阶段推理，按需分块加载。** 各阶段通过独立进程释放内存，去噪权重可逐块读取，大大减少峰值内存占用，为其他应用留出更多内存。

作为参考，实测中 1024×1024 生图的推理进程峰值占用可从 **21.07 GB 降至 2.67 GB**，对应耗时中位数从 **22.31 秒增至 30.90 秒**。[测试条件与完整数据](#实测数据)

### 实测数据

| 设备 / 内存 | 推理方案 | DiT 模式 | 分辨率 | 耗时（秒） | 进程峰值（GB） | 来源 |
| --- | --- | --- | --- | ---: | ---: | --- |
| <sub>M5 Pro MacBook / 48 GB</sub> | H3 Image MLX | `resident` | 512×512 | 16.50 | 20.56 | [Anionex](benchmarks/raw/teapot-512.json) |
| <sub>M5 Pro MacBook / 48 GB</sub> | H3 Image MLX | `resident` | 1024×1024 | 22.31 | 21.07 | [Anionex](benchmarks/raw/teapot-1024-ab.json) |
| <sub>M5 Pro MacBook / 48 GB</sub> | H3 Image MLX | `stream` | 1024×1024 | **30.90** | **2.67** | [Anionex](benchmarks/raw/teapot-1024-ab.json) |
| <sub>M5 Pro MacBook / 48 GB</sub> | H3 Image MLX | `resident` | 2048×2048 | 65.54 | 22.04 | [Anionex](benchmarks/raw/teapot-2048.json) |
| <sub>M5 Pro MacBook / 48 GB</sub> | H3 Image MLX | `stream` | 4096×4096 | **356.84** | **6.86** | [Anionex](benchmarks/raw/teapot-4096.json) |
| <sub>M5 MacBook Air / 16 GB</sub> | H3 Image MLX | `stream` | 1024×1024 | **70.51** | **2.48** | [arielfikru](benchmarks/raw/teapot-1024-m5-air-16gb.json) |
| <sub>M5 MacBook Air / 16 GB</sub> | H3 Image MLX | `stream` | 2048×2048 | **240.78** | **2.78** | [arielfikru](benchmarks/raw/teapot-2048-m5-air-16gb.json) |
| <sub>M5 Pro Mac mini / 64 GB</sub> | ComfyUI + MPS INT8 | — | 1024×1024 | 89.55 | 31.42 | [pratikgx](https://huggingface.co/datasets/pratikgx/minimax-h3-image-m5-pro-mac-mini-bench) |
| <sub>M5 Pro Mac mini / 64 GB</sub> | ComfyUI + MPS INT8 | — | 2048×2048 | 164.90 | 36.21 | [pratikgx](https://huggingface.co/datasets/pratikgx/minimax-h3-image-m5-pro-mac-mini-bench) |
| <sub>M5 Pro Mac mini / 64 GB</sub> | ComfyUI + MPS INT8 | — | 4096×4096 | 547.31 | 48.78 | [pratikgx](https://huggingface.co/datasets/pratikgx/minimax-h3-image-m5-pro-mac-mini-bench) |

<sub>耗时为三轮中位数或单次结果；内存为进程物理占用峰值，单位为十进制 GB。ComfyUI 使用 ComfyUI-MPS-INT8 插件。测试次数、环境与测量方法见[完整记录](BENCHMARKS.md)。</sub>

### 雨夜修表铺

[![修表师用镊子夹起一枚黄铜齿轮，下方是敞开的机芯，身后是挂着雨珠的橱窗。](assets/watchmaker-preview.jpg)](assets/watchmaker-4096.png)

[打开 4096×4096 原图（PNG，22.7 MB）](assets/watchmaker-4096.png?raw=true)

在 M5 Pro（48 GB）上生成这张图耗时 6 分 10 秒，推理进程物理内存峰值为 6.88 GB。

## 生成第一张图

使用支持 Metal 的 Apple Silicon Mac 和 Python 3.11 或更高版本。首次下载模型需要 49.30 GB 磁盘空间。模型的适用地区和使用条件见 [MiniMax 模型许可](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/main/LICENSE)。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install 'h3-image-mlx==0.1.0a1'

h3-image-mlx fetch --models ./models
h3-image-mlx generate --models ./models --output teapot.png \
  --prompt 'A yellow enamel teapot and a white cup on a blue wooden table.'
```

默认输出 1024×1024 的 RGB PNG。用 `--width` 和 `--height` 指定宽高，每边可取 256 到 4096 之间的 32 的倍数。

## 用参考图编辑

假设要把 `teapot.png` 里的白杯换成自己照片里的鸟，把鸟的照片保存为 `bird.png`。提示词里的 `<Picture 1>` 和 `<Picture 2>` 对应命令行传入参考图的顺序。

```bash
h3-image-mlx generate --models ./models --output edit.png \
  --reference teapot.png --reference bird.png \
  --prompt 'Edit <Picture 1>. Replace the white cup with the bird from <Picture 2>. Keep the teapot and table.'
```

最多可传入 16 张参考图。M5 Pro 已完成单图和双图编辑实测；16 图测试在六步去噪完成第三步后取消。[验收记录](VALIDATION.md)

## Python

`generate` 返回 Pillow 图像。用 `references` 列表传入参考图，顺序与命令行一致。

```python
from h3_image_mlx import generate

image = generate(
    "Replace the cup in <Picture 1> with the bird from <Picture 2>.",
    models="./models",
    references=["teapot.png", "bird.png"],
    seed=42,
    dit_mode="stream",
)
image.save("edit.png")
```

## 复用已有权重

将 `--models` 指向包含以下文件的目录，也可以用符号链接指向已有文件。

```text
diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors
text_encoders/qwen3vl_32b_minimax_h3_int8_convrot.safetensors
vae/minimax_h3_image_vae_fp16.safetensors
vae/config.json
tokenizer.json
```

`fetch` 会校验已有文件，并从 ModelScope 下载缺失的文件。如果已经有三个权重文件和 tokenizer，只需再下载 1,826 字节的 VAE 配置。要从 Hugging Face 下载，加上 `--source huggingface`。要按 [SHA-256 清单](src/h3_image_mlx/weights.json) 校验文件，运行 `h3-image-mlx verify --models ./models`。

## 推理原理

管线从三个层面降低内存占用：模型阶段之间分开加载，去噪过程中按需读取权重，图像解码时分块处理。

![H3 Image MLX 推理流程：编码提示词、按块读取去噪权重、重叠分块解码图像。每个阶段完成后释放模型内存，再进入下一阶段。](assets/how-it-works.jpg)

- **一次运行一个阶段。** 文本编码、去噪和 VAE 解码分别在独立进程中执行，保存中间结果后退出，再由下一阶段加载权重，三个模型无需同时驻留内存。图片编辑会在去噪前增加参考图的 VAE 编码阶段。
- **量化线性层，减少计算与存储开销。** 直接读取 Comfy-Org 的 INT8 ConvRot 权重，激活值经过旋转变换和动态 INT8 量化后参与计算。支持的芯片使用 W8A8 Metal TensorOps；通用后端通过标准 MLX 分块运算计算同样的量化投影。其他运算保留适合各自用途的浮点精度。
- **去噪权重按需读取。** `stream` 模式随着 Transformer 各层执行，从 safetensors 读取所需权重，不把全部权重缓存到六步结束。去噪仍处理整张图的潜空间序列；`resident` 模式则缓存权重，减少重复读取。管线还会根据 Comfy-Org 裁剪权重中的压缩曲线，预计算各步的 AdaLN 调制值，随后释放准备阶段的权重。
- **重叠分块解码。** VAE 每次解码一小组潜空间图块，将重叠边缘融合成完整图片，控制高分辨率下解码阶段的工作内存。

分块读取用额外的权重读取时间换取更低的内存占用。峰值还包含激活值、注意力和图像缓冲区，因此也会随分辨率和参考图数量变化。具体取舍见[实测数据](#实测数据)。

## 硬件与内存

| 芯片 | 默认计算后端 | 真机验证 |
| --- | --- | --- |
| M5 系列 | W8A8 Metal TensorOps | M5 Pro 48 GB 已完成文生图、单图和双图编辑；M5 MacBook Air 16 GB 已完成文生图和单图编辑，见[实测记录](BENCHMARKS.md#community-run-macbook-air-m5-16-gb) |
| M1–M4 | 通用 MLX | 已在 M5 Pro 上强制运行通用后端；M1–M4 待测 |
| M6 系列 | W8A8 Metal TensorOps | 待测 |

文本编码、六步去噪和图像解码各用一个进程，完成后释放权重。编辑时，参考图还会经过单独的 VAE 编码阶段。

显存和系统内存由 Apple Silicon 共享。为其他应用留出空间，可在生成命令中加 `--dit-mode stream`，按块读取去噪权重。内存充足时，`--dit-mode resident` 让权重在六步之间常驻，以缩短耗时。默认的 `auto` 在通用后端、机器内存低于 32 GiB 或输入序列超过 8,192 个 token 位置时选择逐块读取。

要对比两条计算路径，可以在 M5 Pro 上分别传入 `--backend w8a8` 和 `--backend portable`。运行参数与逐像素对照结果见[验收记录](VALIDATION.md)。

## 致谢与许可

MiniMax 开发了 [H3 Image](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image) 并提供 [VAE 配置](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image/blob/492fd8c4eca666f1085f57e30738c1e0d86b3439/vae/config.json)。[Comfy-Org](https://huggingface.co/Comfy-Org/MiniMax-H3-Image) 提供量化权重。三份 W8A8 Metal 内核源码来自 [mlx-serve](https://github.com/ddalcu/mlx-serve/tree/9191fbdf697df333cac19ec7a93eaa1098ce3ad5/src/kernels)，沿用其 [MIT 许可](src/h3_image_mlx/kernels/LICENSE)。本项目编写的运行时代码采用 [Apache-2.0 许可](LICENSE)。

## 关于作者

我是 [anionex](https://anionex.me/)，一名 AI 原生开发者，曾上榜 GitHub 全球开发者趋势榜第 3 名，总 Star 数超过 18.7k。如果你想了解我后续的更多工作，欢迎在 [X](https://x.com/anion_ex) 或 [GitHub](https://github.com/Anionex) 关注我～
