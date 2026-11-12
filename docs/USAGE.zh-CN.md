# 使用指南

[README](../README.zh-CN.md) · [English](USAGE.md) · **简体中文** · [性能调优](PERFORMANCE.zh-CN.md)

## 安装与下载

使用支持 Metal 的 Apple Silicon Mac 和 Python 3.11 或更高版本。首次下载模型需要 49.30 GB 磁盘空间。

```bash
python -m pip install --upgrade 'h3-image-mlx==0.1.0a2'
h3-image-mlx fetch --models ./models
```

默认从 ModelScope 下载；在 `fetch` 后加 `--source huggingface` 可切换来源。下载支持断点续传。模型权重和 VAE 配置单独下载，不包含在 Python 软件包里。

## 生图与编辑

```bash
h3-image-mlx generate --models ./models --output teapot.png \
  --prompt 'A yellow enamel teapot and a white cup on a blue wooden table.'

h3-image-mlx generate --models ./models --output edit.png \
  --reference teapot.png --reference bird.png \
  --prompt 'Edit <Picture 1>. Replace the white cup with the bird from <Picture 2>. Keep the teapot and table.'
```

参考图按顺序编号：`<Picture 1>` 对应第一个 `--reference`，以此类推。最多传入 16 张 PNG、JPEG 或 WebP 图片。输出始终为 RGB PNG。默认使用六步采样；GitHub `main` 另提供通过环境变量开启的[实验性少步数 LoRA](PERFORMANCE.zh-CN.md#实验性少步数-lora)，PyPI `0.1.0a2` 尚未包含。

## 命令行参数

| 参数 | 默认值 | 用途 |
| --- | --- | --- |
| `--models` | 必填 | `fetch`、`verify` 和 `generate` 使用的模型目录 |
| `--prompt` | 必填 | 非空的生图或编辑指令 |
| `--output` | 必填 | 以 `.png` 结尾的输出路径 |
| `--width`、`--height` | 各为 `1024` | 256 到 4096 之间的 32 的倍数 |
| `--seed` | `42` | 非负整数 |
| `--reference` | 无 | 每张参考图传一次，与提示词顺序对应 |
| `--backend` | `auto` | `auto`、`w8a8` 或 `portable` |
| `--dit-mode` | `auto` | `auto`、`stream` 或 `resident` |
| `--source`（仅 `fetch`） | `modelscope` | `modelscope` 或 `huggingface` |

需要低内存占用时选择 `--dit-mode stream`。推荐尝试[可选的分块推理加速](PERFORMANCE.zh-CN.md#推荐的分块推理加速)，尤其是 16 GB 的 M5 Mac。硬件选择和性能环境变量也集中在该文档中。

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

`fetch` 会校验已有文件，并从 ModelScope 下载缺失的文件。如果已经有三个权重文件和 tokenizer，只需再下载 1,826 字节的 VAE 配置。要从 Hugging Face 下载，加上 `--source huggingface`。要按 [SHA-256 清单](../src/h3_image_mlx/weights.json) 校验文件，运行 `h3-image-mlx verify --models ./models`。
