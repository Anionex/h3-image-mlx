# 性能调优

[README](../README.zh-CN.md) · [English](PERFORMANCE.md) · **简体中文** · [使用指南](USAGE.zh-CN.md)

## 推荐的分块推理加速

从 `0.1.0a2` 开始，推荐在 `stream` 模式下尝试**同时开启权重预取和异步计算**，尤其是 16 GB 的 M5 Mac。它们**默认关闭**，需要手动开启。

```bash
H3_ASYNC=1 H3_PREFETCH=1 h3-image-mlx generate \
  --models ./models --dit-mode stream --output teapot.png \
  --prompt 'A yellow enamel teapot and a white cup on a blue wooden table.'
```

GPU 计算当前 Transformer 块时，提前读取并准备下一块权重，仍然使用六步采样。异步计算允许最多两块权重同时存活，减少逐块等待。

去掉两个环境变量即可使用默认路径。如果之前在终端中 `export` 过，运行 `unset H3_ASYNC H3_PREFETCH`，或在单次命令中把两个值设为 `0`。命令行和 Python API 都会继承这些环境变量。

## 贡献者实测

来源为 arielfikru 的 [PR #2](https://github.com/Anionex/h3-image-mlx/pull/2)：M5 MacBook Air、16 GB、macOS 26.6.2、MLX 0.32.4、茶壶提示词、seed 42、六步、W8A8 和分块模式。下表仅开启 `H3_ASYNC=1 H3_PREFETCH=1`，没有使用编码缓存。

| 测量项目 | 默认 | 同时开启两个开关 |
| --- | ---: | ---: |
| 1024×1024，整张生成 | 59.8 秒 | 机器休息后 34.8 秒；热机 43.6 秒 |
| 1024×1024，仅 DiT，交错 A/B | 52.3 秒 | 31.3–31.7 秒 |
| 2048×2048，仅 DiT | 144.7 秒 | 124.6 秒 |
| 1024×1024，进程峰值 | 2.5 GB | 2.5 GB |
| 2048×2048，进程峰值 | 2.81 GB | 3.06 GB |

贡献者报告两种分辨率的输出均逐像素一致。整图和 DiT 计时来自不同轮次；2048 对照先跑优化路径，间隔两分钟后跑默认路径。收益受可用内存和统一内存带宽影响；GPU 持续工作也让无风扇 Air 更容易受温度影响。这批优化实测未覆盖 M5 Pro、常驻模式、通用后端、参考图编辑和 4K。

## 环境变量

全部可选，不设置即关闭。

| 变量 | 作用 |
| --- | --- |
| `H3_ASYNC=1` | 用 `mx.async_eval` 排入 DiT 块，让相邻块的工作重叠 |
| `H3_PREFETCH=1` | 工作线程预读下一块的原始权重字节，MLX 调用仍在主线程执行 |
| `H3_EMBED_CACHE=<目录>` | 复用相同提示词和有序参考图的文本阶段输出 |
| `H3_PROFILE=<文件.json>` | 保存 DiT 耗时、权重读取统计和 MLX 分配器峰值 |

### 重复提示词缓存

```bash
H3_EMBED_CACHE=./.h3-embedding-cache h3-image-mlx generate \
  --models ./models --output teapot.png \
  --prompt 'A yellow enamel teapot and a white cup on a blue wooden table.'
```

首轮写入缓存，命中后跳过文本编码。去噪、图像解码和参考图 VAE 编码仍会执行。缓存占用磁盘；并发任务应使用独立缓存目录，更换文本权重或 tokenizer 后应清空或更换目录。移除 `H3_EMBED_CACHE` 可停止使用缓存，删除对应缓存目录可回收空间。

### 性能记录

在生成命令前设置 `H3_PROFILE=./dit-profile.json`。文件记录准备阶段和每步耗时、读取字节数与时间、转换时间、预取等待时间、计算等待时间及 MLX 分配器峰值。重叠阶段的耗时不能直接相加；分配器峰值与进程物理占用是不同指标。需要保留多轮结果时，使用不同文件名。

## 实验性少步数 LoRA

该功能从 `0.1.0a3` 开始随 PyPI 软件包提供。所有实验设置默认关闭；不设置时，生图和编辑仍使用原来的六步链路。升级后即可体验：

```bash
python -m pip install --upgrade 'h3-image-mlx==0.1.0a3'
```

### 直接加载 LoRA

从 [akhaliq/minimax-h3-image-pdmd-2step](https://huggingface.co/akhaliq/minimax-h3-image-pdmd-2step/tree/main) 下载 `minimax_h3_pdmd_2nfe_comfyui.safetensors`，放到 `./loras/`。这是 [PR #3](https://github.com/Anionex/h3-image-mlx/pull/3) 验证过的 ComfyUI 格式转换版，上游原始文件的张量命名不同。LoRA 需要单独下载，`h3-image-mlx fetch` 不会下载它。

```bash
env -u H3_DIT H3_LORA=./loras/minimax_h3_pdmd_2nfe_comfyui.safetensors \
  H3_STEPS=2 H3_SHIFT=12 H3_ASYNC=1 H3_PREFETCH=1 \
  h3-image-mlx generate --models ./models --dit-mode stream \
  --width 1024 --height 1024 --seed 42 --output teapot-2step.png \
  --prompt 'A yellow enamel teapot and a white cup on a blue wooden table.'
```

把 `H3_STEPS` 改为 `3`，并更换输出文件名，即可对比三步效果。[上游 pDMD LoRA](https://huggingface.co/pdmd2026/pdmd_2NFE_lora) 按两步训练，三步是贡献者测试过的推理设置。在他的茶壶示例中，两步在平坦区域留下细密纹理，三步大幅缓解。即使 seed 相同，两者的构图和细节也可能与原版六步不同。

| 设置 | 手动设置后的作用 |
| --- | --- |
| `H3_LORA=<文件>` | 在量化投影旁加载一个 ComfyUI 格式 LoRA |
| `H3_STEPS=2` 或 `3` | 使用偏移后的少步数采样时间表，需搭配对应蒸馏 LoRA 或预合并权重 |
| `H3_SHIFT=12` | 采样时间表偏移；启用少步数但未指定时使用 12 |
| `H3_DIT=<文件>` | 改用张量布局相同的另一份 DiT 权重 |

命令行和 Python API 读取相同的环境变量。运行 `unset H3_LORA H3_STEPS H3_SHIFT H3_DIT` 可恢复原版权重和六步采样；独立的 `H3_ASYNC`、`H3_PREFETCH` 加速开关可以保留。

### 预合并为 INT8 权重

直接加载会在推理时增加低秩计算，不修改现有模型文件。可选的[合并脚本](../scripts/merge_lora_int8.py) 则在 ConvRot 量化前，把 LoRA 加到 BF16 权重中，另存一份 INT8 权重。在本仓库或解压后的源码包目录中，安装好软件包依赖后运行：

```bash
python scripts/merge_lora_int8.py \
  --base ./models/diffusion_models/minimax_h3_image_pruned_int8_convrot.safetensors \
  --lora ./loras/minimax_h3_pdmd_2nfe_comfyui.safetensors \
  --out ./models/diffusion_models/minimax_h3_image_pdmd_merged_int8_convrot.safetensors
```

脚本使用 macOS APFS 克隆，通过 HTTP 读取约 40 GB 的上游 BF16 张量，不保存完整 BF16 下载。克隆逐步改写时会消耗空间，应为新增 INT8 权重预留约 21 GB。可用 `--rev` 固定上游 Hugging Face 版本。输出必须使用独立路径，不能指向原版权重；只在使用相同原版权重、LoRA 和上游版本续跑时复用已有输出。进度保存在输出旁的 `.progress.json` 中，看到 `DONE` 后再用于推理；`--limit` 是开发选项，会留下未完成的权重。

```bash
env -u H3_LORA \
  H3_DIT=./models/diffusion_models/minimax_h3_image_pdmd_merged_int8_convrot.safetensors \
  H3_STEPS=3 H3_SHIFT=12 H3_ASYNC=1 H3_PREFETCH=1 \
  h3-image-mlx generate --models ./models --dit-mode stream \
  --output teapot-merged.png \
  --prompt 'A yellow enamel teapot and a white cup on a blue wooden table.'
```

使用预合并权重时不要再加载同一个 `H3_LORA`，否则会叠加两次。重新量化会让预合并和直接加载的结果存在数值差异。`verify --models` 仍只校验原版固定模型文件，不校验 `H3_DIT` 指向的替代权重。

### 贡献者实测与覆盖范围

[PR #3](https://github.com/Anionex/h3-image-mlx/pull/3) 提供了 M5 MacBook Air 16 GB 上的 1024×1024 茶壶测试：seed 42、W8A8、分块模式，同时开启两个加速开关。

| 模式 | 完整生图耗时 |
| --- | ---: |
| 原版六步 | 34.8–43.6 秒 |
| 直接加载 LoRA，两步 | 23.4 秒 |
| 直接加载 LoRA，三步 | 25.7–28.6 秒 |
| 直接加载 LoRA，三步，命中编码缓存 | 18.7 秒 |

另一次热机且存在 swap 的交错对比中，预合并把两轮 DiT 耗时从 23.9/23.0 秒降至 20.3/18.3 秒；这与上表完整生图计时来自不同测量。实验模式尚无可对比的峰值内存数据。

贡献者只测了 1024×1024 的一个提示词，其他 LoRA、带 LoRA 的参考图编辑、其他分辨率、常驻模式和通用后端仍待验证。我们另行验证了实验设置关闭时的现有生图和编辑链路，见[验收记录](../VALIDATION.md#experimental-lora-integration)。


## 硬件与内存

| 芯片 | 默认计算后端 | 真机验证 |
| --- | --- | --- |
| M5 系列 | W8A8 Metal TensorOps | M5 Pro 48 GB 已完成文生图、单图和双图编辑；M5 MacBook Air 16 GB 已完成文生图和单图编辑，见[实测记录](../BENCHMARKS.md#community-run-macbook-air-m5-16-gb) |
| M1–M4 | 通用 MLX | 已在 M5 Pro 上强制运行通用后端；M1–M4 待测 |
| M6 系列 | W8A8 Metal TensorOps | 待测 |

文本编码、六步去噪和图像解码各用一个进程，完成后释放权重。编辑时，参考图还会经过单独的 VAE 编码阶段。

显存和系统内存由 Apple Silicon 共享。为其他应用留出空间，可在生成命令中加 `--dit-mode stream`，按块读取去噪权重。内存充足时，`--dit-mode resident` 让权重在六步之间常驻，以缩短耗时。默认的 `auto` 在通用后端、机器内存低于 32 GiB 或输入序列超过 8,192 个 token 位置时选择逐块读取。

要对比两条计算路径，可以在 M5 Pro 上分别传入 `--backend w8a8` 和 `--backend portable`。运行参数与逐像素对照结果见[验收记录](../VALIDATION.md)。
