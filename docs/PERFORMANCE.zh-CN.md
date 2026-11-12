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

## 硬件与内存

| 芯片 | 默认计算后端 | 真机验证 |
| --- | --- | --- |
| M5 系列 | W8A8 Metal TensorOps | M5 Pro 48 GB 已完成文生图、单图和双图编辑；M5 MacBook Air 16 GB 已完成文生图和单图编辑，见[实测记录](../BENCHMARKS.md#community-run-macbook-air-m5-16-gb) |
| M1–M4 | 通用 MLX | 已在 M5 Pro 上强制运行通用后端；M1–M4 待测 |
| M6 系列 | W8A8 Metal TensorOps | 待测 |

文本编码、六步去噪和图像解码各用一个进程，完成后释放权重。编辑时，参考图还会经过单独的 VAE 编码阶段。

显存和系统内存由 Apple Silicon 共享。为其他应用留出空间，可在生成命令中加 `--dit-mode stream`，按块读取去噪权重。内存充足时，`--dit-mode resident` 让权重在六步之间常驻，以缩短耗时。默认的 `auto` 在通用后端、机器内存低于 32 GiB 或输入序列超过 8,192 个 token 位置时选择逐块读取。

要对比两条计算路径，可以在 M5 Pro 上分别传入 `--backend w8a8` 和 `--backend portable`。运行参数与逐像素对照结果见[验收记录](../VALIDATION.md)。
