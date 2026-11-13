# Qwen Image 2.1 Turbo

通过同一套 CLI 和 Python API 调用 H3 与 Qwen。Qwen 使用独立的可选原生运行时，不启动生图 App 或 HTTP 服务；H3 用户不需要安装这部分。此次不包含提示词增强、桌面界面和历史管理。

本库提供文生图、有序多图编辑、图生图、透明输出、LoRA 叠加与独立强度、采样预设、CFG/负向提示词、四档内存模式、真实进度及取消。已完成的功能验证见 [验证记录](../VALIDATION.md)，性能数据见 [实测表](../BENCHMARKS.md)。

## 安装与生成

从 PyPI 一次安装主包与可选原生运行时：

```bash
python -m pip install 'h3-image-mlx[qwen]==0.1.0a5'
h3-image-mlx fetch --model qwen-image-2.1-turbo --models ./qwen-models
h3-image-mlx generate --model qwen-image-2.1-turbo --models ./qwen-models \
  --prompt '白色桌面上的红色陶瓷茶壶。' --output teapot.png \
  --memory-mode micro-fast --progress --metrics metrics.json
```

默认从 ModelScope 下载，`--source huggingface` 可切换来源；使用固定修订、文件大小及 SHA-256 校验。可以直接指定已有权重目录。Q8 模型包约 17.73 GB，W8A8 预处理另需约 7.25 GB 的磁盘文件，准备前要求至少 8 GB 空间。编辑可能额外提取视觉权重。`--no-prepare` 可直接读取原始权重，不生成第二份 DiT。准备耗时与生图耗时分开记录。

原生运行时包含动态库和 Metal 资源，要求 Apple Silicon 与 macOS 26.2 或更新版本。M5/M6 默认使用 W8A8，较早芯片选择通用 MLX 路径；通用路径在 M5 上验证，M1–M4 仍待真机验证。

内存模式为 `resident`、`prefetch`、`micro-fast`、`micro`。尺寸接受 256–4096 范围的 32 倍数；实际验收尺寸见实测表。默认保持 Turbo 八步。

## Python API

```python
from threading import Event
from h3_image_mlx import generate_result, QwenOptions, Lora

cancel = Event()  # 另一线程可以调用 cancel.set()
result = generate_result(
    "白色桌面上的红色陶瓷茶壶。",
    model="qwen-image-2.1-turbo", models="./qwen-models",
    width=1024, height=1024, seed=42,
    qwen=QwenOptions(memory_mode="micro-fast"),
    on_progress=print, cancel=cancel,
)
result.image.save("teapot.png")
print(result.metrics)
```

`generate(...)` 只返回 PIL 图像，原有 H3 调用保持兼容。`runtime=` 可指定独立运行时目录，`cache=` 控制派生文件位置，`prepared=` 可复用经底模身份校验的预处理权重。

`private=True` 通过管道传递提示词与参考图，不写临时提示词/潜变量文件，也不自动保存历史。调用者显式保存图片，以及操作系统交换内存，属于各自管理范围。

## 参考图与 LoRA

```bash
h3-image-mlx generate --model qwen-image-2.1-turbo --models ./qwen-models \
  --reference teapot.png --reference palette.png \
  --prompt '将<image1>的茶壶釉色改成<image2>的蓝色。' \
  --lora ./style.safetensors --lora-scale 0.7 --output edited.png
```

最多十张参考图，依输入顺序对应 `<image1>`、`<image2>`。支持 PNG/JPEG/WebP 与透明通道；蒙版或姿势渲染图作为另一张参考图计数。`--reference-resolution` 控制参考图编码分辨率。`--reference-mode variation --strength 0.5` 用单图进行图生图；`--transparent` 请求 RGBA 输出。

LoRA 可重复传入并分别设置强度，上限八个，包含采样预设的适配器。所有强度省略时默认为 1。推理前校验层名、尺寸和上下投影配对；不覆盖源权重。

| 预设 | 底模与适配器 | 步数 |
|---|---|---:|
| `complete` | 已合并的 Turbo 底模 | 默认 8 |
| `turbo4` / `turbo3` | Turbo 底模及缩短的 sigma 网格 | 4 / 3 |
| `standard` | 原版 Qwen Image 2.1 DiT | `--steps` |
| `lightning` | 原版 DiT + LightX2V LoRA | 4 |
| `dmd2` | 原版 DiT + DMD2 LoRA | 2 |
| `turbo8` | 原版 DiT + 官方 Turbo LoRA | 8 |

```bash
h3-image-mlx fetch-profile --profile lightning --directory ./qwen-presets
h3-image-mlx generate --model qwen-image-2.1-turbo --models ./qwen-models \
  --profile lightning --profile-assets ./qwen-presets --auto-strength \
  --prompt '红色陶瓷茶壶。' --output lightning.png
```

`--profile-strength` 设置预设 LoRA 强度；`--auto-strength` 仅用于 Lightning，采用 App 的自动强度规则。预设可叠加额外 LoRA。下载预设须显式执行；部分第三方适配器只有 Hugging Face 来源。

`--cfg` 与 `--negative-prompt` 提供真实 CFG，CFG=1 时不会因填写负向提示词而偷偷开启额外分支。`--sigmas` 为每个 `--steps` 步指定严格递减的 sigma；不能同时覆盖预设的固定网格。

## 内存与测速

普通八步文生图使用文本编码、去噪、VAE 三个独立进程。编辑、LoRA、CFG、自定义采样及透明输出使用完整请求入口，它们的模型生命周期和峰值内存单独验收。

`--weight-cache uncached` 使用有界、按页对齐的无缓存权重读取；`system` 保留 macOS 默认缓存策略。预处理文件是磁盘上的派生权重，进程内显式预读缓冲区计入进程占用。无缓存 benchmark 还必须检查权重驻留页与实际磁盘读取量，不能仅凭开关或运行前清缓存判定。旧的缓存速度数据只作单独标注的补充。

架构与运行时构建细节见 [ARCHITECTURE](ARCHITECTURE.md) 和[英文使用文档](QWEN.md)。

无缓存读取目前限定于 W8A8、`micro` / `micro-fast` 的默认八步纯文生图。编辑、LoRA、私密管道、通用后端及其他模式会明确拒绝该选项，避免把未验证的读取路径标成无缓存。

Lightning、DMD2 与 Turbo 适配器预设沿用 App 的采样配置，固定 CFG 为 1 并清空负向提示词；需要自行控制 CFG 时使用 complete 或 standard。
