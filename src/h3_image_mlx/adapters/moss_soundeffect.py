"""Staged Apple Silicon inference for MOSS-SoundEffect v2.

Reuses xocialize/moss-soundeffect-mlx's Wan blocks, Qwen3 tokenizer and
encoder components, DAC decoder and FlowMatchScheduler.
"""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import asdict
import json
import math
from pathlib import Path
import time
from typing import Sequence

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from .._stage import process_usage
from ..api import AudioResult
from ..streaming import BlockStream, StreamPolicy, TensorStore


@contextmanager
def _stage(paths, policy, uncached, stages, label, *, repeat=False):
    started = time.perf_counter()
    before = process_usage()
    with TensorStore(paths, uncached=uncached) as store:
        stream = BlockStream(store, policy, wrap=repeat) if policy is not None else nullcontext()
        with stream as blocks:
            yield store, blocks
        after = process_usage()
        stages.append({
            "stage": label,
            "seconds": time.perf_counter() - started,
            "weight_io": store.io,
            "disk_read_bytes": after["disk_read_bytes"] - before["disk_read_bytes"],
            "process_lifetime_peak_bytes": after["peak_bytes"],
            **(blocks.metrics if blocks is not None else {}),
        })


class _SeparateForwards(nn.Module):
    """Reuse the same weights for each branch with the original batch shapes."""

    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def __call__(self, inputs):
        return tuple(self.layer(*arguments) for arguments in inputs)


def _bind(layer, prefix, store, stream=None, *, branches=False):
    bound = _SeparateForwards(layer) if branches else layer
    parameter_prefix = "layer." if branches else ""
    mapping = {}
    for name, parameter in tree_flatten(layer.parameters()):
        key = f"{prefix}.{name}"
        if key not in store:
            raise ValueError(f"MOSS checkpoint is missing tensor {key}")
        shape = tuple(store.spec(key)["shape"])
        if shape != parameter.shape:
            raise ValueError(f"MOSS tensor {key} has shape {shape}, expected {parameter.shape}")
        mapping[parameter_prefix + name] = key
    if not mapping:
        return bound
    if stream is not None:
        return stream.wrap(bound, prefix, mapping)
    values = store.arrays(store.read(mapping.values()))
    bound.load_weights([(name, values[key]) for name, key in mapping.items()])
    return bound


class _CachedCrossAttention(nn.Module):
    """Wan audio cross-attention with immutable, per-request K/V projections."""

    def __init__(self, layer, contexts):
        super().__init__()
        self.q = layer.q
        self.k = layer.k
        self.v = layer.v
        self.o = layer.o
        self.norm_q = layer.norm_q
        self.norm_k = layer.norm_k
        self.attn = layer.attn
        self._contexts = contexts
        self._projected = {}

    def __call__(self, x, branch):
        q = self.norm_q(self.q(x))
        if branch not in self._projected:
            context = self._contexts[branch]
            self._projected[branch] = (self.norm_k(self.k(context)), self.v(context))
        k, v = self._projected[branch]
        return self.o(self.attn(q, k, v))


def _text_contexts(root, prompts, policy, uncached, stages):
    from mlx_lm.models.qwen3 import ModelArgs, Qwen3Model, create_attention_mask
    from moss_sfx_mlx.prompter import WanPrompter

    config = json.loads((root / "text_encoder/config.json").read_text())
    encoder = Qwen3Model(ModelArgs.from_dict(config))
    encoder.eval()
    tokenizer = WanPrompter(tokenizer_path=str(root / "tokenizer")).tokenizer
    tokens = tuple(tokenizer(prompt, return_mask=True, add_special_tokens=True) for prompt in prompts)
    paths = sorted((root / "text_encoder").glob("model*.safetensors"))
    if not paths:
        raise FileNotFoundError(f"No Qwen3 weight shards under {root / 'text_encoder'}")
    with _stage(paths, policy, uncached, stages, "text") as (store, stream):
        embedding = _bind(encoder.embed_tokens, "model.embed_tokens", store, stream, branches=True)
        layers = [
            _bind(layer, f"model.layers.{index}", store, stream, branches=True)
            for index, layer in enumerate(encoder.layers)
        ]
        norm = _bind(encoder.norm, "model.norm", store, stream, branches=True)
        hidden = embedding(tuple((mx.array(ids),) for ids, _ in tokens))
        masks = tuple(create_attention_mask(x, None) for x in hidden)
        for layer in layers:
            hidden = layer(tuple(zip(hidden, masks)))
        hidden = norm(tuple((x,) for x in hidden))
        contexts = tuple(
            (x * (mx.arange(x.shape[1])[None, :, None] < mx.array(mask.sum(axis=1))[:, None, None]))
            .astype(mx.bfloat16)
            for x, (_, mask) in zip(hidden, tokens)
        )
        mx.eval(contexts)
    return contexts


def _denoise(root, index, contexts, initial, steps, cfg, sigma_shift, policy, uncached, stages):
    from moss_sfx_mlx.config import WanAudioModelConfig
    from moss_sfx_mlx.model.wan_audio_dit import WanAudioModel
    from moss_sfx_mlx.model.wan_video_dit import sinusoidal_embedding_1d
    from moss_sfx_mlx.scheduler import FlowMatchScheduler

    dit = WanAudioModel(**asdict(WanAudioModelConfig()))
    if quantization := index.get("mlx_quantization"):
        nn.quantize(
            dit, group_size=int(quantization["group_size"]), bits=int(quantization["bits"]),
            class_predicate=lambda path, module: isinstance(module, nn.Linear) and path.startswith("blocks."),
        )
    scheduler = FlowMatchScheduler(shift=5.0, sigma_min=0.0, extra_one_step=True)
    scheduler.set_timesteps(steps, shift=sigma_shift)
    latents = initial
    frame_count = latents.shape[-1] // dit.patch_size[0]
    freqs = tuple(
        mx.concatenate([pair[i][:frame_count].reshape(frame_count, -1) for pair in dit.freqs], axis=-1)
        .reshape(frame_count, 1, -1)
        for i in range(2)
    )
    with _stage([root / "mlx/dit.safetensors"], policy, uncached, stages, "dit", repeat=True) as (store, stream):
        for name in ("text_embedding", "time_embedding", "time_projection", "patch_embedding", "head"):
            _bind(getattr(dit, name), name, store)
        projected = tuple(dit.text_embedding(context) for context in contexts)
        mx.eval(projected, freqs)
        blocks = []
        for block_id, block in enumerate(dit.blocks):
            block.cross_attn = _CachedCrossAttention(block.cross_attn, projected)
            blocks.append(_bind(block, f"blocks.{block_id}", store, stream, branches=True))

        for step in range(steps):
            timestep = scheduler.timesteps[step]
            t = dit.time_embedding(sinusoidal_embedding_1d(dit.freq_dim, timestep.reshape(1)))
            t_mod = dit.time_projection(t).reshape(t.shape[0], 6, dit.dim)
            x, grid = dit.patchify(latents)
            hidden = (x,) * len(projected)
            for block in blocks:
                hidden = block(tuple((h, branch, t_mod, freqs) for branch, h in enumerate(hidden)))
            pos = dit.unpatchify(dit.head(hidden[0], t), grid)
            noise = pos
            if cfg != 1.0:
                neg = dit.unpatchify(dit.head(hidden[1], t), grid).astype(mx.float32)
                noise = neg + cfg * (pos.astype(mx.float32) - neg)
            latents = scheduler.step(noise, timestep, latents.astype(mx.float32)).astype(mx.bfloat16)
            mx.eval(latents)
    return latents


def _decode_frame_count(output_samples, decoder_rates):
    # Walk the DAC decoders' right receptive field back from the final
    # requested sample. Both stem and final convolutions have radius 3.
    last = output_samples - 1 + 3
    for stride in reversed(decoder_rates):
        # Three residual units have dilation 1, 3 and 9 (kernel size 7).
        # The preceding transpose convolution has padding ceil(stride/2).
        last = (last + 3 * (1 + 3 + 9) + (stride + 1) // 2) // stride
    return last + 3 + 1


def _decode(root, latents, output_samples, policy, uncached, stages):
    from moss_sfx_mlx.config import DACConfig
    from moss_sfx_mlx.model.dac_vae import Decoder

    config = DACConfig()
    decoder = Decoder(config.latent_dim, config.decoder_dim, config.decoder_rates)
    projection = nn.Conv1d(config.latent_dim, config.latent_dim, 1)
    latents = latents[:, :, :_decode_frame_count(output_samples, config.decoder_rates)]
    with _stage([root / "mlx/vae.safetensors"], policy, uncached, stages, "vae") as (store, stream):
        project = _bind(projection, "post_quant_conv", store, stream)
        layers = [
            _bind(layer, f"decoder.model.layers.{index}", store, stream)
            for index, layer in enumerate(decoder.model.layers)
        ]
        audio = project(latents.astype(mx.float32).transpose(0, 2, 1))
        for layer in layers:
            audio = layer(audio)
        audio = audio[:, :output_samples, :].transpose(0, 2, 1)
        mx.eval(audio)
    return audio


def generate_sound_effect(
    prompt: str | Sequence[str],
    model_dir: str | Path,
    *,
    seconds: float = 10.0,
    steps: int = 100,
    cfg: float = 4.0,
    sigma_shift: float = 5.0,
    seed: int = 0,
    negative_prompt: str | Sequence[str] = "",
    policy: StreamPolicy | None = None,
    uncached: bool = False,
    latents: mx.array | None = None,
) -> AudioResult:
    """Generate 48 kHz mono audio with the original full-context MOSS solver.

    Pass a stream policy to load layers as needed with bounded read-ahead.
    The default keeps each stage's weights resident during that stage.
    Both paths reuse static attention projections across denoising steps.
    A local converted MOSS v2 checkpoint and the upstream Python MLX
    implementation are required.
    """
    try:
        from moss_sfx_mlx.config import DACConfig, WanAudioModelConfig
    except ModuleNotFoundError as error:
        if error.name != "moss_sfx_mlx":
            raise
        raise RuntimeError(
            "MOSS requires the Python runtime from xocialize/moss-soundeffect-mlx "
            "and the h3-image-mlx[sound-effects] extra"
        ) from error

    began = time.perf_counter()
    usage_before = process_usage()
    root = Path(model_dir).expanduser()
    index = json.loads((root / "model_index.json").read_text())
    if index.get("_class_name", "MossSoundEffectPipeline") != "MossSoundEffectPipeline":
        raise ValueError("Expected a converted MOSS-SoundEffect v2 checkpoint")
    for name in ("dit.safetensors", "vae.safetensors"):
        path = root / "mlx" / name
        if not path.is_file():
            raise FileNotFoundError(f"MOSS weight file is missing: {path}")
    sample_rate = int(index.get("sample_rate", 48000))
    full_seconds = int(index.get("max_inference_seconds", 30))
    seconds = round(float(seconds), 1)
    cfg = float(cfg)
    sigma_shift = float(sigma_shift)
    if not 0 < seconds <= full_seconds:
        raise ValueError(f"seconds must be greater than 0 and at most {full_seconds}")
    if type(steps) is not int or steps <= 0:
        raise ValueError("steps must be a positive integer")
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("Seed must be an unsigned 64-bit integer")
    if not math.isfinite(cfg):
        raise ValueError("cfg must be finite")
    if not math.isfinite(sigma_shift) or sigma_shift <= 0:
        raise ValueError("sigma_shift must be positive and finite")
    positive = [prompt] if isinstance(prompt, str) else list(prompt)
    if not positive:
        raise ValueError("At least one prompt is required")
    prompts = ([f"{text.strip()} duration: {seconds:.1f}s" for text in positive],)
    if cfg != 1.0:
        negative = [negative_prompt] * len(positive) if isinstance(negative_prompt, str) else list(negative_prompt)
        if len(negative) != len(positive):
            raise ValueError("Each prompt requires a matching negative prompt")
        prompts += (negative,)
    latent_shape = (
        len(positive), WanAudioModelConfig().in_dim, sample_rate * full_seconds // DACConfig().hop_length,
    )
    if latents is not None and tuple(latents.shape) != latent_shape:
        raise ValueError(f"Expected latents with shape {latent_shape}, got {latents.shape}")

    stages = []
    contexts = _text_contexts(root, prompts, policy, uncached, stages)
    mx.clear_cache()
    if latents is None:
        # The first upstream normal() after mx.random.seed takes the second
        # split key. Use it directly so concurrent calls do not reset global RNG.
        _, noise_key = mx.random.split(mx.random.key(int(seed)))
        latents = mx.random.normal(latent_shape, key=noise_key)
    latents = _denoise(
        root, index, contexts, latents.astype(mx.bfloat16), steps, cfg, sigma_shift, policy, uncached, stages,
    )
    del contexts
    mx.clear_cache()
    audio = _decode(root, latents, int(sample_rate * seconds), policy, uncached, stages)
    mx.clear_cache()
    usage = process_usage()
    return AudioResult(audio, sample_rate, {
        "generation_seconds": time.perf_counter() - began,
        "process_lifetime_peak_bytes": usage["peak_bytes"],
        "process_disk_read_bytes": usage["disk_read_bytes"] - usage_before["disk_read_bytes"],
        "weight_cache": "uncached" if uncached else "system",
        "stream_execution": "resident" if policy is None else "overlap" if policy.use_overlap() else "sync",
        "stages": stages,
    })
