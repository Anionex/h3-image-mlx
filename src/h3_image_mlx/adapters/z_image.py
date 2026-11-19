"""Staged Z-Image-Turbo inference with the MFlux sampler and MLX operators."""
from __future__ import annotations

import json
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from .._stage import process_usage
from .._subprocess import Cancelled
from ..api import GenerationResult
from ..streaming import BlockStream, StreamPolicy, TensorStore


def _prepare_quantization(module, store):
    def predicate(path, layer):
        scales_key = f"{path}.scales"
        if not hasattr(layer, "to_quantized") or scales_key not in store:
            return False
        width = layer.weight.shape[-1]
        scales_width = store.spec(scales_key)["shape"][-1]
        packed_width = store.spec(f"{path}.weight")["shape"][-1]
        bits, extra_bits = divmod(packed_width * 32, width)
        group_size, extra_scales = divmod(width, scales_width)
        if extra_bits or extra_scales or bits not in (2, 3, 4, 5, 6, 8) or group_size not in (32, 64, 128):
            raise ValueError(f"Invalid affine quantization shapes for {path}")
        return {"bits": bits, "group_size": group_size}

    nn.quantize(module, class_predicate=predicate)


@contextmanager
def _weights(root, component, uncached, stats):
    directory = root / component
    index = json.loads((directory / "model.safetensors.index.json").read_text())
    files = sorted(set(index["weight_map"].values()))
    if not files or any(Path(name).name != name or not name.endswith(".safetensors") for name in files):
        raise ValueError(f"Invalid sharded MFlux weights in {directory}")
    disk_before = process_usage()["disk_read_bytes"]
    started = time.perf_counter()
    with TensorStore([directory / name for name in files], uncached=uncached) as store:
        yield store
        stats.update(
            weight_io=store.io,
            disk_read_bytes=process_usage()["disk_read_bytes"] - disk_before,
            seconds=time.perf_counter() - started,
        )


def _load(module, store, mapping):
    weights = [(name, store.tensor(key)) for name, key in mapping.items()]
    module.load_weights(weights, strict=False)
    mx.eval([array for _, array in weights])


def _load_layers(module, store, stream, *, text_encoder=False):
    _prepare_quantization(module, store)
    params = dict(tree_flatten(module.parameters()))
    layer_count = len(module.layers)
    if text_encoder:
        # MFlux returns the penultimate hidden state. The placeholder keeps the
        # upstream indexing while avoiding the final layer's unused projection.
        layer_count -= 1
    eager = {
        key: key for key in params
        if not key.startswith("layers.")
        or (stream is None and int(key.split(".")[1]) < layer_count)
    }
    _load(module, store, eager)
    layers = module.layers[:layer_count]
    if stream is not None:
        layers = [
            stream.wrap(
                layer, index,
                {name: f"layers.{index}.{name}" for name, _ in tree_flatten(layer.parameters())},
            )
            for index, layer in enumerate(layers)
        ]
    if text_encoder:
        layers.append(lambda hidden_states, **_: hidden_states)
    module.layers = layers
    return store.size(eager.values())


def generate_z_image(
    prompt: str,
    model_dir: str | Path,
    *,
    width: int = 1024,
    height: int = 1024,
    seed: int = 42,
    steps: int = 9,
    policy: StreamPolicy | None = None,
    uncached: bool = False,
    on_progress=None,
    cancel=None,
) -> GenerationResult:
    """Generate from local Z-Image-Turbo weights in MFlux's saved format."""
    from mflux.models.common.compute_precision import ComputePrecision
    from mflux.models.common.config import ModelConfig
    from mflux.models.z_image.variants.z_image import ZImage
    from mflux.models.z_image.z_image_initializer import ZImageInitializer

    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("Provide a nonempty prompt")
    if not all(type(value) is int and value >= 16 and value % 16 == 0 for value in (width, height)):
        raise ValueError("Width and height must be positive multiples of 16")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Seed must be an unsigned 32-bit integer")
    if type(steps) is not int or steps < 1:
        raise ValueError("Steps must be a positive integer")
    root = Path(model_dir).expanduser().resolve(strict=True)
    stats = {}

    def progress(stage, step=0, total=0):
        if cancel is not None and cancel.is_set():
            raise Cancelled("Generation cancelled")
        if on_progress is not None:
            on_progress({"stage": stage, "step": step, "total": total})

    class Progress:
        def call_in_loop(self, t, **_):
            progress("Denoising", t + 1, steps)

    class StagedZImage(ZImage):
        def __init__(self):
            nn.Module.__init__(self)
            ZImageInitializer._init_config(self, ModelConfig.z_image_turbo())
            ZImageInitializer._init_tokenizers(self, str(root))
            ZImageInitializer._init_models(self)
            self.float32 = False
            self.compute_precision = ComputePrecision()
            self.lora_paths = self.lora_scales = None
            index = json.loads((root / "transformer/model.safetensors.index.json").read_text())
            bits = index.get("metadata", {}).get("quantization_level")
            self.bits = None if bits in (None, "None", "null", "") else int(bits)
            self.callbacks.register(Progress())

        def _encode_prompts(self, **kwargs):
            progress("Encoding prompt")
            stage = stats["text"] = {}
            with ExitStack() as stack:
                store = stack.enter_context(_weights(root, "text_encoder", uncached, stage))
                stream = stack.enter_context(BlockStream(store, policy)) if policy is not None else None
                eager_bytes = _load_layers(self.text_encoder, store, stream, text_encoder=True)
                encodings = super()._encode_prompts(**kwargs)
                mx.eval(encodings)
                stage.update(resident_weight_bytes=eager_bytes, stream=None if stream is None else stream.metrics)
            self.text_encoder = None
            mx.clear_cache()
            return encodings

        def _predict(self, transformer):
            progress("Loading denoiser")
            stage = stats["dit"] = {}
            store = denoiser_stack.enter_context(_weights(root, "transformer", uncached, stage))
            stream = denoiser_stack.enter_context(BlockStream(store, policy, wrap=True)) if policy is not None else None
            eager_bytes = _load_layers(transformer, store, stream)
            stage.update(resident_weight_bytes=eager_bytes, stream=None if stream is None else stream.metrics)
            if stream is None:
                return super()._predict(transformer)

            def predict(*, latents, timestep, sigmas, text_encodings, **_):
                return transformer(x=latents, timestep=timestep, sigmas=sigmas, cap_feats=text_encodings)

            return predict

        def _decode_latents(self, **kwargs):
            progress("Decoding image")
            denoiser_stack.close()
            self.transformer = None
            mx.clear_cache()
            self.vae.encoder = None
            stage = stats["vae"] = {}
            with _weights(root, "vae", uncached, stage) as store:
                _prepare_quantization(self.vae, store)
                keys = [key for key, _ in tree_flatten(self.vae.parameters())]
                _load(self.vae, store, {key: key for key in keys})
                stage["resident_weight_bytes"] = store.size(keys)
                decoded = super()._decode_latents(**kwargs)
                mx.eval(decoded)
                return decoded

    progress("Loading model")
    start = time.perf_counter()
    with ExitStack() as denoiser_stack:
        model = StagedZImage()
        result = model.generate_image(seed=seed, prompt=prompt, width=width, height=height, num_inference_steps=steps)
    if cancel is not None and cancel.is_set():
        raise Cancelled("Generation cancelled")
    return GenerationResult(result.image.convert("RGB"), {
        "model": "z-image-turbo",
        "seed": seed,
        "steps": steps,
        "uncached": uncached,
        "stages": stats,
        "generation_seconds": time.perf_counter() - start,
    })
