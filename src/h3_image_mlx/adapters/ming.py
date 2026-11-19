"""Load Ming-3.0 TTS on demand and stream the audio decoder Transformer."""
from __future__ import annotations

import json
import time
from contextlib import ExitStack
from pathlib import Path
from threading import Lock

import mlx.core as mx
from mlx.utils import tree_flatten
from tokenizers import Tokenizer

from .._stage import process_usage
from ..api import AudioResult
from ..streaming import BlockStream, StreamPolicy, TensorStore


_DECODER_LAYERS = "audio.decoder.decoder.layers."
_RANDOM_LOCK = Lock()


class _MingTokenizer:
    def __init__(self, path):
        self._tokenizer = Tokenizer.from_file(str(path))

    def encode(self, text, *, add_special_tokens=False):
        return self._tokenizer.encode(text, add_special_tokens=add_special_tokens).ids

    def convert_tokens_to_ids(self, token):
        return self._tokenizer.token_to_id(token)


def synthesize_speech(
    text: str,
    model_dir: str | Path,
    *,
    seed: int = 42,
    policy: StreamPolicy | None = None,
    uncached: bool = False,
    **generation_options,
) -> AudioResult:
    """Run the Ming-3.0-tts-0.5B MLX checkpoint with its original sampler.

    The autoregressive model, aggregator and small DiTAR stay resident through
    synthesis. A stream policy loads each of the 24 audio decoder layers when
    it is used. The reference encoder is loaded only for acoustic prompts.
    Generation options are forwarded to MLX Audio.
    """
    from mlx_audio.tts.models.dense.dense import Model
    from mlx_audio.utils import apply_quantization

    if not isinstance(text, str) or not text.strip():
        raise ValueError("Provide text to synthesize")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Seed must be an unsigned 32-bit integer")
    root = Path(model_dir).expanduser().resolve(strict=True)
    config = json.loads((root / "config.json").read_text())
    dit_config = config.get("ditar_config", {})
    if config.get("model_type") != "dense" or (
        dit_config.get("patch_size"), dit_config.get("history_patch_size", 1)
    ) != (1, 1):
        raise ValueError("Expected a Ming-3.0-tts-0.5B MLX checkpoint")
    paths = sorted(path for path in root.glob("*.safetensors") if path.name != "campplus.safetensors")
    if not paths:
        raise FileNotFoundError(f"No Ming weight files in {root}")

    start = time.perf_counter()
    disk_before = process_usage()["disk_read_bytes"]
    with _RANDOM_LOCK, TensorStore(paths, uncached=uncached) as store:
        original_random_state = mx.random.state[0]
        try:
            model = Model({**config, "model_path": str(root)})
            if generation_options.get("ref_audio") is None or generation_options.get("ref_text") is None:
                model.audio.encoder = None
            apply_quantization(model, config, store)
            parameters = tree_flatten(model.parameters())
            for name, parameter in parameters:
                if name not in store or tuple(store.spec(name)["shape"]) != parameter.shape:
                    raise ValueError(f"Ming checkpoint is missing a compatible {name} tensor")
            resident = tuple(
                name for name, _ in parameters
                if policy is None or not name.startswith(_DECODER_LAYERS)
            )
            model.load_weights([(name, store.tensor(name)) for name in resident], strict=False)
            model.eval()
            model.tokenizer = _MingTokenizer(root / "tokenizer.json")
            if isinstance(generation_options.get("ref_audio"), Path):
                generation_options["ref_audio"] = str(generation_options["ref_audio"])
            mx.random.seed(seed)
            with ExitStack() as stack:
                stream = None
                if policy is not None:
                    stream = stack.enter_context(BlockStream(store, policy))
                    decoder = model.audio.decoder.decoder
                    decoder.layers = [
                        stream.wrap(
                            layer,
                            index,
                            {
                                name: f"{_DECODER_LAYERS}{index}.{name}"
                                for name, _ in tree_flatten(layer.parameters())
                            },
                        )
                        for index, layer in enumerate(decoder.layers)
                    ]
                mx.eval(model.parameters())
                loaded = time.perf_counter()
                segments = list(model.generate(text=text, **generation_options))
                if not segments:
                    raise RuntimeError("Ming did not generate any audio")
                sample_rate = segments[0].sample_rate
                if any(segment.sample_rate != sample_rate for segment in segments):
                    raise RuntimeError("Ming returned inconsistent audio sample rates")
                waveform = segments[0].audio if len(segments) == 1 else mx.concatenate(
                    [segment.audio for segment in segments], axis=-1,
                )
                mx.eval(waveform)
        finally:
            mx.random.state[0][:] = original_random_state
        metrics = {
            "model": "ming-3.0-tts",
            "uncached": uncached,
            "loading_seconds": loaded - start,
            "synthesis_seconds": time.perf_counter() - loaded,
            "resident_weight_bytes": store.size(resident),
            "decoder_stream": None if stream is None else dict(stream.metrics),
            "weight_io": store.io,
            "disk_read_bytes": process_usage()["disk_read_bytes"] - disk_before,
        }
    return AudioResult(waveform, sample_rate, metrics)
