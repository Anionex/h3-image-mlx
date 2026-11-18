"""Public model dispatch; model-specific options never leak into another adapter."""

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from PIL import Image

if TYPE_CHECKING:
    from .qwen.resident import QwenSession


@dataclass
class GenerationResult:
    image: Image.Image
    metrics: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Lora:
    path: str | Path
    scale: float = 1.0


@dataclass(frozen=True)
class QwenOptions:
    memory_mode: str = "prefetch"
    runtime: str | Path | None = None
    cache: str | Path | None = None
    prepared: str | Path | None = None
    prepare_weights: bool = True
    weight_cache: str = "system"
    profile: str = "complete"
    profile_assets: str | Path | None = None
    profile_strength: float = 1.0
    auto_strength: bool = False
    loras: tuple[Lora, ...] = ()
    steps: int = 8
    guidance_scale: float = 1.0
    negative_prompt: str = ""
    sample_sigmas: tuple[float, ...] | None = None
    mode: str = "edit"
    strength: float = 0.6
    reference_resolution: int = 1024
    transparent: bool = False
    private: bool = False
    timeout: float = 900
    prepared_candidates: tuple[str | Path, ...] = ()
    profile_base: str | Path | None = None


MODEL_IDS = ("h3", "qwen-image-2.1-turbo")


def generate_result(
    prompt,
    *,
    models,
    model="h3",
    width=1024,
    height=1024,
    seed=42,
    references=(),
    backend="auto",
    dit_mode="auto",
    precision="int8",
    qwen: QwenOptions | None = None,
    qwen_session: "QwenSession | None" = None,
    overlap_streaming: bool | None = None,
    private: bool = False,
    on_progress: Callable | None = None,
    cancel=None,
) -> GenerationResult:
    """Render from ordered paths or encoded image bytes.

    on_progress receives stage/step/total dictionaries. Set a threading.Event
    passed as cancel to stop the active stage and raise Cancelled. Private
    requests skip optional request-data caches.
    """
    if model == "h3":
        if qwen is not None or qwen_session is not None:
            raise ValueError("QwenOptions require model='qwen-image-2.1-turbo'")
        from .pipeline import generate_result as h3_generate

        return h3_generate(
            prompt,
            models=models,
            width=width,
            height=height,
            seed=seed,
            references=references,
            backend=backend,
            dit_mode=dit_mode,
            precision=precision,
            private=private,
            overlap_streaming=overlap_streaming,
            on_progress=on_progress,
            cancel=cancel,
        )
    if model == "qwen-image-2.1-turbo":
        if precision != "int8" or dit_mode != "auto" or overlap_streaming is not None:
            raise ValueError(
                "Qwen uses its Q8 model pack and QwenOptions.memory_mode; H3 precision/dit_mode do not apply"
            )
        from .qwen.pipeline import generate as qwen_generate

        options = qwen or QwenOptions()
        if private:
            options = replace(options, private=True)

        return qwen_generate(
            prompt,
            models=models,
            width=width,
            height=height,
            seed=seed,
            references=references,
            backend=backend,
            options=options,
            session=qwen_session,
            on_progress=on_progress,
            cancel=cancel,
        )
    raise ValueError(f"Unknown model: {model}; choose one of {MODEL_IDS}")


def generate(prompt, **kwargs) -> Image.Image:
    """Generate a PIL image; use generate_result to also receive execution metrics."""
    return generate_result(prompt, **kwargs).image
