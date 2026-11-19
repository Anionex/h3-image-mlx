"""Local image and audio inference on Apple Silicon."""
from .api import AudioResult, GenerationResult, Lora, QwenOptions, generate, generate_audio_result, generate_result
from ._subprocess import Cancelled

from .qwen.resident import QwenSession

from .qwen.lora import inspect_adapter as inspect_qwen_lora
from .qwen.vision_assets import extract_vision as prepare_qwen_vision

__all__ = ["generate", "generate_result", "generate_audio_result", "GenerationResult", "AudioResult", "Lora", "QwenOptions", "Cancelled", "QwenSession",
           "inspect_qwen_lora", "prepare_qwen_vision"]
