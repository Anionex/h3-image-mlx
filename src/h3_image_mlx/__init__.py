"""Local H3 and Qwen Image inference on Apple Silicon."""
from .api import generate, generate_result, GenerationResult, Lora, QwenOptions
from ._subprocess import Cancelled

from .qwen.resident import QwenSession

from .qwen.lora import inspect_adapter as inspect_qwen_lora
from .qwen.vision_assets import extract_vision as prepare_qwen_vision

__all__ = ["generate", "generate_result", "GenerationResult", "Lora", "QwenOptions", "Cancelled", "QwenSession",
           "inspect_qwen_lora", "prepare_qwen_vision"]
