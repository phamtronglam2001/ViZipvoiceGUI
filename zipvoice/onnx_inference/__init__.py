"""ViZipVoice ONNX inference (character tokenizer + local export bundle)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from zipvoice.onnx_inference.engine import ViZipVoiceOnnxTTS

__all__ = ["ViZipVoiceOnnxTTS"]


def __getattr__(name: str) -> object:
    if name == "ViZipVoiceOnnxTTS":
        from zipvoice.onnx_inference.engine import ViZipVoiceOnnxTTS

        return ViZipVoiceOnnxTTS
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
