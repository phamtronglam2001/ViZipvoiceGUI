"""ONNX Runtime session helpers (threads, providers)."""

from __future__ import annotations

import os

import onnxruntime as ort


def default_onnx_threads() -> int:
    """ORT threads per session. Override: ZIPVOICE_ONNX_THREADS."""
    raw = os.environ.get("ZIPVOICE_ONNX_THREADS", "").strip()
    if raw:
        try:
            return max(1, int(raw))
        except ValueError:
            pass
    cap = os.cpu_count() or 4
    return min(cap, 8)


def onnx_providers() -> list[str]:
    """Prefer CUDA when available, else CPU."""
    available = set(ort.get_available_providers())
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if "DmlExecutionProvider" in available:
        return ["DmlExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def make_session_options(num_threads: int | None = None) -> ort.SessionOptions:
    opts = ort.SessionOptions()
    threads = default_onnx_threads() if num_threads is None else max(1, int(num_threads))
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = threads
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return opts
