"""ONNX Runtime session helpers (threads, providers)."""

from __future__ import annotations

import logging
import os

import onnxruntime as ort

from zipvoice.onnx_inference.providers import resolve_ort_providers

logger = logging.getLogger(__name__)


def default_onnx_threads() -> int:
    """ORT threads per session. Override: ZIPVOICE_ONNX_THREADS."""
    raw = os.environ.get("ZIPVOICE_ONNX_THREADS", "").strip()
    if raw:
        try:
            return max(1, min(16, int(raw)))
        except ValueError:
            pass
    cap = os.cpu_count() or 4
    return min(cap, 8)


def resolve_onnx_threads(requested: int | None) -> int:
    if requested is None or int(requested) <= 0:
        return default_onnx_threads()
    return max(1, min(16, int(requested)))


def onnx_providers(
    *,
    use_gpu: bool = False,
    force_cpu: bool = False,
) -> list:
    providers, _ = resolve_ort_providers(use_gpu, force_cpu=force_cpu)
    return providers


def make_session_options(num_threads: int | None = None) -> ort.SessionOptions:
    opts = ort.SessionOptions()
    threads = resolve_onnx_threads(num_threads)
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = threads
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.enable_mem_pattern = True
    opts.enable_cpu_mem_arena = True
    return opts


def create_inference_session(
    model_path: str,
    *,
    sess_options: ort.SessionOptions,
    providers: list,
) -> ort.InferenceSession:
    cpu_only = ["CPUExecutionProvider"]
    try:
        return ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=providers,
        )
    except Exception as exc:
        if providers == cpu_only:
            raise
        logger.warning(
            "ORT session failed with requested providers — falling back to CPU: %s",
            exc,
        )
        return ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=cpu_only,
        )
