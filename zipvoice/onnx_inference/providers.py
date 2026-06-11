"""ONNX Runtime execution provider selection (CPU / CUDA / DirectML)."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Any

import onnxruntime as ort

logger = logging.getLogger(__name__)

ProviderEntry = str | tuple[str, dict[str, Any]]

_cuda_loadable: bool | None = None
_cuda_path_prepared: bool = False


def is_force_cpu() -> bool:
    return os.environ.get("ZIPVOICE_FORCE_CPU", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def ensure_cuda_runtime_on_path() -> None:
    """Prepend NVIDIA pip wheel bin dirs so ORT CUDA EP can load DLLs (Windows)."""
    global _cuda_path_prepared
    if _cuda_path_prepared or sys.platform != "win32":
        _cuda_path_prepared = True
        return

    try:
        import nvidia.cublas.lib as cublas_lib  # type: ignore[import-untyped]

        dirs: list[Path] = []
        for mod in (
            cublas_lib,
            _optional_nvidia_lib("nvidia.cudnn.lib"),
            _optional_nvidia_lib("nvidia.cuda_runtime.lib"),
            _optional_nvidia_lib("nvidia.cufft.lib"),
        ):
            if mod is not None:
                dirs.append(Path(mod.__file__).resolve().parent)

        capi = Path(ort.__file__).resolve().parent / "capi"
        if capi.is_dir():
            dirs.append(capi)

        prepend = os.pathsep.join(str(d) for d in reversed(dirs) if d.is_dir())
        if prepend:
            os.environ["PATH"] = prepend + os.pathsep + os.environ.get("PATH", "")
    except ImportError:
        capi = Path(ort.__file__).resolve().parent / "capi"
        if capi.is_dir():
            os.environ["PATH"] = str(capi) + os.pathsep + os.environ.get("PATH", "")

    _cuda_path_prepared = True


def _optional_nvidia_lib(name: str) -> Any | None:
    try:
        return __import__(name, fromlist=["lib"])
    except ImportError:
        return None


def is_cuda_execution_provider_loadable(*, warn: bool = True) -> bool:
    global _cuda_loadable
    if _cuda_loadable is not None:
        return _cuda_loadable

    if "CUDAExecutionProvider" not in ort.get_available_providers():
        _cuda_loadable = False
        return False

    ensure_cuda_runtime_on_path()
    try:
        capi = Path(ort.__file__).resolve().parent / "capi"
        dll = capi / "onnxruntime_providers_cuda.dll"
        if dll.is_file():
            import ctypes

            ctypes.CDLL(str(dll))
            _cuda_loadable = True
            return True
    except OSError as exc:
        if warn:
            logger.warning("CUDA EP DLL probe failed: %s", exc)

    _cuda_loadable = False
    return False


def resolve_ort_providers(
    use_gpu: bool,
    *,
    force_cpu: bool = False,
) -> tuple[list[ProviderEntry], str]:
    if force_cpu or is_force_cpu() or not use_gpu:
        return ["CPUExecutionProvider"], "CPU"

    available = set(ort.get_available_providers())
    providers: list[ProviderEntry] = []
    label = "CPU (fallback)"

    if "CUDAExecutionProvider" in available and is_cuda_execution_provider_loadable():
        providers.append(
            (
                "CUDAExecutionProvider",
                {
                    "device_id": 0,
                    "arena_extend_strategy": "kSameAsRequested",
                },
            )
        )
        label = "CUDA"
    elif sys.platform == "win32" and "DmlExecutionProvider" in available:
        providers.append("DmlExecutionProvider")
        label = "DirectML"
        logger.warning(
            "CUDA unavailable — using DirectML (may run on Intel iGPU on hybrid laptops)."
        )
    elif "CUDAExecutionProvider" in available:
        label = "CPU (CUDA DLL thiếu)"

    providers.append("CPUExecutionProvider")
    return providers, label


def provider_status_message(use_gpu: bool, *, force_cpu: bool = False) -> str:
    if force_cpu or is_force_cpu():
        return "CPU (ZIPVOICE_FORCE_CPU)"
    if not use_gpu:
        return "CPU (GPU tắt trong GUI)"
    _, label = resolve_ort_providers(use_gpu=True, force_cpu=False)
    eps = ort.get_available_providers()
    return f"{label} | EPs: {', '.join(eps)}"


def session_active_provider(session: ort.InferenceSession) -> str:
    try:
        return session.get_providers()[0]
    except (AttributeError, IndexError):
        return "unknown"


def predict_runtime_device_summary(use_gpu: bool, *, force_cpu: bool = False) -> str:
    return provider_status_message(use_gpu, force_cpu=force_cpu)
