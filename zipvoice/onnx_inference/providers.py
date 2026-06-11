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


def ort_available_providers() -> list[str] | None:
    """Return ORT execution providers, or None if onnxruntime is missing/broken."""
    getter = getattr(ort, "get_available_providers", None)
    if not callable(getter):
        return None
    try:
        return list(getter())
    except Exception:
        logger.exception("onnxruntime get_available_providers failed")
        return None


def is_ort_usable() -> bool:
    return ort_available_providers() is not None


def _ort_broken_status() -> str:
    return "CPU (onnxruntime lỗi — chạy setup.bat để cài lại)"


def _ort_module_dir() -> Path | None:
    ort_file = getattr(ort, "__file__", None)
    if not ort_file:
        return None
    return Path(ort_file).resolve().parent


def is_cpu_onnxruntime_wheel() -> bool:
    """True when the CPU onnxruntime wheel is active (Azure EP on Windows)."""
    eps = ort_available_providers()
    if not eps:
        return False
    if "CUDAExecutionProvider" in eps or "DmlExecutionProvider" in eps:
        return False
    if "AzureExecutionProvider" in eps:
        return True
    try:
        import importlib.metadata as md

        has_gpu = True
        has_cpu = True
        try:
            md.version("onnxruntime-gpu")
        except md.PackageNotFoundError:
            has_gpu = False
        try:
            md.version("onnxruntime")
        except md.PackageNotFoundError:
            has_cpu = False
        return has_cpu and not has_gpu
    except Exception:
        return False


def is_force_cpu() -> bool:
    return os.environ.get("ZIPVOICE_FORCE_CPU", "").strip().lower() in {
        "1",
        "true",
        "yes",
    }


def _cuda_library_dirs() -> list[Path]:
    """Dirs that may contain cudart/cublas/cudnn for ORT CUDA EP (Windows)."""
    dirs: list[Path] = []
    seen: set[str] = set()

    def add(path: Path) -> None:
        key = str(path.resolve()).lower()
        if path.is_dir() and key not in seen:
            seen.add(key)
            dirs.append(path)

    try:
        import torch

        if getattr(torch.version, "cuda", None):
            add(Path(torch.__file__).resolve().parent / "lib")
    except ImportError:
        pass

    ort_root = _ort_module_dir()
    if ort_root is not None:
        add(ort_root / "capi")
        nvidia_root = ort_root.parent / "nvidia"
        if nvidia_root.is_dir():
            for bin_dir in nvidia_root.glob("*/bin"):
                add(bin_dir)

    cuda_path = os.environ.get("CUDA_PATH", "").strip()
    if cuda_path:
        add(Path(cuda_path) / "bin")

    return dirs


def ensure_cuda_runtime_on_path() -> None:
    """Prepend CUDA DLL dirs so ORT can load onnxruntime_providers_cuda.dll."""
    global _cuda_path_prepared
    if _cuda_path_prepared:
        return
    if sys.platform != "win32":
        _cuda_path_prepared = True
        return

    dirs = _cuda_library_dirs()
    prepend = os.pathsep.join(str(d) for d in reversed(dirs))
    if prepend:
        os.environ["PATH"] = prepend + os.pathsep + os.environ.get("PATH", "")
    add_dll = getattr(os, "add_dll_directory", None)
    if callable(add_dll):
        for d in dirs:
            try:
                add_dll(str(d))
            except OSError:
                pass

    _cuda_path_prepared = True


def is_cuda_execution_provider_loadable(*, warn: bool = True) -> bool:
    global _cuda_loadable
    if _cuda_loadable is not None:
        return _cuda_loadable

    available = ort_available_providers()
    if available is None or "CUDAExecutionProvider" not in available:
        _cuda_loadable = False
        return False

    ensure_cuda_runtime_on_path()
    try:
        ort_root = _ort_module_dir()
        if ort_root is None:
            _cuda_loadable = False
            return False
        capi = ort_root / "capi"
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

    eps = ort_available_providers()
    if eps is None:
        return ["CPUExecutionProvider"], "CPU (onnxruntime lỗi)"

    available = set(eps)
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
        label = "CPU (CUDA DLL thiếu — setup [3] hoặc requirements-onnx-gpu-cuda-libs.txt)"

    providers.append("CPUExecutionProvider")
    return providers, label


def provider_status_message(use_gpu: bool, *, force_cpu: bool = False) -> str:
    if force_cpu or is_force_cpu():
        return "CPU (ZIPVOICE_FORCE_CPU)"
    if not use_gpu:
        return "CPU (GPU tắt trong GUI)"
    eps = ort_available_providers()
    if eps is None:
        return _ort_broken_status()
    if is_cpu_onnxruntime_wheel():
        return (
            "CPU (onnxruntime CPU — thiếu GPU wheel) | EPs: "
            + ", ".join(eps)
            + " | Chạy setup.bat → [3]"
        )
    _, label = resolve_ort_providers(use_gpu=True, force_cpu=False)
    return f"{label} | EPs: {', '.join(eps)}"


def session_active_provider(session: ort.InferenceSession) -> str:
    try:
        return session.get_providers()[0]
    except (AttributeError, IndexError):
        return "unknown"


def predict_runtime_device_summary(use_gpu: bool, *, force_cpu: bool = False) -> str:
    return provider_status_message(use_gpu, force_cpu=force_cpu)
