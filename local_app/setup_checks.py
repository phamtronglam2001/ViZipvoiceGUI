"""Setup helpers — incremental install, cache-first, no downgrade of existing GPU builds."""

from __future__ import annotations

import importlib.metadata
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

# PyTorch CUDA wheel indices (try cache/offline first, then in order).
_PYTORCH_CUDA_INDEX_URLS = (
    "https://download.pytorch.org/whl/cu128",
    "https://download.pytorch.org/whl/cu124",
    "https://download.pytorch.org/whl/cu126",
    "https://download.pytorch.org/whl/cu121",
)

_EXPORT_PIP_SPECS = ("onnx>=1.16,<1.19", "onnxscript", "librosa")


def _run_uv_pip(args: list[str]) -> int:
    cmd = ["uv", "pip", "install", *args]
    print(f"[setup] {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=REPO_ROOT)


def _installed_dist_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def check_pytorch_cuda() -> int:
    """0 if torch is any working CUDA build already in .venv."""
    try:
        import torch
    except ImportError:
        return 1
    cuda_ver = getattr(torch.version, "cuda", None)
    if not cuda_ver:
        return 1
    ver = torch.__version__
    print(f"[check] PyTorch CUDA OK — torch {ver}, cuda {cuda_ver}")
    if torch.cuda.is_available():
        print(f"[check] GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("[check] CUDA build san sang; runtime chua thay GPU (van OK trong app)")
    return 0


def check_onnx_cpu() -> int:
    """0 if onnxruntime CPU EP works (CPU wheel or GPU wheel)."""
    try:
        import onnxruntime as ort
    except ImportError:
        return 1
    providers_fn = getattr(ort, "get_available_providers", None)
    if not callable(providers_fn):
        return 1
    eps = providers_fn()
    if "CPUExecutionProvider" not in eps:
        return 1
    pkg = "onnxruntime-gpu" if _installed_dist_version("onnxruntime-gpu") else "onnxruntime"
    ver = _installed_dist_version(pkg) or getattr(ort, "__version__", "?")
    print(f"[check] onnxruntime CPU EP OK — {pkg} {ver}, EPs: {eps}")
    return 0


def check_onnx_export_deps() -> int:
    """0 if onnx + librosa present (export tab / ONNX scripts)."""
    try:
        import onnx  # noqa: F401
        import librosa  # noqa: F401
    except ImportError:
        return 1
    print("[check] ONNX export deps OK — onnx, librosa")
    return 0


def check_onnx_gpu() -> int:
    """0 if onnxruntime-gpu is installed and CUDA EP loads."""
    if _installed_dist_version("onnxruntime-gpu") is None:
        return 1
    try:
        from zipvoice.onnx_inference.providers import (
            ensure_cuda_runtime_on_path,
            is_cuda_execution_provider_loadable,
            ort_available_providers,
        )
    except ImportError:
        return 1
    ensure_cuda_runtime_on_path()
    eps = ort_available_providers()
    if not eps or "CUDAExecutionProvider" not in eps:
        return 1
    if not is_cuda_execution_provider_loadable(warn=False):
        return 1
    ver = _installed_dist_version("onnxruntime-gpu") or "?"
    print(f"[check] ONNX GPU OK — onnxruntime-gpu {ver}, EPs: {eps}")
    return 0


def install_pytorch_cuda() -> int:
    """Cache-first CUDA torch; accept any cu12x wheel that works."""
    if check_pytorch_cuda() == 0:
        print("[setup] PyTorch CUDA da co — bo qua cai.")
        return 0

    print("[setup] Thu cai torch CUDA tu cache (offline)...")
    if _run_uv_pip(["--offline", "torch", "torchaudio"]) == 0 and check_pytorch_cuda() == 0:
        return 0

    for index_url in _PYTORCH_CUDA_INDEX_URLS:
        print(f"[setup] Thu index {index_url} (uu tien wheel trong uv/pip cache)...")
        if (
            _run_uv_pip(
                ["torch", "torchaudio", "--index-url", index_url],
            )
            == 0
            and check_pytorch_cuda() == 0
        ):
            return 0

    print("[setup] Khong cai duoc PyTorch CUDA — kiem tra mang hoac cache.")
    return 1


def ensure_onnx_export_deps() -> int:
    if check_onnx_export_deps() == 0:
        return 0
    print("[setup] Cai onnx + librosa + onnxscript (khong doi onnxruntime)...")
    if _run_uv_pip(list(_EXPORT_PIP_SPECS)) != 0:
        return 1
    return check_onnx_export_deps()


def ensure_onnx_cpu() -> int:
    """CPU onnxruntime + export deps; skip ort reinstall if GPU wheel already OK."""
    if check_onnx_gpu() == 0:
        print("[setup] onnxruntime-gpu da co — chi bo sung export deps, khong ha GPU.")
        return ensure_onnx_export_deps()

    if check_onnx_cpu() == 0:
        return ensure_onnx_export_deps()

    print("[setup] Cai ONNX CPU (uv sync --extra onnx --extra export)...")
    sync_cmd = ["uv", "sync", "--extra", "onnx", "--extra", "export"]
    if (REPO_ROOT / "uv.lock").is_file():
        sync_cmd.insert(2, "--frozen")
    if subprocess.call(sync_cmd, cwd=REPO_ROOT) != 0:
        return 1
    if check_onnx_cpu() != 0:
        return 1
    return ensure_onnx_export_deps()


def install_onnx_gpu() -> int:
    if check_onnx_gpu() == 0:
        print("[setup] ONNX GPU da co — bo qua cai.")
        return ensure_onnx_export_deps()

    req = REPO_ROOT / "requirements-onnx-gpu.txt"
    if not req.is_file():
        print(f"[setup] Thieu {req}")
        return 1

    print("[setup] Thu cai onnxruntime-gpu tu cache (offline)...")
    subprocess.call(["uv", "pip", "uninstall", "onnxruntime"], cwd=REPO_ROOT)
    if _run_uv_pip(["--offline", "-r", str(req)]) == 0 and check_onnx_gpu() == 0:
        return ensure_onnx_export_deps()

    print("[setup] Cai onnxruntime-gpu (uu tien wheel trong uv/pip cache)...")
    subprocess.call(["uv", "pip", "uninstall", "onnxruntime"], cwd=REPO_ROOT)
    if _run_uv_pip(["-r", str(req)]) != 0:
        return 1
    if check_onnx_gpu() != 0:
        print("[setup] CANH BAO: CUDA EP chua load — app van fallback CPU.")
        try:
            from zipvoice.onnx_inference.providers import provider_status_message

            print(provider_status_message(True))
        except Exception:
            pass
        return ensure_onnx_export_deps()
    return ensure_onnx_export_deps()


def write_markers() -> int:
    """Write .install_mode_* from actual .venv state (never downgrade on paper)."""
    pytorch_mode = "gpu" if check_pytorch_cuda() == 0 else "cpu"
    if check_onnx_gpu() == 0:
        onnx_mode = "gpu"
    elif check_onnx_cpu() == 0:
        onnx_mode = "cpu"
    else:
        onnx_mode = "cpu"

    (REPO_ROOT / ".install_mode_pytorch").write_text(pytorch_mode + "\n", encoding="utf-8")
    (REPO_ROOT / ".install_mode_onnx").write_text(onnx_mode + "\n", encoding="utf-8")
    legacy = REPO_ROOT / ".install_mode"
    if legacy.exists() or pytorch_mode == "cpu":
        legacy.write_text("cpu\n", encoding="utf-8")

    print(f"[markers] PyTorch: {pytorch_mode}")
    print(f"[markers] ONNX: {onnx_mode}")
    return 0


def print_status() -> int:
    print("--- Trang thai .venv ---")
    if check_pytorch_cuda() != 0:
        try:
            import torch

            print(f"[check] PyTorch CPU — torch {torch.__version__}")
        except ImportError:
            print("[check] PyTorch: chua cai")
    check_onnx_cpu()
    check_onnx_gpu()
    check_onnx_export_deps()
    write_markers()
    return 0


def main() -> None:
    if len(sys.argv) < 2:
        print(
            "usage: setup_checks.py "
            "pytorch_cuda | onnx_cpu | onnx_gpu | "
            "install_pytorch_cuda | ensure_onnx_cpu | install_onnx_gpu | "
            "ensure_export_deps | write_markers | status",
            file=sys.stderr,
        )
        sys.exit(2)

    cmd = sys.argv[1].lower()
    handlers = {
        "pytorch_cuda": check_pytorch_cuda,
        "onnx_cpu": check_onnx_cpu,
        "onnx_gpu": check_onnx_gpu,
        "install_pytorch_cuda": install_pytorch_cuda,
        "ensure_onnx_cpu": ensure_onnx_cpu,
        "install_onnx_gpu": install_onnx_gpu,
        "ensure_export_deps": ensure_onnx_export_deps,
        "write_markers": write_markers,
        "status": print_status,
    }
    fn = handlers.get(cmd)
    if fn is None:
        print(f"unknown: {cmd}", file=sys.stderr)
        sys.exit(2)
    sys.exit(fn())


if __name__ == "__main__":
    main()
