"""Shared Gradio header / branding for local apps."""

from __future__ import annotations

AUTHOR_NAME = "Lam Pham"
AUTHOR_GITHUB = "phamtronglam2001"
REPO_URL = f"https://github.com/{AUTHOR_GITHUB}/ViZipvoiceGUI"
UPSTREAM_URL = "https://github.com/iamdinhthuan/ViZipvoice"
HF_SPACE_URL = "https://huggingface.co/spaces/dinhthuan/ViZipvoice"
HF_MODEL_URL = "https://huggingface.co/contextboxai/ViZipvoice"

AUTHOR_LINE = (
    f"**Author:** [{AUTHOR_NAME}]({REPO_URL}) · "
    f"**Fork of:** [iamdinhthuan/ViZipvoice]({UPSTREAM_URL}) · "
    "**License:** non-commercial use only (see `LICENSE`)"
)

FORK_PURPOSE = (
    "ONNX int4 for smaller footprint · audiobook-oriented long-form TTS "
    "(text normalizer tuning planned)"
)
