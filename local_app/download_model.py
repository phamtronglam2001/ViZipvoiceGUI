#!/usr/bin/env python3
"""Download ViZipVoice model + ref audio for offline use."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from huggingface_hub import snapshot_download

DEFAULT_REPO = "contextboxai/ViZipvoice"

# Copied into the HF model repo for standalone use — redundant when this git repo is installed.
HF_REDUNDANT_FILES = ("vizipvoice.py", "README.md")


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download ViZipVoice model from Hugging Face for offline inference.",
    )
    parser.add_argument(
        "--repo",
        default=DEFAULT_REPO,
        help="Hugging Face model repo id",
    )
    parser.add_argument(
        "--local-dir",
        type=Path,
        default=Path("models/ViZipvoice"),
        help="Destination directory",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional git revision / branch / tag",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = get_parser().parse_args()
    local_dir = args.local_dir.resolve()
    local_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Downloading %s → %s", args.repo, local_dir)
    snapshot_download(
        repo_id=args.repo,
        repo_type="model",
        local_dir=str(local_dir),
        revision=args.revision,
        local_dir_use_symlinks=False,
    )
    for name in HF_REDUNDANT_FILES:
        path = local_dir / name
        if path.is_file():
            path.unlink()
            logging.info("Removed redundant HF artifact: %s", path.name)
    logging.info("Done. Checkpoint + audio/ ready at %s", local_dir)


if __name__ == "__main__":
    main()
