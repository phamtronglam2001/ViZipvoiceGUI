"""Sync bundled reference audio from assets/ref_audio into model_dir/audio/."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BUNDLED_REF_DIR = REPO_ROOT / "assets" / "ref_audio"
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}


def sync_bundled_ref_audio(model_dir: Path) -> int:
    """Copy paired audio+.txt from assets/ref_audio. Returns files copied/updated."""
    if not BUNDLED_REF_DIR.is_dir():
        return 0

    dst = Path(model_dir) / "audio"
    dst.mkdir(parents=True, exist_ok=True)
    copied = 0

    for audio in sorted(BUNDLED_REF_DIR.iterdir()):
        if not audio.is_file() or audio.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        txt = audio.with_suffix(".txt")
        if not txt.is_file():
            logging.warning("Bundled ref missing transcript: %s", txt.name)
            continue
        for src_file in (audio, txt):
            target = dst / src_file.name
            if not target.is_file() or src_file.stat().st_mtime > target.stat().st_mtime:
                shutil.copy2(src_file, target)
                copied += 1

    if copied:
        logging.info("Synced %d bundled ref file(s) → %s", copied, dst)
    return copied
