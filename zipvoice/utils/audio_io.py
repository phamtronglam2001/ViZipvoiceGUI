"""Audio load/save helpers that avoid torchcodec on Windows."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_FFMPEG_DIR = "ffmpeg"
_FFMPEG_EXE_NAMES = ("ffmpeg.exe", "ffmpeg")


@lru_cache(maxsize=8)
def _resolve_ffmpeg_exe(ffmpeg_dir_raw: str = "") -> Path | None:
    env = os.getenv("VIZIPVOICE_FFMPEG_DIR", "").strip()
    raw = (ffmpeg_dir_raw or env or _DEFAULT_FFMPEG_DIR).strip()
    base = Path(raw)
    if not base.is_absolute():
        base = (_REPO_ROOT / base).resolve()

    if base.is_file() and base.name.lower() in {n.lower() for n in _FFMPEG_EXE_NAMES}:
        return base.resolve()

    if not base.is_dir():
        return None

    for name in _FFMPEG_EXE_NAMES:
        for candidate in (base / name, base / "bin" / name):
            if candidate.is_file():
                return candidate.resolve()
    return None


def configure_pydub_ffmpeg(ffmpeg_dir_raw: str = "") -> bool:
    """Point pydub at bundled/portable ffmpeg + ffprobe (via PATH)."""
    ffmpeg_exe = _resolve_ffmpeg_exe(ffmpeg_dir_raw)
    if ffmpeg_exe is None:
        return False

    bin_dir = str(ffmpeg_exe.parent)
    ffprobe_exe = ffmpeg_exe.with_name("ffprobe.exe")
    if not ffprobe_exe.is_file():
        ffprobe_exe = ffmpeg_exe.with_name("ffprobe")
    if not ffprobe_exe.is_file():
        return False

    from pydub import AudioSegment

    AudioSegment.converter = str(ffmpeg_exe)

    path_entries = os.environ.get("PATH", "").split(os.pathsep)
    if bin_dir not in path_entries:
        os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
    return True


def _load_with_pydub(path: str) -> tuple[torch.Tensor, int]:
    from pydub import AudioSegment

    configure_pydub_ffmpeg()
    segment = AudioSegment.from_file(path)
    samples = np.array(segment.get_array_of_samples(), dtype=np.float32)
    if segment.channels > 1:
        samples = samples.reshape(-1, segment.channels).T
    else:
        samples = samples[np.newaxis, :]
    peak = float(1 << (8 * segment.sample_width - 1))
    return torch.from_numpy(samples / peak), int(segment.frame_rate)


def load_audio(path: str | Path) -> tuple[torch.Tensor, int]:
    path = str(path)
    suffix = Path(path).suffix.lower()
    if suffix in {".wav", ".flac"}:
        try:
            import soundfile as sf

            data, sample_rate = sf.read(path, always_2d=True)
            waveform = torch.from_numpy(data.T).float()
            return waveform, int(sample_rate)
        except Exception:
            pass
    return _load_with_pydub(path)


def save_audio(path: str | Path, waveform: torch.Tensor, sample_rate: int) -> None:
    import soundfile as sf

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wav = waveform.detach().cpu().float()
    while wav.ndim > 2:
        wav = wav.squeeze(0)
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    array = wav.numpy().T
    sf.write(str(path), array, int(sample_rate), subtype="PCM_16")
