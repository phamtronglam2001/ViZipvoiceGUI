"""Audio load/save helpers that avoid torchcodec on Windows."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def _load_with_pydub(path: str) -> tuple[torch.Tensor, int]:
    from pydub import AudioSegment

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
    wav = waveform.detach().cpu()
    if wav.ndim == 1:
        wav = wav.unsqueeze(0)
    array = wav.numpy().T
    sf.write(str(path), array, int(sample_rate))
