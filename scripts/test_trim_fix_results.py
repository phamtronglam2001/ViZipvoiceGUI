#!/usr/bin/env python3
"""Verify mel trim fix across test sentences."""
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from zipvoice.onnx_inference import ViZipVoiceOnnxTTS
from zipvoice.utils.audio_io import load_audio

ONNX_DIR = REPO / "models" / "onnx"
VOCODER = REPO / "models" / "vocoder" / "mel_spec_24khz.onnx"
PROMPT_WAV = REPO / "models" / "ViZipvoice" / "audio" / "MC.mp3"
PROMPT_TEXT = Path(PROMPT_WAV.with_suffix(".txt")).read_text(encoding="utf-8").strip()
OUT = REPO / "output" / "trim_fix_test"
OUT.mkdir(parents=True, exist_ok=True)


def stats(path: Path) -> dict:
    wav, sr = load_audio(path)
    w = wav.squeeze().numpy()
    mx = float(np.abs(w).max())
    onset = int(np.argmax(np.abs(w) > max(mx * 0.05, 1e-4))) if mx > 1e-4 else 0
    return {
        "duration_s": round(len(w) / sr, 3),
        "max_amp": round(mx, 4),
        "rms": round(float(np.sqrt((w ** 2).mean())), 4),
        "onset_s": round(onset / sr, 3),
    }


def main():
    tts = ViZipVoiceOnnxTTS(
        onnx_dir=ONNX_DIR,
        use_int4=True,
        vocoder_onnx=str(VOCODER),
    )
    cases = [
        ("counting", "Hai ba bốn năm sáu bảy tám chín mười."),
        ("short8", "Chi so la 8."),
        ("short_dot", "8."),
        ("long", (
            "Đo lường phản ứng của nhà sản xuất với biện pháp điều chỉnh giá "
            "là rất cần thiết cho thị trường trong dài hạn."
        )),
    ]
    print("Test results (int4 + mel_spec vocoder, MC.mp3 prompt):")
    print("-" * 72)
    for tag, text in cases:
        out = OUT / f"{tag}.wav"
        m = tts.synthesize(
            prompt_wav=PROMPT_WAV,
            prompt_text=PROMPT_TEXT,
            text=text,
            output_path=out,
            split_sentences=False,
            remove_long_sil=False,
            seed=666,
        )
        s = stats(out)
        print(f"{tag}:")
        print(f"  text: {text[:60]}{'...' if len(text) > 60 else ''}")
        print(f"  duration={s['duration_s']}s max_amp={s['max_amp']} rms={s['rms']} onset={s['onset_s']}s")
        print(f"  engine wav_seconds={m.get('wav_seconds')}")
        print()


if __name__ == "__main__":
    main()
