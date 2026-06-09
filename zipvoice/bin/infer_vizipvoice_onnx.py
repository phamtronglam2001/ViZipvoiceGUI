#!/usr/bin/env python3
"""ONNX inference for ViZipVoice exported models (FP32 or int4 + Vocos ONNX)."""

from __future__ import annotations

import argparse
import datetime as dt
import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="ViZipVoice ONNX inference from local export directory.",
    )
    parser.add_argument("--onnx-model-dir", required=True)
    parser.add_argument("--use-int4", action="store_true")
    parser.add_argument("--prompt-wav", required=True)
    parser.add_argument("--prompt-text", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--res-wav-path", default="output_onnx.wav")
    parser.add_argument("--num-step", type=int, default=16)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--t-shift", type=float, default=0.5)
    parser.add_argument("--target-rms", type=float, default=0.1)
    parser.add_argument("--feat-scale", type=float, default=0.1)
    parser.add_argument("--no-vietnamese-normalize", action="store_true")
    parser.add_argument(
        "--max-duration",
        type=float,
        default=100,
        help="Max seconds per internal chunk batch (long text); same as Gradio Advanced",
    )
    parser.add_argument(
        "--remove-long-sil",
        action="store_true",
        help="Remove long silences inside generated audio",
    )
    parser.add_argument("--vocoder-path", default=None)
    parser.add_argument(
        "--vocoder-onnx",
        default=None,
        help="Path to mel_spec_24khz.onnx (default: ../vocoder/ next to onnx dir)",
    )
    parser.add_argument(
        "--num-thread",
        type=int,
        default=None,
        help="ORT threads per session (default: min(cpu_count, 8) or ZIPVOICE_ONNX_THREADS)",
    )
    return parser


def resolve_onnx_paths(
    onnx_dir: Path,
    use_int4: bool,
) -> tuple[Path, Path]:
    suffix = "_int4" if use_int4 else ""
    text_encoder = onnx_dir / f"text_encoder{suffix}.onnx"
    fm_decoder = onnx_dir / f"fm_decoder{suffix}.onnx"

    if not text_encoder.is_file():
        raise FileNotFoundError(f"Missing {text_encoder}")
    if not fm_decoder.is_file():
        raise FileNotFoundError(f"Missing {fm_decoder}")

    return text_encoder, fm_decoder


def resolve_vocoder_onnx(onnx_dir: Path, vocoder_onnx: str | None) -> Path | None:
    if vocoder_onnx:
        path = Path(vocoder_onnx)
        if path.is_file():
            return path

    candidates = [
        onnx_dir.parent / "vocoder" / "mel_spec_24khz.onnx",
        onnx_dir / "mel_spec_24khz.onnx",
        onnx_dir / "vocos_decoder.onnx",
        onnx_dir / "vocos_decoder_int4.onnx",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def decode_with_mel_spec_onnx(
    session: ort.InferenceSession,
    features: torch.Tensor,
) -> torch.Tensor:
    import librosa
    import numpy as np

    mel = features.permute(0, 2, 1).contiguous().numpy().astype(np.float32)
    mag, x, y = session.run(None, {"mels": mel})
    if mag.ndim == 3:
        mag, x, y = mag[0], x[0], y[0]
    stft = mag * (x + 1j * y)
    wav = librosa.istft(
        stft,
        hop_length=256,
        win_length=1024,
        n_fft=1024,
        window="hann",
        center=True,
    )
    return torch.from_numpy(np.clip(wav, -1.0, 1.0).astype(np.float32))


def decode_with_vocos_onnx(
    session: ort.InferenceSession,
    features: torch.Tensor,
) -> torch.Tensor:
    outputs = session.get_outputs()
    if len(outputs) >= 3:
        return decode_with_mel_spec_onnx(session, features)

    features = features.permute(0, 2, 1).contiguous()
    input_name = session.get_inputs()[0].name
    audio = session.run(None, {input_name: features.numpy()})[0]
    return torch.from_numpy(audio).squeeze(0)


def main() -> None:
    args = get_parser().parse_args()
    onnx_dir = Path(args.onnx_model_dir)
    vocos_path = resolve_vocoder_onnx(onnx_dir, args.vocoder_onnx)

    from zipvoice.onnx_inference import ViZipVoiceOnnxTTS
    from zipvoice.onnx_inference.runtime import default_onnx_threads, onnx_providers

    num_thread = (
        default_onnx_threads()
        if args.num_thread is None
        else max(1, int(args.num_thread))
    )
    providers = onnx_providers()

    start_t = dt.datetime.now()
    tts = ViZipVoiceOnnxTTS(
        onnx_dir=onnx_dir,
        use_int4=bool(args.use_int4),
        vocoder_onnx=str(vocos_path) if vocos_path else None,
        vocoder_path=args.vocoder_path,
        num_threads=num_thread,
    )
    logging.info("ORT threads=%d | provider=%s", num_thread, providers[0])

    metrics = tts.synthesize(
        prompt_wav=args.prompt_wav,
        prompt_text=args.prompt_text,
        text=args.text,
        output_path=args.res_wav_path,
        num_step=int(args.num_step),
        guidance_scale=float(args.guidance_scale),
        speed=float(args.speed),
        t_shift=float(args.t_shift),
        target_rms=float(args.target_rms),
        feat_scale=float(args.feat_scale),
        max_duration=float(args.max_duration),
        remove_long_sil=bool(args.remove_long_sil),
        normalize_vietnamese=not args.no_vietnamese_normalize,
        split_sentences=False,
    )

    elapsed = (dt.datetime.now() - start_t).total_seconds()
    wav_seconds = float(metrics.get("wav_seconds", 0.0))
    logging.info("Saved to %s", args.res_wav_path)
    logging.info(
        "RTF: %.4f | wav: %.2fs | int4: %s | max_duration: %s | vocos_onnx: %s",
        elapsed / wav_seconds if wav_seconds else 0.0,
        wav_seconds,
        args.use_int4,
        args.max_duration,
        vocos_path is not None,
    )


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
        force=True,
    )
    main()
