"""ViZipVoice ONNX TTS engine — mirrors ViZipVoiceTTS.synthesize() for Gradio/CLI."""

from __future__ import annotations

import json
import logging
import tempfile
import time
from functools import lru_cache
from pathlib import Path
from typing import Optional, Union

import onnxruntime as ort
import torch
from lhotse.utils import fix_random_seed

from zipvoice.bin.infer_zipvoice import get_vocoder
from zipvoice.bin.infer_zipvoice_onnx import OnnxModel, sample
from zipvoice.onnx_inference.runtime import (
    default_onnx_threads,
    make_session_options,
    onnx_providers,
)
from zipvoice.tokenizer.tokenizer import SimpleTokenizer
from zipvoice.tokenizer.vi_normalizer import DEFAULT_PIPELINE
from zipvoice.utils.feature import VocosFbank
from zipvoice.utils.infer import (
    add_punctuation,
    chunk_tokens_punctuation,
    cross_fade_concat,
    load_prompt_wav,
    remove_silence,
    rms_norm,
)
from zipvoice.vizipvoice import (
    get_sentence_inference_params,
    normalize_vietnamese_text,
    postprocess_audio_segments,
    split_text_into_sentences,
    wav_seconds,
)

logger = logging.getLogger(__name__)


def resolve_onnx_paths(onnx_dir: Path, use_int4: bool) -> tuple[Path, Path]:
    suffix = "_int4" if use_int4 else ""
    text_encoder = onnx_dir / f"text_encoder{suffix}.onnx"
    fm_decoder = onnx_dir / f"fm_decoder{suffix}.onnx"
    if not text_encoder.is_file():
        raise FileNotFoundError(f"Missing {text_encoder}")
    if not fm_decoder.is_file():
        raise FileNotFoundError(f"Missing {fm_decoder}")
    return text_encoder, fm_decoder


def resolve_vocoder_onnx(onnx_dir: Path, vocoder_onnx: str | Path | None) -> Path | None:
    if vocoder_onnx:
        path = Path(vocoder_onnx)
        if path.is_file():
            return path

    candidates = [
        onnx_dir.parent / "vocoder" / "mel_spec_24khz.onnx",
        onnx_dir / "mel_spec_24khz.onnx",
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
    if len(session.get_outputs()) >= 3:
        return decode_with_mel_spec_onnx(session, features)

    features = features.permute(0, 2, 1).contiguous()
    input_name = session.get_inputs()[0].name
    audio = session.run(None, {input_name: features.numpy()})[0]
    return torch.from_numpy(audio).squeeze(0)


class ViZipVoiceOnnxTTS:
    """ONNX inference for ViZipVoice export bundle (SimpleTokenizer + int4 optional)."""

    def __init__(
        self,
        onnx_dir: Union[str, Path],
        *,
        use_int4: bool = True,
        vocoder_onnx: Union[str, Path, None] = None,
        vocoder_path: str | None = None,
        num_threads: int | None = None,
    ) -> None:
        self.onnx_dir = Path(onnx_dir).resolve()
        self.use_int4 = bool(use_int4)
        self.vocoder_path = vocoder_path
        self.num_threads = (
            default_onnx_threads() if num_threads is None else max(1, int(num_threads))
        )
        self.providers = onnx_providers()

        token_file = self.onnx_dir / "tokens.txt"
        if not token_file.is_file():
            token_file = self.onnx_dir.parent / "tokens.txt"
        if not token_file.is_file():
            raise FileNotFoundError(f"tokens.txt not found near {self.onnx_dir}")

        config_path = self.onnx_dir / "config.json"
        if not config_path.is_file():
            config_path = self.onnx_dir / "model.json"
        with config_path.open(encoding="utf-8") as handle:
            model_config = json.load(handle)

        self.sampling_rate = int(model_config["feature"]["sampling_rate"])
        self.tokenizer = SimpleTokenizer(token_file=str(token_file))
        self.feature_extractor = VocosFbank()

        te_path, fm_path = resolve_onnx_paths(self.onnx_dir, self.use_int4)
        self.onnx_model = OnnxModel(
            text_encoder_path=str(te_path),
            fm_decoder_path=str(fm_path),
            num_thread=self.num_threads,
            providers=self.providers,
        )
        logger.info(
            "ZipVoice ONNX | threads=%d | providers=%s | int4=%s",
            self.num_threads,
            self.providers[0],
            self.use_int4,
        )

        vocos_path = resolve_vocoder_onnx(self.onnx_dir, vocoder_onnx)
        self._vocos_session: ort.InferenceSession | None = None
        self._vocos_path = vocos_path
        if vocos_path is not None:
            self._vocos_session = ort.InferenceSession(
                str(vocos_path),
                sess_options=make_session_options(self.num_threads),
                providers=self.providers,
            )
            logger.info("Vocoder ONNX: %s", vocos_path.name)
        else:
            logger.warning("No vocoder ONNX — will use PyTorch Vocos fallback")

        self._prompt_cache_key: tuple | None = None
        self._prompt_features: torch.Tensor | None = None
        self._prompt_rms: float = 1.0
        self._prompt_tokens: list[list[int]] | None = None
        self._prompt_text: str = ""
        self._prompt_duration: float = 0.0

    def _prepare_prompt(
        self,
        prompt_wav: Union[str, Path],
        prompt_text: str,
        target_rms: float,
        feat_scale: float,
    ) -> None:
        prompt_text = add_punctuation(prompt_text.strip())
        key = (
            str(Path(prompt_wav).resolve()),
            prompt_text,
            target_rms,
            feat_scale,
        )
        if key == self._prompt_cache_key:
            return

        prompt_wav_t = load_prompt_wav(str(prompt_wav), sampling_rate=self.sampling_rate)
        prompt_wav_t = remove_silence(
            prompt_wav_t,
            self.sampling_rate,
            only_edge=False,
            trail_sil=200,
        )
        prompt_wav_t, prompt_rms = rms_norm(prompt_wav_t, target_rms)
        self._prompt_duration = float(prompt_wav_t.shape[-1]) / self.sampling_rate

        if self._prompt_duration > 10:
            logger.warning(
                "Prompt wav %.1fs — nên dùng 1–3s để clone ổn định.",
                self._prompt_duration,
            )

        prompt_features = self.feature_extractor.extract(
            prompt_wav_t,
            sampling_rate=self.sampling_rate,
        )
        self._prompt_features = prompt_features.unsqueeze(0) * feat_scale
        self._prompt_rms = float(prompt_rms)
        self._prompt_text = prompt_text
        self._prompt_tokens = self.tokenizer.texts_to_token_ids([prompt_text])
        self._prompt_cache_key = key

    def _decode_mel(self, pred_features: torch.Tensor, feat_scale: float) -> torch.Tensor:
        pred_mel = pred_features / feat_scale
        if self._vocos_session is not None:
            wav = decode_with_vocos_onnx(self._vocos_session, pred_mel)
        else:
            vocoder = get_vocoder(self.vocoder_path)
            vocoder.eval()
            pred = pred_mel.permute(0, 2, 1)
            wav = vocoder.decode(pred).squeeze(1).clamp(-1, 1)
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        return wav

    def _synthesize_chunk(
        self,
        token_ids: list[int],
        *,
        num_step: int,
        guidance_scale: float,
        speed: float,
        t_shift: float,
        target_rms: float,
        feat_scale: float,
    ) -> torch.Tensor:
        assert self._prompt_features is not None
        assert self._prompt_tokens is not None

        pred_features = sample(
            model=self.onnx_model,
            tokens=[token_ids],
            prompt_tokens=self._prompt_tokens,
            prompt_features=self._prompt_features,
            speed=float(speed),
            t_shift=float(t_shift),
            guidance_scale=float(guidance_scale),
            num_step=int(num_step),
        )
        wav = self._decode_mel(pred_features, feat_scale).clamp(-1, 1)
        if self._prompt_rms < target_rms:
            wav = wav * self._prompt_rms / target_rms
        return wav

    def _chunk_token_ids(
        self,
        text: str,
        *,
        speed: float,
        max_duration: float,
    ) -> list[list[int]]:
        """Chunk long text in reading order (matches infer_zipvoice_onnx.generate_sentence).

        ONNX runs one chunk per ``sample()`` call, so we must not use ``batchify_tokens``,
        which sorts chunks by length for batched GPU inference and scrambles concat order.
        """
        assert self._prompt_tokens is not None

        text = add_punctuation(text.strip())
        tokens_str = self.tokenizer.texts_to_tokens([text])[0]
        prompt_tokens_str = self.tokenizer.texts_to_tokens([self._prompt_text])[0]

        token_duration = self._prompt_duration / (max(len(prompt_tokens_str), 1) * speed)
        max_tokens = max(1, int((25 - self._prompt_duration) / max(token_duration, 1e-6)))
        chunked_tokens_str = chunk_tokens_punctuation(tokens_str, max_tokens=max_tokens)
        return self.tokenizer.tokens_to_token_ids(chunked_tokens_str)

    def _synthesize_sentence(
        self,
        text: str,
        *,
        num_step: int,
        guidance_scale: float,
        speed: float,
        t_shift: float,
        target_rms: float,
        feat_scale: float,
        max_duration: float,
        remove_long_sil: bool,
    ) -> torch.Tensor:
        chunks = self._chunk_token_ids(text, speed=speed, max_duration=max_duration)
        chunked_wavs: list[torch.Tensor] = []
        for token_ids in chunks:
            chunked_wavs.append(
                self._synthesize_chunk(
                    token_ids,
                    num_step=num_step,
                    guidance_scale=guidance_scale,
                    speed=speed,
                    t_shift=t_shift,
                    target_rms=target_rms,
                    feat_scale=feat_scale,
                )
            )

        if len(chunked_wavs) == 1:
            final_wav = chunked_wavs[0]
        else:
            final_wav = cross_fade_concat(
                chunked_wavs,
                fade_duration=0.1,
                sample_rate=self.sampling_rate,
            )

        return remove_silence(
            final_wav,
            self.sampling_rate,
            only_edge=(not remove_long_sil),
            trail_sil=0,
        )

    def synthesize(
        self,
        prompt_wav: Union[str, Path],
        prompt_text: str,
        text: str,
        output_path: Union[str, Path] = "output_onnx.wav",
        num_step: int = 16,
        guidance_scale: float = 1.0,
        speed: float = 1.0,
        t_shift: float = 0.5,
        target_rms: float = 0.1,
        feat_scale: float = 0.1,
        max_duration: float = 100,
        remove_long_sil: bool = False,
        seed: Optional[int] = 666,
        normalize_vietnamese: bool = True,
        normalize_pipeline: Optional[list[str]] = None,
        split_sentences: bool = True,
        crossfade_ms: int = 80,
        silence_ms: int = 180,
        fade_in_ms: int = 20,
        fade_out_ms: int = 80,
    ) -> dict:
        if seed is not None and seed >= 0:
            fix_random_seed(int(seed))

        norm_pipeline = normalize_pipeline if normalize_pipeline is not None else DEFAULT_PIPELINE
        prompt_text = normalize_vietnamese_text(
            prompt_text,
            enabled=normalize_vietnamese,
            pipeline=norm_pipeline,
        )
        text = normalize_vietnamese_text(
            text,
            enabled=normalize_vietnamese,
            pipeline=norm_pipeline,
        )
        target_sentences = split_text_into_sentences(text) if split_sentences else [text]
        if not target_sentences:
            raise ValueError("No valid text to synthesize.")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        self._prepare_prompt(prompt_wav, prompt_text, target_rms, feat_scale)

        segment_paths: list[Path] = []
        segment_settings: list[dict] = []
        start_time = time.time()

        with tempfile.TemporaryDirectory(
            prefix=f"{output_path.stem}_onnx_segments_",
            dir=str(output_path.parent),
        ) as segment_dir_name:
            segment_dir = Path(segment_dir_name)
            for index, sentence in enumerate(target_sentences, start=1):
                sentence_num_step, sentence_speed, word_count = get_sentence_inference_params(
                    sentence=sentence,
                    base_num_step=int(num_step),
                    base_speed=float(speed),
                )
                wav = self._synthesize_sentence(
                    sentence,
                    num_step=sentence_num_step,
                    guidance_scale=float(guidance_scale),
                    speed=sentence_speed,
                    t_shift=float(t_shift),
                    target_rms=float(target_rms),
                    feat_scale=float(feat_scale),
                    max_duration=float(max_duration),
                    remove_long_sil=bool(remove_long_sil),
                )
                segment_path = segment_dir / f"segment_{index:03d}.wav"
                from zipvoice.utils.audio_io import save_audio

                save_audio(segment_path, wav.cpu(), self.sampling_rate)
                segment_paths.append(segment_path)
                segment_settings.append(
                    {
                        "index": index,
                        "word_count": word_count,
                        "speed": sentence_speed,
                        "num_step": sentence_num_step,
                        "text": sentence,
                    }
                )

            postprocess_audio_segments(
                segment_paths=segment_paths,
                output_path=output_path,
                sampling_rate=self.sampling_rate,
                crossfade_ms=int(crossfade_ms),
                silence_ms=int(silence_ms),
                fade_in_ms=int(fade_in_ms),
                fade_out_ms=int(fade_out_ms),
            )

        elapsed = time.time() - start_time
        audio_seconds = wav_seconds(output_path)
        return {
            "segments": len(target_sentences),
            "wav_seconds": audio_seconds,
            "rtf": elapsed / audio_seconds if audio_seconds else 0.0,
            "elapsed_seconds": elapsed,
            "segment_settings": segment_settings,
            "onnx_dir": str(self.onnx_dir),
            "use_int4": self.use_int4,
            "vocoder_onnx": str(self._vocos_path) if self._vocos_path else None,
            "max_duration": float(max_duration),
            "remove_long_sil": bool(remove_long_sil),
        }


@lru_cache(maxsize=4)
def get_onnx_tts(
    onnx_dir: str,
    use_int4: bool,
    vocoder_onnx: str | None,
    num_threads: int | None = None,
) -> ViZipVoiceOnnxTTS:
    return ViZipVoiceOnnxTTS(
        onnx_dir=onnx_dir,
        use_int4=use_int4,
        vocoder_onnx=vocoder_onnx,
        num_threads=num_threads,
    )
