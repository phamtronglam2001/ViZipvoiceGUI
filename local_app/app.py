#!/usr/bin/env python3
"""
Local offline Gradio GUI — mirror of https://huggingface.co/spaces/dinhthuan/ViZipvoice

Chạy sau khi tải model:
  uv run vizipvoice-download
  uv run vizipvoice-local
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
import uuid
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

import gradio as gr

from local_app.branding import AUTHOR_LINE, FORK_PURPOSE, HF_MODEL_URL, HF_SPACE_URL
from zipvoice.tokenizer.vi_normalizer import (
    DEFAULT_PIPELINE,
    STEP_LABELS,
    build_normalize_pipeline,
    format_pipeline_label,
    format_preview_markdown,
    preview_normalize,
)
from zipvoice.vizipvoice import DEFAULT_REPO_ID, ViZipVoiceTTS

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = REPO_ROOT / "models" / "ViZipvoice"
OUTPUT_DIR = REPO_ROOT / "output" / "gradio"
AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
PREFERRED_REFS = ["Đinh-Quyết", "Nhã-Uyên", "MC"]
MP3_BITRATE_CHOICES = ("64 kbps (mặc định)", "128 kbps", "256 kbps")
DEFAULT_FFMPEG_DIR = "ffmpeg"
FFMPEG_EXE_NAMES = ("ffmpeg.exe", "ffmpeg")

DEMO_TEXT = (
    "Chiến tranh luôn là một chủ đề nặng nề, nhưng cũng rất cần được nhắc đến "
    "để con người hiểu rõ hơn giá trị của hòa bình. Khi một cuộc chiến xảy ra, "
    "những gì bị phá hủy không chỉ là nhà cửa, đường sá, trường học hay bệnh viện. "
    "Điều đau lòng nhất chính là sinh mạng con người, là những gia đình bị chia cắt, "
    "là những đứa trẻ phải lớn lên trong sợ hãi, và là những vùng đất từng yên bình "
    "bỗng trở nên hoang tàn."
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


@dataclass(frozen=True)
class RefPrompt:
    label: str
    audio_path: str
    text: str


def resolve_model_dir() -> Path:
    env = os.getenv("VIZIPVOICE_MODEL_DIR", "").strip()
    if env:
        return Path(env).resolve()
    return DEFAULT_MODEL_DIR.resolve()


def sorted_ref_audio_files(audio_dir: Path) -> list[Path]:
    if not audio_dir.is_dir():
        return []

    files = [
        path
        for path in audio_dir.iterdir()
        if path.is_file() and path.suffix.lower() in AUDIO_EXTENSIONS
    ]

    def sort_key(path: Path) -> tuple[int, str]:
        stem = path.stem
        if stem in PREFERRED_REFS:
            return (PREFERRED_REFS.index(stem), stem)
        return (len(PREFERRED_REFS), stem)

    return sorted(files, key=sort_key)


@lru_cache(maxsize=1)
def load_ref_prompts(model_dir: str) -> tuple[RefPrompt, ...]:
    audio_dir = Path(model_dir) / "audio"
    prompts: list[RefPrompt] = []
    for audio_path in sorted_ref_audio_files(audio_dir):
        text_path = audio_path.with_suffix(".txt")
        if not text_path.is_file():
            logging.warning("Missing transcript for %s", audio_path.name)
            continue
        text = text_path.read_text(encoding="utf-8").strip()
        prompts.append(
            RefPrompt(
                label=audio_path.stem,
                audio_path=str(audio_path.resolve()),
                text=text,
            )
        )

    if not prompts:
        raise FileNotFoundError(
            f"No ref audio in {audio_dir}. Chạy: uv run vizipvoice-download"
        )
    return tuple(prompts)


def default_ffmpeg_dir_str() -> str:
    env = os.getenv("VIZIPVOICE_FFMPEG_DIR", "").strip()
    return env or DEFAULT_FFMPEG_DIR


def resolve_ffmpeg_dir(raw: str) -> Path:
    text = (raw or "").strip() or DEFAULT_FFMPEG_DIR
    path = Path(text)
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    return path


def _ffmpeg_candidates_in(base: Path) -> tuple[Path, ...]:
    if base.is_file() and base.name.lower() in {n.lower() for n in FFMPEG_EXE_NAMES}:
        return (base,)
    if not base.is_dir():
        return ()

    names = FFMPEG_EXE_NAMES
    return (
        base / names[0],
        base / names[1],
        base / "bin" / names[0],
        base / "bin" / names[1],
    )


def resolve_ffmpeg_path(ffmpeg_dir_raw: str = "") -> Optional[Path]:
    """Find ffmpeg.exe under a user-chosen folder (relative to repo or absolute)."""
    base = resolve_ffmpeg_dir(ffmpeg_dir_raw)
    for candidate in _ffmpeg_candidates_in(base):
        if candidate.is_file():
            return candidate.resolve()
    return None


def ffmpeg_available(ffmpeg_dir_raw: str = "") -> bool:
    return resolve_ffmpeg_path(ffmpeg_dir_raw) is not None


def format_ffmpeg_status(ffmpeg_dir_raw: str) -> str:
    exe = resolve_ffmpeg_path(ffmpeg_dir_raw)
    folder = resolve_ffmpeg_dir(ffmpeg_dir_raw)
    if exe is not None:
        return f"**ffmpeg:** `{exe}`"
    return (
        f"**ffmpeg:** không tìm thấy trong `{folder}` "
        f"(cần `ffmpeg.exe` hoặc `bin/ffmpeg.exe`) — chỉ xuất WAV."
    )


def parse_device_choice(choice: str) -> Optional[str]:
    if choice == "auto":
        return None
    return choice


def parse_mp3_bitrate(choice: str) -> int:
    if "256" in choice:
        return 256
    if "128" in choice:
        return 128
    return 64


def wav_to_mp3(wav_path: Path, bitrate_kbps: int, ffmpeg: Path) -> Path:

    mp3_path = wav_path.with_suffix(".mp3")
    cmd = [
        str(ffmpeg),
        "-y",
        "-i",
        str(wav_path),
        "-ar",
        "24000",
        "-ac",
        "1",
        "-b:a",
        f"{bitrate_kbps}k",
        str(mp3_path),
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        raise gr.Error(f"Không chạy được ffmpeg: {exc}") from exc

    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise gr.Error(f"ffmpeg lỗi khi chuyển MP3: {detail or result.returncode}")

    if not mp3_path.is_file():
        raise gr.Error("ffmpeg chạy xong nhưng không tạo được file MP3.")
    return mp3_path


def load_txt_for_preview(file_path: Optional[str]) -> str:
    if not file_path:
        return gr.update()
    try:
        return Path(file_path).read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise gr.Error(
            "Không đọc được file TXT — file phải mã hóa UTF-8."
        ) from exc
    except OSError as exc:
        raise gr.Error(f"Không mở được file TXT: {exc}") from exc


def _load_tts(**kwargs) -> ViZipVoiceTTS:
    model_dir = kwargs.pop("model_dir", None)
    repo_id = kwargs.pop("repo_id", DEFAULT_REPO_ID)
    if model_dir is not None:
        return ViZipVoiceTTS(model_dir=model_dir, **kwargs)
    return ViZipVoiceTTS(repo_id=repo_id, **kwargs)


@lru_cache(maxsize=8)
def get_tts(
    model_dir: str,
    device_choice: str = "auto",
    num_threads: int = 1,
    use_fp16: bool = True,
) -> ViZipVoiceTTS:
    device = parse_device_choice(device_choice)
    path = Path(model_dir)
    kwargs = {
        "checkpoint_name": "latest",
        "device": device,
        "num_threads": int(num_threads),
        "use_fp16": bool(use_fp16),
    }
    if path.is_dir():
        logging.info(
            "Loading ViZipVoice from local %s (device=%s, threads=%s, fp16=%s)",
            path,
            device_choice,
            num_threads,
            use_fp16,
        )
        try:
            return _load_tts(model_dir=path, **kwargs)
        except Exception as exc:
            if device_choice in ("auto", "cuda") and (
                device is None or str(device) != "cpu"
            ):
                logging.warning(
                    "GPU load failed (%s) — retrying on CPU", exc
                )
                kwargs["device"] = "cpu"
                return _load_tts(model_dir=path, **kwargs)
            raise

    logging.info(
        "Loading ViZipVoice from Hugging Face %s (device=%s, threads=%s, fp16=%s)",
        DEFAULT_REPO_ID,
        device_choice,
        num_threads,
        use_fp16,
    )
    try:
        return _load_tts(**kwargs)
    except Exception as exc:
        if device_choice in ("auto", "cuda") and (
            device is None or str(device) != "cpu"
        ):
            logging.warning("GPU load failed (%s) — retrying on CPU", exc)
            kwargs["device"] = "cpu"
            return _load_tts(**kwargs)
        raise


def refs_by_label(model_dir: str) -> dict[str, RefPrompt]:
    return {item.label: item for item in load_ref_prompts(model_dir)}


def default_ref(model_dir: str) -> RefPrompt:
    for item in load_ref_prompts(model_dir):
        if item.label == "Đinh-Quyết":
            return item
    return load_ref_prompts(model_dir)[0]


def select_ref(model_dir: str, label: str) -> tuple[str, str]:
    item = refs_by_label(model_dir)[label]
    return item.audio_path, item.text


def parse_pipeline_arg(raw: str) -> list[str]:
    if not raw or not raw.strip():
        return list(DEFAULT_PIPELINE)
    return build_normalize_pipeline([part.strip() for part in raw.split(",") if part.strip()])


def generate(
    model_dir: str,
    ref_label: str,
    prompt_audio: Optional[str],
    prompt_text: str,
    text: str,
    device_choice: str,
    perf_num_threads: int,
    use_fp16: bool,
    ffmpeg_dir: str,
    mp3_bitrate_choice: str,
    num_step: int,
    guidance_scale: float,
    speed: float,
    t_shift: float,
    max_duration: int,
    seed: int,
    normalize_vietnamese: bool,
    norm_pipeline_raw: str,
    split_sentences: bool,
    remove_long_sil: bool,
    crossfade_ms: int,
    silence_ms: int,
    fade_in_ms: int,
    fade_out_ms: int,
) -> tuple[Optional[str], str]:
    if not text or not text.strip():
        raise gr.Error("Nhập text cần sinh trước khi chạy inference.")

    ref = refs_by_label(model_dir)[ref_label]
    prompt_wav = prompt_audio or ref.audio_path
    final_prompt_text = prompt_text.strip() or ref.text
    norm_pipeline = parse_pipeline_arg(norm_pipeline_raw)
    output_path = OUTPUT_DIR / f"vizipvoice_{int(time.time())}_{uuid.uuid4().hex[:8]}.wav"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    tts = get_tts(
        model_dir,
        device_choice=device_choice,
        num_threads=int(perf_num_threads),
        use_fp16=bool(use_fp16),
    )

    start = time.time()
    try:
        metrics = tts.synthesize(
            prompt_wav=prompt_wav,
            prompt_text=final_prompt_text,
            text=text.strip(),
            output_path=output_path,
            num_step=int(num_step),
            guidance_scale=float(guidance_scale),
            speed=float(speed),
            t_shift=float(t_shift),
            max_duration=float(max_duration),
            remove_long_sil=bool(remove_long_sil),
            seed=int(seed),
            normalize_vietnamese=bool(normalize_vietnamese),
            normalize_pipeline=norm_pipeline,
            split_sentences=bool(split_sentences),
            crossfade_ms=int(crossfade_ms),
            silence_ms=int(silence_ms),
            fade_in_ms=int(fade_in_ms),
            fade_out_ms=int(fade_out_ms),
        )
    except Exception as exc:
        logging.exception("ViZipVoice inference failed")
        raise gr.Error(str(exc)) from exc

    elapsed = time.time() - start
    ffmpeg = resolve_ffmpeg_path(ffmpeg_dir)
    deliver_path: str
    if ffmpeg is not None:
        mp3_file = wav_to_mp3(
            output_path,
            parse_mp3_bitrate(mp3_bitrate_choice),
            ffmpeg,
        )
        try:
            output_path.unlink(missing_ok=True)
        except OSError:
            logging.warning("Could not remove temp WAV %s", output_path)
        deliver_path = str(mp3_file)
        output_format = "mp3"
        output_bitrate_kbps = parse_mp3_bitrate(mp3_bitrate_choice)
    else:
        deliver_path = str(output_path)
        output_format = "wav"
        output_bitrate_kbps = None

    status = {
        "checkpoint": Path(tts.checkpoint_path).name,
        "device": str(tts.device),
        "device_requested": device_choice,
        "fp16_autocast": tts.use_fp16,
        "num_threads": int(perf_num_threads),
        "model_dir": model_dir,
        "prompt": Path(prompt_wav).name,
        "normalize_pipeline": format_pipeline_label(norm_pipeline),
        "segments": metrics.get("segments"),
        "wav_seconds": round(metrics.get("wav_seconds", 0.0), 3),
        "rtf": round(metrics.get("rtf", 0.0), 4),
        "elapsed_seconds": round(elapsed, 3),
        "segment_settings": metrics.get("segment_settings", []),
        "output_format": output_format,
        "output_path": deliver_path,
    }
    if output_bitrate_kbps is not None:
        status["mp3_bitrate_kbps"] = output_bitrate_kbps
    if ffmpeg is not None:
        status["ffmpeg"] = str(ffmpeg)
    else:
        status["ffmpeg_dir"] = str(resolve_ffmpeg_dir(ffmpeg_dir))
    return deliver_path, json.dumps(status, ensure_ascii=False, indent=2)


def refresh_ffmpeg_ui(ffmpeg_dir: str) -> tuple:
    exe = resolve_ffmpeg_path(ffmpeg_dir)
    has_ffmpeg = exe is not None
    return (
        gr.update(visible=has_ffmpeg),
        gr.update(label="Output (MP3)" if has_ffmpeg else "Output (WAV)"),
        format_ffmpeg_status(ffmpeg_dir),
    )


def preview_normalizer(text: str, norm_pipeline_raw: str, enabled: bool) -> str:
    pipeline = parse_pipeline_arg(norm_pipeline_raw)
    result = preview_normalize(text, pipeline, enabled=enabled)
    return format_preview_markdown(result)


def build_app(model_dir: Path) -> gr.Blocks:
    model_dir_str = str(model_dir)
    ref = default_ref(model_dir_str)
    choices = [item.label for item in load_ref_prompts(model_dir_str)]
    default_pipeline_str = ", ".join(DEFAULT_PIPELINE)
    norm_choices = [key for key in STEP_LABELS if key != "none"]
    default_ffmpeg_dir = default_ffmpeg_dir_str()
    has_ffmpeg = ffmpeg_available(default_ffmpeg_dir)
    output_label = "Output (MP3)" if has_ffmpeg else "Output (WAV)"

    with gr.Blocks(title="ViZipVoice Local") as demo:
        gr.Markdown(
            "# ViZipVoice GUI (Local Offline)\n"
            f"{AUTHOR_LINE}\n\n"
            f"_{FORK_PURPOSE}_\n\n"
            f"Model: `{model_dir}`\n\n"
            f"[HF Space]({HF_SPACE_URL}) · [Model weights]({HF_MODEL_URL})"
        )

        with gr.Tabs():
            with gr.Tab("TTS"):
                with gr.Row():
                    with gr.Column(scale=1):
                        ref_label = gr.Dropdown(
                            choices=choices,
                            value=ref.label,
                            label="Ref audio",
                        )
                        prompt_audio = gr.Audio(
                            value=ref.audio_path,
                            type="filepath",
                            label="Prompt audio",
                        )
                        prompt_text = gr.Textbox(
                            value=ref.text,
                            lines=4,
                            label="Prompt text",
                        )

                    with gr.Column(scale=1):
                        txt_upload = gr.File(
                            label="Tải TXT (audiobook)",
                            file_types=[".txt"],
                            type="filepath",
                        )
                        text = gr.Textbox(value=DEMO_TEXT, lines=8, label="Text")
                        generate_btn = gr.Button("Generate", variant="primary")

                        with gr.Accordion("Hiệu năng", open=False):
                            device_choice = gr.Radio(
                                choices=["auto", "cuda", "cpu"],
                                value="auto",
                                label="Thiết bị",
                                info="Tự động: CUDA/MPS nếu có, không thì CPU",
                            )
                            perf_num_threads = gr.Slider(
                                1,
                                16,
                                value=1,
                                step=1,
                                label="Số luồng CPU (PyTorch)",
                            )
                            use_fp16 = gr.Checkbox(
                                value=True,
                                label="FP16 trên CUDA (nhanh hơn, tắt nếu lỗi GPU)",
                            )

                        with gr.Accordion("Xuất MP3 / ffmpeg", open=has_ffmpeg):
                            ffmpeg_dir = gr.Textbox(
                                value=default_ffmpeg_dir,
                                label="Thư mục ffmpeg",
                                info=(
                                    "Tuyệt đối hoặc tương đối repo "
                                    "(vd. ffmpeg, ffmpeg/bin, D:/tools/ffmpeg)"
                                ),
                            )
                            ffmpeg_status = gr.Markdown(
                                format_ffmpeg_status(default_ffmpeg_dir)
                            )
                            mp3_bitrate = gr.Dropdown(
                                choices=list(MP3_BITRATE_CHOICES),
                                value=MP3_BITRATE_CHOICES[0],
                                label="Bitrate MP3",
                                info="24 kHz mono",
                                visible=has_ffmpeg,
                            )

                        with gr.Accordion("Advanced", open=False):
                            with gr.Row():
                                num_step = gr.Slider(4, 64, value=16, step=1, label="Steps")
                                guidance_scale = gr.Slider(
                                    0.0, 5.0, value=1.0, step=0.1, label="Guidance"
                                )
                            with gr.Row():
                                speed = gr.Slider(0.5, 1.5, value=1.0, step=0.05, label="Speed")
                                t_shift = gr.Slider(0.1, 1.0, value=0.5, step=0.05, label="T-shift")
                            with gr.Row():
                                max_duration = gr.Slider(
                                    10, 200, value=100, step=5, label="Max duration"
                                )
                                seed = gr.Number(value=666, precision=0, label="Seed")
                            normalize_vietnamese = gr.Checkbox(
                                value=True,
                                label="Vietnamese normalization",
                            )
                            norm_pipeline_raw = gr.Textbox(
                                value=default_pipeline_str,
                                label="Normalize pipeline (comma-separated)",
                                info=", ".join(norm_choices),
                            )
                            split_sentences = gr.Checkbox(value=True, label="Split sentences")
                            remove_long_sil = gr.Checkbox(value=False, label="Remove long silence")
                            with gr.Row():
                                crossfade_ms = gr.Slider(0, 300, value=80, step=10, label="Crossfade ms")
                                silence_ms = gr.Slider(0, 800, value=180, step=10, label="Silence ms")
                            with gr.Row():
                                fade_in_ms = gr.Slider(0, 500, value=20, step=10, label="Fade in ms")
                                fade_out_ms = gr.Slider(0, 500, value=80, step=10, label="Fade out ms")

                with gr.Row():
                    output_audio = gr.Audio(type="filepath", label=output_label)
                    status = gr.Textbox(lines=12, label="Status")

                txt_upload.change(
                    fn=load_txt_for_preview,
                    inputs=[txt_upload],
                    outputs=[text],
                )
                ffmpeg_dir.change(
                    fn=refresh_ffmpeg_ui,
                    inputs=[ffmpeg_dir],
                    outputs=[mp3_bitrate, output_audio, ffmpeg_status],
                )
                ref_label.change(
                    fn=lambda label: select_ref(model_dir_str, label),
                    inputs=[ref_label],
                    outputs=[prompt_audio, prompt_text],
                )
                generate_btn.click(
                    fn=lambda *args: generate(model_dir_str, *args),
                    inputs=[
                        ref_label,
                        prompt_audio,
                        prompt_text,
                        text,
                        device_choice,
                        perf_num_threads,
                        use_fp16,
                        ffmpeg_dir,
                        mp3_bitrate,
                        num_step,
                        guidance_scale,
                        speed,
                        t_shift,
                        max_duration,
                        seed,
                        normalize_vietnamese,
                        norm_pipeline_raw,
                        split_sentences,
                        remove_long_sil,
                        crossfade_ms,
                        silence_ms,
                        fade_in_ms,
                        fade_out_ms,
                    ],
                    outputs=[output_audio, status],
                )

            with gr.Tab("Text Normalizer"):
                gr.Markdown(
                    "Xem trước pipeline chuẩn hóa trước khi TTS. "
                    "Mặc định: `soe_vinorm` → `spacing`. "
                    "Thử thêm `sea_g2p` hoặc `period_break` nếu cần."
                )
                norm_input = gr.Textbox(
                    value="Hôm nay là 8/6/2026, GDP tăng 6.5%, nhiệt độ 35°C.",
                    lines=6,
                    label="Raw text",
                )
                norm_enabled = gr.Checkbox(value=True, label="Bật chuẩn hóa")
                norm_pipeline_preview = gr.Textbox(
                    value=default_pipeline_str,
                    label="Pipeline (comma-separated)",
                )
                norm_preview_btn = gr.Button("Preview normalize", variant="secondary")
                norm_output = gr.Markdown()

                norm_preview_btn.click(
                    fn=preview_normalizer,
                    inputs=[norm_input, norm_pipeline_preview, norm_enabled],
                    outputs=[norm_output],
                )

    return demo


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="ViZipVoice local Gradio GUI")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=resolve_model_dir(),
        help="Local model directory (default: models/ViZipvoice)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument(
        "--inbrowser",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Tu mo trinh duyet khi server san sang (mac dinh: bat)",
    )
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    if not model_dir.is_dir():
        raise SystemExit(
            f"Model dir không tồn tại: {model_dir}\n"
            "Chạy: uv run vizipvoice-download"
        )

    prompts = load_ref_prompts(str(model_dir))
    allowed = {str(Path(item.audio_path).resolve().parent) for item in prompts}
    allowed.add(str(OUTPUT_DIR.resolve()))
    allowed.add(str(model_dir))

    demo = build_app(model_dir)
    demo.queue(max_size=4).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=args.inbrowser,
        allowed_paths=sorted(allowed),
    )


if __name__ == "__main__":
    main()
