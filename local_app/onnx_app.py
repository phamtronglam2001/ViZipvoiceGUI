#!/usr/bin/env python3
"""Gradio GUI — ViZipVoice ONNX inference (same UX as vizipvoice-local)."""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from pathlib import Path

import gradio as gr

from local_app.branding import AUTHOR_LINE, FORK_PURPOSE, HF_MODEL_URL, HF_SPACE_URL
from local_app.presets_io import (
    BUILTIN_DEFAULT_KEY,
    on_load_preset,
    on_save_preset,
    preset_dropdown_choices,
)
from local_app.app import (
    DEMO_TEXT,
    MP3_BITRATE_CHOICES,
    default_ffmpeg_dir_str,
    default_ref,
    ffmpeg_available,
    format_ffmpeg_status,
    load_ref_prompts,
    load_txt_for_preview,
    parse_mp3_bitrate,
    parse_pipeline_arg,
    preview_normalizer,
    refresh_ffmpeg_ui,
    refs_by_label,
    resolve_ffmpeg_dir,
    resolve_ffmpeg_path,
    resolve_model_dir,
    select_ref,
    wav_to_mp3,
)
from local_app.ref_audio_bundle import sync_bundled_ref_audio
from zipvoice.onnx_inference.engine import get_onnx_tts
from zipvoice.onnx_inference.vocoder_onnx import VOCODER_BASELINE, VOCODER_INT4
from zipvoice.onnx_inference.providers import predict_runtime_device_summary
from zipvoice.utils.audio_io import configure_pydub_ffmpeg
from zipvoice.onnx_inference.runtime import default_onnx_threads
from zipvoice.tokenizer.vi_normalizer import DEFAULT_PIPELINE, STEP_LABELS, format_pipeline_label

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ONNX_DIR = REPO_ROOT / "models" / "onnx"
DEFAULT_VOCODER_ONNX = REPO_ROOT / "models" / "vocoder" / VOCODER_BASELINE
DEFAULT_VOCODER_INT4 = REPO_ROOT / "models" / "vocoder" / VOCODER_INT4
OUTPUT_DIR = REPO_ROOT / "output" / "gradio_onnx"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def default_use_onnx_gpu() -> bool:
    env = os.environ.get("ZIPVOICE_ONNX_GPU", "").strip().lower()
    if env in {"1", "true", "yes"}:
        return True
    marker = REPO_ROOT / ".install_mode_onnx"
    if marker.is_file() and marker.read_text(encoding="utf-8").strip().lower() == "gpu":
        return True
    return False


def resolve_onnx_dir() -> Path:
    env = os.getenv("VIZIPVOICE_ONNX_DIR", "").strip()
    if env:
        return Path(env).resolve()
    return DEFAULT_ONNX_DIR.resolve()


def resolve_vocoder_onnx_path() -> str:
    env = os.getenv("VIZIPVOICE_VOCODER_ONNX", "").strip()
    if env:
        return env
    if DEFAULT_VOCODER_ONNX.is_file():
        return str(DEFAULT_VOCODER_ONNX)
    if DEFAULT_VOCODER_INT4.is_file():
        return str(DEFAULT_VOCODER_INT4)
    return ""


def onnx_ready_report(onnx_dir: Path) -> str:
    lines = [f"**ONNX dir:** `{onnx_dir}`"]
    for name in (
        "text_encoder_int4.onnx",
        "fm_decoder_int4.onnx",
        "tokens.txt",
    ):
        path = onnx_dir / name
        ok = path.is_file()
        lines.append(f"- `{name}`: {'OK' if ok else '**thiếu**'}")
    voc = resolve_vocoder_onnx_path()
    lines.append(
        f"- vocoder: `{'OK — ' + voc if voc else f'thiếu models/vocoder/{VOCODER_BASELINE} hoặc {VOCODER_INT4}'}`"
    )
    return "\n".join(lines)


def generate(
    model_dir: str,
    onnx_dir: str,
    use_int4: bool,
    vocoder_onnx: str,
    use_onnx_gpu: bool,
    force_cpu: bool,
    onnx_threads: int,
    ffmpeg_dir: str,
    mp3_bitrate_choice: str,
    ref_label: str,
    prompt_audio: str | None,
    prompt_text: str,
    text: str,
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
) -> tuple[str, str]:
    if not text or not text.strip():
        raise gr.Error("Nhập text cần sinh trước khi chạy inference.")

    onnx_path = Path(onnx_dir)
    if not onnx_path.is_dir():
        raise gr.Error(f"ONNX dir không tồn tại: {onnx_dir}")

    ref = refs_by_label(model_dir)[ref_label]
    prompt_wav = prompt_audio or ref.audio_path
    final_prompt_text = prompt_text.strip() or ref.text
    norm_pipeline = parse_pipeline_arg(norm_pipeline_raw)
    vocoder = vocoder_onnx.strip() or None

    output_path = OUTPUT_DIR / f"vizipvoice_onnx_{int(time.time())}_{uuid.uuid4().hex[:8]}.wav"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    configure_pydub_ffmpeg(ffmpeg_dir)

    start = time.time()
    try:
        threads = None if int(onnx_threads) <= 0 else int(onnx_threads)
        tts = get_onnx_tts(
            str(onnx_path.resolve()),
            bool(use_int4),
            vocoder,
            num_threads=threads,
            use_gpu=bool(use_onnx_gpu),
            force_cpu=bool(force_cpu),
        )
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
        logging.exception("ViZipVoice ONNX inference failed")
        raise gr.Error(str(exc)) from exc

    elapsed = time.time() - start
    ffmpeg = resolve_ffmpeg_path(ffmpeg_dir)
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
        "backend": "ONNX",
        "onnx_dir": str(onnx_path),
        "use_int4": bool(use_int4),
        "use_gpu": bool(use_onnx_gpu),
        "force_cpu": bool(force_cpu),
        "ort_threads": metrics.get("ort_threads"),
        "provider_label": metrics.get("provider_label"),
        "vocoder_onnx": metrics.get("vocoder_onnx"),
        "ref_model_dir": model_dir,
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


def refresh_onnx_perf_ui(use_onnx_gpu: bool, force_cpu: bool) -> str:
    return predict_runtime_device_summary(
        bool(use_onnx_gpu),
        force_cpu=bool(force_cpu),
    )


def build_app(model_dir: Path, onnx_dir: Path) -> gr.Blocks:
    model_dir_str = str(model_dir)
    onnx_dir_str = str(onnx_dir)
    ref = default_ref(model_dir_str)
    choices = [item.label for item in load_ref_prompts(model_dir_str)]
    default_pipeline_str = ", ".join(DEFAULT_PIPELINE)
    norm_choices = [key for key in STEP_LABELS if key != "none"]
    default_vocoder = resolve_vocoder_onnx_path()
    default_ffmpeg_dir = default_ffmpeg_dir_str()
    has_ffmpeg = ffmpeg_available(default_ffmpeg_dir)
    output_label = "Output (MP3)" if has_ffmpeg else "Output (WAV)"

    with gr.Blocks(title="ViZipVoice ONNX") as demo:
        gr.Markdown(
            "# ViZipVoice ONNX (Local)\n"
            f"{AUTHOR_LINE}\n\n"
            f"_{FORK_PURPOSE}_\n\n"
            f"ONNX: `{onnx_dir}` · Ref audio: `{model_dir / 'audio'}`\n\n"
            f"[HF Space]({HF_SPACE_URL}) · [Model weights]({HF_MODEL_URL})\n\n"
            + onnx_ready_report(onnx_dir)
        )

        with gr.Tabs():
            with gr.Tab("TTS (ONNX)"):
                gr.Markdown(
                    "> **Lưu ý ONNX:** Text **dài / đoạn văn** chạy ổn. "
                    "Câu **rất ngắn (1–3 từ)** có thể sai trim, nhiễu hoặc chậm — "
                    "dùng **PyTorch TTS** (`run.bat` → **[1]**)."
                )
                with gr.Row():
                    with gr.Column(scale=1):
                        onnx_model_dir = gr.Textbox(
                            value=onnx_dir_str,
                            label="ONNX model dir",
                        )
                        use_int4 = gr.Checkbox(value=True, label="Dùng int4 (*_int4.onnx)")
                        vocoder_onnx = gr.Textbox(
                            value=default_vocoder,
                            label="Vocoder ONNX path",
                            placeholder=f"models/vocoder/{VOCODER_BASELINE} hoặc {VOCODER_INT4}",
                        )
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
                        text = gr.Textbox(
                            value=DEMO_TEXT,
                            lines=8,
                            label="Text",
                            info="Câu ngắn (1–3 từ): dùng PyTorch [1] thay vì ONNX.",
                        )
                        generate_btn = gr.Button("Generate (ONNX)", variant="primary")

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
                            _preset_choices = preset_dropdown_choices()
                            with gr.Row():
                                preset_dropdown = gr.Dropdown(
                                    label="Preset",
                                    choices=_preset_choices,
                                    value=BUILTIN_DEFAULT_KEY,
                                    scale=2,
                                )
                                preset_save_name = gr.Textbox(
                                    label="Tên lưu",
                                    placeholder="vd: sach_g2p",
                                    scale=2,
                                )
                                preset_save_btn = gr.Button(
                                    "Save preset",
                                    size="sm",
                                    scale=1,
                                )
                            preset_status = gr.Markdown("")
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
                            remove_long_sil = gr.Checkbox(
                                value=False,
                                label="Remove long silence (trong chunk + giữa chunk)",
                            )
                            with gr.Row():
                                crossfade_ms = gr.Slider(0, 300, value=80, step=10, label="Crossfade ms")
                                silence_ms = gr.Slider(0, 800, value=180, step=10, label="Silence ms")
                            with gr.Row():
                                fade_in_ms = gr.Slider(0, 500, value=20, step=10, label="Fade in ms")
                                fade_out_ms = gr.Slider(0, 500, value=80, step=10, label="Fade out ms")

                with gr.Row():
                    output_audio = gr.Audio(type="filepath", label=output_label)
                    status = gr.Textbox(lines=14, label="Status")

            with gr.Tab("Text Normalizer"):
                gr.Markdown(
                    "Xem trước pipeline chuẩn hóa trước khi TTS. "
                    "Mặc định: `soe_vinorm` → `spacing`."
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

            with gr.Tab("Hiệu năng"):
                gr.Markdown(
                    "Tối ưu ONNX Runtime (tham khảo ZipVoice-Vietnamese-ONNX-GUI). "
                    f"GPU: `setup.bat` → [3]. Vocos ONNX GPU: `models/vocoder/{VOCODER_BASELINE}` hoặc `{VOCODER_INT4}`. "
                    "Env: `ZIPVOICE_ONNX_THREADS`, "
                    "`ZIPVOICE_FORCE_CPU=1`. Không có GPU/DLL → tự fallback CPU."
                )
                with gr.Row():
                    use_onnx_gpu = gr.Checkbox(
                        value=default_use_onnx_gpu(),
                        label="GPU (CUDA / DirectML)",
                        info="Giữ workers=1 khi dùng GPU",
                    )
                    force_cpu = gr.Checkbox(
                        value=False,
                        label="Ép CPU",
                    )
                onnx_threads = gr.Slider(
                    0,
                    min(16, os.cpu_count() or 8),
                    value=0,
                    step=1,
                    label="ORT threads (0 = tự động)",
                    info=f"Mặc định: {default_onnx_threads()}",
                )
                runtime_device = gr.Textbox(
                    label="Thiết bị ONNX Runtime",
                    value=predict_runtime_device_summary(default_use_onnx_gpu()),
                    interactive=False,
                    lines=2,
                )
                for ctrl in (use_onnx_gpu, force_cpu):
                    ctrl.change(
                        fn=refresh_onnx_perf_ui,
                        inputs=[use_onnx_gpu, force_cpu],
                        outputs=[runtime_device],
                    )

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
        _preset_outputs = [
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
            preset_status,
        ]
        preset_dropdown.change(
            fn=on_load_preset,
            inputs=[preset_dropdown],
            outputs=_preset_outputs,
        )
        preset_save_btn.click(
            fn=on_save_preset,
            inputs=[
                preset_save_name,
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
            outputs=[preset_dropdown, preset_status],
        )
        generate_btn.click(
            fn=lambda *args: generate(model_dir_str, *args),
            inputs=[
                onnx_model_dir,
                use_int4,
                vocoder_onnx,
                use_onnx_gpu,
                force_cpu,
                onnx_threads,
                ffmpeg_dir,
                mp3_bitrate,
                ref_label,
                prompt_audio,
                prompt_text,
                text,
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

    return demo


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="ViZipVoice ONNX Gradio GUI")
    parser.add_argument("--model-dir", type=Path, default=resolve_model_dir())
    parser.add_argument("--onnx-dir", type=Path, default=resolve_onnx_dir())
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7861)
    parser.add_argument("--share", action="store_true")
    parser.add_argument(
        "--inbrowser",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    onnx_dir = args.onnx_dir.resolve()
    if not model_dir.is_dir():
        raise SystemExit(
            f"Model dir không tồn tại: {model_dir}\nChạy: uv run vizipvoice-download"
        )
    if not onnx_dir.is_dir():
        raise SystemExit(
            f"ONNX dir không tồn tại: {onnx_dir}\n"
            "Export trước: run.bat → [3] hoặc uv run vizipvoice-export-onnx ..."
        )

    sync_bundled_ref_audio(model_dir)
    configure_pydub_ffmpeg(default_ffmpeg_dir_str())
    load_ref_prompts.cache_clear()
    prompts = load_ref_prompts(str(model_dir))
    allowed = {str(Path(item.audio_path).resolve().parent) for item in prompts}
    allowed.add(str(OUTPUT_DIR.resolve()))
    allowed.add(str(model_dir))
    allowed.add(str(onnx_dir))
    bundled_ref = (Path(__file__).resolve().parents[1] / "assets" / "ref_audio").resolve()
    if bundled_ref.is_dir():
        allowed.add(str(bundled_ref))
    voc = resolve_vocoder_onnx_path()
    if voc:
        allowed.add(str(Path(voc).resolve().parent))

    demo = build_app(model_dir, onnx_dir)
    demo.queue(max_size=4).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=args.inbrowser,
        allowed_paths=sorted(allowed),
    )


if __name__ == "__main__":
    main()
