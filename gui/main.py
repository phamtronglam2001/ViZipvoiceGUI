#!/usr/bin/env python3
"""ViZipVoice local Slint GUI — test CLI & ONNX export."""

from __future__ import annotations

import sys
from pathlib import Path

import slint
from slint import Timer

from runner import CommandRunner, REPO_ROOT

GUI_DIR = Path(__file__).resolve().parent


def pick_file(title: str, filetypes: list[tuple[str, str]] | None = None) -> str:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askopenfilename(title=title, filetypes=filetypes or [("All", "*.*")])
    root.destroy()
    return path or ""


def pick_directory(title: str) -> str:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.askdirectory(title=title, initialdir=str(REPO_ROOT))
    root.destroy()
    return path or ""


def pick_save_file(title: str) -> str:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    path = filedialog.asksaveasfilename(
        title=title,
        defaultextension=".wav",
        filetypes=[("WAV", "*.wav"), ("All", "*.*")],
    )
    root.destroy()
    return path or ""


def main() -> None:
    ui = slint.load_file(
        GUI_DIR / "vizipvoice_gui.slint",
        include_paths=[GUI_DIR],
    )
    window = ui.MainWindow()

    log_buffer: list[str] = ["ViZipVoice GUI ready.\n"]

    def flush_log() -> None:
        window.log_text = "".join(log_buffer[-400:])

    def append_log(text: str) -> None:
        log_buffer.append(text)
        flush_log()

    def on_done(code: int) -> None:
        window.busy = False
        if code == 0:
            window.status_text = "Done ✓"
            append_log("\n✓ Hoàn thành.\n")
        else:
            window.status_text = f"Failed (exit {code})"
            append_log(f"\n✗ Thoát với mã {code}.\n")

    runner = CommandRunner(on_log=append_log, on_done=on_done)

    # Poll busy state from runner thread completion
    def poll_timer() -> None:
        window.busy = runner.busy

    timer = Timer()
    timer.start(slint.TimerMode.Repeated, lambda: poll_timer(), 250)

    # Defaults relative to repo
    window.infer_output_wav = str(REPO_ROOT / "output.wav")
    window.onnx_output_wav = str(REPO_ROOT / "output_onnx.wav")
    window.export_model_dir = str(REPO_ROOT / "models" / "ViZipvoice")
    window.export_onnx_dir = str(REPO_ROOT / "models")
    window.onnx_model_dir = str(REPO_ROOT / "models" / "onnx")
    window.download_dir = str(REPO_ROOT / "models" / "ViZipvoice")

    window.export_deploy_path = ""

    window.on_toggle_theme = lambda: None  # bound via dark-mode property

    window.on_clear_log = lambda: (log_buffer.clear(), flush_log(), setattr(window, "status_text", "Idle"))

    def browse_infer(field: str) -> None:
        if field == "model-dir":
            path = pick_directory("Chọn thư mục model ViZipVoice")
            if path:
                window.infer_model_dir = path
        elif field == "prompt-wav":
            path = pick_file("Chọn prompt WAV", [("Audio", "*.wav *.mp3 *.flac")])
            if path:
                window.infer_prompt_wav = path
        elif field == "output-wav":
            path = pick_save_file("Lưu output WAV")
            if path:
                window.infer_output_wav = path

    window.on_browse_infer = browse_infer

    def browse_onnx(field: str) -> None:
        if field == "model-dir":
            path = pick_directory("Chọn thư mục ONNX")
            if path:
                window.onnx_model_dir = path
        elif field == "prompt-wav":
            path = pick_file("Chọn prompt WAV", [("Audio", "*.wav *.mp3 *.flac")])
            if path:
                window.onnx_prompt_wav = path
        elif field == "output-wav":
            path = pick_save_file("Lưu output ONNX WAV")
            if path:
                window.onnx_output_wav = path

    window.on_browse_onnx = browse_onnx

    def browse_export(field: str) -> None:
        if field in {"model-dir", "onnx-dir"}:
            path = pick_directory("Chọn thư mục")
            if path:
                if field == "model-dir":
                    window.export_model_dir = path
                else:
                    window.export_onnx_dir = path
        elif field == "vocoder-path":
            path = pick_directory("Chọn thư mục Vocos local (optional)")
            if path:
                window.export_vocoder_path = path

    window.on_browse_export = browse_export

    def browse_download(field: str) -> None:
        path = pick_directory("Chọn thư mục tải model")
        if path:
            window.download_dir = path

    window.on_browse_download = browse_download

    def run_inference() -> None:
        if not window.infer_prompt_wav:
            append_log("⚠ Chọn prompt WAV trước.\n")
            return
        window.busy = True
        window.status_text = "Running infer_vizipvoice…"

        args = [
            "--prompt-wav",
            window.infer_prompt_wav,
            "--prompt-text",
            window.infer_prompt_text,
            "--text",
            window.infer_text,
            "--res-wav-path",
            window.infer_output_wav,
            "--num-step",
            str(window.infer_num_step),
            "--guidance-scale",
            window.infer_guidance_scale,
            "--speed",
            window.infer_speed,
            "--seed",
            str(window.infer_seed),
        ]
        if not window.infer_use_hf and window.infer_model_dir:
            args.extend(["--model-dir", window.infer_model_dir])
        if window.infer_no_vn_normalize:
            args.append("--no-vietnamese-normalize")

        runner.python_module("zipvoice.bin.infer_vizipvoice", *args)

    window.on_run_inference = run_inference

    def run_onnx_inference() -> None:
        if not window.onnx_prompt_wav:
            append_log("⚠ Chọn prompt WAV trước.\n")
            return
        window.busy = True
        window.status_text = "Running ONNX inference…"

        args = [
            "--onnx-model-dir",
            window.onnx_model_dir,
            "--prompt-wav",
            window.onnx_prompt_wav,
            "--prompt-text",
            window.onnx_prompt_text,
            "--text",
            window.onnx_text,
            "--res-wav-path",
            window.onnx_output_wav,
            "--num-step",
            str(window.onnx_num_step),
            "--guidance-scale",
            window.onnx_guidance_scale,
            "--speed",
            window.onnx_speed,
        ]
        if window.onnx_use_int4:
            args.append("--use-int4")

        runner.python_module("zipvoice.bin.infer_vizipvoice_onnx", *args)

    window.on_run_onnx_inference = run_onnx_inference

    def run_export() -> None:
        window.busy = True
        window.status_text = "Exporting ONNX 4-bit…"

        args = [
            "--model-dir",
            window.export_model_dir,
            "--export-root",
            window.export_onnx_dir,
            "--checkpoint-name",
            window.export_checkpoint,
        ]
        if window.export_quantize_4bit:
            args.extend(["--quantize-int4", "1"])
        else:
            args.extend(["--quantize-int4", "0"])
        if window.export_vocoder_path:
            args.extend(["--vocoder-path", window.export_vocoder_path])
        if window.export_skip_vocos:
            args.append("--skip-vocos")
        if window.export_deploy_onnx_gui:
            args.extend(["--deploy-onnx-gui", window.export_deploy_path])

        runner.python_module("zipvoice.bin.export_vizipvoice_onnx", *args)

    window.on_run_export = run_export

    def run_download() -> None:
        window.busy = True
        window.status_text = "Downloading from Hugging Face…"
        runner.huggingface_download(window.download_repo, window.download_dir)

    window.on_run_download = run_download

    window.run()


if __name__ == "__main__":
    main()
