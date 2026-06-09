#!/usr/bin/env python3
"""Gradio GUI — export ViZipVoice + Vocos ONNX int4 (local ViZipvoiceGUI)."""

from __future__ import annotations

import json
import logging
import traceback
from pathlib import Path

import gradio as gr

from local_app.branding import AUTHOR_LINE, FORK_PURPOSE
from zipvoice.export.onnx_bundle import export_onnx_bundle

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = REPO_ROOT / "models" / "ViZipvoice"
DEFAULT_EXPORT_ROOT = REPO_ROOT / "models"
DEFAULT_ONNX_GUI = Path("")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def run_export(
    model_dir: str,
    export_root: str,
    checkpoint_name: str,
    vocoder_path: str,
    quantize_int4: bool,
    keep_baseline: bool,
    skip_vocos: bool,
    block_size: int,
    opset_version: int,
    deploy_enabled: bool,
    deploy_path: str,
) -> tuple[str, str]:
    logs: list[str] = []

    def log(msg: str) -> None:
        logs.append(msg.rstrip())

    try:
        deploy_root = Path(deploy_path).resolve() if deploy_enabled and deploy_path.strip() else None
        result = export_onnx_bundle(
            model_dir=Path(model_dir),
            export_root=Path(export_root),
            checkpoint_name=checkpoint_name.strip() or "latest",
            vocoder_path=vocoder_path.strip() or None,
            quantize_int4=quantize_int4,
            keep_baseline=keep_baseline,
            skip_vocos=skip_vocos,
            block_size=int(block_size),
            opset_version=int(opset_version),
            deploy_onnx_gui_root=deploy_root,
            log=log,
        )
        summary = {
            "onnx_dir": str(result.onnx_dir),
            "vocoder_dir": str(result.vocoder_dir),
            "files": result.files,
            "deployed_to": str(result.deployed_to) if result.deployed_to else None,
            "backup_dir": str(result.backup_dir) if result.backup_dir else None,
        }
        return "\n".join(logs), json.dumps(summary, indent=2, ensure_ascii=False)
    except Exception as exc:
        log(f"ERROR: {exc}")
        log(traceback.format_exc())
        return "\n".join(logs), json.dumps({"error": str(exc)}, indent=2)


def build_app() -> gr.Blocks:
    with gr.Blocks(title="ViZipVoice ONNX Export") as demo:
        gr.Markdown(
            "# ViZipVoice → ONNX int4 Export\n"
            f"{AUTHOR_LINE}\n\n"
            f"_{FORK_PURPOSE}_\n\n"
            "Export **ZipVoice** (`text_encoder_int4`, `fm_decoder_int4`) và **Vocos** "
            "(`mel_spec_24khz.onnx`). Test bằng `run_onnx_local.bat` hoặc "
            "`uv run python -m zipvoice.bin.infer_vizipvoice_onnx`.\n\n"
            "**Lưu ý:** ViZipVoice dùng character tokenizer — không tương thích "
            "ZipVoice-Vietnamese-ONNX-GUI (Espeak/phoneme).\n\n"
            "Output: `export-root/onnx/` + `export-root/vocoder/`"
        )

        with gr.Row():
            with gr.Column():
                model_dir = gr.Textbox(
                    value=str(DEFAULT_MODEL_DIR),
                    label="ViZipVoice model dir",
                )
                checkpoint = gr.Textbox(value="latest", label="Checkpoint (latest or checkpoint-*.pt)")
                vocoder_path = gr.Textbox(
                    value="",
                    label="Vocos local path (optional)",
                    placeholder="Để trống = charactr/vocos-mel-24khz",
                )
                export_root = gr.Textbox(
                    value=str(DEFAULT_EXPORT_ROOT),
                    label="Export root (tạo onnx/ + vocoder/)",
                )

            with gr.Column():
                quantize_int4 = gr.Checkbox(value=True, label="Quantize int4 (MatMulNBits)")
                skip_vocos = gr.Checkbox(value=False, label="Bỏ qua Vocos export")
                keep_baseline = gr.Checkbox(
                    value=False,
                    label="Giữ bản FP32 baseline (*.onnx không _int4)",
                )
                with gr.Row():
                    block_size = gr.Slider(64, 256, value=128, step=64, label="Block size")
                    opset = gr.Slider(15, 19, value=17, step=1, label="ONNX opset")

        gr.Markdown("### Deploy (tùy chọn — thường không cần)")
        deploy_enabled = gr.Checkbox(value=False, label="Copy sang repo khác sau export")
        deploy_path = gr.Textbox(
            value="",
            label="Repo path (optional)",
            placeholder="Để trống = chỉ export local",
        )

        export_btn = gr.Button("Export ONNX int4", variant="primary")
        log_box = gr.Textbox(label="Log", lines=16)
        status_json = gr.Textbox(label="Kết quả", lines=8)

        export_btn.click(
            fn=run_export,
            inputs=[
                model_dir,
                export_root,
                checkpoint,
                vocoder_path,
                quantize_int4,
                keep_baseline,
                skip_vocos,
                block_size,
                opset,
                deploy_enabled,
                deploy_path,
            ],
            outputs=[log_box, status_json],
        )

    return demo


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="ViZipVoice ONNX export Gradio GUI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7862)
    parser.add_argument("--share", action="store_true")
    parser.add_argument(
        "--inbrowser",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Tu mo trinh duyet khi server san sang (mac dinh: bat)",
    )
    args = parser.parse_args()

    build_app().queue(max_size=2).launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=args.inbrowser,
    )


if __name__ == "__main__":
    main()
