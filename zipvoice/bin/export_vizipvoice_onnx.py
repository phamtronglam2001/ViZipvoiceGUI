#!/usr/bin/env python3
"""CLI: export ViZipVoice + Vocos ONNX int4 bundle (local inference via infer_vizipvoice_onnx)."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from zipvoice.export.onnx_bundle import export_onnx_bundle
from zipvoice.vizipvoice import DEFAULT_CHECKPOINT_NAME


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Export ZipVoice checkpoint + Vocos to ONNX int4 (ViZipVoice character tokenizer).",
    )
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument(
        "--export-root",
        type=Path,
        default=Path("models"),
        help="Creates export-root/onnx/ and export-root/vocoder/",
    )
    parser.add_argument("--checkpoint-name", default=DEFAULT_CHECKPOINT_NAME)
    parser.add_argument("--vocoder-path", default=None)
    parser.add_argument("--opset-version", type=int, default=17)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--quantize-int4", type=int, default=1)
    parser.add_argument("--keep-baseline", action="store_true")
    parser.add_argument("--skip-vocos", action="store_true")
    parser.add_argument(
        "--deploy-onnx-gui",
        type=Path,
        default=None,
        help="Copy bundle into ZipVoice-Vietnamese-ONNX-GUI repo (models/onnx + models/vocoder).",
    )
    return parser


def main() -> None:
    args = get_parser().parse_args()
    export_onnx_bundle(
        model_dir=args.model_dir,
        export_root=args.export_root,
        checkpoint_name=args.checkpoint_name,
        vocoder_path=args.vocoder_path,
        quantize_int4=bool(args.quantize_int4),
        block_size=args.block_size,
        opset_version=args.opset_version,
        skip_vocos=args.skip_vocos,
        keep_baseline=args.keep_baseline,
        deploy_onnx_gui_root=args.deploy_onnx_gui,
    )


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        level=logging.INFO,
        force=True,
    )
    main()
