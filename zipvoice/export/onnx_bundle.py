"""Export ViZipVoice checkpoint + Vocos to ONNX int4 bundle for ZipVoice-Vietnamese-ONNX-GUI."""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

import onnx
import safetensors.torch
import torch
from torch import Tensor, nn

from zipvoice.bin.onnx_export import (
    OnnxFlowMatchingModel,
    OnnxTextModel,
    add_meta_data,
    export_fm_decoder,
    export_text_encoder,
)
from zipvoice.bin.infer_zipvoice import get_vocoder
from zipvoice.models.zipvoice import ZipVoice
from zipvoice.tokenizer.tokenizer import SimpleTokenizer
from zipvoice.utils.checkpoint import load_checkpoint
from zipvoice.utils.scaling_converter import convert_scaled_to_non_scaled
from zipvoice.vizipvoice import (
    _resolve_local_checkpoint_path,
    _resolve_local_config_path,
    DEFAULT_CHECKPOINT_NAME,
)

logger = logging.getLogger(__name__)

ZIPVOICE_COMPONENTS = ("text_encoder", "fm_decoder")
VOCODER_BASELINE = "mel_spec_24khz.onnx"
VOCODER_INT4 = "mel_spec_24khz_int4.onnx"
QUANT_MANIFEST = "quantization.json"


@dataclass
class ExportResult:
    onnx_dir: Path
    vocoder_dir: Path
    files: list[str] = field(default_factory=list)
    deployed_to: Optional[Path] = None
    backup_dir: Optional[Path] = None


class VocosMelSpecOnnx(nn.Module):
    """Vocos backbone + ISTFT head mag/x/y — matches ZipVoice-Vietnamese-ONNX-GUI."""

    def __init__(self, vocoder: nn.Module) -> None:
        super().__init__()
        self.backbone = vocoder.backbone
        self.head_linear = vocoder.head.out

    def forward(self, mels: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        hidden = self.backbone(mels)
        stft_params = self.head_linear(hidden).transpose(1, 2)
        mag, phase = stft_params.chunk(2, dim=1)
        mag = torch.exp(mag).clamp(max=1e2)
        return mag, torch.cos(phase), torch.sin(phase)


def resolve_model_paths(model_dir: Path, checkpoint_name: str) -> tuple[Path, Path, Path]:
    checkpoint_path = _resolve_local_checkpoint_path(model_dir, checkpoint_name)
    config_path = _resolve_local_config_path(model_dir)
    token_file = model_dir / "tokens.txt"
    if not token_file.is_file():
        raise FileNotFoundError(f"Missing tokens.txt in {model_dir}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    return checkpoint_path, config_path, token_file


def load_zipvoice_model(
    checkpoint_path: Path,
    config_path: Path,
    token_file: Path,
) -> tuple[ZipVoice, dict]:
    tokenizer = SimpleTokenizer(token_file=str(token_file))
    with config_path.open("r", encoding="utf-8") as handle:
        model_config = json.load(handle)

    model = ZipVoice(
        **model_config["model"],
        vocab_size=tokenizer.vocab_size,
        pad_id=tokenizer.pad_id,
    )

    suffix = checkpoint_path.suffix.lower()
    if suffix == ".safetensors":
        safetensors.torch.load_model(model, str(checkpoint_path))
    elif suffix == ".pt":
        load_checkpoint(filename=checkpoint_path, model=model, strict=True)
    else:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")

    model.eval()
    return model, model_config


def quantize_matmul_4bit(src: Path, dst: Path, block_size: int = 128) -> None:
    try:
        from onnxruntime.quantization.matmul_4bits_quantizer import (
            DefaultWeightOnlyQuantConfig,
            MatMul4BitsQuantizer,
        )
        from onnxruntime.quantization.quant_utils import QuantFormat
    except ImportError as exc:
        raise RuntimeError(
            "4-bit export requires onnxruntime>=1.18. Install: uv sync --extra export"
        ) from exc

    model = onnx.load(str(src))
    quant_config = DefaultWeightOnlyQuantConfig(
        block_size=block_size,
        is_symmetric=True,
        accuracy_level=4,
        quant_format=QuantFormat.QOperator,
    )
    quantizer = MatMul4BitsQuantizer(model, algo_config=quant_config)
    quantizer.process()
    quantizer.model.save_model_to_file(str(dst), use_external_data_format=False)
    logger.info("4-bit: %s → %s", src.name, dst.name)


def export_vocos_mel_spec(
    vocoder: nn.Module,
    vocoder_dir: Path,
    opset_version: int,
    feat_dim: int = 100,
) -> Path:
    wrapper = VocosMelSpecOnnx(vocoder)
    wrapper.eval()

    seq_len = 200
    dummy = torch.randn(1, feat_dim, seq_len, dtype=torch.float32)
    output_path = vocoder_dir / VOCODER_BASELINE

    torch.onnx.export(
        wrapper,
        dummy,
        str(output_path),
        verbose=False,
        opset_version=opset_version,
        input_names=["mels"],
        output_names=["mag", "x", "y"],
        dynamic_axes={
            "mels": {0: "N", 2: "T"},
            "mag": {0: "N", 2: "T"},
            "x": {0: "N", 2: "T"},
            "y": {0: "N", 2: "T"},
        },
        dynamo=False,
    )

    add_meta_data(
        filename=str(output_path),
        meta_data={
            "version": "1",
            "model_author": "ViZipVoice",
            "comment": "Vocos mel→mag/x/y for librosa ISTFT (ZipVoice-Vietnamese-ONNX-GUI)",
            "feat_dim": str(feat_dim),
            "sample_rate": "24000",
            "n_fft": "1024",
            "hop_length": "256",
        },
    )
    logger.info("Exported vocoder %s", output_path.name)
    return output_path


def copy_zipvoice_assets(model_dir: Path, onnx_dir: Path) -> None:
    for name in ("tokens.txt", "config.json", "model.json"):
        src = model_dir / name
        if src.is_file():
            shutil.copy2(src, onnx_dir / name)

    if not (onnx_dir / "model.json").is_file() and (onnx_dir / "config.json").is_file():
        shutil.copy2(onnx_dir / "config.json", onnx_dir / "model.json")


def write_quantization_manifest(onnx_dir: Path, created: list[str], model_config: dict | None = None) -> Path:
    payload = {
        "mode": "int4",
        "text_encoder": "int4",
        "fm_decoder": "int4",
        "filenames": {
            "text_encoder": "text_encoder_int4.onnx",
            "fm_decoder": "fm_decoder_int4.onnx",
        },
        "created": created,
        "source": "ViZipvoiceGUI export_onnx_bundle",
    }
    if model_config and "tokenizer" in model_config:
        payload["tokenizer"] = model_config["tokenizer"]
    path = onnx_dir / QUANT_MANIFEST
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def copy_vocoder_config(vocoder_dir: Path, vocoder_path: Optional[str]) -> None:
    candidates: list[Path] = []
    if vocoder_path:
        candidates.append(Path(vocoder_path) / "config.yaml")
    try:
        from huggingface_hub import hf_hub_download

        cached = Path(
            hf_hub_download(
                repo_id="charactr/vocos-mel-24khz",
                filename="config.yaml",
            )
        )
        candidates.append(cached)
    except Exception:
        pass

    for src in candidates:
        if src.is_file():
            shutil.copy2(src, vocoder_dir / "config.yaml")
            return


def backup_onnx_gui_models(onnx_gui_root: Path, log: Optional[Callable[[str], None]] = None) -> Optional[Path]:
    """Backup models/onnx and models/vocoder before overwriting."""

    def emit(message: str) -> None:
        logger.info(message)
        if log:
            log(message + "\n")

    onnx_gui_root = onnx_gui_root.resolve()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_root = onnx_gui_root / "models_backup" / stamp
    backed_up = False

    for sub in ("onnx", "vocoder"):
        src = onnx_gui_root / "models" / sub
        if not src.is_dir():
            continue
        files = [p for p in src.iterdir() if p.is_file()]
        if not files:
            continue
        dst = backup_root / sub
        dst.mkdir(parents=True, exist_ok=True)
        for path in files:
            shutil.copy2(path, dst / path.name)
            emit(f"Backup: {sub}/{path.name}")
        backed_up = True

    if not backed_up:
        emit("Không có file model cũ để backup — bỏ qua.")
        return None

    emit(f"Backup hoàn tất: {backup_root}")
    return backup_root


def deploy_to_onnx_gui(
    export_root: Path,
    onnx_gui_root: Path,
    log: Optional[Callable[[str], None]] = None,
) -> Optional[Path]:
    backup_dir = backup_onnx_gui_models(onnx_gui_root, log=log)

    src_onnx = export_root / "onnx"
    src_vocoder = export_root / "vocoder"
    dst_onnx = onnx_gui_root / "models" / "onnx"
    dst_vocoder = onnx_gui_root / "models" / "vocoder"

    if not src_onnx.is_dir():
        raise FileNotFoundError(f"Missing export onnx dir: {src_onnx}")

    dst_onnx.mkdir(parents=True, exist_ok=True)
    dst_vocoder.mkdir(parents=True, exist_ok=True)

    for path in src_onnx.iterdir():
        if path.is_file():
            shutil.copy2(path, dst_onnx / path.name)

    if src_vocoder.is_dir():
        for path in src_vocoder.iterdir():
            if path.is_file():
                shutil.copy2(path, dst_vocoder / path.name)

    logger.info("Deployed bundle → %s", onnx_gui_root)
    return backup_dir


@torch.no_grad()
def export_onnx_bundle(
    model_dir: Path,
    export_root: Path,
    *,
    checkpoint_name: str = DEFAULT_CHECKPOINT_NAME,
    vocoder_path: Optional[str] = None,
    quantize_int4: bool = True,
    block_size: int = 128,
    opset_version: int = 17,
    skip_vocos: bool = False,
    keep_baseline: bool = False,
    deploy_onnx_gui_root: Optional[Path] = None,
    log: Optional[Callable[[str], None]] = None,
) -> ExportResult:
    def emit(message: str) -> None:
        logger.info(message)
        if log:
            log(message + "\n")

    model_dir = model_dir.resolve()
    export_root = export_root.resolve()
    onnx_dir = export_root / "onnx"
    vocoder_dir = export_root / "vocoder"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    vocoder_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path, config_path, token_file = resolve_model_paths(model_dir, checkpoint_name)
    emit(f"Loading checkpoint: {checkpoint_path.name}")

    model, model_config = load_zipvoice_model(checkpoint_path, config_path, token_file)
    model = model.to(torch.device("cpu"))
    convert_scaled_to_non_scaled(model, inplace=True, is_onnx=True)

    created: list[str] = []

    text_fp32 = onnx_dir / "text_encoder.onnx"
    fm_fp32 = onnx_dir / "fm_decoder.onnx"
    export_text_encoder(
        model=OnnxTextModel(model=model),
        filename=str(text_fp32),
        opset_version=opset_version,
        meta_data={
            "version": "1",
            "model_author": "ViZipVoice",
            "comment": "ViZipVoice text encoder (character tokenizer)",
            "use_espeak": "0",
            "use_pinyin": "0",
            "tokenizer_type": "SimpleTokenizer",
        },
    )
    export_fm_decoder(
        model=OnnxFlowMatchingModel(model=model, distill=False),
        filename=str(fm_fp32),
        opset_version=opset_version,
    )
    created.extend([text_fp32.name, fm_fp32.name])
    emit("Exported ZipVoice text_encoder + fm_decoder (FP32 baseline)")

    if quantize_int4:
        for component in ZIPVOICE_COMPONENTS:
            src = onnx_dir / f"{component}.onnx"
            dst = onnx_dir / f"{component}_int4.onnx"
            quantize_matmul_4bit(src, dst, block_size=block_size)
            created.append(dst.name)
        emit("Quantized ZipVoice → int4 (MatMulNBits)")

        if not keep_baseline:
            for component in ZIPVOICE_COMPONENTS:
                baseline = onnx_dir / f"{component}.onnx"
                if baseline.is_file():
                    baseline.unlink()
                    emit(f"Removed baseline {baseline.name}")

    vocoder_files: list[str] = []
    if not skip_vocos:
        emit("Exporting Vocos mel_spec_24khz.onnx (mag/x/y for ONNX-GUI)")
        vocoder = get_vocoder(vocoder_path)
        vocoder = vocoder.to(torch.device("cpu"))
        vocoder.eval()
        feat_dim = int(model_config["model"]["feat_dim"])
        vocos_fp32 = export_vocos_mel_spec(
            vocoder=vocoder,
            vocoder_dir=vocoder_dir,
            opset_version=opset_version,
            feat_dim=feat_dim,
        )
        vocoder_files.append(vocos_fp32.name)

        if quantize_int4:
            vocos_int4 = vocoder_dir / VOCODER_INT4
            quantize_matmul_4bit(vocos_fp32, vocos_int4, block_size=block_size)
            vocoder_files.append(vocos_int4.name)
            emit(f"Quantized vocoder → {VOCODER_INT4}")

            if not keep_baseline:
                vocos_fp32.unlink()
                shutil.copy2(vocos_int4, vocoder_dir / VOCODER_BASELINE)
                emit(f"Deployed vocoder as {VOCODER_BASELINE} (int4 weights)")
                vocoder_files.append(f"{VOCODER_BASELINE} (from int4)")

        copy_vocoder_config(vocoder_dir, vocoder_path)

    copy_zipvoice_assets(model_dir, onnx_dir)
    if quantize_int4:
        write_quantization_manifest(onnx_dir, created, model_config=model_config)
        created.append(QUANT_MANIFEST)

    emit(f"Export complete:\n  ONNX: {onnx_dir}\n  Vocoder: {vocoder_dir}")

    deployed = None
    backup_dir = None
    if deploy_onnx_gui_root is not None:
        backup_dir = deploy_to_onnx_gui(export_root, deploy_onnx_gui_root.resolve(), log=log)
        deployed = deploy_onnx_gui_root.resolve()
        emit(f"Copied to ZipVoice-Vietnamese-ONNX-GUI: {deployed}")
        if backup_dir:
            emit(f"Backup cũ tại: {backup_dir}")

    all_files = created + vocoder_files
    return ExportResult(
        onnx_dir=onnx_dir,
        vocoder_dir=vocoder_dir,
        files=all_files,
        deployed_to=deployed,
        backup_dir=backup_dir,
    )
