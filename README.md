<div align="center">

# 🎙️ ViZipVoice GUI — Local Offline

**Fork of [ViZipVoice](https://github.com/iamdinhthuan/ViZipvoice) for smaller ONNX deployment and audiobook-style long-form TTS — offline Gradio, export, and CLI.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-native-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/UI-Gradio-F97316)](https://www.gradio.app/)
[![uv](https://img.shields.io/badge/setup-uv-DE5FE9)](https://github.com/astral-sh/uv)
[![License](https://img.shields.io/badge/license-Non--Commercial-red)](LICENSE)

**Author:** [Lam Pham](https://github.com/phamtronglam2001) · **Upstream:** [iamdinhthuan/ViZipvoice](https://github.com/iamdinhthuan/ViZipvoice) · [Tiếng Việt](README_VI.md)

[Model](https://huggingface.co/contextboxai/ViZipvoice) · [HF Space](https://huggingface.co/spaces/dinhthuan/ViZipvoice) · [ZipVoice paper](https://arxiv.org/abs/2506.13053)

</div>

---

## 📖 Overview

**ViZipVoice** ([iamdinhthuan/ViZipvoice](https://github.com/iamdinhthuan/ViZipvoice)) is a Vietnamese fine-tune of [ZipVoice](https://github.com/k2-fsa/ZipVoice) for zero-shot TTS and voice cloning. **This repository** is a community fork focused on:

| Goal | What we added |
|------|----------------|
| **Smaller footprint** | ONNX int4 export + `ViZipVoiceOnnxTTS`, Gradio export/test apps, ORT tuning |
| **Audiobook direction** | Long-text chunking (`max_duration`), sentence split, segment join — **text normalizer will be tuned for book narration later** |
| **Local product** | Windows `.bat` launchers, `uv` project, optional Slint GUI |

Unlike cloud APIs, everything runs **offline**: download the PyTorch checkpoint once, pick from **30 bundled reference voices**, and synthesize without sending data off your machine.

The **Gradio GUIs** mirror the [Hugging Face Space](https://huggingface.co/spaces/dinhthuan/ViZipvoice): Vietnamese normalization, sentence splitting, short-text rules, and post-processing (silence, crossfade, fade in/out). PyTorch and ONNX tabs share the same controls (`max_duration`, `remove_long_sil`, steps, speed, …).

> **License:** this fork is **non-commercial only** — see [LICENSE](LICENSE). Upstream model/code licenses still apply to their respective components.

> **In one line:** upstream ViZipVoice quality + ONNX int4 + offline Gradio — install with **`uv`**, run with one batch file.

---

## ✨ Key Features

- 🗣️ **Zero-shot voice cloning** — ~3–10 s of clean reference audio + transcript, or choose from **30 ref voices** in `models/ViZipvoice/audio/`.
- 🎯 **Native PyTorch quality** — full ZipVoice flow-matching model (24 kHz, Vocos vocoder); generally higher fidelity than quantized ONNX ports.
- 🧩 **Configurable normalization pipeline** — composable steps (`soe_vinorm`, spacing cleanup, `period_break`, `sea_g2p`, …) with live preview tab.
- 📝 **Sentence-aware synthesis** — auto split long text; adaptive `num_step` / `speed` for 1-word and 2–4-word sentences.
- 🎛️ **Offline Gradio (PyTorch)** — ref picker, advanced controls, status JSON, same flow as the public HF Space; **auto-opens browser**.
- ⚡ **ONNX Gradio** — same tabs and controls as PyTorch Gradio for `models/onnx` int4 + Vocos ONNX (`run_onnx_local.bat`).
- 📤 **ONNX export Gradio** — export ZipVoice + Vocos int4 to `models/onnx` / `models/vocoder` (`run_export_gui.bat`).
- 🖥️ **Slint GUI (optional)** — native desktop window for CLI inference, download, and export (no browser).
- 💾 **CLI + Python wrapper** — `infer_vizipvoice`, `ViZipVoiceTTS`, `ViZipVoiceOnnxTTS`, metrics (RTF, segments).

---

## 🛠️ Engineering Highlights

| Area | What was built |
|------|----------------|
| **Inference wrapper** | `ViZipVoiceTTS` — HF or local checkpoint, latest `checkpoint-<step>.pt` auto-select, FP16 on CUDA |
| **Text processing** | `vi_normalizer` registry — chainable steps, preview API, CLI `--normalize-pipeline` |
| **Audio I/O (Windows)** | `audio_io` — soundfile + pydub fallback (mp3/wav) without FFmpeg / torchcodec |
| **Segment post-process** | Per-sentence synth → silence + crossfade + fade in/out join |
| **Local productization** | `uv` project (`pyproject.toml`), Windows `.bat` launchers with step logs + `pause` on error |
| **ONNX path** | `export_onnx_bundle` (int4), `ViZipVoiceOnnxTTS`, Gradio + CLI infer; **character tokenizer only** |

---

## 🏗️ Architecture

```
  Ref audio + transcript
            │
            ▼
  ┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
  │ Normalize       │──► │ Split sentences  │──► │ Per segment:         │
  │ (soe_vinorm →   │    │ + short-text     │    │ ZipVoice flow-match  │
  │  spacing, …)    │    │   speed rules    │    │ → Vocos 24 kHz       │
  └─────────────────┘    └──────────────────┘    └──────────┬──────────┘
                                                             ▼
                                               ┌─────────────────────────┐
                                               │ Join segments          │
                                               │ (silence, crossfade,   │
                                               │  fade in/out) → WAV    │
                                               └─────────────────────────┘
```

<details>
<summary><strong>Repository layout</strong></summary>

```
local_app/               # Gradio apps
  app.py                 # PyTorch TTS (vizipvoice-local)
  onnx_app.py            # ONNX TTS test (vizipvoice-onnx-local)
  export_app.py          # ONNX int4 export GUI (vizipvoice-export-gui)
  download_model.py
gui/                     # Slint desktop GUI (optional)
zipvoice/
  vizipvoice.py          # ViZipVoiceTTS (PyTorch)
  onnx_inference/        # ViZipVoiceOnnxTTS
  export/onnx_bundle.py  # Export + optional deploy
  tokenizer/vi_normalizer.py
  bin/                   # CLI infer / export
models/ViZipvoice/       # PyTorch weights + audio/ (30 ref voices)
models/onnx/             # Exported ZipVoice ONNX + tokens.txt
models/vocoder/          # mel_spec_24khz.onnx
output/                  # Generated WAV files
setup_local.bat          # uv sync + download model
run_local.bat            # Gradio PyTorch :7860
run_onnx_local.bat       # Gradio ONNX test :7861
run_export_gui.bat       # Gradio export :7862
```

</details>

---

## 🚀 Quickstart (Windows)

**Prerequisites:** [uv](https://github.com/astral-sh/uv) installed, ~4 GB disk for model weights.

```bat
git clone https://github.com/phamtronglam2001/ViZipvoiceGUI.git
cd ViZipvoiceGUI

setup_local.bat        REM uv sync + download model (first run)
run_local.bat          REM PyTorch Gradio → http://127.0.0.1:7860 (opens browser)
```

**ONNX workflow** (after PyTorch works):

```bat
run_export_gui.bat     REM export int4 → models\onnx + models\vocoder (:7862)
run_onnx_local.bat     REM test ONNX with same Gradio UX (:7861)
```

Manual equivalent:

```bat
uv sync
uv run vizipvoice-download
uv run vizipvoice-local
uv sync --extra export && uv run vizipvoice-export-gui
uv sync --extra onnx && uv run vizipvoice-onnx-local
```

### Launchers (Windows)

| Batch file | Gradio / UI | URL | Browser |
|------------|-------------|-----|---------|
| `setup_local.bat` | Setup only (uv sync + download) | — | — |
| `run_local.bat` | PyTorch TTS | `:7860` | Auto |
| `run_onnx_local.bat` | ONNX TTS test | `:7861` | Auto |
| `run_export_gui.bat` | ONNX int4 export | `:7862` | Auto |
| `gui\run_gui.bat` | Slint desktop | native window | No |

All Gradio apps accept `--no-inbrowser` to skip opening a tab. Each `.bat` prints what it does, shows `[OK]` / `[LOI]`, and **pauses on error** so the window stays open.

| Entry point | Command |
|-------------|---------|
| **Gradio PyTorch** | `run_local.bat` or `uv run vizipvoice-local` |
| **Gradio ONNX test** | `run_onnx_local.bat` or `uv run vizipvoice-onnx-local` |
| **Gradio ONNX export** | `run_export_gui.bat` or `uv run vizipvoice-export-gui` |
| **CLI PyTorch** | `uv run python -m zipvoice.bin.infer_vizipvoice --model-dir models/ViZipvoice ...` |
| **CLI ONNX** | `uv run python -m zipvoice.bin.infer_vizipvoice_onnx --onnx-model-dir models/onnx --use-int4 ...` |
| **Slint GUI** | `gui\run_gui.bat` (`uv sync --extra gui`) |
| **Download model** | `uv run vizipvoice-download` |

<details>
<summary><strong>Linux / macOS</strong></summary>

```bash
uv sync
uv run vizipvoice-download
uv run vizipvoice-local --host 0.0.0.0 --port 7860
```

Set `VIZIPVOICE_MODEL_DIR` to override the default `models/ViZipvoice` path.

</details>

---

## 🖥️ Usage

### Gradio tabs (PyTorch & ONNX)

Both `run_local.bat` and `run_onnx_local.bat` expose the same structure:

| Tab | Contents |
|-----|----------|
| **TTS** / **TTS (ONNX)** | Ref voice dropdown (30 presets), prompt audio/text, target text, Generate, advanced sliders |
| **Text Normalizer** | Raw text → pipeline preview (`soe_vinorm`, `spacing`, …) before synthesis |

ONNX tab adds fields: ONNX model dir, int4 toggle, vocoder ONNX path (`models/vocoder/mel_spec_24khz.onnx`).

### Reference voices (`models/ViZipvoice/audio/`)

Sidecar format — one audio + same-stem `.txt` transcript:

```text
audio/Đinh-Quyết.mp3
audio/Đinh-Quyết.txt
```

Supported: `.wav`, `.mp3`, `.flac`, `.m4a`, `.ogg`. Preferred presets: **Đinh-Quyết**, **Nhã-Uyên**, **MC**.

### CLI example (local model)

```bat
uv run python -m zipvoice.bin.infer_vizipvoice ^
  --model-dir models/ViZipvoice ^
  --prompt-wav "models/ViZipvoice/audio/Đinh-Quyết.mp3" ^
  --prompt-text "Transcript khớp prompt." ^
  --text "ViZipVoice chạy offline trên máy local." ^
  --res-wav-path output/demo.wav ^
  --num-step 16 --seed 666
```

Custom normalization pipeline:

```bat
uv run python -m zipvoice.bin.infer_vizipvoice ^
  ... ^
  --normalize-pipeline soe_vinorm,spacing,period_break
```

Optional **sea-g2p** step: `uv sync --extra g2p` first, then add `sea_g2p` to the pipeline.

### Python wrapper

```python
from zipvoice.vizipvoice import ViZipVoiceTTS

tts = ViZipVoiceTTS(model_dir="models/ViZipvoice")
metrics = tts.synthesize(
    prompt_wav="models/ViZipvoice/audio/Đinh-Quyết.mp3",
    prompt_text="Xin chào, đây là giọng mẫu.",
    text="Câu tiếng Việt cần đọc.",
    output_path="output.wav",
    normalize_pipeline=["soe_vinorm", "spacing"],
)
print(metrics)  # rtf, wav_seconds, segments, …
```

### Inference defaults (same as HF Space)

- Normalize with **`soe-vinorm`** then **spacing** cleanup around punctuation.
- Split long text into sentences; merge segments with **80 ms crossfade**, **180 ms silence**, fade in/out.
- **1-word** sentences: `num_step ≥ 24`, `speed = 0.6`.
- **2–4 words**: `speed = 0.8`.

### Normalization pipeline (`vi_normalizer`)

| Key | Role |
|-----|------|
| `soe_vinorm` | NSW: numbers, dates, units, abbreviations → spoken Vietnamese |
| `spacing` | Punctuation spacing cleanup |
| `vieneu` | Collapse extra spaces (per line) |
| `period_break` | Brackets / numbered items → newlines (book structure) |
| `lowercase` | Lowercase entire string |
| `strip_quotes` | Remove `"` and `'` |
| `sea_g2p` | Stronger NSW (`uv sync --extra g2p`) |

Default: `soe_vinorm, spacing`. Tokenizer vocab: `models/ViZipvoice/tokens.txt` (244 character tokens; no dedicated silence token — use punctuation or `silence_ms` between segments).

### ONNX export & test

```bat
uv sync --extra export
uv run vizipvoice-export-onnx --model-dir models/ViZipvoice --export-root models --quantize-int4 1
```

Output layout:

| Path | Contents |
|------|----------|
| `models/onnx/text_encoder_int4.onnx` | ZipVoice text encoder (int4) |
| `models/onnx/fm_decoder_int4.onnx` | Flow-matching decoder (int4) |
| `models/onnx/tokens.txt` | Character tokenizer (SimpleTokenizer) |
| `models/onnx/config.json` | Model metadata |
| `models/vocoder/mel_spec_24khz.onnx` | Vocos mag/x/y → librosa ISTFT |

> **Important:** ViZipVoice uses a **character** tokenizer (`SimpleTokenizer`). Do **not** copy this ONNX bundle into [ZipVoice-Vietnamese-ONNX-GUI](https://github.com/phamtronglam2001/ZipVoice-Vietnamese-ONNX-GUI) — that app expects **Espeak phoneme** tokens. Test ViZipVoice ONNX only via `run_onnx_local.bat` or `infer_vizipvoice_onnx` in this repo.

```python
from zipvoice.onnx_inference import ViZipVoiceOnnxTTS

tts = ViZipVoiceOnnxTTS("models/onnx", use_int4=True)
metrics = tts.synthesize(
    prompt_wav="models/ViZipvoice/audio/Đinh-Quyết.mp3",
    prompt_text="…",
    text="…",
    output_path="output/onnx.wav",
)
```

Env overrides: `VIZIPVOICE_ONNX_DIR`, `VIZIPVOICE_VOCODER_ONNX`, `VIZIPVOICE_MODEL_DIR`.

---

## 📦 Model

Weights: [contextboxai/ViZipvoice](https://huggingface.co/contextboxai/ViZipvoice)

| Asset | Description |
|-------|-------------|
| `checkpoint-1300000.pt` | Latest FP16 inference checkpoint (auto-selected) |
| `checkpoint-700000.pt`, `checkpoint-920000.pt` | Earlier checkpoints for comparison |
| `config.json` / `model.json` | Model config |
| `tokens.txt` | Character tokenizer, **244** Vietnamese tokens |
| `audio/` | **30** reference clips + transcripts |
| `demo/` | Sample outputs |

- **Sample rate:** 24 kHz · **Vocoder:** [charactr/vocos-mel-24khz](https://huggingface.co/charactr/vocos-mel-24khz)
- **Training data:** ~7000 h (~6500 h Vietnamese + ~500 h English)

### Demo audio (HF)

**Đinh-Quyết** — [listen](https://huggingface.co/contextboxai/ViZipvoice/resolve/main/demo/demo_01_%C4%90inh-Quy%E1%BA%BFt.wav)

**Nhã-Uyên** — [listen](https://huggingface.co/contextboxai/ViZipvoice/resolve/main/demo/demo_02_Nh%C3%A3-Uy%C3%AAn.wav)

**MC** — [listen](https://huggingface.co/contextboxai/ViZipvoice/resolve/main/demo/demo_03_MC.wav)

---

## ⚙️ Performance

| Device | Notes |
|--------|-------|
| **CUDA** | FP16 autocast enabled by default; best RTF |
| **CPU** | Works out of the box; expect higher RTF (~7× on typical desktop) |
| **MPS** | Supported via PyTorch device auto-select |

Tuning levers: `--num-step` (quality vs speed), `--speed`, `--guidance-scale`, post-process ms sliders in Gradio **Advanced**.

---

## 💡 Quality tips

- Use **one speaker**, clean audio, minimal reverb/music; **3–10 s** prompt is ideal.
- **Transcript must match** the prompt — mismatches hurt cloning quality.
- For very short phrases, try `--speed 0.7` or `--num-step 24`.
- Use clear punctuation in long text for natural sentence splits.
- Characters outside `tokens.txt` vocab are dropped by the character tokenizer.
- Compare normalization pipelines in the **Text Normalizer** tab before batch runs.

---

## 🧪 Development

```bat
uv sync
uv run python -c "from zipvoice.tokenizer.vi_normalizer import preview_normalize; print(preview_normalize('Hôm nay là 8/6/2026.').normalized)"
```

Legacy install (without uv):

```bash
pip install -r requirements.txt
export PYTHONPATH="$PWD:$PYTHONPATH"
```

Optional extras:

```bat
uv sync --extra export  REM ONNX export (onnx<1.19, onnxruntime, onnxscript, librosa)
uv sync --extra onnx    REM ONNX Gradio test + infer
uv sync --extra gui     REM Slint desktop GUI
uv sync --extra g2p     REM sea-g2p normalizer
```

---

## ⚖️ Ethics & use

ViZipVoice can clone voices from a short prompt. Use only voices you have rights to or explicit consent. Do not use for impersonation, fraud, harm, or disinformation.

**This fork is for non-commercial use only** (research, personal projects, audiobook experiments). Contact the author for commercial licensing.

---

## 🙏 Acknowledgments & License

### Upstream (thank you)

| Component | Source | License |
|-----------|--------|---------|
| **ViZipVoice** (model & original repo) | [iamdinhthuan/ViZipvoice](https://github.com/iamdinhthuan/ViZipvoice) · [contextboxai/ViZipvoice](https://huggingface.co/contextboxai/ViZipvoice) | Apache-2.0 |
| ZipVoice | [k2-fsa/ZipVoice](https://github.com/k2-fsa/ZipVoice) | Apache-2.0 |
| Vocos | [gemelo-ai/vocos](https://github.com/gemelo-ai/vocos) · [charactr/vocos-mel-24khz](https://huggingface.co/charactr/vocos-mel-24khz) | MIT |
| soe-vinorm | [soe-vinorm](https://pypi.org/project/soe-vinorm/) | per upstream |
| sea-g2p (optional) | [pnnbao97/sea-g2p](https://github.com/pnnbao97/sea-g2p) | per upstream |

### This fork (ViZipvoiceGUI)

**Author:** [Lam Pham](https://github.com/phamtronglam2001) · **Repo:** [phamtronglam2001/ViZipvoiceGUI](https://github.com/phamtronglam2001/ViZipvoiceGUI)

Code and documentation in **this repository** are distributed under **Apache License 2.0 with an additional non-commercial restriction** — you may **not** use this fork for commercial purposes without written permission. See **[LICENSE](LICENSE)** for the full text. Model weights and third-party libraries keep their original licenses.

Please cite **ZipVoice** and credit **ViZipVoice** when building on this work.

### Citation

```bibtex
@article{zhu2025zipvoice,
  title={ZipVoice: Fast and High-Quality Zero-Shot Text-to-Speech with Flow Matching},
  author={Zhu, Han and others},
  journal={arXiv preprint arXiv:2506.13053},
  year={2025}
}
```

---

## 👤 Authors

| Role | Link |
|------|------|
| **ViZipVoice GUI (this fork)** | [phamtronglam2001/ViZipvoiceGUI](https://github.com/phamtronglam2001/ViZipvoiceGUI) — Lam Pham |
| **ViZipVoice (upstream model)** | [iamdinhthuan/ViZipvoice](https://github.com/iamdinhthuan/ViZipvoice) · [HF Space](https://huggingface.co/spaces/dinhthuan/ViZipvoice) |
