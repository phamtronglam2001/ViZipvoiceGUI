<div align="center">

# 🎙️ ViZipVoice GUI — Local Offline

**Fork of [ViZipVoice](https://github.com/iamdinhthuan/ViZipvoice) for smaller ONNX deployment and audiobook-style long-form TTS — offline Gradio, export, and CLI.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-native-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/UI-Gradio-F97316)](https://www.gradio.app/)
[![uv](https://img.shields.io/badge/setup-uv-DE5FE9)](https://github.com/astral-sh/uv)
[![License](https://img.shields.io/badge/license-Non--Commercial-red)](LICENSE)

**Author:** [Pham Trong Lam](https://github.com/phamtronglam2001) · **Upstream:** [iamdinhthuan/ViZipvoice](https://github.com/iamdinhthuan/ViZipvoice) · [Tiếng Việt](README_VI.md)

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

> **In one line:** upstream ViZipVoice quality + ONNX int4 + offline Gradio — copy folder, run **`setup.bat`**, then **`run.bat`**.

---

## ✨ Key Features

- 🗣️ **Zero-shot voice cloning** — ~3–10 s of clean reference audio + transcript, or choose from **30 ref voices** in `models/ViZipvoice/audio/`.
- 🎯 **Native PyTorch quality** — full ZipVoice flow-matching model (24 kHz, Vocos vocoder); generally higher fidelity than quantized ONNX ports.
- 🧩 **Configurable normalization pipeline** — composable steps (`soe_vinorm`, spacing cleanup, `period_break`, `sea_g2p`, …) with live preview tab.
- 📝 **Sentence-aware synthesis** — auto split long text; adaptive `num_step` / `speed` for 1-word and 2–4-word sentences.
- 🎛️ **Offline Gradio (PyTorch)** — ref picker, **TXT audiobook upload**, **Hiệu năng** (device, CPU threads, FP16), optional **MP3 export** via bundled `ffmpeg/bin`, advanced controls, status JSON; **auto-opens browser**.
- ⚡ **ONNX Gradio** — same TTS controls as PyTorch for `models/onnx` int4 + Vocos ONNX; dedicated **Hiệu năng** tab (GPU, ORT threads, force CPU); **mel trim fix** for short sentences (`run.bat` → [2]).
- 📤 **ONNX export Gradio** — export ZipVoice + Vocos int4 to `models/onnx` / `models/vocoder` (`run.bat` → [3]).
- 🚀 **Unified launchers** — `setup.bat` (CPU or GPU) + `run.bat` menu (PyTorch / ONNX / Export / Slint); GPU auto-fallback to CPU — no separate CPU batch files.
- 🖥️ **Slint GUI (optional)** — native desktop window for CLI inference, download, and export (no browser).
- 💾 **CLI + Python wrapper** — `infer_vizipvoice`, `ViZipVoiceTTS`, `ViZipVoiceOnnxTTS`, metrics (RTF, segments).

---

## 🛠️ Engineering Highlights

| Area | What was built |
|------|----------------|
| **Inference wrapper** | `ViZipVoiceTTS` — HF or local checkpoint, latest `checkpoint-<step>.pt` auto-select, FP16 on CUDA |
| **Text processing** | `vi_normalizer` registry — chainable steps, preview API, CLI `--normalize-pipeline` |
| **Audio I/O (Windows)** | `audio_io` — soundfile + pydub for read; optional **MP3 export** via portable `ffmpeg/bin/ffmpeg.exe` (64/128/256 kbps) |
| **Segment post-process** | Per-sentence synth → silence + crossfade + fade in/out join |
| **Local productization** | `uv` project (`pyproject.toml`), Windows `.bat` launchers with step logs + `pause` on error |
| **ONNX path** | `export_onnx_bundle` (int4), `ViZipVoiceOnnxTTS`, Gradio + CLI infer; **mel trim** aligned with PyTorch for short sentences; **character tokenizer only** |

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
ffmpeg/bin/              # Portable ffmpeg.exe for MP3 export (optional)
output/                  # Generated WAV / MP3 files
.venv/                   # uv virtualenv (created by setup.bat)
setup.bat                # First-time install: CPU [1] or GPU [2]
run.bat                  # App menu: PyTorch :7860 / ONNX :7861 / Export :7862 / Slint
```

</details>

---

## 🚀 Quickstart (Windows)

**Prerequisites:** [uv](https://github.com/astral-sh/uv) installed, ~4 GB disk for model weights. Copy the whole folder or clone the repo.

```bat
cd ViZipvoiceGUI

setup.bat              REM first run: [1] CPU (default) or [2] GPU (NVIDIA CUDA)
run.bat                REM menu: [1] PyTorch :7860  [2] ONNX :7861  [3] Export :7862  [4] Slint
```

**ONNX workflow** (after PyTorch works):

```bat
run.bat                REM choose [3] Export ONNX → models\onnx + models\vocoder
run.bat                REM choose [2] Test ONNX Gradio (:7861)
```

> **Removed launchers:** `setup_local.bat`, `run_local.bat`, `run_onnx_local.bat`, `run_onnx_standalone.bat`, `run_export_gui.bat` — replaced by **`setup.bat`** + **`run.bat`**.

### CPU vs GPU setup (`setup.bat`)

| Choice | What gets installed | Notes |
|--------|---------------------|-------|
| **[1] CPU** (default) | `uv sync` + model download | No NVIDIA GPU required; PyTorch and ONNX Runtime use CPU |
| **[2] GPU** | PyTorch **CUDA cu128** + `onnxruntime-gpu` + CUDA DLLs | Large download; needs compatible NVIDIA driver |

If GPU setup fails or CUDA is unavailable at runtime, apps **fall back to CPU automatically** — no separate CPU batch files.

Manual equivalent:

```bat
uv sync
uv run vizipvoice-download
uv run vizipvoice-local
uv sync --extra export && uv run vizipvoice-export-gui
uv sync --extra onnx && uv run vizipvoice-onnx-local
```

GPU ONNX path: `setup.bat` option [2] uses `requirements-onnx-gpu.txt` and `pyproject.toml` extra **`onnx-gpu`**.

### Launchers (Windows)

| Batch file | Role |
|------------|------|
| `setup.bat` | First-time install: `uv sync`, download model, optional GPU stack |
| `run.bat` | App picker — PyTorch TTS `:7860`, ONNX `:7861`, Export `:7862`, Slint desktop |

`run.bat` menu:

| Choice | App | URL / UI | Browser |
|--------|-----|----------|---------|
| **[1] PyTorch** | Gradio TTS | `:7860` | Auto |
| **[2] ONNX** | Gradio ONNX test | `:7861` | Auto |
| **[3] Export** | Gradio ONNX int4 export | `:7862` | Auto |
| **[4] Slint** | Native desktop GUI | window | No |

Pass extra CLI flags after the menu, e.g. `--no-inbrowser` or `--port 7863`. Each `.bat` prints steps, shows `[OK]` / `[LOI]`, and **pauses on error**.

| Entry point | Command |
|-------------|---------|
| **Gradio PyTorch** | `run.bat` → [1] or `uv run vizipvoice-local` |
| **Gradio ONNX test** | `run.bat` → [2] or `uv run vizipvoice-onnx-local` |
| **Gradio ONNX export** | `run.bat` → [3] or `uv run vizipvoice-export-gui` |
| **CLI PyTorch** | `uv run python -m zipvoice.bin.infer_vizipvoice --model-dir models/ViZipvoice ...` |
| **CLI ONNX** | `uv run python -m zipvoice.bin.infer_zipvoice_onnx --onnx-model-dir models/onnx --use-int4 ...` |
| **Slint GUI** | `run.bat` → [4] (`uv sync --extra gui`) |
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

**PyTorch** (`run.bat` → [1]) — tab **TTS**:

| Section | Contents |
|---------|----------|
| **TTS** | Ref voice (30 presets), prompt audio/text, **TXT audiobook upload** (UTF-8 → fills text box), Generate |
| **Hiệu năng** (accordion) | Device `auto` / `cuda` / `cpu`, PyTorch CPU threads, FP16 on CUDA; GPU load failure → auto CPU retry |
| **Xuất MP3 / ffmpeg** | `ffmpeg` folder (default `ffmpeg` → `ffmpeg/bin/ffmpeg.exe`), bitrate 64 / 128 / 256 kbps; **MP3 only when ffmpeg found**, else WAV |
| **Advanced** | Steps, guidance, speed, chunking, post-process ms sliders |
| **Text Normalizer** | Pipeline preview before synthesis |

**ONNX** (`run.bat` → [2]):

| Tab | Contents |
|-----|----------|
| **Hiệu năng** | GPU (CUDA / DirectML), force CPU, ORT threads (0 = auto); reads `.install_mode_onnx` after GPU `setup.bat` |
| **TTS (ONNX)** | Same synthesis controls as PyTorch + ONNX model dir, int4 toggle, vocoder path (`models/vocoder/mel_spec_24khz.onnx`) |
| **Text Normalizer** | Same preview tab |

ONNX inference trims generated mel to match PyTorch duration and **active-speech alignment** — fixes bleed/artifacts on **short sentences** (e.g. one-word prompts).

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

> **Important:** ViZipVoice uses a **character** tokenizer (`SimpleTokenizer`). Do **not** copy this ONNX bundle into [ZipVoice-Vietnamese-ONNX-GUI](https://github.com/phamtronglam2001/ZipVoice-Vietnamese-ONNX-GUI) — that app expects **Espeak phoneme** tokens. Test ViZipVoice ONNX only via `run.bat` → [2] or `infer_vizipvoice_onnx` in this repo.

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

### Environment variables

| Variable | Applies to | Purpose |
|----------|------------|---------|
| `VIZIPVOICE_MODEL_DIR` | PyTorch, download | Override default `models/ViZipvoice` |
| `VIZIPVOICE_FFMPEG_DIR` | PyTorch Gradio | Folder containing `ffmpeg.exe` or `bin/ffmpeg.exe` (default `ffmpeg`) |
| `VIZIPVOICE_ONNX_DIR` | ONNX | Exported ONNX bundle directory |
| `VIZIPVOICE_VOCODER_ONNX` | ONNX | Path to `mel_spec_24khz.onnx` |
| `ZIPVOICE_ONNX_GPU` | ONNX | `1` / `true` to prefer GPU providers |
| `ZIPVOICE_ONNX_THREADS` | ONNX | ORT intra-op threads per session |
| `ZIPVOICE_FORCE_CPU` | ONNX | `1` to force CPU execution provider |

`run.bat` sets `ZIPVOICE_ONNX_GPU=1` when `.install_mode_onnx` contains `gpu` (from `setup.bat` → [2]).

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
| **CUDA** | Install via `setup.bat` → [2]; PyTorch FP16 on by default (toggle in **Hiệu năng**); best RTF |
| **CPU** | Default `setup.bat` → [1]; works without NVIDIA; expect higher RTF (~7× on typical desktop) |
| **MPS** | PyTorch `auto` device on Apple Silicon |
| **ONNX GPU** | `onnxruntime-gpu` + CUDA DLLs from `setup.bat` [2] or `uv sync --extra onnx-gpu`; auto-fallback if EP unavailable |

**Auto fallback:** PyTorch retries on CPU if CUDA load fails; ONNX Runtime falls back when GPU/DLLs missing — no extra launchers.

Tuning levers: Gradio **Hiệu năng** (device, threads, FP16 / ORT threads / force CPU), `--num-step`, `--speed`, `--guidance-scale`, post-process sliders in **Advanced**.

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
uv sync --extra export    REM ONNX export (onnx<1.19, onnxruntime, onnxscript, librosa)
uv sync --extra onnx      REM ONNX Gradio test + infer (CPU ORT)
uv sync --extra onnx-gpu  REM onnxruntime-gpu + NVIDIA CUDA 12 DLLs (see requirements-onnx-gpu.txt)
uv sync --extra gui       REM Slint desktop GUI
uv sync --extra g2p       REM sea-g2p normalizer
```

On Windows, prefer **`setup.bat` → [2]** for GPU — it reinstalls PyTorch cu128 and applies the `onnx-gpu` stack in one flow.

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

**Author:** [Pham Trong Lam](https://github.com/phamtronglam2001) · **Repo:** [phamtronglam2001/ViZipvoiceGUI](https://github.com/phamtronglam2001/ViZipvoiceGUI)

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
| **ViZipVoice GUI (this fork)** | [phamtronglam2001/ViZipvoiceGUI](https://github.com/phamtronglam2001/ViZipvoiceGUI) — Pham Trong Lam |
| **ViZipVoice (upstream model)** | [iamdinhthuan/ViZipvoice](https://github.com/iamdinhthuan/ViZipvoice) · [HF Space](https://huggingface.co/spaces/dinhthuan/ViZipvoice) |
