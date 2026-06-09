<div align="center">

# 🎙️ ViZipVoice GUI — Offline Local

**Fork [ViZipVoice](https://github.com/iamdinhthuan/ViZipvoice) — giảm dung lượng ONNX, hướng đọc sách nói (audiobook), Gradio/CLI offline.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-native-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/UI-Gradio-F97316)](https://www.gradio.app/)
[![uv](https://img.shields.io/badge/setup-uv-DE5FE9)](https://github.com/astral-sh/uv)
[![License](https://img.shields.io/badge/license-Non--Commercial-red)](LICENSE)

**Tác giả:** [Lam Pham](https://github.com/phamtronglam2001) · **Gốc:** [iamdinhthuan/ViZipvoice](https://github.com/iamdinhthuan/ViZipvoice) · [English](README.md)

[Model](https://huggingface.co/contextboxai/ViZipvoice) · [HF Space](https://huggingface.co/spaces/dinhthuan/ViZipvoice) · [Paper ZipVoice](https://arxiv.org/abs/2506.13053)

</div>

---

## 📖 Giới thiệu

**ViZipVoice** ([iamdinhthuan/ViZipvoice](https://github.com/iamdinhthuan/ViZipvoice)) là bản fine-tune tiếng Việt của [ZipVoice](https://github.com/k2-fsa/ZipVoice) cho zero-shot TTS và voice cloning. **Repo này** là fork tập trung vào:

| Mục tiêu | Nội dung |
|----------|----------|
| **Giảm dung lượng** | Export ONNX int4, `ViZipVoiceOnnxTTS`, Gradio export/test, tối ưu ORT |
| **Hướng audiobook** | Chunk text dài (`max_duration`), tách câu, ghép segment — **sẽ chỉnh text normalizer cho đọc sách sau** |
| **Chạy local** | File `.bat`, project `uv`, Slint GUI tùy chọn |

Stack **offline**: tải checkpoint PyTorch một lần, chọn **30 giọng mẫu**, synth không gửi dữ liệu ra ngoài.

**Gradio** mirror [HF Space](https://huggingface.co/spaces/dinhthuan/ViZipvoice): chuẩn hóa, tách câu, rule text ngắn, ghép segment. Tab PyTorch và ONNX dùng chung tham số (`max_duration`, `remove_long_sil`, steps, speed, …).

> **License:** fork này **không cho phép dùng thương mại** — xem [LICENSE](LICENSE). Model và thư viện gốc giữ license riêng.

> **Tóm gọn:** chất lượng ViZipVoice gốc + ONNX int4 + Gradio offline — cài **`uv`**, chạy một file `.bat`.

---

## ✨ Tính năng nổi bật

- 🗣️ **Sao chép giọng zero-shot** — ~3–10 giây audio sạch + transcript, hoặc chọn **30 giọng** trong `models/ViZipvoice/audio/`.
- 🎯 **Chất lượng PyTorch gốc** — full model flow-matching ZipVoice (24 kHz, Vocos); thường tốt hơn bản ONNX lượng tử hóa.
- 🧩 **Pipeline chuẩn hóa tùy biến** — các bước ghép nối (`soe_vinorm`, dọn dấu câu, `period_break`, `sea_g2p`, …) + tab xem trước.
- 📝 **Synth theo câu** — tách text dài; điều chỉnh `num_step` / `speed` cho câu 1 từ và 2–4 từ.
- 🎛️ **Gradio PyTorch** — chọn ref, tham số nâng cao, JSON trạng thái — cùng flow Space; **tự mở browser**.
- ⚡ **Gradio ONNX** — cùng tab/controls cho `models/onnx` int4 (`run_onnx_local.bat`).
- 📤 **Gradio export ONNX** — export ZipVoice + Vocos int4 (`run_export_gui.bat`).
- 🖥️ **Slint GUI (tùy chọn)** — desktop native, không browser.
- 💾 **CLI + Python** — `infer_vizipvoice`, `ViZipVoiceTTS`, `ViZipVoiceOnnxTTS`, RTF/segments.

---

## 🛠️ Điểm nhấn kỹ thuật

| Lĩnh vực | Đã xây dựng |
|----------|-------------|
| **Wrapper inference** | `ViZipVoiceTTS` — HF hoặc local, tự chọn `checkpoint-<step>.pt` mới nhất, FP16 trên CUDA |
| **Xử lý văn bản** | Registry `vi_normalizer` — chain bước, API preview, CLI `--normalize-pipeline` |
| **Audio I/O (Windows)** | `audio_io` — soundfile + pydub (mp3/wav), không cần FFmpeg / torchcodec |
| **Hậu xử lý segment** | Synth từng câu → nối silence + crossfade + fade in/out |
| **Cài đặt local** | Project `uv`, file `.bat` có log từng bước + `pause` khi lỗi |
| **Nhánh ONNX** | `export_onnx_bundle`, `ViZipVoiceOnnxTTS`, Gradio + CLI; **tokenizer ký tự** |

---

## 🏗️ Kiến trúc

```
  Giọng mẫu + transcript
            │
            ▼
  ┌─────────────────┐    ┌──────────────────┐    ┌─────────────────────┐
  │ Chuẩn hóa       │──► │ Tách câu         │──► │ Mỗi segment:         │
  │ (soe_vinorm →   │    │ + rule text ngắn │    │ ZipVoice flow-match  │
  │  spacing, …)    │    │                  │    │ → Vocos 24 kHz       │
  └─────────────────┘    └──────────────────┘    └──────────┬──────────┘
                                                             ▼
                                               ┌─────────────────────────┐
                                               │ Ghép segment             │
                                               │ (silence, crossfade,     │
                                               │  fade) → WAV             │
                                               └─────────────────────────┘
```

<details>
<summary><strong>Cấu trúc thư mục</strong></summary>

```
local_app/               # Gradio
  app.py                 # PyTorch TTS (vizipvoice-local)
  onnx_app.py            # Test ONNX (vizipvoice-onnx-local)
  export_app.py          # Export int4 (vizipvoice-export-gui)
  download_model.py
gui/                     # Slint desktop (tùy chọn)
zipvoice/
  vizipvoice.py          # ViZipVoiceTTS (PyTorch)
  onnx_inference/        # ViZipVoiceOnnxTTS
  export/onnx_bundle.py
  tokenizer/vi_normalizer.py
models/ViZipvoice/       # Weights PyTorch + audio/ (30 giọng)
models/onnx/             # ONNX đã export + tokens.txt
models/vocoder/          # mel_spec_24khz.onnx
output/
setup_local.bat          # uv sync + tải model
run_local.bat            # Gradio PyTorch :7860
run_onnx_local.bat       # Gradio test ONNX :7861
run_export_gui.bat       # Gradio export :7862
```

</details>

---

## 🚀 Bắt đầu nhanh (Windows)

**Yêu cầu:** đã cài [uv](https://github.com/astral-sh/uv), ~4 GB ổ đĩa cho weights.

```bat
git clone https://github.com/phamtronglam2001/ViZipvoiceGUI.git
cd ViZipvoiceGUI

setup_local.bat        REM uv sync + tải model (lần đầu)
run_local.bat          REM Gradio PyTorch → http://127.0.0.1:7860 (tự mở browser)
```

**Luồng ONNX** (sau khi PyTorch chạy ổn):

```bat
run_export_gui.bat     REM export int4 → models\onnx + models\vocoder (:7862)
run_onnx_local.bat     REM test ONNX, giao diện giống Gradio gốc (:7861)
```

Tương đương thủ công:

```bat
uv sync
uv run vizipvoice-download
uv run vizipvoice-local
uv sync --extra export && uv run vizipvoice-export-gui
uv sync --extra onnx && uv run vizipvoice-onnx-local
```

### File `.bat` (Windows)

| File | Chức năng | URL | Browser |
|------|-----------|-----|---------|
| `setup_local.bat` | Cài đặt + tải model | — | — |
| `run_local.bat` | Gradio PyTorch | `:7860` | Tự mở |
| `run_onnx_local.bat` | Gradio test ONNX | `:7861` | Tự mở |
| `run_export_gui.bat` | Gradio export ONNX | `:7862` | Tự mở |
| `gui\run_gui.bat` | Slint desktop | cửa sổ native | Không |

Gradio hỗ trợ `--no-inbrowser`. Mỗi `.bat` in mô tả trước khi chạy, báo `[OK]`/`[LOI]`, **dừng màn hình khi lỗi**.

| Điểm vào | Lệnh |
|----------|------|
| **Gradio PyTorch** | `run_local.bat` hoặc `uv run vizipvoice-local` |
| **Gradio ONNX** | `run_onnx_local.bat` hoặc `uv run vizipvoice-onnx-local` |
| **Gradio export** | `run_export_gui.bat` hoặc `uv run vizipvoice-export-gui` |
| **CLI PyTorch** | `uv run python -m zipvoice.bin.infer_vizipvoice --model-dir models/ViZipvoice ...` |
| **CLI ONNX** | `uv run python -m zipvoice.bin.infer_vizipvoice_onnx --onnx-model-dir models/onnx --use-int4 ...` |
| **Slint GUI** | `gui\run_gui.bat` (`uv sync --extra gui`) |
| **Tải model** | `uv run vizipvoice-download` |

<details>
<summary><strong>Linux / macOS</strong></summary>

```bash
uv sync
uv run vizipvoice-download
uv run vizipvoice-local --host 0.0.0.0 --port 7860
```

Đặt `VIZIPVOICE_MODEL_DIR` để đổi đường dẫn mặc định `models/ViZipvoice`.

</details>

---

## 🖥️ Hướng dẫn dùng

### Các tab Gradio (PyTorch & ONNX)

`run_local.bat` và `run_onnx_local.bat` dùng cùng cấu trúc:

| Tab | Nội dung |
|-----|----------|
| **TTS** / **TTS (ONNX)** | 30 giọng, prompt, text, Generate, slider nâng cao |
| **Text Normalizer** | Xem trước pipeline trước khi synth |

Tab ONNX thêm: thư mục ONNX, bật int4, đường dẫn vocoder (`models/vocoder/mel_spec_24khz.onnx`).

### Giọng mẫu (`models/ViZipvoice/audio/`)

Format sidecar — một file audio + `.txt` cùng tên:

```text
audio/Đinh-Quyết.mp3
audio/Đinh-Quyết.txt
```

Hỗ trợ: `.wav`, `.mp3`, `.flac`, `.m4a`, `.ogg`. Giọng hay dùng: **Đinh-Quyết**, **Nhã-Uyên**, **MC**.

### Ví dụ CLI (model local)

```bat
uv run python -m zipvoice.bin.infer_vizipvoice ^
  --model-dir models/ViZipvoice ^
  --prompt-wav "models/ViZipvoice/audio/Đinh-Quyết.mp3" ^
  --prompt-text "Transcript khớp prompt." ^
  --text "ViZipVoice chạy offline trên máy local." ^
  --res-wav-path output/demo.wav ^
  --num-step 16 --seed 666
```

Pipeline chuẩn hóa tùy chỉnh:

```bat
uv run python -m zipvoice.bin.infer_vizipvoice ^
  ... ^
  --normalize-pipeline soe_vinorm,spacing,period_break
```

Bước **sea-g2p** (tùy chọn): `uv sync --extra g2p` trước, rồi thêm `sea_g2p` vào pipeline.

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

### Mặc định inference (giống HF Space)

- Chuẩn hóa **`soe-vinorm`** rồi **dọn spacing** quanh dấu câu.
- Tách câu; ghép segment với **crossfade 80 ms**, **silence 180 ms**, fade in/out.
- Câu **1 từ**: `num_step ≥ 24`, `speed = 0.6`.
- Câu **2–4 từ**: `speed = 0.8`.

### Các bước chuẩn hóa (`vi_normalizer`)

| Key | Chức năng |
|-----|-----------|
| `soe_vinorm` | Số, ngày, đơn vị, viết tắt → dạng đọc |
| `spacing` | Dọn khoảng trắng quanh dấu câu |
| `vieneu` | Gộp khoảng trắng thừa |
| `period_break` | Ngoặc / số+chấm → xuống dòng (cấu trúc TTS) |
| `lowercase` | Chuyển chữ thường |
| `strip_quotes` | Bỏ dấu ngoặc kép |
| `sea_g2p` | NSW mạnh hơn (cần extra `g2p`) |

Mặc định: `soe_vinorm, spacing`.

Vocab tokenizer: `models/ViZipvoice/tokens.txt` (244 ký tự). **Không có token silence riêng** — dùng dấu câu hoặc `silence_ms` giữa segment.

### Export & test ONNX

```bat
uv sync --extra export
uv run vizipvoice-export-onnx --model-dir models/ViZipvoice --export-root models --quantize-int4 1
```

| Đường dẫn | Nội dung |
|-----------|----------|
| `models/onnx/text_encoder_int4.onnx` | Text encoder int4 |
| `models/onnx/fm_decoder_int4.onnx` | FM decoder int4 |
| `models/onnx/tokens.txt` | Tokenizer ký tự (SimpleTokenizer) |
| `models/vocoder/mel_spec_24khz.onnx` | Vocos → librosa ISTFT |

> **Lưu ý:** ViZipVoice dùng tokenizer **ký tự**. **Không** copy bundle ONNX sang [ZipVoice-Vietnamese-ONNX-GUI](https://github.com/phamtronglam2001/ZipVoice-Vietnamese-ONNX-GUI) (app đó dùng **Espeak phoneme**). Chỉ test ONNX trong repo này qua `run_onnx_local.bat` hoặc `infer_vizipvoice_onnx`.

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

Biến môi trường: `VIZIPVOICE_ONNX_DIR`, `VIZIPVOICE_VOCODER_ONNX`, `VIZIPVOICE_MODEL_DIR`.

---

## 📦 Model

Weights: [contextboxai/ViZipvoice](https://huggingface.co/contextboxai/ViZipvoice)

| Tài sản | Mô tả |
|---------|-------|
| `checkpoint-1300000.pt` | Checkpoint FP16 mới nhất (tự chọn) |
| `checkpoint-700000.pt`, `checkpoint-920000.pt` | Checkpoint cũ để so sánh |
| `config.json` / `model.json` | Cấu hình model |
| `tokens.txt` | Tokenizer ký tự, **244** token tiếng Việt |
| `audio/` | **30** clip mẫu + transcript |
| `demo/` | Output mẫu |

- **Sample rate:** 24 kHz · **Vocoder:** [charactr/vocos-mel-24khz](https://huggingface.co/charactr/vocos-mel-24khz)
- **Dữ liệu train:** ~7000 giờ (~6500 h Việt + ~500 h Anh)

### Audio demo (HF)

**Đinh-Quyết** — [nghe](https://huggingface.co/contextboxai/ViZipvoice/resolve/main/demo/demo_01_%C4%90inh-Quy%E1%BA%BFt.wav)

**Nhã-Uyên** — [nghe](https://huggingface.co/contextboxai/ViZipvoice/resolve/main/demo/demo_02_Nh%C3%A3-Uy%C3%AAn.wav)

**MC** — [nghe](https://huggingface.co/contextboxai/ViZipvoice/resolve/main/demo/demo_03_MC.wav)

---

## ⚙️ Hiệu năng

| Thiết bị | Ghi chú |
|----------|---------|
| **CUDA** | FP16 autocast mặc định; RTF tốt nhất |
| **CPU** | Chạy được ngay; RTF cao hơn (~7× trên desktop thường) |
| **MPS** | Hỗ trợ qua auto-select device PyTorch |

Các đòn bẩy: `--num-step`, `--speed`, `--guidance-scale`, slider post-process trong Gradio **Advanced**.

---

## 💡 Gợi ý chất lượng

- **Một người nói**, audio sạch, ít vang/nhạc nền; prompt **3–10 giây** là lý tưởng.
- **Transcript phải khớp** prompt — sai transcript làm clone kém.
- Câu rất ngắn: thử `--speed 0.7` hoặc `--num-step 24`.
- Text dài: dấu câu rõ để tách câu tự nhiên.
- Ký tự ngoài vocab `tokens.txt` bị tokenizer bỏ qua.
- So sánh pipeline trong tab **Text Normalizer** trước khi chạy hàng loạt.

---

## 🧪 Phát triển

```bat
uv sync
uv run python -c "from zipvoice.tokenizer.vi_normalizer import preview_normalize; print(preview_normalize('Hôm nay là 8/6/2026.').normalized)"
```

Cài đặt legacy (không dùng uv):

```bash
pip install -r requirements.txt
export PYTHONPATH="$PWD:$PYTHONPATH"
```

Extra tùy chọn:

```bat
uv sync --extra export  REM export ONNX (onnx<1.19, onnxruntime, onnxscript, librosa)
uv sync --extra onnx    REM Gradio test ONNX + infer
uv sync --extra gui     REM Slint desktop GUI
uv sync --extra g2p     REM normalizer sea-g2p
```

---

## ⚖️ Đạo đức & sử dụng

ViZipVoice có thể clone giọng từ prompt ngắn. Chỉ dùng giọng khi bạn có quyền hoặc được đồng ý. Không dùng để giả mạo, lừa đảo, gây hại, hoặc phát tán thông tin sai lệch.

**Fork này chỉ cho phép dùng phi thương mại** (nghiên cứu, cá nhân, thử audiobook). Liên hệ tác giả nếu cần license thương mại.

---

## 🙏 Lời cảm ơn & License

### Upstream (cảm ơn)

| Thành phần | Nguồn | License |
|------------|-------|---------|
| **ViZipVoice** (model & repo gốc) | [iamdinhthuan/ViZipvoice](https://github.com/iamdinhthuan/ViZipvoice) · [contextboxai/ViZipvoice](https://huggingface.co/contextboxai/ViZipvoice) | Apache-2.0 |
| ZipVoice | [k2-fsa/ZipVoice](https://github.com/k2-fsa/ZipVoice) | Apache-2.0 |
| Vocos | [gemelo-ai/vocos](https://github.com/gemelo-ai/vocos) · [charactr/vocos-mel-24khz](https://huggingface.co/charactr/vocos-mel-24khz) | MIT |
| soe-vinorm | [soe-vinorm](https://pypi.org/project/soe-vinorm/) | theo repo gốc |
| sea-g2p (tùy chọn) | [pnnbao97/sea-g2p](https://github.com/pnnbao97/sea-g2p) | theo repo gốc |

### Fork này (ViZipvoiceGUI)

**Tác giả:** [Lam Pham](https://github.com/phamtronglam2001) · **Repo:** [phamtronglam2001/ViZipvoiceGUI](https://github.com/phamtronglam2001/ViZipvoiceGUI)

Mã nguồn và tài liệu trong **repo này** phát hành theo **Apache 2.0 kèm điều khoản cấm dùng thương mại** — không được dùng cho mục đích thương mại nếu chưa có sự đồng ý bằng văn bản. Chi tiết: **[LICENSE](LICENSE)**. Weights và thư viện bên thứ ba giữ license gốc.

Vui lòng cite **ZipVoice** và ghi nhận **ViZipVoice** khi tái sử dụng.

### Trích dẫn

```bibtex
@article{zhu2025zipvoice,
  title={ZipVoice: Fast and High-Quality Zero-Shot Text-to-Speech with Flow Matching},
  author={Zhu, Han and others},
  journal={arXiv preprint arXiv:2506.13053},
  year={2025}
}
```

---

## 👤 Tác giả

| Vai trò | Liên kết |
|---------|----------|
| **ViZipVoice GUI (fork này)** | [phamtronglam2001/ViZipvoiceGUI](https://github.com/phamtronglam2001/ViZipvoiceGUI) — Lam Pham |
| **ViZipVoice (model gốc)** | [iamdinhthuan/ViZipvoice](https://github.com/iamdinhthuan/ViZipvoice) · [HF Space](https://huggingface.co/spaces/dinhthuan/ViZipvoice) |
