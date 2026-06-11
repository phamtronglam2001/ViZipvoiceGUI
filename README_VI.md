<div align="center">

# 🎙️ ViZipVoice GUI — Offline Local

**Fork [ViZipVoice](https://github.com/iamdinhthuan/ViZipvoice) — giảm dung lượng ONNX, hướng đọc sách nói (audiobook), Gradio/CLI offline.**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-native-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Gradio](https://img.shields.io/badge/UI-Gradio-F97316)](https://www.gradio.app/)
[![uv](https://img.shields.io/badge/setup-uv-DE5FE9)](https://github.com/astral-sh/uv)
[![License](https://img.shields.io/badge/license-Non--Commercial-red)](LICENSE)

**Tác giả:** [Pham Trong Lam](https://github.com/phamtronglam2001) · **Gốc:** [iamdinhthuan/ViZipvoice](https://github.com/iamdinhthuan/ViZipvoice) · [English](README.md)

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

> **Tóm gọn:** chất lượng ViZipVoice gốc + ONNX int4 + Gradio offline — copy thư mục, chạy **`setup.bat`**, rồi **`run.bat`**.

---

## ✨ Tính năng nổi bật

- 🗣️ **Sao chép giọng zero-shot** — ~3–10 giây audio sạch + transcript, hoặc chọn **30 giọng** trong `models/ViZipvoice/audio/`.
- 🎯 **Chất lượng PyTorch gốc** — full model flow-matching ZipVoice (24 kHz, Vocos); thường tốt hơn bản ONNX lượng tử hóa.
- 🧩 **Pipeline chuẩn hóa tùy biến** — các bước ghép nối (`soe_vinorm`, dọn dấu câu, `period_break`, `sea_g2p`, …) + tab xem trước.
- 📝 **Synth theo câu** — tách text dài; điều chỉnh `num_step` / `speed` cho câu 1 từ và 2–4 từ.
- 🎛️ **Gradio PyTorch** — chọn ref, **tải TXT audiobook**, tab **Hiệu năng** (thiết bị, luồng CPU, FP16), **xuất MP3** qua `ffmpeg/bin`, tham số nâng cao, JSON trạng thái — cùng flow Space; **tự mở browser**.
- ⚡ **Gradio ONNX** — cùng controls TTS cho `models/onnx` int4 + Vocos ONNX (`mel_spec_24khz.onnx` hoặc `mel_spec_24khz_int4.onnx`); **tải TXT audiobook**, **xuất MP3/ffmpeg** (dùng chung flow PyTorch), tab **Hiệu năng** cuối cùng (GPU, ORT threads, ép CPU); phù hợp **text dài / audiobook** (`run.bat` → [2]).
- 📤 **Gradio export ONNX** — export ZipVoice + Vocos int4 (`run.bat` → [3]).
- 🚀 **Launcher thống nhất** — `setup.bat` (chỉ uv, profile [1]–[7]: ONNX/PyTorch CPU hoặc GPU, export) + `run.bat` mở app trực tiếp (không `uv sync`); GPU lỗi → tự fallback CPU — không còn file `.bat` CPU riêng.
- 🖥️ **Slint GUI (tùy chọn)** — desktop native, không browser.
- 💾 **CLI + Python** — `infer_vizipvoice`, `ViZipVoiceTTS`, `ViZipVoiceOnnxTTS`, RTF/segments.

---

## 🛠️ Điểm nhấn kỹ thuật

| Lĩnh vực | Đã xây dựng |
|----------|-------------|
| **Wrapper inference** | `ViZipVoiceTTS` — HF hoặc local, tự chọn `checkpoint-<step>.pt` mới nhất, FP16 trên CUDA |
| **Xử lý văn bản** | Registry `vi_normalizer` — chain bước, API preview, CLI `--normalize-pipeline` |
| **Audio I/O (Windows)** | `audio_io` — soundfile + pydub đọc file; **xuất MP3** tùy chọn qua `ffmpeg/bin/ffmpeg.exe` (64/128/256 kbps) |
| **Hậu xử lý segment** | Synth từng câu → nối silence + crossfade + fade in/out |
| **Cài đặt local** | Project `uv`, file `.bat` có log từng bước + `pause` khi lỗi |
| **Nhánh ONNX** | `export_onnx_bundle`, `ViZipVoiceOnnxTTS`, Gradio + CLI; **tokenizer ký tự**; nên dùng cho text dài |

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
models/vocoder/          # mel_spec_24khz.onnx hoặc mel_spec_24khz_int4.onnx
ffmpeg/bin/              # ffmpeg.exe portable cho xuất MP3 (tùy chọn)
output/                  # WAV / MP3 đã synth
.venv/                   # virtualenv uv (tạo bởi setup.bat)
setup.bat                # Chỉ cài .venv — profile [1]–[7] (ONNX/PyTorch CPU/GPU, export)
run.bat                  # Menu (không cài thêm): PyTorch :7860 / ONNX :7861 / Export :7862 / Slint
requirements-onnx-inference.txt   # librosa cho vocoder ONNX ISTFT ([1]/[3])
requirements-onnx-gpu-cuda-libs.txt  # DLL NVIDIA CUDA khi probe GPU lỗi ([3]/[6])
```

</details>

---

## 🚀 Bắt đầu nhanh (Windows)

**Yêu cầu:** đã cài [uv](https://github.com/astral-sh/uv), ~4 GB ổ đĩa cho weights. Copy nguyên thư mục hoặc clone repo.

```bat
cd ViZipvoiceGUI

setup.bat              REM lần đầu: chọn profile [1]–[7] (mặc định [1] ONNX CPU)
uv run vizipvoice-download   REM một lần: weights PyTorch + giọng mẫu
run.bat                REM menu: [1] PyTorch :7860  [2] ONNX :7861  [3] Export :7862  [4] Slint
```

**Luồng ONNX** (sau khi PyTorch chạy ổn):

```bat
setup.bat              REM [7] cho export, hoặc [3]/[6] cho ONNX GPU
run.bat                REM chọn [3] Export ONNX → models\onnx + models\vocoder
run.bat                REM chọn [2] Test ONNX Gradio (:7861)
```

> **Đã bỏ:** `setup_local.bat`, `run_local.bat`, `run_onnx_local.bat`, `run_onnx_standalone.bat`, `run_export_gui.bat`, `setup_checks.py` — thay bằng **`setup.bat`** + **`run.bat`**.

### Profile cài đặt (`setup.bat`)

`setup.bat` chỉ tạo/cập nhật `.venv` (uv-only). **Không** mở app hay tải weights.

| Profile | Cài gì | Ghi chú |
|---------|--------|---------|
| **[1] ONNX CPU** (mặc định) | `onnxruntime` + **librosa** (`requirements-onnx-inference.txt`) | Vocoder `mel_spec_*.onnx` → ISTFT sau ORT |
| **[2] PyTorch CPU** | `uv sync` cơ bản | PyTorch CPU từ lockfile |
| **[3] ONNX GPU** | `onnxruntime-gpu` + librosa; **probe DLL CUDA** tự động | Probe lỗi → `requirements-onnx-gpu-cuda-libs.txt`; đóng `run.bat` trước nếu DLL bị khóa |
| **[4] PyTorch GPU** | PyTorch **CUDA cu128** (fallback cu124) | Probe kiểm tra `torch.version.cuda` |
| **[5] ONNX CPU + PyTorch CPU** | [1] + [2] | Stack CPU đầy đủ |
| **[6] ONNX GPU + PyTorch GPU** | [4] + [3] | Stack NVIDIA; ghi `.install_mode_onnx` / `.install_mode_pytorch` |
| **[7] Export ONNX** | `onnx`, `onnxscript`, `librosa` | Cho `run.bat` → [3] export Gradio |

GPU lỗi hoặc không có CUDA lúc chạy → app **tự fallback CPU**. DLL CUDA ONNX load qua `zipvoice.onnx_inference.providers` (`add_dll_directory` + `nvidia/*/bin`).

Tương đương thủ công:

```bat
uv sync
uv run vizipvoice-download
uv run vizipvoice-local
uv pip install -r requirements-onnx-inference.txt   REM infer ONNX + vocoder
uv pip install -r requirements-onnx-gpu.txt -r requirements-onnx-gpu-cuda-libs.txt   REM ONNX GPU
uv pip install onnx onnxscript librosa              REM export [7]
```

### File `.bat` (Windows)

| File | Vai trò |
|------|---------|
| `setup.bat` | Chỉ cài deps `.venv` — chọn profile [1]–[7] |
| `run.bat` | Chọn app — **không `uv sync`**; kiểm tra deps ONNX/export trước khi chạy |

Menu `run.bat`:

| Chọn | App | URL / UI | Browser |
|------|-----|----------|---------|
| **[1] PyTorch** | Gradio TTS | `:7860` | Tự mở |
| **[2] ONNX** | Gradio test ONNX | `:7861` | Tự mở |
| **[3] Export** | Gradio export int4 | `:7862` | Tự mở |
| **[4] Slint** | Desktop native | cửa sổ | Không |

Tham số thêm sau menu, vd. `--no-inbrowser` hoặc `--port 7863`. Mỗi `.bat` in bước, báo `[OK]`/`[LOI]`, **dừng màn hình khi lỗi**.

| Điểm vào | Lệnh |
|----------|------|
| **Gradio PyTorch** | `run.bat` → [1] hoặc `uv run vizipvoice-local` |
| **Gradio ONNX** | `run.bat` → [2] hoặc `uv run vizipvoice-onnx-local` |
| **Gradio export** | `run.bat` → [3] hoặc `uv run vizipvoice-export-gui` |
| **CLI PyTorch** | `uv run python -m zipvoice.bin.infer_vizipvoice --model-dir models/ViZipvoice ...` |
| **CLI ONNX** | `uv run python -m zipvoice.bin.infer_zipvoice_onnx --onnx-model-dir models/onnx --use-int4 ...` |
| **Slint GUI** | `run.bat` → [4] (`uv sync --extra gui`) |
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

**PyTorch** (`run.bat` → [1]) — tab **TTS**:

| Mục | Nội dung |
|-----|----------|
| **TTS** | 30 giọng, prompt, **tải TXT audiobook** (UTF-8 → điền text), Generate |
| **Hiệu năng** (accordion) | Thiết bị `auto` / `cuda` / `cpu`, số luồng CPU PyTorch, FP16 trên CUDA; load GPU lỗi → tự thử CPU |
| **Xuất MP3 / ffmpeg** | Thư mục ffmpeg (mặc định `ffmpeg` → `ffmpeg/bin/ffmpeg.exe`), bitrate 64/128/256 kbps; **chỉ MP3 khi có ffmpeg**, không thì WAV |
| **Advanced** | Steps, guidance, speed, chunking, slider hậu xử lý |
| **Text Normalizer** | Xem trước pipeline trước khi synth |

**ONNX** (`run.bat` → [2]) — thứ tab: **TTS (ONNX)** → **Text Normalizer** → **Hiệu năng** (cuối):

| Tab | Nội dung |
|-----|----------|
| **TTS (ONNX)** | Cùng controls synth như PyTorch + thư mục ONNX, int4, vocoder (`mel_spec_24khz.onnx` hoặc `mel_spec_24khz_int4.onnx`), **tải TXT audiobook**, accordion **MP3/ffmpeg** |
| **Text Normalizer** | Tab preview giống PyTorch |
| **Hiệu năng** | GPU (CUDA / DirectML), ép CPU, ORT threads (0 = tự động); đọc `.install_mode_onnx` sau `setup.bat` → [3] hoặc [6] |

### Giới hạn ONNX (câu ngắn)

Inference ONNX **ổn với câu dài và đoạn văn** (kiểu audiobook). **Input ngắn (1–3 từ)** — vd. `một`, hoặc `một. hai.` khi bật tách câu — có thể sai trim mel, nhiễu, hoặc synth chậm. Đây là hạn chế của bản export ONNX, repo **không** vá workaround.

**Khuyến nghị:** cụm ngắn → dùng **PyTorch TTS** (`run.bat` → [1] hoặc `uv run vizipvoice-local`).

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
| `models/vocoder/mel_spec_24khz.onnx` | Vocos → librosa ISTFT (FP32) |
| `models/vocoder/mel_spec_24khz_int4.onnx` | Cùng graph, lượng tử int4 (nhỏ hơn; tự chọn khi có) |

> **Lưu ý:** ViZipVoice dùng tokenizer **ký tự**. **Không** copy bundle ONNX sang [ZipVoice-Vietnamese-ONNX-GUI](https://github.com/phamtronglam2001/ZipVoice-Vietnamese-ONNX-GUI) (app đó dùng **Espeak phoneme**). Chỉ test ONNX trong repo này qua `run.bat` → [2] hoặc `infer_vizipvoice_onnx`.

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

### Biến môi trường

| Biến | Áp dụng | Mục đích |
|------|---------|----------|
| `VIZIPVOICE_MODEL_DIR` | PyTorch, download | Đổi đường dẫn mặc định `models/ViZipvoice` |
| `VIZIPVOICE_FFMPEG_DIR` | Gradio PyTorch | Thư mục chứa `ffmpeg.exe` hoặc `bin/ffmpeg.exe` (mặc định `ffmpeg`) |
| `VIZIPVOICE_ONNX_DIR` | ONNX | Thư mục bundle ONNX đã export |
| `VIZIPVOICE_VOCODER_ONNX` | ONNX | Đường dẫn `mel_spec_24khz.onnx` hoặc `mel_spec_24khz_int4.onnx` |
| `ZIPVOICE_ONNX_GPU` | ONNX | `1` / `true` để ưu tiên GPU |
| `ZIPVOICE_ONNX_THREADS` | ONNX | Số thread ORT mỗi session |
| `ZIPVOICE_FORCE_CPU` | ONNX | `1` để ép CPU execution provider |

`run.bat` đặt `ZIPVOICE_ONNX_GPU=1` khi `.install_mode_onnx` là `gpu` (từ `setup.bat` → [3] hoặc [6]).

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
| **CUDA** | PyTorch: `setup.bat` → [4] hoặc [6]; FP16 mặc định (tắt trong **Hiệu năng**); RTF tốt nhất |
| **CPU** | `setup.bat` → [1]/[2]/[5]; không cần NVIDIA; RTF cao hơn (~7× trên desktop thường) |
| **MPS** | PyTorch `auto` trên Apple Silicon |
| **ONNX GPU** | `setup.bat` → [3] hoặc [6]: `onnxruntime-gpu` + probe DLL CUDA (`requirements-onnx-gpu-cuda-libs.txt`); tự fallback nếu EP không dùng được |

**Tự fallback:** PyTorch thử lại CPU nếu load CUDA lỗi; ONNX Runtime fallback khi thiếu GPU/DLL — không cần launcher riêng.

Các đòn bẩy: tab **Hiệu năng** (thiết bị, threads, FP16 / ORT / ép CPU), `--num-step`, `--speed`, `--guidance-scale`, slider trong **Advanced**.

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
uv sync --extra export    REM export ONNX (onnx<1.19, onnxruntime, onnxscript, librosa)
uv sync --extra onnx      REM Gradio test ONNX + infer (ORT CPU)
uv sync --extra onnx-gpu  REM onnxruntime-gpu + DLL NVIDIA CUDA 12 (xem requirements-onnx-gpu.txt)
uv sync --extra gui       REM Slint desktop GUI
uv sync --extra g2p       REM normalizer sea-g2p
```

Trên Windows, nên dùng **`setup.bat`** profile — [1]/[3] cho ONNX (+ librosa `requirements-onnx-inference.txt`), [3]/[6] cho ONNX GPU với probe DLL, [7] cho export.

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

**Tác giả:** [Pham Trong Lam](https://github.com/phamtronglam2001) · **Repo:** [phamtronglam2001/ViZipvoiceGUI](https://github.com/phamtronglam2001/ViZipvoiceGUI)

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
| **ViZipVoice GUI (fork này)** | [phamtronglam2001/ViZipvoiceGUI](https://github.com/phamtronglam2001/ViZipvoiceGUI) — Pham Trong Lam |
| **ViZipVoice (model gốc)** | [iamdinhthuan/ViZipvoice](https://github.com/iamdinhthuan/ViZipvoice) · [HF Space](https://huggingface.co/spaces/dinhthuan/ViZipvoice) |
