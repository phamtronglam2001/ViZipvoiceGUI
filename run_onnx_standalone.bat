@echo off
setlocal EnableExtensions
chcp 65001 >nul 2>&1
cd /d "%~dp0"

echo.
echo ============================================================
echo   ViZipVoice ONNX — standalone (copy folder + chay)
echo ============================================================
echo.
echo Can co trong thu muc:
echo   models\onnx\     text_encoder*.onnx + fm_decoder*.onnx + tokens.txt
echo   models\vocoder\  mel_spec_24khz.onnx
echo   models\ViZipvoice\audio\  ref voices + transcript .txt
echo   .venv\           (chay setup_local.bat mot lan truoc)
echo.
echo CLI vi du ^(mac dinh models\onnx, khong can --onnx-model-dir^):
echo   run_onnx_standalone.bat --prompt-wav models\ViZipvoice\audio\MC.mp3 ^
echo     --prompt-text "transcript day du cua ref" --text "Chi so la 8." ^
echo     --res-wav-path output\test.wav
echo.
echo Gradio GUI:
echo   run_onnx_local.bat
echo ============================================================
echo.

if not exist ".venv\" (
    echo [LOI] Chua co .venv — chay setup_local.bat truoc.
    pause
    exit /b 1
)

if not exist "models\onnx\tokens.txt" (
    echo [LOI] Thieu models\onnx\tokens.txt — export ONNX truoc.
    pause
    exit /b 1
)

if not exist "models\onnx\text_encoder.onnx" if not exist "models\onnx\text_encoder_int4.onnx" (
    echo [LOI] Thieu models\onnx\text_encoder*.onnx
    pause
    exit /b 1
)

if not exist "models\vocoder\mel_spec_24khz.onnx" (
    echo [CANH BAO] Thieu models\vocoder\mel_spec_24khz.onnx
)

where uv >nul 2>&1
if errorlevel 1 (
    echo [LOI] Khong tim thay lenh "uv".
    pause
    exit /b 1
)

if "%~1"=="" (
    echo [DANG CHAY] Gradio ONNX — run_onnx_local.bat
    call run_onnx_local.bat %*
    exit /b %ERRORLEVEL%
)

echo [DANG CHAY] uv run python -m zipvoice.bin.infer_vizipvoice_onnx %*
uv run python -m zipvoice.bin.infer_vizipvoice_onnx %*
set EXIT_CODE=%ERRORLEVEL%
if %EXIT_CODE% neq 0 pause
exit /b %EXIT_CODE%
