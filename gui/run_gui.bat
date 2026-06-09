@echo off
setlocal EnableExtensions
chcp 65001 >nul 2>&1
cd /d "%~dp0.."

echo.
echo ============================================================
echo   ViZipVoice — gui\run_gui.bat  ^(Slint GUI^)
echo ============================================================
echo.
echo Muc dich:
echo   GUI desktop ^(Slint^) — cua so native, KHONG mo browser.
echo   Test PyTorch TTS, ONNX inference, export ONNX.
echo   Gradio + browser: run_local.bat / run_onnx_local.bat / run_export_gui.bat
echo.
echo Tab chinh:
echo   TTS ^(PyTorch^)     — infer_vizipvoice
echo   ONNX Inference     — models\onnx ^(ViZipVoice character tokenizer^)
echo   ONNX Export        — export int4 vao models\
echo.
echo Yeu cau: da chay setup_local.bat ^(+ run_export_gui.bat neu export^)
echo ============================================================
echo.

if not exist ".venv\" (
    echo [LOI] Chua co .venv — chay setup_local.bat truoc.
    echo.
    pause
    exit /b 1
)

set PYTHONPATH=%CD%;%PYTHONPATH%
echo [DANG CHAY] python gui\main.py
echo.

python gui\main.py
set EXIT_CODE=%ERRORLEVEL%

echo.
if %EXIT_CODE% neq 0 (
    echo [LOI] Slint GUI thoat voi ma loi: %EXIT_CODE%
    echo       Thu: uv sync --extra gui
    echo.
    pause
    exit /b %EXIT_CODE%
)

echo [XONG] GUI da dong.
pause
exit /b 0
