@echo off
setlocal EnableExtensions
chcp 65001 >nul 2>&1
cd /d "%~dp0"

echo.
echo ============================================================
echo   ViZipVoice — run_export_gui.bat
echo ============================================================
echo.
echo Muc dich:
echo   [1] Cai them package export ^(onnx, onnxruntime, ...^)
echo   [2] Mo Gradio export ONNX int4
echo   URL mac dinh: http://127.0.0.1:7862  ^(tu mo trinh duyet^)
echo.
echo Output export ^(mac dinh^):
echo   models\onnx\     — text_encoder_int4, fm_decoder_int4, tokens.txt
echo   models\vocoder\  — mel_spec_24khz.onnx
echo.
echo Test ONNX sau export:
echo   run_onnx_local.bat  — Gradio ONNX ^(giong run_local.bat^) :7861
echo   gui\run_gui.bat     — Slint tab ONNX Inference
echo.
echo Tham so tuy chon: --port 7863  ^(truyen vao cuoi dong lenh^)
echo   Tat tu mo browser: run_export_gui.bat --no-inbrowser
echo.
echo Nhan Ctrl+C de dung server.
echo ============================================================
echo.

if not exist ".venv\" (
    echo [LOI] Chua co .venv — chay setup_local.bat truoc.
    echo.
    pause
    exit /b 1
)

where uv >nul 2>&1
if errorlevel 1 (
    echo [LOI] Khong tim thay lenh "uv".
    echo.
    pause
    exit /b 1
)

echo ------------------------------------------------------------
echo [1/2] uv sync --extra export
echo ------------------------------------------------------------
uv sync --extra export
if errorlevel 1 (
    echo.
    echo [LOI] uv sync --extra export that bai ^(ma loi: %ERRORLEVEL%^).
    echo.
    pause
    exit /b 1
)
echo [OK] Package export da san sang.

if not exist "models\ViZipvoice\config.json" (
    echo.
    echo [CANH BAO] Chua thay models\ViZipvoice — can checkpoint de export.
    echo            Chay setup_local.bat hoac tai model truoc.
    echo.
)

echo.
echo ------------------------------------------------------------
echo [2/2] Mo Gradio export GUI...
echo ------------------------------------------------------------
echo [DANG CHAY] uv run vizipvoice-export-gui %*
echo.

uv run vizipvoice-export-gui %*
set EXIT_CODE=%ERRORLEVEL%

echo.
if %EXIT_CODE% neq 0 (
    echo ============================================================
    echo [LOI] vizipvoice-export-gui thoat voi ma loi: %EXIT_CODE%
    echo ============================================================
    echo Kiem tra log phia tren. Goi y:
    echo   - Thieu onnx/onnxruntime: uv sync --extra export
    echo   - Port 7862 bi chiem: run_export_gui.bat --port 7863
    echo.
    pause
    exit /b %EXIT_CODE%
)

echo ============================================================
echo [XONG] Export GUI da dong binh thuong.
echo ============================================================
pause
exit /b 0
