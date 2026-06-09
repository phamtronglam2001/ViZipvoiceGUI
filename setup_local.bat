@echo off
setlocal EnableExtensions
chcp 65001 >nul 2>&1
cd /d "%~dp0"

echo.
echo ============================================================
echo   ViZipVoice — setup_local.bat
echo ============================================================
echo.
echo Muc dich:
echo   [1] Cai dependency Python vao .venv  (uv sync)
echo   [2] Tai model ViZipVoice neu chua co   (models\ViZipvoice\)
echo.
echo Sau khi xong, chay tiep:
echo   run_local.bat       — Gradio TTS PyTorch   (http://127.0.0.1:7860, tu mo browser)
echo   run_export_gui.bat  — Gradio export ONNX   (http://127.0.0.1:7862, tu mo browser)
echo   run_onnx_local.bat  — Gradio test ONNX     (http://127.0.0.1:7861, tu mo browser)
echo   gui\run_gui.bat     — Slint desktop ^(cua so native, khong phai browser^)
echo.
echo Nhan phim bat ky de bat dau...
pause >nul

where uv >nul 2>&1
if errorlevel 1 (
    echo.
    echo [LOI] Khong tim thay lenh "uv".
    echo       Cai uv: https://docs.astral.sh/uv/getting-started/installation/
    echo.
    pause
    exit /b 1
)

echo.
echo ------------------------------------------------------------
echo [1/2] uv sync — cai package tu pyproject.toml...
echo ------------------------------------------------------------
uv sync
if errorlevel 1 (
    echo.
    echo [LOI] uv sync that bai ^(ma loi: %ERRORLEVEL%^).
    echo       Kiem tra mang, proxy, hoac chay lai tu thu muc: %CD%
    echo.
    pause
    exit /b 1
)
echo [OK] uv sync thanh cong.

echo.
echo ------------------------------------------------------------
echo [2/2] Model ViZipVoice...
echo ------------------------------------------------------------
if exist "models\ViZipvoice\config.json" (
    echo [OK] Model da co — bo qua download.
    echo      Thu muc: %CD%\models\ViZipvoice
) else (
    echo Model chua co — dang tai tu Hugging Face ^(co the mat vai phut^)...
    uv run vizipvoice-download
    if errorlevel 1 (
        echo.
        echo [LOI] vizipvoice-download that bai ^(ma loi: %ERRORLEVEL%^).
        echo       Thu lai sau hoac tai thu cong vao models\ViZipvoice
        echo.
        pause
        exit /b 1
    )
    if exist "models\ViZipvoice\config.json" (
        echo [OK] Download model thanh cong.
    ) else (
        echo [CANH BAO] Download chay xong nhung khong thay config.json.
        echo            Kiem tra thu muc models\ViZipvoice
    )
)

echo.
echo ============================================================
echo   HOAN TAT — setup_local.bat
echo ============================================================
echo   Buoc tiep: double-click run_local.bat de test TTS
echo.
pause
exit /b 0
