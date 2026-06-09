@echo off
setlocal EnableExtensions
chcp 65001 >nul 2>&1
cd /d "%~dp0"

echo.
echo ============================================================
echo   ViZipVoice — run_local.bat
echo ============================================================
echo.
echo Muc dich:
echo   Mo Gradio TTS local ^(PyTorch^) de test doc tieng Viet.
echo   URL mac dinh: http://127.0.0.1:7860  ^(tu mo trinh duyet^)
echo.
echo Yeu cau:
echo   - Da chay setup_local.bat ^(co .venv^)
echo   - Model tai models\ViZipvoice ^(hoac tu Hugging Face^)
echo.
echo Tham so tuy chon: truyen them vao day, vd. --port 7861
echo   Vi du: run_local.bat --port 7861
echo   Tat tu mo browser: run_local.bat --no-inbrowser
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

if not exist "models\ViZipvoice\config.json" (
    echo [CANH BAO] Chua thay models\ViZipvoice\config.json
    echo            App co the tu tai tu Hugging Face hoac bao loi khi mo.
    echo.
)

echo [DANG CHAY] uv run vizipvoice-local %*
echo.

uv run vizipvoice-local %*
set EXIT_CODE=%ERRORLEVEL%

echo.
if %EXIT_CODE% neq 0 (
    echo ============================================================
    echo [LOI] vizipvoice-local thoat voi ma loi: %EXIT_CODE%
    echo ============================================================
    echo Kiem tra log phia tren. Goi y:
    echo   - Chua setup: chay setup_local.bat
    echo   - Thieu model: chay setup_local.bat hoac vizipvoice-download
    echo   - Port 7860 bi chiem: run_local.bat --port 7861
    echo.
    pause
    exit /b %EXIT_CODE%
)

echo ============================================================
echo [XONG] Server da dong binh thuong.
echo ============================================================
pause
exit /b 0
