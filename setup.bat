@echo off
setlocal EnableExtensions
chcp 65001 >nul 2>&1
cd /d "%~dp0"

:menu
echo.
echo ============================================================
echo   ViZipVoice — setup.bat
echo ============================================================
echo.
echo Moi muc CHI bo sung phan thieu — KHONG ha GPU da cai.
echo Cache: uv/pip dung wheel da co, khong tai lai neu .venv OK.
echo.
echo   [1] Bo sung CPU         — PyTorch/ONNX CPU neu thieu
echo   [2] Bo sung GPU PyTorch — chi torch CUDA neu thieu
echo   [3] Bo sung GPU ONNX    — chi onnxruntime-gpu neu thieu
echo   [4] GPU ca hai          — PyTorch CUDA + ONNX GPU
echo   [5] Cai lai tu dau      — xoa .venv, cai lai (fix env hong)
echo.
set "SETUP_MODE=cpu"
set /p SETUP_CHOICE="Nhap 1-5 [mac dinh 1]: "
if "%SETUP_CHOICE%"=="" set "SETUP_CHOICE=1"
if "%SETUP_CHOICE%"=="2" set "SETUP_MODE=gpu_pytorch"
if "%SETUP_CHOICE%"=="3" set "SETUP_MODE=gpu_onnx"
if "%SETUP_CHOICE%"=="4" set "SETUP_MODE=gpu_both"
if "%SETUP_CHOICE%"=="5" set "SETUP_MODE=reinstall"
if /i "%SETUP_CHOICE%"=="pytorch" set "SETUP_MODE=gpu_pytorch"
if /i "%SETUP_CHOICE%"=="onnx" set "SETUP_MODE=gpu_onnx"
if /i "%SETUP_CHOICE%"=="gpu" set "SETUP_MODE=gpu_both"
if /i "%SETUP_CHOICE%"=="reinstall" set "SETUP_MODE=reinstall"

echo.
echo Ban chon: %SETUP_MODE%
if /i "%SETUP_MODE%"=="reinstall" (
    echo.
    echo [5] Se XOA thu muc .venv va cai lai tu dau.
    set /p REINSTALL_OK="Chac chan? Go Y roi Enter: "
    if /i not "%REINSTALL_OK%"=="Y" (
        echo Huy.
        pause
        exit /b 0
    )
) else (
    echo Nhan phim bat ky de bat dau...
    pause >nul
)

where uv >nul 2>&1
if errorlevel 1 (
    echo.
    echo [LOI] Khong tim thay lenh "uv".
    echo       Cai uv: https://docs.astral.sh/uv/getting-started/installation/
    echo.
    pause
    exit /b 1
)

if /i "%SETUP_MODE%"=="reinstall" (
    call :wipe_venv
    call :base_setup
    if errorlevel 1 goto :fail
    echo.
    echo [OK] Da cai lai base. Chon muc bo sung tiep theo:
    goto :menu
)

call :base_setup
if errorlevel 1 goto :fail

if /i "%SETUP_MODE%"=="gpu_pytorch" (
    call :do_gpu_pytorch
    if errorlevel 1 goto :fail
) else if /i "%SETUP_MODE%"=="gpu_onnx" (
    call :do_gpu_onnx
    if errorlevel 1 goto :fail
) else if /i "%SETUP_MODE%"=="gpu_both" (
    call :do_gpu_pytorch
    if errorlevel 1 goto :fail
    call :do_gpu_onnx
    if errorlevel 1 goto :fail
) else (
    call :do_cpu
    if errorlevel 1 goto :fail
)

call :finalize
exit /b 0

:wipe_venv
echo.
echo ------------------------------------------------------------
echo Xoa .venv cu...
echo ------------------------------------------------------------
if exist ".venv\" (
    rmdir /s /q ".venv"
    echo [OK] Da xoa .venv
) else (
    echo [OK] Khong co .venv — bo qua.
)
del ".install_mode" 2>nul
del ".install_mode_pytorch" 2>nul
del ".install_mode_onnx" 2>nul
exit /b 0

:base_setup
echo.
echo ------------------------------------------------------------
echo [1/2] uv sync — package co ban tu pyproject.toml...
echo ------------------------------------------------------------
if exist "uv.lock" (
    uv sync --frozen
) else (
    uv sync
)
if errorlevel 1 (
    echo.
    echo [LOI] uv sync that bai ^(ma loi: %ERRORLEVEL%^).
    echo       Kiem tra mang, proxy, hoac chay lai tu thu muc: %CD%
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
echo Dong bo ref audio bundled ^(assets\ref_audio^)...
uv run python -c "from pathlib import Path; from local_app.ref_audio_bundle import sync_bundled_ref_audio; sync_bundled_ref_audio(Path('models/ViZipvoice'))"
exit /b 0

:do_cpu
echo.
echo ------------------------------------------------------------
echo Bo sung CPU — khong doi PyTorch CUDA / ONNX GPU da co
echo ------------------------------------------------------------
uv run python -m local_app.setup_checks ensure_onnx_cpu
exit /b %ERRORLEVEL%

:do_gpu_pytorch
echo.
echo ------------------------------------------------------------
echo Bo sung GPU PyTorch — cache-first, moi CUDA build deu OK
echo ------------------------------------------------------------
uv run python -m local_app.setup_checks install_pytorch_cuda
exit /b %ERRORLEVEL%

:do_gpu_onnx
echo.
echo ------------------------------------------------------------
echo Bo sung GPU ONNX — khong doi PyTorch da cai
echo ------------------------------------------------------------
echo [CANH BAO] Dong Gradio/Python cu truoc khi cai GPU ONNX ^(tranh loi DLL^).
uv run python -m local_app.setup_checks install_onnx_gpu
exit /b %ERRORLEVEL%

:finalize
uv run python -m local_app.setup_checks write_markers
echo.
echo ============================================================
echo   HOAN TAT — setup.bat
echo ============================================================
echo   Che do: %SETUP_MODE%
if exist ".install_mode_pytorch" (
    echo   PyTorch: 
    type ".install_mode_pytorch"
)
if exist ".install_mode_onnx" (
    echo   ONNX:    
    type ".install_mode_onnx"
)
echo   Buoc tiep: double-click run.bat
echo.
pause
exit /b 0

:fail
echo.
echo [LOI] setup.bat that bai.
pause
exit /b 1
