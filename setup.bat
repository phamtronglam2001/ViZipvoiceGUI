@echo off
setlocal EnableExtensions
chcp 65001 >nul 2>&1
cd /d "%~dp0"

echo.
echo ============================================================
echo   ViZipVoice — setup.bat  (cai dat lan dau)
echo ============================================================
echo.
echo Chon che do:
echo   [1] CPU  — nhanh, khong can NVIDIA  ^(mac dinh^)
echo   [2] GPU  — NVIDIA CUDA: PyTorch + ONNX  ^(tai lon hon^)
echo.
echo Sau khi xong: double-click run.bat de mo ung dung.
echo.
set "SETUP_MODE=cpu"
set /p SETUP_CHOICE="Nhap 1 hoac 2 [mac dinh 1]: "
if "%SETUP_CHOICE%"=="2" set "SETUP_MODE=gpu"
if /i "%SETUP_CHOICE%"=="gpu" set "SETUP_MODE=gpu"

echo.
echo Ban chon: %SETUP_MODE%
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

call :base_setup
if errorlevel 1 goto :fail

if /i "%SETUP_MODE%"=="gpu" (
    call :install_pytorch_gpu
    if errorlevel 1 goto :fail
    call :install_onnx_gpu
    if errorlevel 1 goto :fail
) else (
    echo cpu>"%CD%\.install_mode"
    echo cpu>"%CD%\.install_mode_pytorch"
    echo cpu>"%CD%\.install_mode_onnx"
    call :install_onnx_cpu
    if errorlevel 1 goto :fail
)

echo.
echo ============================================================
echo   HOAN TAT — setup.bat
echo ============================================================
echo   Che do: %SETUP_MODE%
echo   Buoc tiep: double-click run.bat
echo.
pause
exit /b 0

:base_setup
echo.
echo ------------------------------------------------------------
echo [1/2] uv sync — cai package tu pyproject.toml...
echo ------------------------------------------------------------
uv sync
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

:install_pytorch_gpu
echo.
echo ------------------------------------------------------------
echo GPU [1/2] PyTorch CUDA cu128...
echo ------------------------------------------------------------
echo gpu>"%CD%\.install_mode"
echo gpu>"%CD%\.install_mode_pytorch"
uv pip install --reinstall torch torchaudio --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 exit /b 1
echo Kiem tra CUDA...
uv run python -c "import torch; print('torch', torch.__version__); print('cuda available', torch.cuda.is_available()); print('device', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A (se fallback CPU khi chay)')"
if errorlevel 1 exit /b 1
exit /b 0

:install_onnx_cpu
echo.
echo ------------------------------------------------------------
echo CPU — ONNX Runtime + export deps...
echo ------------------------------------------------------------
uv sync --extra onnx --extra export
if errorlevel 1 exit /b 1
uv run python -c "import onnxruntime as ort; print('onnxruntime OK'); print('EPs', ort.get_available_providers())"
if errorlevel 1 exit /b 1
exit /b 0

:install_onnx_gpu
echo.
echo ------------------------------------------------------------
echo GPU [2/2] ONNX Runtime GPU + CUDA DLLs...
echo ------------------------------------------------------------
echo gpu>"%CD%\.install_mode_onnx"
uv sync --extra onnx --extra export
if errorlevel 1 exit /b 1
uv pip uninstall onnxruntime 2>nul
uv pip install -r requirements-onnx-gpu.txt
if errorlevel 1 exit /b 1
uv run python -c "import onnxruntime as ort; from zipvoice.onnx_inference.providers import ensure_cuda_runtime_on_path, is_cuda_execution_provider_loadable, provider_status_message; ensure_cuda_runtime_on_path(); print('EPs', ort.get_available_providers()); print('CUDA loadable', is_cuda_execution_provider_loadable()); print(provider_status_message(True))"
if errorlevel 1 exit /b 1
exit /b 0

:fail
echo.
echo [LOI] setup.bat that bai.
pause
exit /b 1
