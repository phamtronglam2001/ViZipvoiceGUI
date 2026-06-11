@echo off
setlocal EnableExtensions
chcp 65001 >nul 2>&1
cd /d "%~dp0"

set "VENV_PY=%CD%\.venv\Scripts\python.exe"

echo.
echo ============================================================
echo   ViZipVoice — setup.bat  ^(chi cai .venv^)
echo ============================================================
echo.
echo   [1] ONNX CPU
echo   [2] PyTorch CPU        ^(mac dinh tu uv sync^)
echo   [3] ONNX GPU
echo   [4] PyTorch GPU
echo   [5] ONNX CPU + PyTorch CPU
echo   [6] ONNX GPU + PyTorch GPU
echo   [7] Export ONNX        ^(onnx + librosa + onnxscript^)
echo.
echo Mo app: run.bat ^(khong chay o day^).
echo [3]/[6]: dong run.bat truoc neu DLL onnxruntime bi khoa.
echo.
set "CHOICE=1"
set /p CHOICE="Nhap 1-7 [mac dinh 1]: "
if "%CHOICE%"=="" set "CHOICE=1"

where uv >nul 2>&1
if errorlevel 1 (
    echo [LOI] Khong tim thay uv — https://docs.astral.sh/uv/
    pause
    exit /b 1
)

echo.
echo [uv sync] tao / cap nhat .venv...
if exist "uv.lock" (uv sync --frozen) else (uv sync)
if errorlevel 1 goto :fail

if not exist "%VENV_PY%" (
    echo [LOI] Khong co .venv\Scripts\python.exe
    goto :fail
)

if "%CHOICE%"=="1" goto :p1
if "%CHOICE%"=="2" goto :p2
if "%CHOICE%"=="3" goto :p3
if "%CHOICE%"=="4" goto :p4
if "%CHOICE%"=="5" goto :p5
if "%CHOICE%"=="6" goto :p6
if "%CHOICE%"=="7" goto :p7
echo [LOI] Lua chon khong hop le: %CHOICE%
goto :fail

:p1
call :onnx_cpu
if errorlevel 1 goto :fail
echo cpu>".install_mode_onnx"
goto :ok

:p2
echo cpu>".install_mode_pytorch"
goto :ok

:p3
call :onnx_gpu
if errorlevel 1 goto :fail
echo gpu>".install_mode_onnx"
goto :ok

:p4
call :pytorch_gpu
if errorlevel 1 goto :fail
echo gpu>".install_mode_pytorch"
goto :ok

:p5
call :onnx_cpu
if errorlevel 1 goto :fail
echo cpu>".install_mode_onnx"
echo cpu>".install_mode_pytorch"
goto :ok

:p6
call :pytorch_gpu
if errorlevel 1 goto :fail
call :onnx_gpu
if errorlevel 1 goto :fail
echo gpu>".install_mode_pytorch"
echo gpu>".install_mode_onnx"
goto :ok

:p7
echo [uv pip] export deps...
uv pip install "onnx>=1.16,<1.19" onnxscript librosa
if errorlevel 1 goto :fail
goto :ok

:onnx_cpu
echo [uv pip] onnxruntime CPU + librosa ^(vocoder mel_spec ONNX^)...
uv pip uninstall onnxruntime-gpu
uv pip install "onnxruntime>=1.18,<1.24"
if errorlevel 1 exit /b 1
uv pip install -r requirements-onnx-inference.txt
exit /b %ERRORLEVEL%

:onnx_gpu
echo [uv pip] onnxruntime-gpu...
uv pip uninstall onnxruntime onnxruntime-gpu
uv pip install --offline -r requirements-onnx-gpu.txt
if errorlevel 1 uv pip install -r requirements-onnx-gpu.txt
if errorlevel 1 exit /b 1
call :check_onnx_gpu_dll
if errorlevel 1 (
    echo [uv pip] CUDA runtime DLL...
    uv pip install --offline -r requirements-onnx-gpu-cuda-libs.txt
    if errorlevel 1 uv pip install -r requirements-onnx-gpu-cuda-libs.txt
    call :check_onnx_gpu_dll
)
echo [uv pip] librosa ^(vocoder mel_spec ONNX^)...
uv pip install -r requirements-onnx-inference.txt
exit /b %ERRORLEVEL%

:pytorch_gpu
echo [uv pip] torch CUDA...
uv pip install --offline torch torchaudio
if errorlevel 1 uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
if errorlevel 1 exit /b 1
call :check_pytorch_gpu
exit /b %ERRORLEVEL%

:check_pytorch_gpu
echo [check] PyTorch CUDA...
"%VENV_PY%" -c "import torch,sys; c=getattr(torch.version,'cuda',None); print('OK torch',torch.__version__,'cuda',c) if c else None; sys.exit(0 if c else 1)"
if errorlevel 1 (
    echo [CANH BAO] PyTorch chua co CUDA build.
    exit /b 1
)
exit /b 0

:check_onnx_gpu_dll
echo [check] ONNX CUDA DLL...
"%VENV_PY%" -c "from zipvoice.onnx_inference.providers import is_cuda_execution_provider_loadable; import sys; ok=is_cuda_execution_provider_loadable(); print('CUDA EP load:',ok); sys.exit(0 if ok else 1)"
if errorlevel 1 (
    echo [CANH BAO] CUDA EP chua load duoc DLL.
    exit /b 1
)
exit /b 0

:ok
echo.
echo ============================================================
echo   XONG — profile [%CHOICE%]
echo   Tiep theo: run.bat
echo ============================================================
pause
exit /b 0

:fail
echo.
echo [LOI] setup.bat that bai.
pause
exit /b 1
