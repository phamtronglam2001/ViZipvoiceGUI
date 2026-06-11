@echo off
setlocal EnableExtensions
chcp 65001 >nul 2>&1
cd /d "%~dp0"

set "VENV_PY=%CD%\.venv\Scripts\python.exe"

echo.
echo ============================================================
echo   ViZipVoice — run.bat  (mo ung dung)
echo ============================================================
echo.
echo Luu y: run.bat KHONG cai them goi — dung setup.bat truoc.
echo.

if not exist "%VENV_PY%" (
    echo [LOI] Chua co .venv — chay setup.bat truoc.
    echo.
    pause
    exit /b 1
)

echo Chon ung dung:
echo   [1] PyTorch TTS   — Gradio http://127.0.0.1:7860
echo   [2] ONNX TTS      — Gradio http://127.0.0.1:7861
echo   [3] Export ONNX   — Gradio http://127.0.0.1:7862
echo   [4] Slint Desktop
echo.
echo GPU: theo setup.bat, khong co thi fallback CPU.
echo.
set "RUN_CHOICE=1"
set /p RUN_CHOICE="Nhap 1-4 [mac dinh 1]: "
if "%RUN_CHOICE%"=="" set "RUN_CHOICE=1"

if "%RUN_CHOICE%"=="1" goto :run_pytorch
if "%RUN_CHOICE%"=="2" goto :run_onnx
if "%RUN_CHOICE%"=="3" goto :run_export
if "%RUN_CHOICE%"=="4" goto :run_slint
if /i "%RUN_CHOICE%"=="pytorch" goto :run_pytorch
if /i "%RUN_CHOICE%"=="onnx" goto :run_onnx
if /i "%RUN_CHOICE%"=="export" goto :run_export
if /i "%RUN_CHOICE%"=="slint" goto :run_slint
if /i "%RUN_CHOICE%"=="gui" goto :run_slint

echo [LOI] Lua chon khong hop le: %RUN_CHOICE%
pause
exit /b 1

:run_pytorch
if not exist "models\ViZipvoice\config.json" (
    echo [CANH BAO] Chua co model — chay: .venv\Scripts\vizipvoice-download.exe
    echo.
)
echo [DANG CHAY] PyTorch Gradio — http://127.0.0.1:7860
echo.
if exist ".venv\Scripts\vizipvoice-local.exe" (
    ".venv\Scripts\vizipvoice-local.exe"
) else (
    "%VENV_PY%" -m local_app.app
)
goto :done

:run_onnx
if exist ".install_mode_onnx" (
    for /f "usebackq delims=" %%M in (".install_mode_onnx") do (
        if /i "%%M"=="gpu" set "ZIPVOICE_ONNX_GPU=1"
    )
)
if not exist "models\onnx\text_encoder_int4.onnx" (
    echo [CANH BAO] Chua thay models\onnx\ — chon [3] Export ONNX truoc.
    echo.
)
call :check_onnx_deps
if errorlevel 1 goto :fail
echo [DANG CHAY] ONNX Gradio — http://127.0.0.1:7861
echo.
if exist ".venv\Scripts\vizipvoice-onnx-local.exe" (
    ".venv\Scripts\vizipvoice-onnx-local.exe"
) else (
    "%VENV_PY%" -m local_app.onnx_app
)
goto :done

:run_export
call :check_export_deps
if errorlevel 1 goto :fail
if not exist "models\ViZipvoice\config.json" (
    echo [CANH BAO] Chua thay models\ViZipvoice — can checkpoint de export.
    echo.
)
echo [DANG CHAY] Export Gradio — http://127.0.0.1:7862
echo.
if exist ".venv\Scripts\vizipvoice-export-gui.exe" (
    ".venv\Scripts\vizipvoice-export-gui.exe"
) else (
    "%VENV_PY%" -m local_app.export_app
)
goto :done

:run_slint
set "PYTHONPATH=%CD%;%PYTHONPATH%"
echo [DANG CHAY] Slint desktop GUI...
echo.
"%VENV_PY%" gui\main.py
goto :done

:done
set EXIT_CODE=%ERRORLEVEL%
echo.
if %EXIT_CODE% neq 0 (
    echo ============================================================
    echo [LOI] Ung dung thoat voi ma loi: %EXIT_CODE%
    echo ============================================================
    echo Goi y: chay setup.bat / export ONNX / dong Gradio cu neu DLL bi khoa.
    echo.
    pause
    exit /b %EXIT_CODE%
)
echo ============================================================
echo [XONG] Ung dung da dong binh thuong.
echo ============================================================
pause
exit /b 0

:check_onnx_deps
"%VENV_PY%" -c "import onnxruntime as ort; import librosa; p=getattr(ort,'get_available_providers',None); assert callable(p), 'onnxruntime broken'; p()" >nul 2>&1
if not errorlevel 1 exit /b 0
echo [LOI] Chua cai ONNX inference — setup.bat [1] hoac [3] ^(can librosa cho vocoder ONNX^).
exit /b 1

:check_export_deps
"%VENV_PY%" -c "import onnx, onnxruntime as ort; p=getattr(ort,'get_available_providers',None); assert callable(p), 'onnxruntime broken'; p()" >nul 2>&1
if not errorlevel 1 exit /b 0
echo [LOI] Chua cai export deps — setup.bat [7].
exit /b 1

:fail
echo [LOI] run.bat that bai.
pause
exit /b 1
