@echo off
setlocal EnableExtensions
chcp 65001 >nul 2>&1
cd /d "%~dp0"

echo.
echo ============================================================
echo   ViZipVoice — run.bat  (mo ung dung)
echo ============================================================
echo.

if not exist ".venv\" (
    echo [LOI] Chua co .venv — chay setup.bat truoc.
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

echo Chon ung dung:
echo   [1] PyTorch TTS   — Gradio :7860  ^(mac dinh^)
echo   [2] ONNX TTS      — Gradio :7861  ^(can export ONNX truoc^)
echo   [3] Export ONNX   — Gradio :7862
echo   [4] Slint Desktop — cua so native ^(khong browser^)
echo.
echo GPU: theo setup.bat ^(GPU PyTorch / GPU ONNX^), khong co thi fallback CPU.
echo Tham so them: truyen sau menu, vd. --port 7863 --no-inbrowser
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
    echo [CANH BAO] Chua thay models\ViZipvoice\config.json — chay setup.bat.
    echo.
)
echo [DANG CHAY] PyTorch Gradio — http://127.0.0.1:7860
echo.
uv run vizipvoice-local %*
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
uv run vizipvoice-onnx-local %*
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
uv run vizipvoice-export-gui %*
goto :done

:run_slint
set "PYTHONPATH=%CD%;%PYTHONPATH%"
echo [DANG CHAY] Slint desktop GUI...
echo.
python gui\main.py %*
goto :done

:done
set EXIT_CODE=%ERRORLEVEL%
echo.
if %EXIT_CODE% neq 0 (
    echo ============================================================
    echo [LOI] Ung dung thoat voi ma loi: %EXIT_CODE%
    echo ============================================================
    echo Goi y:
    echo   - Chua setup: chay setup.bat
    echo   - Port bi chiem: run.bat — chon lai menu, them --port 7863
    echo   - DLL Access denied: dong Gradio/Python cu, roi chay setup.bat
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
uv run python -c "import onnxruntime as ort; p=getattr(ort,'get_available_providers',None); assert callable(p), 'onnxruntime broken'; p()" >nul 2>&1
if not errorlevel 1 exit /b 0
echo [LOI] Chua cai ONNX dependencies ^(onnxruntime^) hoac cai dat bi loi.
echo       Chay setup.bat de cai --extra onnx.
echo       Neu setup.bat bao "Access is denied" tren file .dll:
echo         dong tat ca cua so Gradio/Python cu roi chay setup.bat lai.
exit /b 1

:check_export_deps
uv run python -c "import onnx, onnxruntime as ort; p=getattr(ort,'get_available_providers',None); assert callable(p), 'onnxruntime broken'; p()" >nul 2>&1
if not errorlevel 1 exit /b 0
echo [LOI] Chua cai export dependencies ^(onnx, onnxruntime^) hoac cai dat bi loi.
echo       Chay setup.bat de cai --extra export.
echo       Neu setup.bat bao "Access is denied" tren file .dll:
echo         dong tat ca cua so Gradio/Python cu roi chay setup.bat lai.
exit /b 1

:fail
echo [LOI] run.bat that bai.
pause
exit /b 1
