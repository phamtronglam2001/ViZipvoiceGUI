@echo off

setlocal EnableExtensions

chcp 65001 >nul 2>&1

cd /d "%~dp0"



echo.

echo ============================================================

echo   ViZipVoice — run_onnx_local.bat

echo ============================================================

echo.

echo Muc dich:

echo   Mo Gradio test ONNX ^(int4 + Vocos ONNX^) — giao dien giong run_local.bat

echo   URL mac dinh: http://127.0.0.1:7861

echo.

echo Yeu cau:

echo   - Da chay setup_local.bat

echo   - Da export ONNX: models\onnx\ + models\vocoder\mel_spec_24khz.onnx

echo     ^(chay run_export_gui.bat neu chua co^)

echo.

echo Ref audio: models\ViZipvoice\audio\
echo Tang toc CPU: set ZIPVOICE_ONNX_THREADS=8  ^(mac dinh = so core, toi da 8^)

echo Tham so: --port 7863  ^(truyen vao cuoi dong lenh^)

echo Tat tu mo browser: run_onnx_local.bat --no-inbrowser

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



if not exist "models\onnx\text_encoder_int4.onnx" (

    echo [CANH BAO] Chua thay models\onnx\text_encoder_int4.onnx

    echo            Chay run_export_gui.bat truoc khi test ONNX.

    echo.

)



echo ------------------------------------------------------------

echo uv sync --extra onnx

echo ------------------------------------------------------------

uv sync --extra onnx

if errorlevel 1 (

    echo [LOI] uv sync --extra onnx that bai.

    echo.

    pause

    exit /b 1

)



echo.

echo [DANG CHAY] uv run vizipvoice-onnx-local %*

echo.



uv run vizipvoice-onnx-local %*

set EXIT_CODE=%ERRORLEVEL%



echo.

if %EXIT_CODE% neq 0 (

    echo ============================================================

    echo [LOI] vizipvoice-onnx-local thoat voi ma loi: %EXIT_CODE%

    echo ============================================================

    echo Goi y:

    echo   - Export ONNX: run_export_gui.bat

    echo   - Port 7861 bi chiem: run_onnx_local.bat --port 7863

    echo.

    pause

    exit /b %EXIT_CODE%

)



echo ============================================================

echo [XONG] ONNX Gradio da dong binh thuong.

echo ============================================================

pause

exit /b 0

