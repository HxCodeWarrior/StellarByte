@echo off
echo ===================================
echo  StellarByte 环境安装脚本 (Windows)
echo ===================================
echo.

REM 检查 Python 是否已安装
python --version >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo 错误: 未检测到 Python。请安装 Python 3.8 或更高版本。
    echo 您可以从 https://www.python.org/downloads/ 下载 Python。
    exit /b 1
)

REM 解析命令行参数
set INSTALL_DEV=0
set INSTALL_CUDA=0
set VENV_PATH=.\venv

:parse_args
if "%~1"=="" goto :done_args
if /i "%~1"=="--dev" set INSTALL_DEV=1
if /i "%~1"=="--cuda" set INSTALL_CUDA=1
if /i "%~1"=="--venv" (
    set VENV_PATH=%~2
    shift
)
shift
goto :parse_args
:done_args

echo 使用虚拟环境路径: %VENV_PATH%

REM 构建 Python 脚本参数
set SCRIPT_ARGS=--venv "%VENV_PATH%"
if %INSTALL_DEV%==1 set SCRIPT_ARGS=%SCRIPT_ARGS% --dev
if %INSTALL_CUDA%==1 set SCRIPT_ARGS=%SCRIPT_ARGS% --cuda

REM 运行 Python 安装脚本
echo 运行安装脚本...
python scripts\setup_env.py %SCRIPT_ARGS%

if %ERRORLEVEL% NEQ 0 (
    echo 安装失败。请查看上面的错误信息。
    exit /b 1
)

echo.
echo 安装完成！
echo 要激活虚拟环境，请运行: %VENV_PATH%\Scripts\activate
echo.

pause