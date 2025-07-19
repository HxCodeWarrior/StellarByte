#!/bin/bash

echo "==================================="
echo " StellarByte 环境安装脚本 (Unix)"
echo "==================================="
echo 

# 检查 Python 是否已安装
if ! command -v python3 &> /dev/null; then
    echo "错误: 未检测到 Python。请安装 Python 3.8 或更高版本。"
    echo "您可以从 https://www.python.org/downloads/ 下载 Python。"
    exit 1
fi

# 解析命令行参数
INSTALL_DEV=0
INSTALL_CUDA=0
VENV_PATH="./venv"

while [[ $# -gt 0 ]]; do
    case $1 in
        --dev)
            INSTALL_DEV=1
            shift
            ;;
        --cuda)
            INSTALL_CUDA=1
            shift
            ;;
        --venv)
            VENV_PATH="$2"
            shift 2
            ;;
        *)
            echo "未知参数: $1"
            exit 1
            ;;
    esac
done

echo "使用虚拟环境路径: $VENV_PATH"

# 构建 Python 脚本参数
SCRIPT_ARGS="--venv \"$VENV_PATH\""
if [ $INSTALL_DEV -eq 1 ]; then
    SCRIPT_ARGS="$SCRIPT_ARGS --dev"
fi
if [ $INSTALL_CUDA -eq 1 ]; then
    SCRIPT_ARGS="$SCRIPT_ARGS --cuda"
fi

# 运行 Python 安装脚本
echo "运行安装脚本..."
eval "python3 scripts/setup_env.py $SCRIPT_ARGS"

if [ $? -ne 0 ]; then
    echo "安装失败。请查看上面的错误信息。"
    exit 1
fi

echo 
echo "安装完成！"
echo "要激活虚拟环境，请运行: source $VENV_PATH/bin/activate"
echo