#!/usr/bin/env python
"""
StellarByte 环境设置脚本

此脚本帮助用户快速设置 StellarByte 项目的开发环境，包括：
1. 检查系统依赖
2. 创建虚拟环境
3. 安装项目依赖
4. 验证安装

使用方法：
    python scripts/setup_env.py [--dev] [--cuda] [--venv PATH]

选项：
    --dev       安装开发依赖
    --cuda      安装 CUDA 支持
    --venv PATH 指定虚拟环境路径（默认：./venv）
"""

import os
import sys
import platform
import subprocess
import argparse
from pathlib import Path

# 获取项目根目录
ROOT_DIR = Path(__file__).parent.parent.absolute()

def print_step(message):
    """打印带格式的步骤信息"""
    print(f"\n{'=' * 50}")
    print(f"  {message}")
    print(f"{'=' * 50}\n")

def run_command(command, cwd=ROOT_DIR):
    """运行命令并实时显示输出"""
    print(f"执行: {' '.join(command)}")
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=cwd
    )
    
    for line in process.stdout:
        print(line, end='')
        
    process.wait()
    return process.returncode

def check_python_version():
    """检查 Python 版本"""
    print_step("检查 Python 版本")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print(f"错误: 需要 Python 3.8 或更高版本，当前版本为 {sys.version}")
        sys.exit(1)
    print(f"Python 版本: {sys.version}")
    return True

def create_venv(venv_path):
    """创建虚拟环境"""
    print_step(f"创建虚拟环境: {venv_path}")
    venv_dir = Path(venv_path)
    
    if venv_dir.exists():
        print(f"虚拟环境已存在: {venv_dir}")
        return True
    
    try:
        import venv
        venv.create(venv_dir, with_pip=True)
        print(f"成功创建虚拟环境: {venv_dir}")
        return True
    except Exception as e:
        print(f"创建虚拟环境失败: {e}")
        return False

def get_pip_path(venv_path):
    """获取虚拟环境中的 pip 路径"""
    if platform.system() == "Windows":
        return os.path.join(venv_path, "Scripts", "pip.exe")
    else:
        return os.path.join(venv_path, "bin", "pip")

def install_dependencies(venv_path, dev=False, cuda=False):
    """安装项目依赖"""
    print_step("安装项目依赖")
    pip_path = get_pip_path(venv_path)
    
    # 升级 pip
    run_command([pip_path, "install", "--upgrade", "pip"])
    
    # 安装基本依赖
    if dev:
        print("安装开发依赖...")
        run_command([pip_path, "install", "-e", ".[dev]"])
    else:
        print("安装基本依赖...")
        run_command([pip_path, "install", "-r", "requirements.txt"])
    
    # 安装 CUDA 支持
    if cuda:
        print("安装 CUDA 支持...")
        if platform.system() == "Windows":
            run_command([pip_path, "install", "torch==2.5.1", "torchvision==0.20.1", "--index-url", "https://download.pytorch.org/whl/cu118"])
        else:
            run_command([pip_path, "install", "torch==2.5.1", "torchvision==0.20.1", "--index-url", "https://download.pytorch.org/whl/cu118"])
    
    return True

def verify_installation(venv_path):
    """验证安装"""
    print_step("验证安装")
    
    # 获取 Python 解释器路径
    if platform.system() == "Windows":
        python_path = os.path.join(venv_path, "Scripts", "python.exe")
    else:
        python_path = os.path.join(venv_path, "bin", "python")
    
    # 验证 PyTorch 安装
    print("验证 PyTorch 安装...")
    cmd = [
        python_path, "-c", 
        "import torch; print(f'PyTorch 版本: {torch.__version__}'); "
        "print(f'CUDA 可用: {torch.cuda.is_available()}'); "
        "print(f'GPU 数量: {torch.cuda.device_count()}'); "
        "if torch.cuda.is_available(): print(f'GPU 型号: {torch.cuda.get_device_name(0)}')"
    ]
    run_command(cmd)
    
    # 验证 Transformers 安装
    print("\n验证 Transformers 安装...")
    cmd = [
        python_path, "-c", 
        "import transformers; print(f'Transformers 版本: {transformers.__version__}')"
    ]
    run_command(cmd)
    
    # 运行测试（如果有）
    test_dir = ROOT_DIR / "test"
    if test_dir.exists() and len(list(test_dir.glob("test_*.py"))) > 0:
        print("\n运行测试...")
        run_command([python_path, "-m", "pytest", "-xvs", "test/test_Attention.py"])
    
    return True

def print_activation_instructions(venv_path):
    """打印激活虚拟环境的指令"""
    print_step("环境设置完成")
    
    venv_path = Path(venv_path).absolute()
    if platform.system() == "Windows":
        activate_cmd = f"{venv_path}\Scripts\activate"
    else:
        activate_cmd = f"source {venv_path}/bin/activate"
    
    print(f"\n要激活虚拟环境，请运行:\n\n    {activate_cmd}\n")
    print("开始使用 StellarByte:\n")
    print("    # 运行预训练")
    print("    python model_pretrain.py --config configs/pretrain_config.yaml\n")
    print("    # 运行微调")
    print("    python model_stf_train.py --config configs/sft_config.yaml\n")
    print("更多信息请参阅 README.md 和 INSTALL.md\n")

def main():
    parser = argparse.ArgumentParser(description="StellarByte 环境设置脚本")
    parser.add_argument("--dev", action="store_true", help="安装开发依赖")
    parser.add_argument("--cuda", action="store_true", help="安装 CUDA 支持")
    parser.add_argument("--venv", default="./venv", help="虚拟环境路径（默认：./venv）")
    args = parser.parse_args()
    
    # 检查 Python 版本
    check_python_version()
    
    # 创建虚拟环境
    if not create_venv(args.venv):
        sys.exit(1)
    
    # 安装依赖
    if not install_dependencies(args.venv, args.dev, args.cuda):
        sys.exit(1)
    
    # 验证安装
    if not verify_installation(args.venv):
        sys.exit(1)
    
    # 打印激活指令
    print_activation_instructions(args.venv)

if __name__ == "__main__":
    main()