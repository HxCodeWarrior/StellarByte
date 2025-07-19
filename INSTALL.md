# StellarByte 安装指南

本文档提供了详细的 StellarByte 项目安装和环境配置指南。

## 系统要求

- **操作系统**：Windows 10/11、Linux (Ubuntu 20.04+)、macOS 12+
- **Python**：Python 3.8 或更高版本
- **GPU**：NVIDIA GPU (CUDA 11.8+) 用于加速训练和推理（可选）
- **内存**：至少 8GB RAM，推荐 16GB 或更高
- **存储**：至少 2GB 可用空间

## 安装步骤

### 1. 克隆仓库

```bash
git clone https://github.com/yourusername/StellarByte.git
cd StellarByte
```

### 2. 创建虚拟环境（推荐）

#### 使用 venv (Python 内置)

```bash
# 创建虚拟环境
python -m venv venv

# 激活虚拟环境
# Windows
venv\Scripts\activate
# Linux/macOS
source venv/bin/activate
```

#### 使用 conda

```bash
# 创建虚拟环境
conda create -n stellarbyte python=3.10

# 激活虚拟环境
conda activate stellarbyte
```

### 3. 安装依赖

#### 基本安装

```bash
# 安装基本依赖
pip install -r requirements.txt
```

#### 开发环境安装

```bash
# 安装开发依赖
pip install -e ".[dev]"
```

#### 文档开发安装

```bash
# 安装文档开发依赖
pip install -e ".[docs]"
```

### 4. 验证安装

```bash
# 运行测试确认安装成功
python -m pytest test/
```

## 依赖说明

StellarByte 的依赖按功能模块分类：

### 核心依赖

- **PyTorch (2.5.1+)**：深度学习框架
- **torchvision**：PyTorch 视觉库
- **transformers**：Hugging Face Transformers 库
- **numpy/pandas**：数据处理库

### 模型组件

- **timm**：提供 DropPath 实现

### 可视化与监控

- **swanlab**：实验追踪与可视化
- **matplotlib/seaborn**：绘图库
- **tabulate**：表格格式化
- **rich**：终端美化输出

### 测试与开发

- **pytest**：单元测试框架
- **mypy**：静态类型检查

### 分布式训练

- **torch-xla**：TPU 支持（非 Windows 平台）

### 性能优化

- **accelerator**：内存优化

## 常见问题

### 1. CUDA 相关错误

如果遇到 CUDA 相关错误，请确保：
- 已安装兼容的 NVIDIA 驱动
- CUDA 版本与 PyTorch 版本兼容
- 尝试使用 `torch.cuda.is_available()` 检查 CUDA 是否可用

### 2. 内存不足

如果遇到内存不足错误：
- 减小批处理大小
- 启用梯度检查点 (gradient checkpointing)
- 使用混合精度训练

### 3. 依赖冲突

如果遇到依赖冲突：
- 创建新的虚拟环境
- 按照依赖列表顺序安装
- 考虑使用 `pip-tools` 解决依赖冲突

## 更新

定期更新项目以获取最新功能和修复：

```bash
git pull
pip install -r requirements.txt
```

## 联系与支持

如有安装问题，请通过以下方式获取支持：
- 提交 GitHub Issue
- 联系项目维护者