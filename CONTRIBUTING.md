# StellarByte 贡献指南

感谢您对 StellarByte 项目的关注！本文档将指导您如何为项目做出贡献。

## 项目结构

```
StellarByte/
├── model/                  # 核心模型实现
│   ├── Attention.py        # 注意力机制实现
│   ├── DecoderLayer.py     # Transformer解码器层
│   ├── MLP.py              # 多层感知机实现
│   ├── MoE.py              # 混合专家模型实现
│   ├── Model.py            # 主模型架构
│   ├── Position_Embedding.py # 位置编码实现
│   ├── RMSNorm.py          # RMSNorm归一化层
│   ├── config.py           # 模型配置类
│   └── utils/              # 工具模块
│       ├── DropPath.py     # DropPath正则化
│       ├── KVCache.py      # KV缓存实现
│       ├── LoRA.py         # LoRA低秩适应
│       └── Memory.py       # 内存优化工具
├── test/                   # 单元测试
├── utils/                  # 通用工具
│   ├── checkpoint.py       # 检查点管理
│   ├── config_params.py    # 配置参数处理
│   ├── logger.py           # 日志工具
│   ├── metrics.py          # 评估指标
│   ├── model_info.py       # 模型信息分析
│   └── progressbar.py      # 进度条
├── configs/                # 配置文件
├── scripts/                # 实用脚本
├── model_pretrain.py       # 预训练脚本
├── model_stf_train.py      # 微调训练脚本
└── datasets.py             # 数据集实现
```

## 开发环境设置

1. 克隆仓库并安装开发依赖：

```bash
git clone https://github.com/yourusername/StellarByte.git
cd StellarByte
pip install -e ".[dev]"
```

2. 安装预提交钩子（可选但推荐）：

```bash
pre-commit install
```

## 代码风格

我们遵循以下代码风格规范：

- **PEP 8**：Python 代码风格指南
- **类型注解**：使用 Python 类型提示增强代码可读性和可维护性
- **文档字符串**：所有公共 API 必须有文档字符串
- **注释**：复杂逻辑需要添加注释说明

可以使用以下工具自动格式化代码：

```bash
# 代码格式化
black .

# 导入排序
isort .

# 类型检查
mypy .

# 代码质量检查
flake8 .
```

## 提交流程

1. **创建分支**：从 `main` 分支创建新的功能分支

```bash
git checkout -b feature/your-feature-name
```

2. **编写代码**：实现您的功能或修复

3. **编写测试**：为新功能添加测试用例

4. **运行测试**：确保所有测试通过

```bash
python -m pytest
```

5. **提交更改**：使用描述性的提交消息

```bash
git add .
git commit -m "feat: 添加新功能 XYZ"
```

6. **推送分支**：将分支推送到远程仓库

```bash
git push origin feature/your-feature-name
```

7. **创建 Pull Request**：在 GitHub 上创建 PR 并等待审核

## 提交消息规范

我们使用 [Conventional Commits](https://www.conventionalcommits.org/) 规范：

```
<类型>[可选作用域]: <描述>

[可选正文]

[可选脚注]
```

类型包括：
- **feat**：新功能
- **fix**：错误修复
- **docs**：文档更改
- **style**：不影响代码含义的更改（空格、格式等）
- **refactor**：既不修复错误也不添加功能的代码更改
- **perf**：提高性能的代码更改
- **test**：添加或修正测试
- **build**：影响构建系统或外部依赖的更改
- **ci**：CI 配置文件和脚本的更改

## 测试指南

- 所有新功能必须有单元测试
- 测试文件应放在 `test/` 目录中
- 测试文件名应以 `test_` 开头
- 使用 `pytest` 运行测试

## 文档指南

- 所有公共 API 必须有文档字符串
- 文档字符串应遵循 [Google 风格](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html)
- 复杂功能应在 `docs/` 目录中有详细文档

## 发布流程

1. 更新版本号（在 `model/__init__.py` 中）
2. 更新 CHANGELOG.md
3. 创建发布分支 `release/vX.Y.Z`
4. 创建 PR 并合并到 `main`
5. 在 GitHub 上创建新的发布

## 行为准则

请参阅 [行为准则](CODE_OF_CONDUCT.md)。

## 许可证

通过贡献代码，您同意您的贡献将根据项目的 MIT 许可证进行许可。