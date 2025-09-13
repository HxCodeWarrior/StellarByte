import os
import re
from setuptools import setup, find_packages

# 读取版本号
def get_version():
    init_py = open(os.path.join('model', '__init__.py')).read()
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", init_py, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("无法从 __init__.py 获取版本信息")

# 读取README作为长描述
with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

# 读取requirements.txt
with open('requirements.txt', encoding='utf-8') as f:
    requirements = f.read().splitlines()
    # 过滤掉注释行和空行
    requirements = [line for line in requirements if line and not line.startswith('#')]

setup(
    name="stellarbyte",
    version=get_version(),
    author="ByteWyrm",
    author_email="bytewyrm@163.com",
    description="高性能Transformer语言模型实现",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/HxCodeWarrior/StellarByte",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Creative Commons Attribution-NonCommercial 4.0 International License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=8.3.4",
            "pytest-cov>=4.1.0",
            "mypy>=1.5.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
        ],
        "docs": [
            "sphinx>=7.0.0",
            "sphinx-rtd-theme>=1.3.0",
            "myst-parser>=2.0.0",
        ],
    },
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "stellarbyte-pretrain=model_pretrain:main",
            "stellarbyte-tokenizer=tokenizer_pretrain:main",
        ],
    },
)