"""模型信息分析工具
=================================================
本文件提供了分析 ByteTransformer 模型详细信息的工具，
包括参数统计、层级结构、内存使用、计算复杂度等，
并生成详细的可视化报告。

作者：ByteWyrm
日期：2025-07-14
=================================================
"""

import os
import sys
import time
import math
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
from tabulate import tabulate
from torch.nn import Module

# 将项目根目录添加到sys.path
root_dir = Path(__file__).parent.parent
sys.path.append(str(root_dir))

# 导入项目模块
from model.config import ByteModelConfig
from model.Model import ByteTransformer


class ModelInfo:
    """模型信息分析类
    
    提供全面的模型分析功能，包括：
    1. 参数统计（总量、分布、稀疏度等）
    2. 层级结构（各层参数量、计算量等）
    3. 内存使用（权重、激活值、梯度等）
    4. 计算复杂度（FLOPs、MACs等）
    5. 推理性能（吞吐量、延迟等）
    6. 可视化报告生成
    """
    
    def __init__(self, model: Union[ByteTransformer, str, Path], config: Optional[ByteModelConfig] = None):
        """初始化模型信息分析器
        
        Args:
            model: ByteTransformer模型实例、模型路径或模型名称
            config: 模型配置，如果model是路径或名称，则必须提供
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 加载模型
        if isinstance(model, ByteTransformer):
            self.model = model
            self.config = model.config
        elif isinstance(model, (str, Path)):
            if config is None:
                raise ValueError("当model是路径或名称时，必须提供config参数")
            self.config = config
            self.model = ByteTransformer(config)
            # 尝试加载模型权重
            try:
                self.model.load_state_dict(torch.load(model, map_location="cpu"))
                print(f"成功加载模型权重: {model}")
            except Exception as e:
                print(f"警告: 无法加载模型权重 ({e})，使用随机初始化权重")
        else:
            raise TypeError("model参数必须是ByteTransformer实例、模型路径或模型名称")
        
        # 初始化分析结果存储
        self.param_stats = {}
        self.layer_stats = {}
        self.memory_stats = {}
        self.compute_stats = {}
        self.perf_stats = {}
        
        # 分析标志
        self._analyzed_params = False
        self._analyzed_layers = False
        self._analyzed_memory = False
        self._analyzed_compute = False
        self._analyzed_perf = False
    
    def analyze_parameters(self) -> Dict[str, Any]:
        """分析模型参数统计信息
        
        Returns:
            包含参数统计信息的字典
        """
        if self._analyzed_params:
            return self.param_stats
        
        # 1. 总参数量
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        # 2. 非嵌入层参数量
        embed_params = self.model.embed_tokens.weight.numel()
        if self.config.tie_word_embeddings:
            non_embed_params = total_params - embed_params
        else:
            lm_head_params = self.model.lm_head.weight.numel()
            non_embed_params = total_params - (embed_params + lm_head_params)
        
        # 3. 按模块统计参数
        module_params = {}
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # 只统计叶子模块
                params = sum(p.numel() for p in module.parameters(recurse=False))
                if params > 0:
                    module_params[name] = params
        
        # 4. 按层统计参数
        layer_params = {}
        for i, layer in enumerate(self.model.layers):
            layer_params[f"layer_{i}"] = sum(p.numel() for p in layer.parameters())
        
        # 5. 按参数类型统计
        param_types = {
            "embedding": embed_params,
            "attention": sum(p.numel() for name, p in self.model.named_parameters() 
                            if any(n in name for n in ["q_proj", "k_proj", "v_proj", "o_proj", "W_q", "W_k", "W_v", "W_o"])),
            "mlp": sum(p.numel() for name, p in self.model.named_parameters() 
                      if any(n in name for n in ["w1", "w2", "w3", "gate_proj", "up_proj", "down_proj"])),
            "norm": sum(p.numel() for name, p in self.model.named_parameters() 
                       if "norm" in name.lower()),
            "lm_head": self.model.lm_head.weight.numel() if not self.config.tie_word_embeddings else 0
        }
        
        # 6. 参数精度统计
        precision_stats = {}
        for dtype_name in ["float32", "float16", "bfloat16", "int8", "int4"]:
            dtype = getattr(torch, dtype_name, None)
            if dtype:
                precision_stats[dtype_name] = sum(p.numel() for p in self.model.parameters() 
                                                if p.dtype == dtype)
        
        # 7. 参数稀疏度（接近零的参数比例）
        sparsity = {}
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.dim() > 1:  # 只检查矩阵参数
                    # 计算接近零的参数比例（绝对值小于1e-4）
                    zero_count = (param.abs() < 1e-4).sum().item()
                    sparsity[name] = zero_count / param.numel()
        
        # 8. 参数范围统计
        range_stats = {}
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.dim() > 0:
                    range_stats[name] = {
                        "min": param.min().item(),
                        "max": param.max().item(),
                        "mean": param.mean().item(),
                        "std": param.std().item()
                    }
        
        # 存储结果
        self.param_stats = {
            "total": total_params,
            "trainable": trainable_params,
            "non_embedding": non_embed_params,
            "by_module": module_params,
            "by_layer": layer_params,
            "by_type": param_types,
            "by_precision": precision_stats,
            "sparsity": sparsity,
            "range": range_stats
        }
        
        self._analyzed_params = True
        return self.param_stats
    
    def analyze_layers(self) -> Dict[str, Any]:
        """分析模型层级结构
        
        Returns:
            包含层级结构信息的字典
        """
        if self._analyzed_layers:
            return self.layer_stats
        
        # 1. 基本层级信息
        basic_info = {
            "num_layers": self.config.num_layers,
            "model_dim": self.config.model_dim,
            "num_heads": self.config.num_attention_heads,
            "num_kv_heads": self.config.num_kv_heads,
            "hidden_dim": self.config.hidden_dim,
            "max_seq_len": self.config.max_seq_len,
        }
        
        # 2. 每层详细信息
        layers_info = []
        for i, layer in enumerate(self.model.layers):
            # print(layer)
            # 注意力层信息
            attn = layer.self_attn
            attn_info = {
                "num_heads": attn.num_heads,
                "head_dim": attn.head_dim,
                "qkv_params": sum(p.numel() for p in [attn.W_q.weight, attn.W_k.weight, attn.W_v.weight]),
                "output_params": attn.W_o.weight.numel()
            }
            
            # MLP层信息
            mlp = layer.mlp
            mlp_info = {
                "hidden_dim": mlp.w1.out_features,
                "up_params": mlp.w1.weight.numel(),
                "gate_params": mlp.w2.weight.numel() if hasattr(mlp, "w2") else 0,
                "down_params": mlp.w3.weight.numel()
            }
            
            layers_info.append({
                "id": i,
                "attention": attn_info,
                "mlp": mlp_info,
                "total_params": sum(p.numel() for p in layer.parameters())
            })
        
        # 3. 层间连接信息
        connections = []
        for i in range(self.config.num_layers):
            connections.append({
                "from": f"layer_{i}",
                "to": f"layer_{i+1}" if i < self.config.num_layers - 1 else "output",
                "type": "residual"
            })
        
        # 存储结果
        self.layer_stats = {
            "basic_info": basic_info,
            "layers": layers_info,
            "connections": connections
        }
        
        self._analyzed_layers = True
        return self.layer_stats
    
    def analyze_memory(self, batch_size: int = 1, seq_len: int = 512) -> Dict[str, Any]:
        """分析模型内存使用情况
        
        Args:
            batch_size: 批次大小
            seq_len: 序列长度
            
        Returns:
            包含内存使用情况的字典
        """
        if self._analyzed_memory:
            return self.memory_stats
        
        # 1. 模型参数内存
        param_memory = sum(p.nelement() * p.element_size() for p in self.model.parameters())
        
        # 2. 模型缓冲区内存
        buffer_memory = sum(b.nelement() * b.element_size() for b in self.model.buffers())
        
        # 3. 激活值内存估计（前向传播）
        # 假设每个token在每层都产生model_dim大小的激活值
        D = self.config.model_dim
        activation_size = 4  # float32 = 4 bytes
        
        # 嵌入层激活值
        embed_activation = batch_size * seq_len * D * activation_size
        
        # 每个Transformer层的激活值
        # 注意力层: Q, K, V, attn_weights, attn_output
        attn_activation = batch_size * seq_len * D * 5 * activation_size
        # MLP层: hidden_states, intermediate, output
        mlp_activation = batch_size * seq_len * (D + self.config.hidden_dim * 2) * activation_size
        
        layer_activation = (attn_activation + mlp_activation) * self.config.num_layers
        
        # 输出层激活值
        output_activation = batch_size * seq_len * self.config.vocab_size * activation_size
        
        total_activation = embed_activation + layer_activation + output_activation
        
        # 4. KV缓存内存（如果启用）
        kv_cache_memory = 0
        if self.config.use_cache:
            # 每个层的K和V缓存
            k_size = self.config.key_cache_dtype.itemsize
            v_size = self.config.value_cache_dtype.itemsize
            
            head_dim = self.config.model_dim // self.config.num_attention_heads
            kv_cache_memory = (
                self.config.num_layers * 
                batch_size * 
                self.config.max_seq_len * 
                self.config.num_kv_heads * 
                head_dim * 
                (k_size + v_size)
            )
        
        # 5. 梯度内存（训练时）
        grad_memory = sum(p.nelement() * p.element_size() for p in self.model.parameters() if p.requires_grad)
        
        # 6. 优化器状态内存（假设使用Adam）
        # Adam为每个参数存储两个状态（一阶矩和二阶矩）
        optimizer_memory = grad_memory * 2
        
        # 存储结果
        self.memory_stats = {
            "parameters": {
                "bytes": param_memory,
                "megabytes": param_memory / (1024 * 1024)
            },
            "buffers": {
                "bytes": buffer_memory,
                "megabytes": buffer_memory / (1024 * 1024)
            },
            "activations": {
                "bytes": total_activation,
                "megabytes": total_activation / (1024 * 1024)
            },
            "kv_cache": {
                "bytes": kv_cache_memory,
                "megabytes": kv_cache_memory / (1024 * 1024)
            },
            "gradients": {
                "bytes": grad_memory,
                "megabytes": grad_memory / (1024 * 1024)
            },
            "optimizer": {
                "bytes": optimizer_memory,
                "megabytes": optimizer_memory / (1024 * 1024)
            },
            "total": {
                "bytes": param_memory + buffer_memory + total_activation + kv_cache_memory + grad_memory + optimizer_memory,
                "megabytes": (param_memory + buffer_memory + total_activation + kv_cache_memory + grad_memory + optimizer_memory) / (1024 * 1024),
                "gigabytes": (param_memory + buffer_memory + total_activation + kv_cache_memory + grad_memory + optimizer_memory) / (1024 * 1024 * 1024)
            }
        }
        
        self._analyzed_memory = True
        return self.memory_stats
    
    def analyze_compute(self, batch_size: int = 1, seq_len: int = 512) -> Dict[str, Any]:
        """分析模型计算复杂度
        
        Args:
            batch_size: 批次大小
            seq_len: 序列长度
            
        Returns:
            包含计算复杂度信息的字典
        """
        if self._analyzed_compute:
            return self.compute_stats
        
        # 模型维度参数
        D = self.config.model_dim
        H = self.config.hidden_dim
        L = self.config.num_layers
        V = self.config.vocab_size
        A = self.config.num_attention_heads
        
        # 1. 嵌入层计算量
        # 查表操作，计算量很小，忽略不计
        embed_flops = 0
        
        # 2. 每个Transformer层的计算量
        # 2.1 自注意力层
        # Q, K, V投影: 3 * seq_len * D * D
        qkv_proj_flops = 3 * batch_size * seq_len * D * D
        # 注意力计算: seq_len * seq_len * D
        attn_flops = batch_size * A * seq_len * seq_len * (D // A)
        # 输出投影: seq_len * D * D
        out_proj_flops = batch_size * seq_len * D * D
        
        attn_total_flops = qkv_proj_flops + attn_flops + out_proj_flops
        
        # 2.2 MLP层
        # 上投影: seq_len * D * H
        up_proj_flops = batch_size * seq_len * D * H
        # 激活函数: seq_len * H (忽略不计)
        # 下投影: seq_len * H * D
        down_proj_flops = batch_size * seq_len * H * D
        
        mlp_total_flops = up_proj_flops + down_proj_flops
        
        # 2.3 层归一化 (忽略不计)
        norm_flops = 0
        
        # 每层总计算量
        layer_flops = attn_total_flops + mlp_total_flops + norm_flops
        
        # 所有层总计算量
        all_layers_flops = layer_flops * L
        
        # 3. 输出层计算量 (LM头)
        # 如果权重共享，则只计算偏置部分
        if self.config.tie_word_embeddings:
            lm_head_flops = batch_size * seq_len * V
        else:
            lm_head_flops = batch_size * seq_len * D * V
        
        # 总计算量
        total_flops = embed_flops + all_layers_flops + lm_head_flops
        
        # 计算每秒浮点运算次数 (FLOPS)
        # 假设生成速度为10 tokens/s
        gen_tokens_per_sec = 10
        gen_flops_per_token = (attn_total_flops + mlp_total_flops + norm_flops) / seq_len
        gen_flops_per_sec = gen_flops_per_token * gen_tokens_per_sec
        
        # 存储结果
        self.compute_stats = {
            "embedding": embed_flops,
            "attention": {
                "qkv_projection": qkv_proj_flops,
                "attention_compute": attn_flops,
                "output_projection": out_proj_flops,
                "total": attn_total_flops
            },
            "mlp": {
                "up_projection": up_proj_flops,
                "down_projection": down_proj_flops,
                "total": mlp_total_flops
            },
            "per_layer": layer_flops,
            "all_layers": all_layers_flops,
            "lm_head": lm_head_flops,
            "total": {
                "flops": total_flops,
                "gflops": total_flops / 1e9,
                "tflops": total_flops / 1e12
            },
            "generation": {
                "flops_per_token": gen_flops_per_token,
                "flops_per_sec": gen_flops_per_sec,
                "gflops_per_sec": gen_flops_per_sec / 1e9
            }
        }
        
        self._analyzed_compute = True
        return self.compute_stats
    
    def analyze_performance(self, batch_sizes: List[int] = [1, 4, 16], 
                       seq_lengths: List[int] = [128, 512, 2048]) -> Dict[str, Any]:
        """分析模型推理性能
        
        Args:
            batch_sizes: 批次大小列表
            seq_lengths: 序列长度列表
            
        Returns:
            包含性能分析信息的字典
        """
        if self._analyzed_perf:
            return self.perf_stats
        
        # 检查是否有GPU
        if not torch.cuda.is_available():
            print("警告: 没有可用的GPU，性能分析可能不准确")
        
        # 将模型移至设备并设置为评估模式
        self.model.to(self.device)
        self.model.eval()
        
        # 获取模型参数的数据类型
        model_dtype = next(self.model.parameters()).dtype
        
        results = {}
        
        # 预热
        warmup_bs = batch_sizes[0]          # 用第一个 batch_size 预热
        dummy_input = torch.randint(0, 100, (warmup_bs, 128), device=self.device)
        with torch.no_grad():
                _ = self.model(dummy_input)
        
        # 测试不同批次大小和序列长度的组合
        for batch_size in batch_sizes:
            if getattr(self.model, "kv_cache", None) is not None:
                self.model.kv_cache.reset()          # 清 token
                self.model.kv_cache.batch_size = None  # 允许重新分配

            batch_results = {}
            for seq_len in seq_lengths:
                # 跳过可能导致OOM的组合
                if batch_size * seq_len > 32768 and torch.cuda.is_available():
                    batch_results[seq_len] = {"error": "可能导致OOM，已跳过"}
                    continue
                
                # 生成随机输入，使用模型的数据类型
                input_ids = torch.randint(0, 100, (batch_size, seq_len), device=self.device, dtype=torch.long)
                
                # 测量前向传播时间
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                start_time = time.time()
                
                with torch.no_grad():
                    for _ in range(10):
                        _ = self.model(input_ids)
                
                torch.cuda.synchronize() if torch.cuda.is_available() else None
                end_time = time.time()
                
                # 计算平均时间和吞吐量
                avg_time = (end_time - start_time) / 10
                throughput = batch_size * seq_len / avg_time
                
                batch_results[seq_len] = {
                    "latency_ms": avg_time * 1000,
                    "throughput_tokens_per_sec": throughput
                }
            
            results[batch_size] = batch_results
        
        # 测量生成速度（自回归解码）
        gen_results = {}
        for batch_size in [1, 2, 4]:
            if getattr(self.model, "kv_cache", None) is not None:
                self.model.kv_cache.reset()
                self.model.kv_cache.batch_size = None
            # 生成随机输入，使用模型的数据类型
            input_ids = torch.randint(0, 100, (batch_size, 10), device=self.device, dtype=torch.long)
            
            # 测量生成时间
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            start_time = time.time()
            
            with torch.no_grad():
                for _ in range(5):
                    # 模拟生成50个token
                    for i in range(50):
                        outputs = self.model(input_ids)
                        logits = outputs.logits if hasattr(outputs, 'logits') else outputs[1]
                        next_token = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
                        input_ids = torch.cat([input_ids, next_token], dim=1)
            
            torch.cuda.synchronize() if torch.cuda.is_available() else None
            end_time = time.time()
            
            # 计算生成速度
            total_new_tokens = 5 * 50 * batch_size
            total_time = end_time - start_time
            tokens_per_sec = total_new_tokens / total_time
            
            gen_results[batch_size] = {
                "tokens_per_sec": tokens_per_sec,
                "time_per_token_ms": (total_time / total_new_tokens) * 1000
            }
        
        # 存储结果
        self.perf_stats = {
            "forward_pass": results,
            "generation": gen_results,
            "device": str(self.device),
            "dtype": str(model_dtype),  # 记录使用的数据类型
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        self._analyzed_perf = True
        return self.perf_stats

    def generate_report(self, output_dir: Optional[str] = None, 
                       include_plots: bool = True) -> str:
        """生成详细的模型分析报告
        
        Args:
            output_dir: 输出目录，如果为None则使用当前目录
            include_plots: 是否包含可视化图表
            
        Returns:
            报告文件路径
        """
        # 确保所有分析都已完成
        self.analyze_parameters()
        self.analyze_layers()
        self.analyze_memory()
        self.analyze_compute()
        
        # 创建输出目录
        if output_dir is None:
            output_dir = os.path.join(os.getcwd(), "model_reports")
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成报告文件名
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(output_dir, f"model_report_{timestamp}.md")
        
        # 开始生成报告
        with open(report_file, "w", encoding="utf-8") as f:
            # 1. 报告标题
            f.write(f"# ByteTransformer 模型分析报告\n\n")
            f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # 2. 模型概览
            f.write("## 1. 模型概览\n\n")
            f.write(f"- **模型类型**: ByteTransformer\n")
            f.write(f"- **词汇表大小**: {self.config.vocab_size:,}\n")
            f.write(f"- **模型维度**: {self.config.model_dim}\n")
            f.write(f"- **层数**: {self.config.num_layers}\n")
            f.write(f"- **注意力头数**: {self.config.num_attention_heads}\n")
            f.write(f"- **KV头数**: {self.config.num_kv_heads}\n")
            f.write(f"- **隐藏层维度**: {self.config.hidden_dim}\n")
            f.write(f"- **最大序列长度**: {self.config.max_seq_len}\n")
            f.write(f"- **总参数量**: {self.param_stats['total']:,}\n")
            f.write(f"- **非嵌入参数量**: {self.param_stats['non_embedding']:,}\n\n")
            
            # 3. 参数统计
            f.write("## 2. 参数统计\n\n")
            
            # 3.1 参数类型分布
            f.write("### 2.1 参数类型分布\n\n")
            param_type_data = []
            for ptype, count in self.param_stats["by_type"].items():
                param_type_data.append([ptype, count, f"{count/self.param_stats['total']*100:.2f}%"])
            
            f.write(tabulate(param_type_data, 
                            headers=["参数类型", "参数数量", "占比"], 
                            tablefmt="pipe") + "\n\n")
            
            # 3.2 层参数分布
            f.write("### 2.2 层参数分布\n\n")
            layer_param_data = []
            for layer_name, count in self.param_stats["by_layer"].items():
                layer_param_data.append([layer_name, count, f"{count/self.param_stats['total']*100:.2f}%"])
            
            f.write(tabulate(layer_param_data[:10], 
                            headers=["层名称", "参数数量", "占比"], 
                            tablefmt="pipe") + "\n\n")
            
            if len(layer_param_data) > 10:
                f.write(f"*注: 仅显示前10层，共{len(layer_param_data)}层*\n\n")
            
            # 4. 内存使用
            f.write("## 3. 内存使用\n\n")
            memory_data = [
                ["参数", f"{self.memory_stats['parameters']['megabytes']:.2f} MB"],
                ["缓冲区", f"{self.memory_stats['buffers']['megabytes']:.2f} MB"],
                ["激活值", f"{self.memory_stats['activations']['megabytes']:.2f} MB"],
                ["KV缓存", f"{self.memory_stats['kv_cache']['megabytes']:.2f} MB"],
                ["梯度", f"{self.memory_stats['gradients']['megabytes']:.2f} MB"],
                ["优化器状态", f"{self.memory_stats['optimizer']['megabytes']:.2f} MB"],
                ["总计", f"{self.memory_stats['total']['gigabytes']:.2f} GB"]
            ]
            
            f.write(tabulate(memory_data, 
                            headers=["内存类型", "大小"], 
                            tablefmt="pipe") + "\n\n")
            
            # 5. 计算复杂度
            f.write("## 4. 计算复杂度\n\n")
            
            # 5.1 总计算量
            f.write("### 4.1 总计算量\n\n")
            f.write(f"- **总浮点运算次数**: {self.compute_stats['total']['flops']:,.0f} FLOPS\n")
            f.write(f"- **总浮点运算次数(G)**: {self.compute_stats['total']['gflops']:.2f} GFLOPS\n")
            f.write(f"- **总浮点运算次数(T)**: {self.compute_stats['total']['tflops']:.4f} TFLOPS\n\n")
            
            # 5.2 计算分布
            f.write("### 4.2 计算分布\n\n")
            compute_data = [
                ["注意力层", f"{self.compute_stats['attention']['total']:,.0f}", 
                 f"{self.compute_stats['attention']['total']/self.compute_stats['total']['flops']*100:.2f}%"],
                ["MLP层", f"{self.compute_stats['mlp']['total']:,.0f}", 
                 f"{self.compute_stats['mlp']['total']/self.compute_stats['total']['flops']*100:.2f}%"],
                ["LM头", f"{self.compute_stats['lm_head']:,.0f}", 
                 f"{self.compute_stats['lm_head']/self.compute_stats['total']['flops']*100:.2f}%"]
            ]
            
            f.write(tabulate(compute_data, 
                            headers=["计算部分", "FLOPS", "占比"], 
                            tablefmt="pipe") + "\n\n")
            
            # 5.3 生成计算量
            f.write("### 4.3 生成计算量\n\n")
            f.write(f"- **每个token的浮点运算次数**: {self.compute_stats['generation']['flops_per_token']:,.0f} FLOPS\n")
            f.write(f"- **每秒浮点运算次数(估计)**: {self.compute_stats['generation']['flops_per_sec']:,.0f} FLOPS\n")
            f.write(f"- **每秒浮点运算次数(G)(估计)**: {self.compute_stats['generation']['gflops_per_sec']:.2f} GFLOPS\n\n")
            
            # 6. 性能分析（如果已执行）
            if self._analyzed_perf:
                f.write("## 5. 性能分析\n\n")
                f.write(f"设备: {self.perf_stats['device']}\n\n")
                
                # 6.1 前向传播性能
                f.write("### 5.1 前向传播性能\n\n")
                for batch_size, seq_results in self.perf_stats["forward_pass"].items():
                    f.write(f"#### 批次大小 = {batch_size}\n\n")
                    perf_data = []
                    for seq_len, metrics in seq_results.items():
                        if "error" in metrics:
                            perf_data.append([seq_len, metrics["error"], "-"])
                        else:
                            perf_data.append([
                                seq_len, 
                                f"{metrics['latency_ms']:.2f} ms", 
                                f"{metrics['throughput_tokens_per_sec']:,.0f} tokens/s"
                            ])
                    
                    f.write(tabulate(perf_data, 
                                    headers=["序列长度", "延迟", "吞吐量"], 
                                    tablefmt="pipe") + "\n\n")
                
                # 6.2 生成性能
                f.write("### 5.2 生成性能\n\n")
                gen_data = []
                for batch_size, metrics in self.perf_stats["generation"].items():
                    gen_data.append([
                        batch_size,
                        f"{metrics['tokens_per_sec']:.2f} tokens/s",
                        f"{metrics['time_per_token_ms']:.2f} ms/token"
                    ])
                
                f.write(tabulate(gen_data, 
                                headers=["批次大小", "生成速度", "每token时间"], 
                                tablefmt="pipe") + "\n\n")
            
            # 7. 结论和建议
            f.write("## 6. 结论和建议\n\n")
            
            # 7.1 参数效率
            f.write("### 6.1 参数效率\n\n")
            embed_ratio = self.param_stats["by_type"]["embedding"] / self.param_stats["total"] * 100
            if embed_ratio > 30:
                f.write("- **嵌入层占比过高**: 嵌入层参数占总参数的比例为 {:.2f}%，考虑使用参数共享或低秩分解技术\n".format(embed_ratio))
            
            # 7.2 计算效率
            f.write("### 6.2 计算效率\n\n")
            attn_ratio = self.compute_stats["attention"]["total"] / self.compute_stats["total"]["flops"] * 100
            if attn_ratio > 60:
                f.write("- **注意力计算占比过高**: 注意力计算占总计算量的 {:.2f}%，考虑使用稀疏注意力或线性注意力机制\n".format(attn_ratio))
            
            # 7.3 内存优化
            f.write("### 6.3 内存优化\n\n")
            if self.memory_stats["kv_cache"]["megabytes"] > 1000:  # 超过1GB
                f.write("- **KV缓存内存占用过大**: KV缓存占用 {:.2f} MB，考虑使用KV缓存裁剪或量化技术\n".format(
                    self.memory_stats["kv_cache"]["megabytes"]))
            
            # 8. 附录：模型架构图
            f.write("## 7. 附录：模型架构\n\n")
            f.write("```\n")
            f.write(f"ByteTransformer(\n")
            f.write(f"  (embed_tokens): Embedding({self.config.vocab_size}, {self.config.model_dim})\n")
            
            # 简化的层结构
            for i in range(min(3, self.config.num_layers)):
                f.write(f"  (layers.{i}): DecoderLayer(\n")
                f.write(f"    (attn): MultiHeadSelfAttention(...)\n")
                f.write(f"    (mlp): MLP(...)\n")
                f.write(f"  )\n")
            
            if self.config.num_layers > 3:
                f.write(f"  ... {self.config.num_layers - 3} more layers ...\n")
            
            f.write(f"  (norm): LayerNorm({self.config.model_dim})\n")
            f.write(f"  (lm_head): Linear({self.config.model_dim}, {self.config.vocab_size})\n")
            f.write(f")\n")
            f.write("```\n\n")
            
            # 9. 生成可视化图表（如果启用）
            if include_plots:
                plots_dir = os.path.join(output_dir, "plots")
                os.makedirs(plots_dir, exist_ok=True)
                
                # 9.1 参数分布饼图
                param_plot_file = self._plot_parameter_distribution(plots_dir)
                if param_plot_file:
                    rel_path = os.path.relpath(param_plot_file, os.path.dirname(report_file))
                    f.write(f"## 8. 可视化图表\n\n")
                    f.write(f"### 8.1 参数分布\n\n")
                    f.write(f"![参数分布]({rel_path})\n\n")
                
                # 9.2 计算分布饼图
                compute_plot_file = self._plot_compute_distribution(plots_dir)
                if compute_plot_file:
                    rel_path = os.path.relpath(compute_plot_file, os.path.dirname(report_file))
                    f.write(f"### 8.2 计算分布\n\n")
                    f.write(f"![计算分布]({rel_path})\n\n")
                
                # 9.3 层参数分布条形图
                layer_plot_file = self._plot_layer_parameters(plots_dir)
                if layer_plot_file:
                    rel_path = os.path.relpath(layer_plot_file, os.path.dirname(report_file))
                    f.write(f"### 8.3 层参数分布\n\n")
                    f.write(f"![层参数分布]({rel_path})\n\n")
        
        print(f"报告已生成: {report_file}")
        return report_file
    
    def _plot_parameter_distribution(self, output_dir: str) -> Optional[str]:
        """绘制参数分布饼图
        
        Args:
            output_dir: 输出目录
            
        Returns:
            图表文件路径，如果失败则返回None
        """
        try:
            # 准备数据
            labels = []
            sizes = []
            for ptype, count in self.param_stats["by_type"].items():
                if count > 0:  # 只包含非零值
                    labels.append(ptype)
                    sizes.append(count)
            
            # 创建饼图
            plt.figure(figsize=(10, 8))
            plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, shadow=True)
            plt.axis('equal')  # 保持饼图为圆形
            plt.title('模型参数分布')
            
            # 保存图表
            plot_file = os.path.join(output_dir, "parameter_distribution.png")
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            return plot_file
        except Exception as e:
            print(f"绘制参数分布图表失败: {e}")
            return None
    
    def _plot_compute_distribution(self, output_dir: str) -> Optional[str]:
        """绘制计算分布饼图
        
        Args:
            output_dir: 输出目录
            
        Returns:
            图表文件路径，如果失败则返回None
        """
        try:
            # 准备数据
            labels = ['注意力层', 'MLP层', 'LM头']
            sizes = [
                self.compute_stats['attention']['total'],
                self.compute_stats['mlp']['total'],
                self.compute_stats['lm_head']
            ]
            
            # 创建饼图
            plt.figure(figsize=(10, 8))
            plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, shadow=True)
            plt.axis('equal')  # 保持饼图为圆形
            plt.title('模型计算分布')
            
            # 保存图表
            plot_file = os.path.join(output_dir, "compute_distribution.png")
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            return plot_file
        except Exception as e:
            print(f"绘制计算分布图表失败: {e}")
            return None
    
    def _plot_layer_parameters(self, output_dir: str) -> Optional[str]:
        """绘制层参数分布条形图
        
        Args:
            output_dir: 输出目录
            
        Returns:
            图表文件路径，如果失败则返回None
        """
        try:
            # 准备数据
            layer_names = []
            param_counts = []
            
            # 只取前15层，避免图表过于拥挤
            for layer_name, count in list(self.param_stats["by_layer"].items())[:15]:
                layer_names.append(layer_name)
                param_counts.append(count / 1_000_000)  # 转换为百万参数
            
            # 创建条形图
            plt.figure(figsize=(12, 8))
            sns.barplot(x=layer_names, y=param_counts)
            plt.title('层参数分布')
            plt.xlabel('层名称')
            plt.ylabel('参数数量 (百万)')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # 保存图表
            plot_file = os.path.join(output_dir, "layer_parameters.png")
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            return plot_file
        except Exception as e:
            print(f"绘制层参数分布图表失败: {e}")
            return None


def main():
    """命令行接口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="StellarByte模型分析工具")
    parser.add_argument("--model_path", type=str, default= "",help="模型路径或名称")
    parser.add_argument("--config_path", type=str, default="", help="配置文件路径")
    parser.add_argument("--output_dir", type=str, default="D:/Objects/StellarByte/model_info", help="报告输出目录")
    parser.add_argument("--generate_plots", type=bool, default=True, help="是否生成可视化图表")
    parser.add_argument("--analyze_perf", type=bool, default=True , help="分析性能（可能需要较长时间）")
    
    args = parser.parse_args()
    
    # 加载配置
    if args.config_path:
        config = ByteModelConfig.from_json(args.config_path)
    else:
        print("未提供配置文件路径，使用默认配置")
        config = ByteModelConfig()
    
    # 创建模型信息分析器
    if args.model_path:
        model_info = ModelInfo(args.model_path, config)
    else:
        print("未提供模型路径，使用随机初始化的模型")
        model = ByteTransformer(config)
        model_info = ModelInfo(model)
    
    # 分析参数
    print("分析模型参数...")
    model_info.analyze_parameters()
    
    # 分析层级结构
    print("分析模型层级结构...")
    model_info.analyze_layers()
    
    # 分析内存使用
    print("分析模型内存使用...")
    model_info.analyze_memory()
    
    # 分析计算复杂度
    print("分析模型计算复杂度...")
    model_info.analyze_compute()
    
    # 分析性能（可选）
    if args.analyze_perf:
        print("分析模型性能（可能需要较长时间）...")
        model_info.analyze_performance()
    
    # 生成报告
    print("生成分析报告...")
    report_file = model_info.generate_report(
        output_dir=args.output_dir,
        include_plots=args.generate_plots
    )
    
    print(f"分析完成！报告已保存至: {report_file}")


if __name__ == "__main__":
    main()