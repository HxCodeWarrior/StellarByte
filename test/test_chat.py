import sys
import torch
from transformers import AutoTokenizer
from pathlib import Path

# 将项目根目录添加到sys.path
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from model.Model import ByteModel, ByteModelConfig

# 1. 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 2. 加载tokenizer
tokenizer_path = "./tokenizer"  # 替换为你的tokenizer目录路径
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# 3. 创建模型配置（需要与训练时保持一致）
config = ByteModelConfig(
    vocab_size=32768,
    # 以下是示例配置，你需要根据实际训练配置调整这些参数
    model_dim=768,
    num_layers=12,
    num_attention_heads=16,
    num_kv_heads=8,
    ffn_hidden_dim=3072,
    max_seq_len=2048,
    hidden_dropout_prob=0.0,  # 推理时设为0
    attention_dropout_prob=0.0,
    residual_dropout_prob=0.0,
    layer_norm_eps=1e-5,
    base_theta=10000.0,
    ntk_alpha=1.0,
    initializer_range=0.02,
    moe_loss_coefficient=0.1,
    tensor_parallel_size=1  # 单卡推理设为1
)

# 4. 初始化模型
model = ByteModel(config)
model.to(device)

# 5. 加载预训练权重
weight_path = "./checkpoints/StellarByte_v1_380M.pth"  # 替换为你的权重文件路径
checkpoint = torch.load(weight_path, map_location=device, weights_only=True)

# 从检查点中提取模型状态字典
if 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
else:
    # 如果检查点直接是模型状态字典
    state_dict = checkpoint

# 处理可能的权重键名不匹配（如多GPU训练保存的权重）
if any(k.startswith('module.') for k in state_dict.keys()):
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

model.load_state_dict(state_dict)
model.eval()  # 设置为评估模式

print("模型加载完成!")

# 6. 对话测试函数
def chat_with_model(prompt, max_length=200, temperature=0.7, top_p=0.9):
    # 格式化输入
    input_text = f"{tokenizer.bos_token}{prompt}{tokenizer.eos_token}"
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    
    # 生成回复
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_seq_len=input_ids.shape[1] + max_length,
            temperature=temperature,
            top_p=top_p,
            top_k=50,
            repetition_penalty=1.2,
            repetition_context=64,
            eos_token_id=tokenizer.eos_token_id
        )
    
    # 提取生成的回复（去掉输入部分）
    response_ids = output_ids[0, input_ids.shape[1]:]
    response = tokenizer.decode(response_ids, skip_special_tokens=True)
    
    return response

# 7. 开始对话
print("开始对话测试! 输入'退出'结束对话")
while True:
    user_input = input("你: ")
    if user_input.lower() in ['退出', 'exit', 'quit']:
        break
        
    response = chat_with_model(user_input)
    print(f"AI: {response}")
    print("-" * 50)