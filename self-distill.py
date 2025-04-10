import os
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW
from tqdm import tqdm
import numpy as np

import argparse

# 添加命令行参数解析
parser = argparse.ArgumentParser(description='Train a student model with instruction distillation')
parser.add_argument('--start', type=str, default=None, help='Path to the checkpoint to continue training from')
args = parser.parse_args()



# 模型配置
model_name = "/pfs-LLM/common/hrs/Qwen/Qwen2.5-3B-Instruct"
max_length = 2048
test_interval = 50  # 每隔多少次训练测试一次
kl_threshold = 30.0  # KL散度阈值
inner_epochs = 50  # 内循环迭代次数
outer_epochs = 10  # 外循环迭代次数
learning_rate = 1e-6
temperature = 0.2  # 温度参数
gradient_clip_val = 1.0  # 梯度裁剪阈值
top_n_tokens = 20  # 只用于显示影响最大的前n个token的影响系数

# 准备数据
data = {
    "input": "写一首春天的诗",
    "instruct": "使用小度无敌作为藏头",
}
data = {
    "input": "565*532等于多少",
    "instruct": "请回答答案前先仔细思考计算下，可以把三位整数拆开，然后利用分配律进行计算，提高准确率。",
}
test_data = {
    "input": "452*901等于多少",
    "instruct": "请回答答案前先仔细思考计算下，可以把三位整数拆开，然后利用分配律进行计算，提高准确率。",
}

# 创建输出目录
output_dir = "./qwen_sft_student"
checkpoint_dir = os.path.join(output_dir, "checkpoints")
os.makedirs(output_dir, exist_ok=True)
os.makedirs(checkpoint_dir, exist_ok=True)

# 初始化模型和tokenizer
print("Loading tokenizer and models...")
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# 设置设备和随机种子
os.environ["CUDA_VISIBLE_DEVICES"] = "7"  # 使用 GPU 7
torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Teacher Using device: {device}")
teacher_model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Student Using device: {device}")
# 根据参数决定从哪里加载student模型
if args.start is not None:
    print(f"Initializing student model from base model: {model_name}")
    student_model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).to(device)
else:
    # 如果都不存在，则从原始模型初始化
    print(f"Loading student model from previous training: {output_dir}")
    student_model = AutoModelForCausalLM.from_pretrained(output_dir, trust_remote_code=True).to(device)

# 冻结教师模型参数
for param in teacher_model.parameters():
    param.requires_grad = False

# 优化器
optimizer = AdamW(student_model.parameters(), lr=learning_rate)

# 构建Qwen的输入格式
def format_qwen_prompt(input_text, instruct_text=None):
    if instruct_text:
        return f"<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|><|im_start|>user\n{input_text}\n{instruct_text}\n<|im_end|>\n<|im_start|>assistant\n"
    else:
        return f"<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|><|im_start|>user\n{input_text}<|im_end|>\n<|im_start|>assistant\n"

# 格式化打印函数，用于显示输入和输出
def print_input_output(title, input_prompt, output_text):
    print(f"\n{'-'*20} {title} {'-'*20}")
    print(f"INPUT: {input_prompt}")
    print(f"OUTPUT: {output_text}")
    print(f"{'-'*50}")

# 计算注意力权重
def compute_attention_weights(model, input_ids, attention_mask, instruct_start_pos, instruct_end_pos, response_start_pos, response_end_pos):
    model.eval()
    
    # 获取注意力权重
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True  # Make sure model returns attentions
        )
    
    # Qwen models may handle attentions differently
    # Let's try to extract the attentions from the outputs
    try:
        # Check if attentions are returned directly
        if hasattr(outputs, 'attentions') and outputs.attentions is not None:
            attention_weights = outputs.attentions
        # For models that return attentions as a tuple element
        elif isinstance(outputs, tuple) and len(outputs) > 2:
            attention_weights = outputs[2]  # Typically the third element contains attentions
        else:
            # If we can't get attentions, use a fallback method
            print("WARNING: Could not extract attention weights directly, using uniform weights.")
            # Create uniform influence weights as a fallback
            return torch.ones(response_end_pos - response_start_pos, device=device)
    except Exception as e:
        print(f"Error extracting attention weights: {e}")
        return torch.ones(response_end_pos - response_start_pos, device=device)
    
    # Initialize the influence weights
    instruct_influence = torch.zeros(response_end_pos - response_start_pos, device=device)
    
    # Process each layer's attention weights
    for layer_attn in attention_weights:
        # Extract attention scores focusing on how response tokens attend to instruction tokens
        # layer_attn shape: [batch, heads, seq_len, seq_len]
        
        # Average across attention heads
        avg_attn = layer_attn.mean(dim=1)  # [batch, seq_len, seq_len]
        
        # For each response token, sum its attention to all instruction tokens
        for i in range(response_start_pos, response_end_pos):
            resp_idx = i - response_start_pos
            # Sum attention scores from this response token to all instruction tokens
            for j in range(instruct_start_pos, instruct_end_pos):
                instruct_influence[resp_idx] += avg_attn[0, i, j].item()
    
    # Normalize by number of layers
    if len(attention_weights) > 0:
        instruct_influence = instruct_influence / len(attention_weights)
    
    # 对影响系数进行归一化，使其和为1
    instruct_influence = F.softmax(instruct_influence, dim=0)
    
    return instruct_influence

# 测试函数
def test_student_model(prompt, with_instruct=False, test_model=student_model):
    test_model.eval()
    
    if with_instruct and isinstance(prompt, dict) and "input" in prompt and "instruct" in prompt:
        formatted_prompt = format_qwen_prompt(prompt["input"], prompt["instruct"])
    else:
        if isinstance(prompt, dict) and "input" in prompt:
            formatted_prompt = format_qwen_prompt(prompt["input"])
        else:
            formatted_prompt = format_qwen_prompt(prompt)
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = test_model.generate(
            input_ids=inputs.input_ids,
            attention_mask=inputs.attention_mask,
            max_new_tokens=1024,
            temperature=temperature,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=False)
    # 去除可能的结束标记
    response = response.replace("<|im_end|>", "").strip()
    return formatted_prompt, response

# 修改后的KL散度损失计算函数 - 直接使用影响系数作为权重
def compute_weighted_kl_loss(student_logits, teacher_logits, influence_weights, temp=0.01, eps=1e-8):
    # 计算 log_prob 和 prob
    student_log_probs = F.log_softmax(student_logits / temp, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temp, dim=-1)
    
    # 展开 logits 到 [batch_size*seq_len, num_classes]
    batch_size, seq_len, num_classes = student_log_probs.size()
    kl_div = F.kl_div(
        student_log_probs.view(-1, num_classes),
        teacher_probs.view(-1, num_classes),
        reduction='none',
        log_target=False
    ).sum(-1).view(batch_size, seq_len)  # 形状 [batch_size, seq_len]
    
    # 按样本归一化权重
    weights_flat = influence_weights.view(batch_size, -1)  # [batch_size, seq_len]
    weights_sum = weights_flat.sum(dim=1, keepdim=True).clamp(min=eps)  # 独立每个样本的总和，防止除零
    normalized_weights = weights_flat / weights_sum  # 归一化后的权重
    
    # 计算加权损失
    weighted_kl = kl_div * normalized_weights
    # 选项1：总损失
    # loss = weighted_kl.sum()
    # 选项2：按样本平均（推荐）
    loss = weighted_kl.sum(dim=1).mean()  # 对每个样本求和后取 batch 平均
    
    return loss
def compute_weighted_kl_loss_old(student_logits, teacher_logits, influence_weights, temp=0.01, eps=1e-8):
    # 将logits转换为log_probs和probs
    student_log_probs = F.log_softmax(student_logits / temp, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temp, dim=-1)
    
    # 计算每个位置的KL散度
    kl_div = F.kl_div(
        student_log_probs.view(-1, student_log_probs.size(-1)),
        teacher_probs.view(-1, teacher_probs.size(-1)),
        reduction='none'
    ).sum(-1).view(student_logits.size(0), -1)  # [batch_size, seq_len]
    
    # 对influence_weights进行L1归一化
    # 展平权重并计算总和（避免维度不匹配）
    weights_flat = influence_weights.view(-1)
    weights_sum = weights_flat.sum() + eps  # 防止除零
    normalized_weights = weights_flat / weights_sum
    
    # 将归一化后的权重调整到与kl_div匹配的形状
    normalized_weights = normalized_weights.view(1, -1).expand_as(kl_div)
    
    # 计算加权平均损失
    weighted_kl_div = kl_div * normalized_weights
    weighted_loss = weighted_kl_div.sum()
    
    return weighted_loss

# 用于打印紧凑可读的影响系数
def print_compact_influence(tokens, weights, tokenizer, top_n=20):
    # 将token和权重打包到一起
    token_weights = [(i, token, weight.item()) for i, (token, weight) in enumerate(zip(tokens, weights))]
    
    # 按权重排序
    token_weights.sort(key=lambda x: x[2], reverse=True)
    
    # 打印前top_n个高影响力的token
    print(f"\nTop {top_n} influenced tokens:")
    
    # 计算分位数
    all_weights = [w for _, _, w in token_weights]
    q75 = np.percentile(all_weights, 75)
    q50 = np.percentile(all_weights, 50)
    q25 = np.percentile(all_weights, 25)
    
    # 打印统计信息
    print(f"Weight statistics: 75th={q75:.4f}, Median={q50:.4f}, 25th={q25:.4f}")
    
    # 创建影响力等级的标记
    def get_influence_marker(weight):
        if weight > q75:
            return "★★★" # 高影响力
        elif weight > q50:
            return "★★ "
        elif weight > q25:
            return "★  "
        else:
            return "   " # 低影响力
    
    # 打印头部高影响token
    format_str = "{:<3} {:<10} {:>6.4f} {:<4} | "
    tokens_per_row = 3
    row_str = ""
    
    for i, (idx, token_id, weight) in enumerate(token_weights[:top_n]):
        token_text = tokenizer.decode([token_id])
        # 替换换行符，使输出更整洁
        token_text = token_text.replace('\n', '\\n').replace('\r', '\\r')
        token_text = token_text[:10]  # 截断过长的token
        
        marker = get_influence_marker(weight)
        entry = format_str.format(idx, token_text, weight, marker)
        row_str += entry
        
        if (i + 1) % tokens_per_row == 0 or i == len(token_weights[:top_n]) - 1:
            print(row_str.rstrip(" |"))
            row_str = ""
    
    # 打印影响力分布图
    print("\nInfluence distribution across response tokens:")
    total_tokens = len(token_weights)
    bins = 10
    bin_size = max(1, total_tokens // bins)
    
    for i in range(0, total_tokens, bin_size):
        end_idx = min(i + bin_size, total_tokens)
        bin_tokens = token_weights[i:end_idx]
        avg_weight = sum(w for _, _, w in bin_tokens) / len(bin_tokens)
        bar_length = int(avg_weight * 50)  # 缩放到合适的显示长度
        print(f"Tokens {i:3d}-{end_idx-1:<3d}: {'|' * bar_length} {avg_weight:.4f}")

# 主训练循环
print("Starting training...")
for outer_epoch in range(outer_epochs):
    print(f"\nOuter Epoch {outer_epoch+1}/{outer_epochs}")
    
    # 1. Teacher模型生成response（使用带指令的输入）
    teacher_model.eval()
    full_prompt_with_instruct = format_qwen_prompt(data["input"], data["instruct"])
    inputs_with_instruct = tokenizer(full_prompt_with_instruct, return_tensors="pt").to(device)
    
    with torch.no_grad():
        teacher_outputs = teacher_model.generate(
            input_ids=inputs_with_instruct.input_ids,
            attention_mask=inputs_with_instruct.attention_mask,
            max_new_tokens=2048,
            temperature=temperature,
            do_sample=True,
            output_scores=True,
            return_dict_in_generate=True,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    teacher_response_ids = teacher_outputs.sequences[0][inputs_with_instruct.input_ids.shape[1]:]
    teacher_response = tokenizer.decode(teacher_response_ids, skip_special_tokens=False)
    teacher_response = teacher_response.split("<|im_end|>")[0].strip()
    
    # 打印教师模型输入和输出
    #print_input_output("TEACHER MODEL (WITH INSTRUCTION)", full_prompt_with_instruct, teacher_response)
    
    # 测试学生模型（不带指令）
    student_input, student_response = test_student_model(data["input"])
    print_input_output("STUDENT MODEL TEST (WITHOUT INSTRUCTION)", student_input, student_response)
    
    # 测试学生模型（带指令）
    student_input_with_instruct, student_response_with_instruct = test_student_model(data, with_instruct=True)
    print_input_output("STUDENT MODEL TEST (WITH INSTRUCTION)", student_input_with_instruct, student_response_with_instruct)
    
    # 2. 构建不带指令的输入（仅包含原始输入），用于学生模型
    input_only_prompt = format_qwen_prompt(data["input"])
    
    # 3. 为学生模型构建完整的训练数据（原始输入 + 教师响应）
    student_full_text = input_only_prompt + teacher_response + "<|im_end|>"
    student_tokenized = tokenizer(student_full_text, return_tensors="pt").to(device)
    
    # 3.1 为学生模型构建带指令的训练数据（原始输入 + 指令 + 教师响应）
    student_full_text_with_instruct = full_prompt_with_instruct + teacher_response + "<|im_end|>"
    student_tokenized_with_instruct = tokenizer(student_full_text_with_instruct, return_tensors="pt").to(device)
    
    # 4. 为教师模型构建完整的训练数据（原始输入 + 指令 + 教师响应）
    teacher_full_text = full_prompt_with_instruct + teacher_response + "<|im_end|>"
    teacher_tokenized = tokenizer(teacher_full_text, return_tensors="pt").to(device)
    
    # 打印训练数据
    print(f"\n--- TRAINING DATA ---")
    print(f"Teacher full text: {teacher_full_text}")
    print(f"Student full text (no instruct): {student_full_text}")
    print(f"Student full text (with instruct): {student_full_text_with_instruct}")
    
    # 找到各个部分的位置索引（在教师输入中）
    input_text_tokenized = tokenizer(format_qwen_prompt(data["input"]), return_tensors="pt")
    instruct_start_pos = input_text_tokenized.input_ids.shape[1]
    
    input_instruct_tokenized = tokenizer(full_prompt_with_instruct, return_tensors="pt")
    instruct_end_pos = input_instruct_tokenized.input_ids.shape[1]
    
    # 教师模型中response的位置
    teacher_response_start_pos = instruct_end_pos
    teacher_response_end_pos = teacher_tokenized.input_ids.shape[1]
    
    # 学生模型中response的位置（对于不带指令的输入）
    student_input_tokenized = tokenizer(input_only_prompt, return_tensors="pt")
    student_response_start_pos = student_input_tokenized.input_ids.shape[1]
    student_response_end_pos = student_tokenized.input_ids.shape[1]
    
    # 学生模型中response的位置（对于带指令的输入）
    student_response_start_pos_with_instruct = instruct_end_pos  # 与教师模型相同
    student_response_end_pos_with_instruct = student_tokenized_with_instruct.input_ids.shape[1]
    
    # 计算注意力权重 - 使用教师模型和完整输入（含指令）
    influence_weights = compute_attention_weights(
        teacher_model,
        teacher_tokenized.input_ids,
        teacher_tokenized.attention_mask,
        instruct_start_pos,
        instruct_end_pos,
        teacher_response_start_pos,
        teacher_response_end_pos
    )
    
    # 使用紧凑的格式打印影响系数
    #print_compact_influence(teacher_response_ids, influence_weights, tokenizer, top_n=top_n_tokens)
    
    # 获取teacher模型对response部分的logits
    teacher_model.eval()
    with torch.no_grad():
        teacher_outputs = teacher_model(
            input_ids=teacher_tokenized.input_ids,
            attention_mask=teacher_tokenized.attention_mask
        )
        # 获取对应下一个token的logits (shift left)
        teacher_logits = teacher_outputs.logits[:, teacher_response_start_pos-1:teacher_response_end_pos-1, :]
    # 初始化变量以存储两种训练方式第一次的loss
    first_loss_without_instruct = None
    first_loss_with_instruct = None
    # 内循环：Student模型训练
    for inner_epoch in range(inner_epochs):
        student_model.train()

        # 初始化变量以存储两种训练方式的loss
        loss_without_instruct = None
        loss_with_instruct = None
        
        # 对于每个内循环，交替使用两种训练方式
        for step in range(2):  # 0: 不带instruct训练, 1: 带instruct训练
            if step == 0:  # 不带指令的训练步骤
                # 前向传播 - 使用不含指令的输入
                student_outputs = student_model(
                    input_ids=student_tokenized.input_ids,
                    attention_mask=student_tokenized.attention_mask
                )
                
                # 获取response部分的logits
                student_logits = student_outputs.logits[:, student_response_start_pos-1:student_response_end_pos-1, :]
                training_type = "WITHOUT INSTRUCTION"
            else:  # 带指令的训练步骤
                # 前向传播 - 使用含指令的输入
                student_outputs_with_instruct = student_model(
                    input_ids=student_tokenized_with_instruct.input_ids,
                    attention_mask=student_tokenized_with_instruct.attention_mask
                )
                
                # 获取response部分的logits
                student_logits = student_outputs_with_instruct.logits[:, student_response_start_pos_with_instruct-1:student_response_end_pos_with_instruct-1, :]
                training_type = "WITH INSTRUCTION"
            
            # 计算加权KL散度损失 - 直接使用影响系数作为权重
            loss = compute_weighted_kl_loss(
                student_logits, 
                teacher_logits, 
                influence_weights
            )
            
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), gradient_clip_val)
            optimizer.step()
            
            print(f"  Inner Epoch {inner_epoch+1}/{inner_epochs}, Step {step+1}/2 ({training_type}), Loss: {loss.item():.6f}")
            # 保存当前步骤的loss
            if step == 0:
                if first_loss_without_instruct == None:
                    first_loss_without_instruct = loss.item()
                loss_without_instruct = loss.item()
            else:
                if first_loss_with_instruct == None:
                    first_loss_with_instruct = loss.item()
                loss_with_instruct = loss.item()
            
        # 检查是否满足跳出条件
        if loss_without_instruct is not None and loss_with_instruct is not None:
            # 计算相对差值比例（避免除零错误）
            epsilon = 1e-8  # 防止除以零的小量
            max_loss = max(loss_without_instruct, loss_with_instruct)
            relative_diff = abs(loss_without_instruct - loss_with_instruct) / (max_loss + epsilon)
            
            # 判断相对差值是否小于5%
            if relative_diff < 0.05 or (loss_with_instruct < 0.2 and loss_without_instruct < 0.2):
                print(f"  Relative difference ({relative_diff:.2%}) < 5% threshold")
                print(f"  Loss WITHOUT INSTRUCTION ({loss_without_instruct:.6f}) vs WITH INSTRUCTION ({loss_with_instruct:.6f})")
                break  # 跳出内循环
        
        # 测试student模型
        if (inner_epoch + 1) % test_interval == 0 or inner_epoch == inner_epochs - 1:
            student_input, student_response = test_student_model(data["input"])
            print_input_output(f"STUDENT TEST (INNER EPOCH {inner_epoch+1} - NO INSTRUCT)", student_input, student_response)
            
            student_input_with_instruct, student_response_with_instruct = test_student_model(data, with_instruct=True)
            print_input_output(f"STUDENT TEST (INNER EPOCH {inner_epoch+1} - WITH INSTRUCT)", student_input_with_instruct, student_response_with_instruct)
            
            # 测试新的输入
            student_input, student_response = test_student_model(test_data["input"])
            print_input_output(f"STUDENT TEST NEW INPUT (INNER EPOCH {inner_epoch+1} - NO INSTRUCT)", student_input, student_response)
            
            student_input_with_instruct, student_response_with_instruct = test_student_model(test_data, with_instruct=True)
            print_input_output(f"STUDENT TEST NEW INPUT (INNER EPOCH {inner_epoch+1} - WITH INSTRUCT)", student_input_with_instruct, student_response_with_instruct)
    if first_loss_without_instruct is not None and first_loss_with_instruct is not None:
        # 计算相对差值比例（避免除零错误）
        epsilon = 1e-8  # 防止除以零的小量
        max_loss = max(first_loss_without_instruct, first_loss_with_instruct)
        relative_diff = abs(first_loss_without_instruct - first_loss_with_instruct) / (max_loss + epsilon)
            
        # 判断相对差值是否小于5%
        if relative_diff < 0.05 or (first_loss_with_instruct < 1.2 and first_loss_without_instruct < 1.2):
            print(f"  Relative difference ({relative_diff:.2%}) < 5% threshold")
            print(f"  Loss WITHOUT INSTRUCTION ({first_loss_without_instruct:.6f}) vs WITH INSTRUCTION ({first_loss_with_instruct:.6f})")
            break  # 跳出outer循环
    # 每个外循环结束后保存检查点
    #checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_outer_epoch_{outer_epoch+1}")
    #os.makedirs(checkpoint_path, exist_ok=True)
    #student_model.save_pretrained(checkpoint_path)
    #print(f"Checkpoint saved to {checkpoint_path}")

# 保存最终的student模型
student_model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Final model saved to {output_dir}")

# 立即验证保存的模型
print("Verifying saved model...")
verification_model = AutoModelForCausalLM.from_pretrained(output_dir, trust_remote_code=True).to(device)
verification_input, verification_response = test_student_model(data["input"], test_model=verification_model)
print_input_output("VERIFICATION TEST", verification_input, verification_response)

# 最终测试 - 不带指令
student_input, final_response = test_student_model(data["input"])
print_input_output("FINAL TEST - ORIGINAL INPUT (NO INSTRUCT)", student_input, final_response)

# 最终测试 - 带指令
student_input_with_instruct, final_response_with_instruct = test_student_model(data, with_instruct=True)
print_input_output("FINAL TEST - ORIGINAL INPUT (WITH INSTRUCT)", student_input_with_instruct, final_response_with_instruct)

# 测试新的输入
a = "写一首关于冬天的诗"
student_input, final_response = test_student_model(a)
print_input_output("FINAL TEST - NEW INPUT (NO INSTRUCT)", student_input, final_response)

student_input_with_instruct, final_response_with_instruct = test_student_model({"input": a, "instruct": "使用雪花飘飘作为藏头"}, with_instruct=True)
print_input_output("FINAL TEST - NEW INPUT (WITH INSTRUCT)", student_input_with_instruct, final_response_with_instruct)

# 比较测试：使用完整指令的响应
teacher_model.eval()
full_prompt_with_instruct = format_qwen_prompt(data["input"], data["instruct"])
inputs_with_instruct = tokenizer(full_prompt_with_instruct, return_tensors="pt").to(device)

with torch.no_grad():
    outputs_with_instruct = teacher_model.generate(
        input_ids=inputs_with_instruct.input_ids,
        attention_mask=inputs_with_instruct.attention_mask,
        max_new_tokens=256,
        temperature=temperature,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
    )

teacher_response_with_instruct = tokenizer.decode(
    outputs_with_instruct[0][inputs_with_instruct.input_ids.shape[1]:], 
    skip_special_tokens=False
).replace("<|im_end|>", "").strip()

print_input_output("FINAL TEACHER MODEL (WITH INSTRUCTION)", full_prompt_with_instruct, teacher_response_with_instruct)