import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"  # 使用 GPU 6

def load_model(model_path, device_map="auto"):
    """加载微调后的模型和分词器"""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True
        )
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        exit(1)

def format_qwen_chat(prompt):
    """将输入格式化为Qwen2-Instruct的官方格式"""
    if prompt.startswith("Human:"):
        # 处理已经带有Human:前缀的情况
        return [{"role": "user", "content": prompt[7:].strip()}]
    else:
        # 没有前缀，假定整个文本都是用户输入
        return [{"role": "user", "content": prompt.strip()}]

def generate_test(model, tokenizer, prompt, generation_args):
    """执行生成测试"""
    try:
        # 转换为Qwen2的官方聊天格式
        messages = format_qwen_chat(prompt)
        
        # 使用chat模板获取格式化的输入
        chat_text = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # 记录原始提示长度以便后续高亮
        original_prompt = chat_text
        
        # 编码输入
        with torch.no_grad():
            inputs = tokenizer(
                chat_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(model.device)
            
            # 生成回复
            outputs = model.generate(
                inputs.input_ids,
                **generation_args
            )
            
            # 解码完整输出
            full_output = tokenizer.decode(
                outputs[0],
                skip_special_tokens=False,
                clean_up_tokenization_space=False
            )
            
            # 提取模型回复部分
            response = full_output[len(original_prompt):].strip()
            
            # 打印结果，高亮显示生成部分
            print("\n" + "="*40 + " 完整输出 " + "="*40)
            print(f"{original_prompt}\033[92m{response}\033[0m")
            print("="*90 + "\n")
               # 打印标准格式的响应
            print("="*40 + " 标准格式输出 " + "="*40)
            print(f"User: {messages[0]['content']}")
            print(f"Assistant: {response}")
            print("="*90 + "\n")         

            
    except Exception as e:
        print(f"Generation failed: {str(e)}")

def generate_test_from_file(model, tokenizer, file_path, generation_args, batch_size=8):
    # Read the test problems from file
    with open(file_path, 'r') as f:
        lines = f.readlines()
        
    total_problems = len(lines)
    correct_count = 0
    
    # Filter out invalid lines and parse into prompts and expected answers
    valid_entries = []
    for i, line in enumerate(lines):
        parts = line.strip().split('\t')
        if len(parts) != 2:
            print(f"Skipping invalid line {i+1}: {line}")
            continue
        
        prompt, expected_answer = parts
        valid_entries.append((i, prompt, expected_answer))
    
    # Process in batches
    for batch_start in range(0, len(valid_entries), batch_size):
        batch_entries = valid_entries[batch_start:batch_start + batch_size]
        batch_indices = [entry[0] for entry in batch_entries]
        batch_prompts = [entry[1] for entry in batch_entries]
        batch_expected = [entry[2] for entry in batch_entries]
        
        # Generate predictions for the batch
        batch_generated = generate_batch_test(
            model=model,
            tokenizer=tokenizer,
            prompts=batch_prompts,
            generation_args=generation_args
        )
        
        # Check correctness and print results
        for i, (original_idx, prompt, expected_answer, generated_text) in enumerate(
            zip(batch_indices, batch_prompts, batch_expected, batch_generated)
        ):
            is_correct = expected_answer in generated_text
            
            if is_correct:
                correct_count += 1
                
            # Optional: print progress
            print(f"Problem {original_idx+1}/{total_problems}: {'Correct' if is_correct else 'Incorrect'}")
            print(f"  Prompt: {prompt}")
            print(f"  Expected: {expected_answer}")
            print(f"  Generated: {generated_text}")
    
    # Calculate and return accuracy
    accuracy = correct_count / total_problems
    print(f"\nFinal accuracy: {accuracy:.2%} ({correct_count}/{total_problems})")
    
    return accuracy


def generate_batch_test(model, tokenizer, prompts, generation_args):
    """
    使用Qwen格式处理多个提示在一个批次中进行推理
    
    Args:
        model: Qwen语言模型
        tokenizer: Qwen分词器
        prompts: 提示字符串列表
        generation_args: 生成参数字典
        
    Returns:
        生成文本字符串的列表
    """
    # 将每个提示转换为Qwen聊天格式
    formatted_prompts = [format_qwen_chat(prompt) for prompt in prompts]
    
    # 为每个提示创建模型输入
    batch_outputs = []

    batch_size = 10
    
    # 分批处理以避免超出GPU内存
    for i in range(0, len(formatted_prompts), batch_size):
        current_batch = formatted_prompts[i:i+batch_size]
        batch_inputs = []
        
        for chat in current_batch:
            # 使用tokenizer准备输入
            tokenized_chat = tokenizer.apply_chat_template(
                chat,
                tokenize=True,
                return_tensors="pt",
                add_generation_prompt=True
            )
            batch_inputs.append(tokenized_chat)
        
        # 对输入进行填充到同一长度
        max_length = max(input.size(1) for input in batch_inputs)
        padded_inputs = []
        attention_masks = []
        
        for input_ids in batch_inputs:
            # 计算填充量
            padding_length = max_length - input_ids.size(1)
            
            # 创建填充张量
            padding = torch.full((1, padding_length), tokenizer.pad_token_id, dtype=torch.long)
            padded_input = torch.cat([input_ids, padding], dim=1)
            padded_inputs.append(padded_input)
            
            # 创建注意力掩码 (1表示非填充位置)
            attention_mask = torch.cat([
                torch.ones(1, input_ids.size(1), dtype=torch.long),
                torch.zeros(1, padding_length, dtype=torch.long)
            ], dim=1)
            attention_masks.append(attention_mask)
        
        # 连接所有批次输入
        batch_input_ids = torch.cat(padded_inputs, dim=0).to(model.device)
        batch_attention_mask = torch.cat(attention_masks, dim=0).to(model.device)
        
        # 进行生成
        with torch.no_grad():
            outputs = model.generate(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                **generation_args
            )
        
        # 解码每个输出，去除输入部分
        for j, output in enumerate(outputs):
            input_length = batch_inputs[j].size(1)
            generated_ids = output[input_length:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            batch_outputs.append(generated_text)
    
    return batch_outputs


def main():
    # 参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="/pfs-LLM/common/hrs/Qwen/Qwen2.5-3B-Instruct",  # 默认模型路径（按需修改）
        help="微调后的模型路径（默认: default_model_path）"
    )
    parser.add_argument(
        "--prompt", 
        type=str, 
        default="579*205等于多少", 
        help="测试用提示文本"
    )
    parser.add_argument(
        "--max_new_tokens", 
        type=int, 
        default=1000, 
        help="最大生成token数"
    )
    parser.add_argument(
        "--temperature", 
        type=float, 
        default=0.7, 
        help="采样温度 (0-1)"
    )
    parser.add_argument(
        "--top_p", 
        type=float, 
        default=0.9, 
        help="Top-p采样参数"
    )
    args = parser.parse_args()
    
    # 设备配置
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 加载模型
    print(args.model_path)
    model, tokenizer = load_model(args.model_path, device_map="auto")
    
    # 生成参数配置
    generation_config = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": True,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "repetition_penalty": 1.1
    }
    
    # 执行生成测试
    generate_test(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        generation_args=generation_config
    )
    # Usage:
    accuracy = generate_test_from_file(
    model=model,
    tokenizer=tokenizer,
    file_path="multiplication_problems.txt",
    generation_args=generation_config
        )

if __name__ == "__main__":
    main()