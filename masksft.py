# 严格遵循Qwen指令格式的SFT代码
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

# 1. 加载模型和分词器（必须使用官方指定参数）
model_name = "/pfs-LLM/common/hrs/Qwen/Qwen2.5-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    trust_remote_code=True,
    pad_token='<|endoftext|>',  # 必须显式设置pad_token
    eos_token='<|endoftext|>',
    padding_side="left"
)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    use_cache=False  # 训练时需要关闭cache
)

# 2. 数据处理函数（严格遵循Qwen指令格式）
def qwen_format(example):
    """
    转换示例到Qwen标准指令格式：
    <|im_start|>system
    You are a helpful assistant.<|im_end|>
    <|im_start|>user
    {instruction}<|im_end|>
    <|im_start|>assistant
    {response}<|im_end|>
    """
    system_msg = "You are a helpful assistant."
    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["output"]}
    ]
    
    # 使用Qwen专用模板生成文本
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text": text}

# 3. 加载并处理数据集（示例数据应包含instruction和output字段）
dataset = load_dataset("json", data_files="sft_data.json")
dataset = dataset.map(qwen_format, remove_columns=dataset["train"].column_names)

# 4. 分词处理（使用Qwen专用设置）
def tokenize_function(examples):
    tokenized = tokenizer(
        examples["text"],
        truncation=True,
        max_length=1024,
        padding="max_length",
        return_tensors="pt",
        add_special_tokens=False  # 模板中已包含特殊token
    )
    
    # 创建labels（只计算assistant部分的loss）
    labels = tokenized["input_ids"].clone()
    # 找到所有<|im_end|>的位置作为mask边界
    sep_token_id = tokenizer.encode("<|im_end|>")[0]
    sep_positions = (labels == sep_token_id).nonzero(as_tuple=True)[1]
    
    # 将非assistant部分设为-100
    for i in range(labels.size(0)):
        # 第一个<|im_end|>之后是assistant内容
        first_sep = sep_positions[i*3 + 1]  # system(0), user(1), assistant(2)
        labels[i, :first_sep+1] = -100
        
    tokenized["labels"] = labels
    return tokenized

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# 5. 训练参数配置（Qwen推荐参数）
training_args = TrainingArguments(
    output_dir="./qwen_sft",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=1e-5,
    num_train_epochs=3,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    logging_steps=10,
    optim="adamw_torch",
    fp16=True,
    gradient_checkpointing=True,  # 必须开启梯度检查点
    report_to="none",
    save_strategy="epoch",
    remove_unused_columns=False
)

# 6. 创建训练器
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
)

# 7. 开始训练
trainer.train()

# 8. 保存适配Qwen格式的模型
model.save_pretrained("./qwen_sft_final")
tokenizer.save_pretrained("./qwen_sft_final")