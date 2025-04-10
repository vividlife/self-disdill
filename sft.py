import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import SFTTrainer, SFTConfig

# 加载JSON数据集
def load_dataset_from_json(json_file_path):
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # 确保数据集有正确的格式
    formatted_data = []
    for item in data:
        if "question" in item and "answer" in item:
            formatted_data.append({
                "question": str(item["question"]),  # 确保是字符串
                "answer": str(item["answer"])       # 确保是字符串
            })
    return Dataset.from_list(formatted_data)

# 定义格式化函数 - 返回纯文本字符串，而不是字典
# 修改格式化函数 - 返回列表，而不是单个字符串
# 定义格式化函数处理单个样本
def formatting_func(example):
    return f"### Question: {example['question']}\n### Answer: {example['answer']}"



# 或者，定义用于批处理的格式化函数
def batch_formatting_func(examples):
    return [
        f"### Question: {item['question']}\n### Answer: {item['answer']}"
        for item in examples
    ]

# 加载数据集
json_file_path = "sft_data.json"
dataset = load_dataset_from_json(json_file_path)

# 打印数据集大小和示例，用于调试
print(f"Dataset size: {len(dataset)}")
if len(dataset) > 0:
    print(f"Sample entry: {dataset[0]}")
    print(f"Formatted sample: {formatting_func(dataset[0])}")

# 加载Qwen模型和分词器
model_name = "/pfs-LLM/common/hrs/Qwen/Qwen2.5-3B-Instruct"  # 替换为你的具体Qwen模型
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

# 确保tokenizer有正确的padding token
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 配置训练参数
training_args = SFTConfig(
    output_dir="./sft_qwen_model",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    max_seq_length=256,
    packing=False,
    gradient_checkpointing=True,
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    weight_decay=0.01,
    fp16=True,
    logging_steps=100,
    optim="adamw_torch",
    max_grad_norm=1.0,
    save_strategy="epoch"
)

# 初始化trainer - 注意这里的参数
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=training_args,
    formatting_func=batch_formatting_func,
)

# 开始训练
trainer.train()

# 保存最终模型
trainer.save_model("./sft_qwen_final")