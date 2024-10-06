import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import pandas as pd
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from trl import SFTTrainer
from transformers import TrainingArguments

# 设置命令行参数解析
def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen LLM with LoRA")
    parser.add_argument("--model_name", type=str, required=True, help="Path to the model directory")
    parser.add_argument("--max_seq_length", type=int, default=1000, help="Maximum sequence length for training")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save the output model and logs")
    parser.add_argument("--if_resume_from_checkpoint", action="store_true", help="Whether to resume from a checkpoint")
    parser.add_argument("--checkpoint_dir", type=str, default=None, help="Directory of the checkpoint to resume from")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 加载模型和标记器
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype="auto",
        device_map="auto",
        quantization_config=quantization_config
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    print(model)

    # 数据处理
    max_seq_length = args.max_seq_length
    EOS_TOKEN = tokenizer.eos_token

    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {}
    ### Input:
    {}
    ### Response:
    {}"""

    def process_func(example):
        instructions = example["instruction"]
        inputs = example["input"]
        outputs = example["output"]
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            # 必须添加EOS_TOKEN，否则无限生成
            text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
            texts.append(text)
        return {"text": texts}

    df = pd.read_json('./h-corpus_combined.json')
    # 删除 instruction 没有背景和情节的行
    df = df[df['instruction'] != '请按照以下描述续写小说：']
    ds = Dataset.from_pandas(df)

    tokenized_id = ds.map(process_func, batched=True)

    # 开启梯度检查点时，要执行该方法
    model.enable_input_require_grads()

    # LoRA 配置
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        inference_mode=False,  # 训练模式
        r=8,  # Lora 秩
        lora_alpha=32,  # Lora alpha
        lora_dropout=0.1  # Dropout 比例
    )

    model = get_peft_model(model, lora_config)

    # 设置是否从检查点恢复训练
    resume_from_checkpoint = args.checkpoint_dir if args.if_resume_from_checkpoint and args.checkpoint_dir else None

    # SFT Trainer 配置
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=tokenized_id,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        dataset_num_proc=2,
        packing=False,  # 可以让短序列的训练速度提高5倍。
        args=TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            learning_rate=2e-4,  # 学习率
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=args.output_dir,
            resume_from_checkpoint=resume_from_checkpoint,
        ),
    )

    trainer_stats = trainer.train()
    model.save_pretrained(f"{args.output_dir}/lora_model")

if __name__ == "__main__":
    main()
