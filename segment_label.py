import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import os
import json
import re
import time
import logging

# 定义数据预处理函数
def preprocess_data(corpus_path, output_dir, n=500, batch_size=100): 
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_json_path = os.path.join(output_dir, "corpus_processed.json")
    processed_files_log = os.path.join(output_dir, "processed_files.log")
    log_file_path = os.path.join(output_dir, 'process_log.log')

    data = []
    file_count = 0
    max_files = 205027

    # 设置日志配置
    logging.basicConfig(
        filename=log_file_path,  # 日志文件名
        level=logging.INFO,       # 日志级别
        format='%(asctime)s - %(message)s',  # 日志格式
        filemode='a'              # 追加模式
    )

    # 检查是否有记录已处理文件的日志文件
    if os.path.exists(processed_files_log):
        with open(processed_files_log, 'r') as log_file:
            processed_files = set(log_file.read().splitlines())
    else:
        processed_files = set()

    start_time = time.time()
    # 遍历语料库路径下的所有txt文件
    for filename in os.listdir(corpus_path):
        if filename.endswith(".txt") and filename not in processed_files:
            file_path = os.path.join(corpus_path, filename)

            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()

            # 以换行符分割段落
            paragraphs = [para.strip() for para in content.split("\n") if para.strip()]

            # 合并段落，保证段落字数大于 n/2 小于 2n
            segments = []
            current_segment = ""

            for para in paragraphs:
                if len(current_segment) + len(para) > 1.5 * n:
                    if n / 2 <= len(current_segment) <= 1.5 * n:
                        segments.append(current_segment.strip())
                    current_segment = para
                else:
                    current_segment += "\n" + para

            if n / 2 <= len(current_segment) <= 1.5 * n:
                segments.append(current_segment.strip())

            # 创建 input, output 数据对
            for segment in segments:
                summary = summarize_segment(segment)
                # 将段落按句号分割为句子
                sentences = [s for s in segment.split("。") if s.strip()]
                sentences = [sentence + '。' for sentence in sentences]

                if len(sentences) > 1:
                    input_sentence = sentences[0].strip()  # 第一句作为input
                    output_text = "".join(sentences[1:]).strip()  # 其余作为output

                    data.append({
                        "instruction": '请按照以下描述续写小说：' + summary,
                        "input": input_sentence,
                        "output": output_text
                    })

            # 将数据和日志写入文件
            with open(output_json_path, 'a', encoding='utf-8') as json_file:
                json.dump(data, json_file, ensure_ascii=False, indent=4)
            with open(processed_files_log, 'a') as log_file:
                log_file.write(filename + "\n")

            data = []  # 清空数据以进行下一个文件的处理
            file_count += 1

            elapsed_time = time.time() - start_time  # 计算已运行的时间
            logging.info(f"{file_count} files 处理完毕，数据保存到文件 {output_json_path}，已运行时间：{elapsed_time:.2f} 秒")

            if file_count >= max_files:
                break


# 定义命令行参数解析
def main():
    parser = argparse.ArgumentParser(description="Extract story backgrounds and summaries using Qwen2-7B")
    parser.add_argument("--corpus_path", type=str, required=True, help="Path to the corpus directory containing txt files")
    parser.add_argument("--output_dir", type=str, default="segment_label", help="Directory to save output JSON and log files")
    parser.add_argument("--n", type=int, default=800, help="Threshold for segment length")
    parser.add_argument("--model_dir", type=str, required=True, help="Model directory or model name")

    args = parser.parse_args()

    # 加载标记器
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)

    # 加载模型（确保设备支持大模型运行）
    model = AutoModelForCausalLM.from_pretrained(args.model_dir, device_map="auto", torch_dtype=torch.float16)

    # 定义段落总结函数
    def summarize_segment(segment):
        prompt = "\n你是一个熟读各类小说的专家，请将以上小说内容确定其背景并且用一句话总结故事情节不要进行续写。严格按照以下格式回复：背景：XXXX剧情梗概：XXXX(一句话)。注意：只需提供背景和剧情梗概，不要续写正文。"
        prompt = '小说内容如下：' + segment + prompt

        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return response

    # 预处理数据
    preprocess_data(args.corpus_path, args.output_dir, args.n)

if __name__ == "__main__":
    main()
