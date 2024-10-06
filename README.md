# Completing-Stories-Following-Instruct-with-LLMs-LoRA

Train an LLM LoRA using a specific dataset, allowing the LLM to continue writing stories in a specific style based on plot and background. Inspired by the [Tianchi Novel Writing Challenge](https://tianchi.aliyun.com/competition/entrance/532210/introduction), the method involves providing an instruction with the plot background and the opening sentence of the novel, enabling the model to generate the rest of the story. This repo builds a simple and user-friendly pipeline, allowing users to easily train the model's LoRA with just two lines of code after collecting novels of the desired style. The repo consists of two main steps: generating an instruction-structured dataset from novels, and training the LoRA model based on the dataset.


通过特定数据集训练LLM LoRA，使LLM基于情节和背景续写出特定风格的故事。受到[天池小说创作大模型挑战赛](https://tianchi.aliyun.com/competition/entrance/532210/introduction)的启发，通过指示输入情节背景与小说开头句，以此让大模型生成小说。本repo构建了一个简单易用的pipeline，使用者可以在收集自己想要让大模型学习的特定风格的小说后，通过两行代码轻松训练模型的lora。本repo包含两个主要步骤，从小说生成指令结构的数据集，与基于数据集训练lora模型。

## Environment Setup  环境配置

Configure the Python environment using the environment.yaml file:


使用 `environment.yaml` 文件配置 Python 环境：

```bash
conda env create -f environment.yaml
```

## Dataset Generation 数据集生成

Divide paragraphs from the novel corpus and extract the background and plot of each paragraph using the Qwen2.5-3B model. Run the following command to generate the labeled dataset, where the parameter `corpus_path` is the path to the novel data file, `n` is the number of characters per segment, and `model_dir` can be either a local model path or a model name:


从小说数据集中划分段落，并使用 Qwen2.5-3B 模型提取段落的背景与情节。运行以下命令生成带有标注的数据集，其中参数`corpus_path`为小说数据文件路径，`n`为每段的字数，`model_dir`可以为本地的模型路径或者模型名称：

   ```bash
   python segment_label.py --corpus_path "./sample_data" --output_dir "segment_label" --n 800 --model_dir "Qwen/Qwen2.5-3B-Instruct"
   ```
The format of the converted data is as follows:

转换后的数据格式如下：


   ```

   {
    "instruction": "Please continue writing the novel based on the following description: Background: Modern high school campus. Plot summary: A young girl develops feelings for her literature teacher, who has been assigned as the new homeroom teacher.",
    "input": "The first class in the afternoon was the literature class I've been looking forward to for so long! Mr. Yan entered the classroom with his usual grace, and from the moment he stepped in, every move he made captivated my attention. But I didn't dare make it too obvious, afraid my classmates would notice. I could only sneak glances at him from behind my textbook.",
    "output": "Today, the teacher had prepared another poem for us: Zheng Chouyu's 'Love, Begins': Ever since the awkward beginning of love, Xiao Lianli, your bright eyes have truly become enchanting!\nAh, seventeen, what a wonderful first love!\nI happened to have read this poem yesterday, and for a moment, it felt like the teacher was reciting it just for me. My heart started to race, and I could no longer look at him. I lowered my head, and suddenly the feeling I had in the morning returned. My heart was pounding as if it was about to leap out, my breath grew uneven, and I felt a warm current surging inside me, making my whole body heat up..."
}
    {
        "instruction": "请按照以下描述续写小说：背景：现代高中校园。剧情梗概：一位少女对国文老师产生爱慕之情，而老师被指派为新班主任。",
        "input": "下午第一堂课是我期待好久的国文课！颜老师又是那么风采翩翩的进了教室，从他进来开始，他的一举一动就牵引着我的目光，可是我又不敢太明显，怕给同学知道了，我只敢从课本后悄悄的伸出头来偷看他。",
        "output": "今天老师又帮我们准备了一首郑愁予的“爱，开始”∶自从爱情忸怩的开始，小莲莉，你生命底盈盈的眼，才算迷人了！\n哟，十七岁，好一个动人的初恋呀！\n我昨天正好看过这首，霎时之间，我觉得老师好象是在对着我吟诵，我觉得心慌意乱，不敢再看着他，低下头来，忽然早上那种感觉又回来了，我的心又碰碰碰的好象要跳出来，呼吸也乱了起来，觉得身体里有股热流在乱窜着，弄得我浑身发烫..........................。"
    }
   ```

## LoRA 训练

Use the following command to train the LoRA model:
使用以下命令训练 LoRA 模型：

```bash
python train_lora.py --model_name "Qwen/Qwen2.5-7B-Instruct" --max_seq_length 1000 --output_dir "./output"
```

Finally, thanks to the anonymous contributor for using this pipeline on Hugging Face to create the [novel instruction dataset]((https://huggingface.co/datasets/cgxjdzz/h-corpus-Instruct)) and train the [Qwen-2.5-7B LoRA model](https://huggingface.co/cgxjdzz/Qwen-2.5-7B-Instruct-novel-lora).



最后感谢不知名贡献者在hugging face上使用本pipeline提取的[小说指示数据集](https://huggingface.co/datasets/cgxjdzz/h-corpus-Instruct)与训练的[Qwen-2.5-7B Lora](https://huggingface.co/cgxjdzz/Qwen-2.5-7B-Instruct-novel-lora)。
