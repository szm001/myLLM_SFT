# -*- coding: utf-8 -*-
# @Time    : 2025/3/23 9:18
# @Author  : shaozhumin
# @FileName: sft.py
# @Software: PyCharm

# 需求：用SFT方法训练一个白话文转文言文的大模型
# 数据集：YeungNLP/firefly-train-1.1m
# 基础模型：Qwen/Qwen2.5-0.5B-Instruct

'''
什么是SFT======================
supervised Finetuning
和FT(fine tuning)有什么区别？
本质是一样的，都是用新数据进行大模型的优化
区别===========================
SFT的方法：  LoRA
参数高效PEFT训练的一种方式
LoRA
Prefix Tuning
P-Tuning
adapter turning
prompt tuning
优点================
节约内存
训练速度快
效果损失小
Learn less，forget less
用训练好的模型的模型权重的基础上进行训练

'''
import accelerate
import os
from datasets import load_dataset
from transformers import AutoTokenizer,AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from transformers import TrainingArguments
from trl import SFTTrainer
from peft import AutoPeftModelForCausalLM
from transformers import pipeline

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["HF_HOME"] = "E:/python_projiect/models/"

def trainLLMBySFT():
    # 第一步下载数据集，加载数据
    # 下载地址
    # https://hf-mirror.com/datasets/YeungNLP/firefly-train-1.1M
    # 加载数据
    test_dataset = load_dataset(
        "E:/python_projiect/voice_chat/models/datasets/YeungNLP/firefly-train-1.1m/",
        split="train[:500]")

    print(test_dataset)
    '''
    Dataset({
        features: ['kind', 'input', 'target'],
        num_rows: 500
    })
    '''
    print(test_dataset[0])
    '''
    {'kind': 'NLI',
     'input': '自然语言推理：\n前提：家里人心甘情愿地养他,还有几家想让他做女婿的\n假设：他是被家里人收养的孤儿',
     'target': '中立'}
    '''
    print(test_dataset[100])
    '''
    {'kind': 'ClassicalChinese',
     'input': '我当时在三司，访求太祖、仁宗的手书敕令没有见到，然而人人能传诵那些话，禁止私盐的建议也最终被搁置。\n翻译成文言文：',
     'target': '余时在三司，求访两朝墨敕不获，然人人能诵其言，议亦竟寝。'}
    '''

    # 第二步目标格式转化
    '''
    目标格式
    要把数据转成qwen2-0.5b-instruct模型的输入格式
    <|im_start|>system
    你是一个非常棒的人工智能助手<|im_end|>
    <|im_start|>user
    我当时在三司，访求太祖、仁宗的手书敕令没有见到，然而人人能传诵那些话，禁止私盐的建议也最终被搁置。
    翻译成文言文：<|im_end|>
    <|im_start|>assistant
    余时在三司，求访两朝墨敕不获，然人人能诵其言，议亦竟寝。<|im_end|>
    '''

    tokenizer = AutoTokenizer.from_pretrained("E:/python_projiect/voice_chat/models/Qwen/Qwen2.5-0.5B-Instruct")

    def format_prompt(example):
        chat = [
            {"role": "system", "content": "你是一个非常棒的人工智能助手"},
            {"role": "user", "content": example["input"]},
            {"role": "assistant", "content": example["target"]},
        ]
        prompt = tokenizer.apply_chat_template(chat, tokenize=False)
        return {"text": prompt}
        # return {"text":example["input"],"label":example["target"]}

    dataset = test_dataset.map(
        format_prompt,
        remove_columns = test_dataset.column_names
    )
    print(dataset)
    '''
    Dataset({
        features: ['text'],
        num_rows: 500
    })
    '''

    print(dataset[100])
    '''
    {
        'text': '<|im_start|>system\n你是一个非常棒的人工智能助手<|im_end|>\n<|im_start|>user\n我当时在三司，访求太祖、仁宗的手书敕令没有见到，然而人人能传诵那些话，禁止私盐的建议也最终被搁置。\n翻译成文言文：<|im_end|>\n<|im_start|>assistant\n余时在三司，求访两朝墨敕不获，然人人能诵其言，议亦竟寝。<|im_end|>\n'
    }
    '''

    # 第三步 加载模型

    model = AutoModelForCausalLM.from_pretrained("E:/python_projiect/voice_chat/models/Qwen/Qwen2.5-0.5B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained("E:/python_projiect/voice_chat/models/Qwen/Qwen2.5-0.5B-Instruct")
    # tokenizer.padding_size="left"

    # 第四步 配置lara
    peft_config = LoraConfig(
        lora_alpha=32,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=['k_proj', 'v_proj', 'q_proj']
    )
    model_lora = get_peft_model(model, peft_config)

    # 第5步 训练配置
    output_dir = "./results"
    training_arguments = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        # optim="adamw_torch",
        learning_rate=2e-4,
        # lr_scheduler_type="cosine",
        # num_train_epochs=1,
        logging_steps=10,
        # fp16=False,
        # gradient_checkpointing=False,
        save_steps=1,  # 15个step保存一个checkpoint
        max_steps=2,
    )

    trainer = SFTTrainer(
        model=model_lora,
        args=training_arguments,
        train_dataset=dataset,
        # dataset_text_field="text",
        # tokenizer = tokenizer,
        peft_config=peft_config,
    )

    # 第6步 训练
    trainer.train()
    trainer.model.save_pretrained('./results/final-result')
    # 保存路径：E:\python_projiect\nanoGPT\course_bbruceyuan\results\final-result

    trainer.model.print_trainable_parameters()
    '''
    trainable params: 5,898,240 || all params: 499,931,008 || trainable%: 1.1798
    '''

    # 第6步 训练结果和基础模型合并为新模型

    model_peft = AutoPeftModelForCausalLM.from_pretrained(
        "./results/final-result",
        # device_map="aoto",
    )
    merged_model = model_peft.merge_and_unload()

    # 第7步 新模型的推理验证,此时新模型还没保存到文件，还是临时验证
    pipe = pipeline(task="text-generation", model=merged_model, tokenizer=tokenizer)
    prompt_example = """
    <|im_start|>system
    你是一个非常棒的人工智能助手<|im_end|>
    <|im_start|>user
    人很贪婪的，需要要预防自己犯错。
    翻译成文言文：<|im_end|>
    <|im_start|>assistant
    """
    dd = pipe(prompt_example, max_new_tokens=50)
    ret = dd[0]["generated_text"]
    print(ret)
    '''
    '\n<|im_start|>system\n你是一个非常棒的人工智能助手<|im_end|>\n<|im_start|>user\n人很贪婪的，需要要预防自己犯错。\n翻译成文言文：<|im_end|>\n<|im_start|>assistant\n人之贪心甚重，当防其过失。'
    '''
    print(dd)
    '''
    [{'generated_text': '\n<|im_start|>system\n你是一个非常棒的人工智能助手<|im_end|>\n<|im_start|>user\n人很贪婪的，需要要预防自己犯错。\n翻译成文言文：<|im_end|>\n<|im_start|>assistant\n人之贪心甚重，当防其过失。'}]
    '''

    # 第8步 新模型保存
    merged_model.save_pretrained("./results/merged_model", )

    # 第9步 模型合并后的使用
    model = AutoModelForCausalLM.from_pretrained("E:/python_projiect/nanoGPT/course_bbruceyuan/results/merged_model")

    # 第10步 新模型的推理验证
    pipe = pipeline(task="text-generation",model = model,tokenizer=tokenizer)
    prompt_example = """
    <|im_start|>system
    你是一个非常棒的人工智能助手<|im_end|>
    <|im_start|>user
    人很贪婪的，管理团队要注意这个问题。
    翻译成文言文：<|im_end|>
    <|im_start|>assistant
    """
    dd = pipe(prompt_example,max_new_tokens=50)
    ret = dd[0]["generated_text"]
    print(ret)
    '''
    '\n<|im_start|>system\n你是一个非常棒的人工智能助手<|im_end|>\n<|im_start|>user\n人很贪婪的，管理团队要注意这个问题。\n翻译成文言文：<|im_end|>\n<|im_start|>assistant\n人极贪心，管理团队当务之急。'
    '''
    print(dd)
    '''
    [{'generated_text': '\n<|im_start|>system\n你是一个非常棒的人工智能助手<|im_end|>\n<|im_start|>user\n人很贪婪的，管理团队要注意这个问题。\n翻译成文言文：<|im_end|>\n<|im_start|>assistant\n人极贪心，管理团队当务之急。'}]
    '''

    prompt_example = "人很贪婪的，管理团队要注意这个问题。翻译成文言文"
    dd = pipe(prompt_example, max_new_tokens=50, num_return_sequences=1)
    ret = dd[0]["generated_text"]
    print(ret)
    '人很贪婪的，管理团队要注意这个问题。翻译成文言文\n“人极贪鄙，管理团队要加以注意。”这句话翻译成文言文是：\n\n子甚贪鄙，治群务者当慎之。'

if __name__ == "__main__":
    trainLLMBySFT()




