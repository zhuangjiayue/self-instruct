import os
import json
import random
import re
import string
import tqdm
import argparse
import numpy as np
import pandas as pd
from multiprocessing import Pool
from functools import partial
from rouge_score import rouge_scorer
from gpt3_api import make_requests as make_gpt3_requests

# 设置随机种子

random.seed(42)


def encode_prompt(prompt_instructions, classification=False):
    """对prompt进行编码"""
    if classification:
        prompt = "Come up with a series of classification tasks. Try to specify the possible output labels when possible.\n"
    else:
        prompt = "Come up with a series of tasks:\n"
    for idx, instruction in enumerate(prompt_instructions):
        instruction = re.sub(r"\s+", " ", instruction).strip().rstrip(":")
        prompt += f"{idx+1}. {instruction}\n"
    prompt += f"{len(prompt_instructions) + 1}."
    return prompt


def sample_machine_instructions(machine_instructions, similarities, n):
    """在字符串 s 中查找字符串 w"""
    return random.sample(machine_instructions, min(n, len(machine_instructions)))


def find_word_in_string(w, s):

    return re.compile(r'\b({0})\b'.format(w), flags=re.IGNORECASE).search(s)


def post_process_gpt3_response(response):
    """对 GPT-3 的响应进行后处理，包括分割生成的文本，对文本进行清洗，以及根据一系列规则过滤掉不合适的指令"""
    if response is None or response["choices"][0]["finish_reason"] == "length":
        return []
    raw_instructions = re.split(r"\n\d+\s?\. ", response["choices"][0]["text"])
    instructions = []
    for inst in raw_instructions:
        inst = re.sub(r"\s+", " ", inst).strip()
        inst = inst.strip().capitalize()
        if inst == "":
            continue
        # filter out too short or too long instructions
        if len(inst.split()) <= 3 or len(inst.split()) > 150:
            continue
        # filter based on keywords that are not suitable for language models.
        if any(find_word_in_string(word, inst) for word in ["image", "images", "graph", "graphs", "picture", "pictures", "file", "files", "map", "maps", "draw", "plot", "go to"]):
            continue
        # We found that the model tends to add "write a program" to some existing instructions, which lead to a lot of such instructions.
        # And it's a bit comfusing whether the model need to write a program or directly output the result. 
        # Here we filter them out.
        # Note this is not a comprehensive filtering for all programming instructions.
        if inst.startswith("Write a program"):
            continue
        # filter those starting with punctuation
        if inst[0] in string.punctuation:
            continue
        # filter those starting with non-english character
        if not inst[0].isascii():
            continue
        instructions.append(inst)
    return instructions


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_dir",
        type=str,
        required=True,
        default="data/gpt3_generations/",
        help="The directory where the batch is stored.",
    )
    parser.add_argument(
        "--seed_tasks_path",
        type=str,
        required=True,
        default="data/seed_tasks.jsonl",
        help="The path to the human written data.",
    )
    parser.add_argument(
        "--num_instructions_to_generate",
        type=int,
        default=100,
        help="th",
    )
    parser.add_argument(
        "--use_clf_seed_tasks_only",
        action="store_true",
        help="If specified, we will only use the classification seed tasks to prompt new instructions. This will lead to more classification instructions.",
    )
    parser.add_argument(
        "--engine",
        type=str,
        default="davinci",
        help="The engine to use."
    )
    parser.add_argument(
        "--num_prompt_instructions",
        type=int,
        default=8,
        help="The number of instructions to use in the prompt."
    )
    parser.add_argument(
        "--request_batch_size",
        type=int,
        default=5,
        help="The number of requests to send to GPT3 at a time."
    )
    parser.add_argument(
        "--api_key",
        type=str,
        help="The API key to use. If not specified, the key will be read from the environment variable OPENAI_API_KEY."
    )
    parser.add_argument(
        "--organization",
        type=str,
        help="The organization to use. If not specified, the default organization id will be used."
    )
    return parser.parse_args()


if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()
    # 读取种子任务文件，其中每一行都是一个json格式的任务信息，通过json.loads解析每一行的内容，得到一个包含所有任务信息的列表
    # default="data/seed_tasks.jsonl" 最初的175个指令
    seed_tasks = [json.loads(l) for l in open(args.seed_tasks_path, "r")]

    # 如果参数设置了只使用分类任务，那么就从种子任务中筛选出分类任务
    if args.use_clf_seed_tasks_only:
        seed_tasks = [t for t in seed_tasks if t["is_classification"]]

    # 获取每个种子任务的指令，得到一个包含所有种子指令的列表
    seed_instructions = [t["instruction"] for t in seed_tasks]
    print(f"Loaded {len(seed_instructions)} human-written seed instructions")

    # 如果不存在批处理目录，就创建一个 default="data/gpt3_generations/",
    os.makedirs(args.batch_dir, exist_ok=True)
    request_idx = 0
    # load the LM-generated instructions
    # 初始化一个列表，用于存储机器生成的指令
    machine_instructions = []

    # 如果存在已经生成的机器指令文件，就从文件中加载这些指令 default="data/gpt3_generations/"
    # 解析每一行的json内容，将机器指令添加到列表中
    if os.path.exists(os.path.join(args.batch_dir, "machine_generated_instructions.jsonl")):
        with open(os.path.join(args.batch_dir, "machine_generated_instructions.jsonl"), "r") as fin:
            for line in fin:
                instruction_info = json.loads(line)
                machine_instructions.append(instruction_info["instruction"])
                request_idx = instruction_info["request_idx"] + 1
        print(f"Loaded {len(machine_instructions)} machine-generated instructions")

    # similarities = {}
    # 初始化一个Rouge评分器，用于计算新生成的指令与已有指令的相似度
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=False)
    
    # now let's generate new instructions!
    # 初始化一个进度条，总长度为需要生成的指令数量
    progress_bar = tqdm.tqdm(total=args.num_instructions_to_generate)
    # 如果已经存在机器生成的指令，就更新进度条
    if machine_instructions:
        progress_bar.update(len(machine_instructions))
    # 打开一个文件，用于存储新生成的机器指令
    with open(os.path.join(args.batch_dir, "machine_generated_instructions.jsonl"), "a") as fout:
        # 如果生成的指令数还没有达到设定的值，就持续生成新的指令
        while len(machine_instructions) < args.num_instructions_to_generate:
            batch_inputs = []
            for _ in range(args.request_batch_size):
                # sample machine instructions from the pool
                # 从机器生成的指令中抽样出一部分，加入到提示指令中
                prompt_instructions = sample_machine_instructions(
                    machine_instructions, 
                    similarities=None,
                    n=2)
                # sample human instructions from the pool
                # 从人类编写的指令中抽样出一部分，加入到提示指令中
                prompt_instructions += random.sample(seed_instructions, args.num_prompt_instructions - len(prompt_instructions))
                random.shuffle(prompt_instructions)
                prompt = encode_prompt(prompt_instructions, classification=args.use_clf_seed_tasks_only)
                batch_inputs.append(prompt)
            results = make_gpt3_requests(
                engine=args.engine,
                prompts=batch_inputs,
                max_tokens=1024,
                temperature=0.7,
                top_p=0.5,
                frequency_penalty=0,
                presence_penalty=2,
                stop_sequences=["\n\n", "\n16", "16.", "16 ."],
                logprobs=1,
                n=1,
                best_of=1,
                api_key=args.api_key,
                organization=args.organization,
            )
            instructions = []
            all_metadata = []
            for result in results:
                new_instructions = post_process_gpt3_response(result["response"])
                instructions += new_instructions
                all_metadata += [result] * len(new_instructions)

            for inst, metadata in zip(instructions, all_metadata):
                with Pool(4) as p:
                    rouge_scores = p.map(partial(scorer.score, inst), seed_instructions + machine_instructions)
                rouge_scores = [score["rougeL"].fmeasure for score in rouge_scores]
                # rouge_scores = [scorer.score(inst, e_inst)["rougeL"].fmeasure for e_inst in human_instructions + machine_instructions]
                if max(rouge_scores) > 0.7:
                    continue
                all_instructions = seed_instructions + machine_instructions
                most_similar_instructions = {
                        all_instructions[i] : rouge_scores[i] for i in np.argsort(rouge_scores)[-10:][::-1]
                    }
                machine_instructions.append(inst)
                fout.write(json.dumps({
                    "instruction": inst,
                    "most_similar": most_similar_instructions,
                    "avg_similarity_score": float(np.mean(rouge_scores)),
                    "metadata": metadata,
                    "request_idx": request_idx
                }) + "\n")
                progress_bar.update(1)
            request_idx += 1
