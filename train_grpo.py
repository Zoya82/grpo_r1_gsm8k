# -*- coding: utf-8 -*-
"""
用 GRPO 复现 DeepSeek-R1 的"思维链"雏形
================================================
目标:让一个很小的模型(Qwen2.5-0.5B)通过【强化学习】学会:
  1) 按指定格式输出 <reasoning>推理</reasoning><answer>答案</answer>
  2) 把 gsm8k 小学数学题算对

为什么这是"强化学习后训练"而不是 SFT/DPO:
  - SFT:给模型看"标准答案",让它模仿  -> 教它"说什么"
  - DPO:给模型看"好/坏两个回答"的偏好对 -> 教它"怎么选"
  - GRPO(本项目):不给标准答案,只给一个【reward 函数】当"裁判"。
    模型自己对同一道题生成一组(G个)回答,裁判给每个打分,
    模型朝"得分高于组内平均"的回答方向更新 -> 教它"怎么想"
    这就是 DeepSeek-R1 的核心思路:用 RL + 规则奖励逼出推理能力。
"""
import os
# MPS(苹果 GPU)有少数算子不支持,设置回退到 CPU,避免训练崩溃
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import re
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# 可用环境变量调规模:跑通用小,正式跑用大
N_DATA   = int(os.environ.get("N_DATA", 200))    # 用多少条 gsm8k 题
MAX_STEPS = int(os.environ.get("MAX_STEPS", 30))  # 训练多少步

# ============ 1. 数据:把 gsm8k 包装成"问题 + 标准数字答案" ============
# 系统提示规定输出格式 —— 格式 reward 就是检查模型有没有照做
SYSTEM_PROMPT = (
    "You are a math assistant. Solve the problem step by step.\n"
    "Respond in EXACTLY this format:\n"
    "<reasoning>\nyour step-by-step reasoning\n</reasoning>\n"
    "<answer>\nthe final numeric answer only\n</answer>"
)

def extract_gt(answer_text: str) -> str:
    # gsm8k 的答案形如 "....#### 72",#### 后面是最终数字答案
    return answer_text.split("####")[-1].strip().replace(",", "")

def build_dataset(n: int):
    ds = load_dataset("openai/gsm8k", "main", split=f"train[:{n}]")
    def fmt(ex):
        return {
            # GRPO 要求数据集有 "prompt" 列;用对话格式,trl 会自动套 chat template
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": ex["question"]},
            ],
            "answer_gt": extract_gt(ex["answer"]),  # 标准答案,给正确性裁判用
        }
    return ds.map(fmt, remove_columns=ds.column_names)

# ============ 2. reward 函数(裁判):GRPO 的灵魂 ============
# trl 会一次传入一个 batch 的 completions(list),返回等长的分数 list。
# 对话格式下,每个 completion 是 [{"role":"assistant","content":"..."}]
def _text(completion) -> str:
    return completion[0]["content"]

# --- 裁判一:格式奖励(渐进式软奖励)。出现一对标签给一点分,引导模型逐步学格式 ---
# 一开始就要求严格完整格式,0.5B 小模型 zero-shot 根本做不到 -> reward 恒 0 学不动。
# 改成分层:有 <reasoning>..</reasoning> 给 0.25,有 <answer>..</answer> 再给 0.25。
RE_REASON = re.compile(r"<reasoning>.*?</reasoning>", re.DOTALL)
RE_ANSWER = re.compile(r"<answer>.*?</answer>", re.DOTALL)
def format_reward(completions, **kwargs):
    out = []
    for c in completions:
        t = _text(c)
        r = 0.0
        if RE_REASON.search(t): r += 0.25
        if RE_ANSWER.search(t): r += 0.25
        out.append(r)
    return out

# --- 裁判二:正确性奖励。优先从 <answer> 抠数字;没有标签就兜底取全文最后一个数字 ---
# 这样"算对了"就能立刻拿到正确分(产生组内方差/学习信号),不被"没按格式"卡死。
def _extract_pred(text: str):
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
    src = m.group(1) if m else text          # 没标签 -> 用全文兜底
    nums = re.findall(r"-?\d+\.?\d*", src.replace(",", ""))
    return nums[-1].rstrip(".") if nums else None  # 取最后一个数字

def correctness_reward(completions, answer_gt, **kwargs):
    out = []
    for c, gt in zip(completions, answer_gt):
        pred = _extract_pred(_text(c))
        out.append(2.0 if (pred is not None and pred == gt) else 0.0)
    return out
# 注:格式分(0.5) < 正确分(2.0),是有意的 —— 先用小奖励把"格式"带出来,
#     再用大奖励逼"答对"。reward 数值的相对大小,就是你在"设计奖励信号"。

# ============ 3. GRPO 训练配置 ============
def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    dataset = build_dataset(N_DATA)
    print(f"数据集: {len(dataset)} 条")

    # LoRA:只训练插入的低秩小矩阵,冻结 0.5B 主体 —— 省显存,雏形足够
    lora = LoraConfig(
        r=8, lora_alpha=16, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )

    args = GRPOConfig(
        output_dir="./grpo_out",
        per_device_train_batch_size=4,  # 一个 batch 4 个 completion
        num_generations=4,              # 每道题生成 4 个回答组成一组(GRPO 的 G)
        max_completion_length=256,      # 每个回答最多 256 token(防答案被截断)
        max_steps=MAX_STEPS,
        learning_rate=1e-5,
        logging_steps=1,                # 每步都打印 reward,方便看曲线/讲
        beta=0.04,                      # KL 惩罚系数:防止训歪、离原模型太远
        temperature=0.9,                # 采样温度高一点 -> 多样性,利于探索
        use_vllm=False,                 # MPS 不支持 vllm,用 transformers 原生生成
        bf16=False, fp16=False,         # MPS 上用 fp32 最稳
        save_strategy="no",
        report_to="none",
    )

    trainer = GRPOTrainer(
        model=MODEL,
        reward_funcs=[format_reward, correctness_reward],  # 两个裁判一起打分
        args=args,
        train_dataset=dataset,
        peft_config=lora,
    )

    print("开始 GRPO 训练...(看 reward 列有没有往上走)")
    trainer.train()

    # 训练后把每步的 reward 存下来,方便画曲线 / 面试讲
    import json
    logs = [l for l in trainer.state.log_history if "reward" in l]
    with open("reward_log.json", "w") as f:
        json.dump(logs, f, ensure_ascii=False, indent=2)
    print("\n=== 每步 reward(总) ===")
    for l in logs:
        print(f"step {l.get('step'):>3} | reward={l.get('reward'):.3f}")
    print("\n训练完成,日志已存 reward_log.json")

if __name__ == "__main__":
    main()
