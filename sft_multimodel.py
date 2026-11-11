#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

The SFT target preserves the paper's structured response format while avoiding
fabricated gold reasoning traces: <think> None </think><answer>...</answer>.
Adapters are written to ./adapters/<model-key>/final_adapter by default.
"""

import argparse
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

MODEL_REGISTRY = {
    "gemma2_2b": "google/gemma-2-2b-it",
    "qwen25_3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen25_7b": "Qwen/Qwen2.5-7B-Instruct",
}
LANGS = ["english", "hindi", "bengali", "marathi", "gujarati", "tamil"]
OPTION_LABEL_MAP = {
    "english":  {"a":"a","b":"b","c":"c","d":"d"},
    "hindi":    {"a":"क","b":"ख","c":"ग","d":"घ"},
    "bengali":  {"a":"ক","b":"খ","c":"গ","d":"ঘ"},
    "marathi":  {"a":"क","b":"ख","c":"ग","d":"घ"},
    "gujarati": {"a":"ક","b":"ખ","c":"ગ","d":"ઘ"},
    "tamil":    {"a":"௧","b":"௨","c":"௩","d":"௪"},
}
SYSTEM_PROMPT = (
    "You are solving a multilingual multiple-choice financial reasoning problem. "
    "Read the question and all four options carefully. First reason about the problem, "
    "then provide exactly one final option. Use the following output format exactly: "
    "<think> reasoning process </think><answer> option-label. option-text </answer>. "
    "Use the language of the question for the reasoning and final answer."
)


def norm(x):
    return re.sub(r"\s+", " ", "" if x is None else str(x).strip())


def canonical_gold(language: str, value) -> Optional[str]:
    raw = norm(value); low = raw.lower()
    if low in set("abcd"): return low
    rev = {v:k for k,v in OPTION_LABEL_MAP.get(language,{}).items()}
    if raw in rev: return rev[raw]
    cleaned = re.sub(r"[\s\.\,\:\;\-\(\)\[\]\{\}]+", "", raw)
    for k,v in OPTION_LABEL_MAP.get(language,{}).items():
        if cleaned == re.sub(r"[\s\.\,\:\;\-\(\)\[\]\{\}]+", "", v): return k
    return None


def prepare_df(path: str):
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    required = {"question","option_a","option_b","option_c","option_d","language","final_answer"}
    missing = required-set(df.columns)
    if missing: raise ValueError(f"Missing columns: {sorted(missing)}")
    df["language"] = df.language.str.strip().str.lower()
    df = df[df.language.isin(LANGS)].copy()
    if "gold_letter" not in df.columns:
        df["gold_letter"] = [canonical_gold(l,a) for l,a in zip(df.language,df.final_answer)]
    else:
        df["gold_letter"] = [canonical_gold(l,g) or canonical_gold(l,a) for l,g,a in zip(df.language,df.gold_letter,df.final_answer)]
    return df[df.gold_letter.isin(list("abcd"))].reset_index(drop=True)


def check_group_leakage(train_df, val_df):
    if "item_id" not in train_df.columns or "item_id" not in val_df.columns:
        return
    overlap = set(train_df.item_id.astype(str)) & set(val_df.item_id.astype(str))
    if overlap:
        raise ValueError(f"Source-group leakage detected between train and validation: {len(overlap)} item_id values")


def build_messages(row):
    lang=row["language"]; m=OPTION_LABEL_MAP[lang]
    user=(f"{SYSTEM_PROMPT}\n\nQuestion: {row['question']}\nOptions:\n"
          f"{m['a']}. {row['option_a']}\n{m['b']}. {row['option_b']}\n"
          f"{m['c']}. {row['option_c']}\n{m['d']}. {row['option_d']}\n")
    return [{"role":"user","content":user}]


def target_text(row):
    g=row["gold_letter"]; label=OPTION_LABEL_MAP[row["language"]][g]
    return f"<think> None </think><answer>{label}. {row[f'option_{g}']}</answer>"


class SFTDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.examples=[]
        for row in df.to_dict("records"):
            prompt=tokenizer.apply_chat_template(build_messages(row), tokenize=False, add_generation_prompt=True)
            p=tokenizer(prompt, add_special_tokens=False)["input_ids"]
            t=tokenizer(target_text(row), add_special_tokens=False)["input_ids"]
            if tokenizer.eos_token_id is not None: t=t+[tokenizer.eos_token_id]
            ids=(p+t)[:max_length]
            labels=([-100]*len(p)+t)[:max_length]
            if all(x == -100 for x in labels):
                continue
            self.examples.append({"input_ids":ids,"attention_mask":[1]*len(ids),"labels":labels})
    def __len__(self): return len(self.examples)
    def __getitem__(self,i): return self.examples[i]


@dataclass
class CausalSFTCollator:
    pad_token_id: int
    def __call__(self, features):
        maxlen=max(len(x["input_ids"]) for x in features)
        ids=[]; masks=[]; labels=[]
        for x in features:
            n=maxlen-len(x["input_ids"])
            ids.append(x["input_ids"]+[self.pad_token_id]*n)
            masks.append(x["attention_mask"]+[0]*n)
            labels.append(x["labels"]+[-100]*n)
        return {"input_ids":torch.tensor(ids,dtype=torch.long),"attention_mask":torch.tensor(masks,dtype=torch.long),"labels":torch.tensor(labels,dtype=torch.long)}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",choices=MODEL_REGISTRY,default="gemma2_2b")
    ap.add_argument("--train_csv",default="./data/train.csv")
    ap.add_argument("--val_csv",default="./data/val.csv")
    ap.add_argument("--adapter_root",default="./adapters")
    ap.add_argument("--epochs",type=float,default=3.0)
    ap.add_argument("--learning_rate",type=float,default=2e-4)
    ap.add_argument("--batch_size",type=int,default=2)
    ap.add_argument("--grad_accum",type=int,default=8)
    ap.add_argument("--max_length",type=int,default=2048)
    ap.add_argument("--lora_r",type=int,default=8)
    ap.add_argument("--lora_alpha",type=int,default=32)
    ap.add_argument("--lora_dropout",type=float,default=0.10)
    ap.add_argument("--seed",type=int,default=42)
    args=ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    train_df=prepare_df(args.train_csv)
    val_path=Path(args.val_csv)
    val_df=prepare_df(str(val_path)) if val_path.exists() else None
    if val_df is not None: check_group_leakage(train_df,val_df)

    model_id=MODEL_REGISTRY[args.model]
    tok=AutoTokenizer.from_pretrained(model_id,use_fast=True,padding_side="right",trust_remote_code=False)
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token
    dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    try:
        model=AutoModelForCausalLM.from_pretrained(model_id,dtype=dtype,attn_implementation="sdpa",low_cpu_mem_usage=True,trust_remote_code=False)
    except TypeError:
        model=AutoModelForCausalLM.from_pretrained(model_id,torch_dtype=dtype,attn_implementation="sdpa",low_cpu_mem_usage=True,trust_remote_code=False)

    available={name.rsplit(".",1)[-1] for name,_ in model.named_modules()}
    targets=[x for x in ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"] if x in available]
    if not targets: raise RuntimeError("No supported LoRA target modules found")
    peft_cfg=LoraConfig(task_type="CAUSAL_LM",r=args.lora_r,lora_alpha=args.lora_alpha,lora_dropout=args.lora_dropout,target_modules=targets,bias="none")
    model=get_peft_model(model,peft_cfg)
    model.config.use_cache=False
    model.print_trainable_parameters()

    train_ds=SFTDataset(train_df,tok,args.max_length)
    val_ds=SFTDataset(val_df,tok,args.max_length) if val_df is not None else None
    out_dir=Path(args.adapter_root)/args.model
    out_dir.mkdir(parents=True,exist_ok=True)

    train_args=TrainingArguments(
        output_dir=str(out_dir/"checkpoints"), num_train_epochs=args.epochs,
        learning_rate=args.learning_rate, per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum, logging_steps=10,
        save_strategy="epoch", save_total_limit=2, report_to=[],
        bf16=bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
        fp16=bool(torch.cuda.is_available() and not torch.cuda.is_bf16_supported()),
        remove_unused_columns=False, gradient_checkpointing=True,
        seed=args.seed, data_seed=args.seed,
    )
    trainer=Trainer(model=model,args=train_args,train_dataset=train_ds,eval_dataset=val_ds,data_collator=CausalSFTCollator(tok.pad_token_id))
    trainer.train()
    final=out_dir/"final_adapter"
    trainer.save_model(str(final)); tok.save_pretrained(str(final))
    (out_dir/"sft_config.json").write_text(json.dumps({
        "model_key":args.model,"model_id":model_id,"train_examples":len(train_ds),
        "val_examples":0 if val_ds is None else len(val_ds),"epochs":args.epochs,
        "learning_rate":args.learning_rate,"lora_r":args.lora_r,"lora_alpha":args.lora_alpha,
        "lora_dropout":args.lora_dropout,"target_modules":targets,"seed":args.seed,
        "prompt":SYSTEM_PROMPT,
    },indent=2,ensure_ascii=False),encoding="utf-8")
    print("Saved anonymous SFT adapter to",final)


if __name__=="__main__":
    main()
