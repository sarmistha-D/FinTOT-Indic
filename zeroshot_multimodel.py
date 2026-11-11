#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Supported model keys:
  gemma2_2b, qwen25_3b, qwen25_7b

The prompt implements the structured response protocol described in the paper:
<think> ... </think><answer> option-label. option-text </answer>

Example:
  python zeroshot_multimodel.py --model qwen25_3b --test_csv data/test.csv
"""

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_REGISTRY = {
    "gemma2_2b": "google/gemma-2-2b-it",
    "qwen25_3b": "Qwen/Qwen2.5-3B-Instruct",
    "qwen25_7b": "Qwen/Qwen2.5-7B-Instruct",
}

LANGS = ["english", "hindi", "bengali", "marathi", "gujarati", "tamil"]
OPTION_LABEL_MAP = {
    "english":  {"a": "a", "b": "b", "c": "c", "d": "d"},
    "hindi":    {"a": "क", "b": "ख", "c": "ग", "d": "घ"},
    "bengali":  {"a": "ক", "b": "খ", "c": "গ", "d": "ঘ"},
    "marathi":  {"a": "क", "b": "ख", "c": "ग", "d": "घ"},
    "gujarati": {"a": "ક", "b": "ખ", "c": "ગ", "d": "ઘ"},
    "tamil":    {"a": "௧", "b": "௨", "c": "௩", "d": "௪"},
}

SYSTEM_PROMPT = (
    "You are solving a multilingual multiple-choice financial reasoning problem. "
    "Read the question and all four options carefully. "
    "First reason about the problem, then provide exactly one final option. "
    "Use the following output format exactly: "
    "<think> reasoning process </think><answer> option-label. option-text </answer>. "
    "Use the language of the question for the reasoning and final answer."
)

ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.I | re.S)


def norm(x) -> str:
    return re.sub(r"\s+", " ", "" if x is None else str(x).strip())


def canonical_gold(language: str, value) -> Optional[str]:
    raw = norm(value)
    low = raw.lower()
    if low in {"a", "b", "c", "d"}:
        return low
    reverse = {v: k for k, v in OPTION_LABEL_MAP.get(language, {}).items()}
    if raw in reverse:
        return reverse[raw]
    cleaned = re.sub(r"[\s\.\,\:\;\-\(\)\[\]\{\}]+", "", raw)
    for k, v in OPTION_LABEL_MAP.get(language, {}).items():
        if cleaned == re.sub(r"[\s\.\,\:\;\-\(\)\[\]\{\}]+", "", v):
            return k
    return None


def prepare_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    required = {"question", "option_a", "option_b", "option_c", "option_d", "language", "final_answer"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    df["language"] = df["language"].str.strip().str.lower()
    df = df[df["language"].isin(LANGS)].copy()
    if "gold_letter" not in df.columns:
        df["gold_letter"] = [canonical_gold(l, y) for l, y in zip(df.language, df.final_answer)]
    else:
        df["gold_letter"] = [canonical_gold(l, y) or canonical_gold(l, z) for l, y, z in zip(df.language, df.gold_letter, df.final_answer)]
    df = df[df["gold_letter"].isin(list("abcd"))].reset_index(drop=True)
    if "item_id" not in df.columns:
        df["item_id"] = np.arange(len(df)).astype(str)
    return df


def build_messages(row: dict):
    lang = row["language"]
    labels = OPTION_LABEL_MAP[lang]
    user = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Question: {row['question']}\n"
        "Options:\n"
        f"{labels['a']}. {row['option_a']}\n"
        f"{labels['b']}. {row['option_b']}\n"
        f"{labels['c']}. {row['option_c']}\n"
        f"{labels['d']}. {row['option_d']}\n"
    )
    # User-only chat is portable across Gemma and Qwen templates.
    return [{"role": "user", "content": user}]


def extract_answer(text: str) -> str:
    m = ANSWER_RE.search(text or "")
    return m.group(1).strip() if m else norm(text)


def parse_letter(text: str, language: str, options: Dict[str, str]) -> Optional[str]:
    answer = extract_answer(text)
    mapping = OPTION_LABEL_MAP[language]
    for key in "abcd":
        native = re.escape(mapping[key])
        if re.search(rf"^\s*[\(\[]?\s*(?:{native}|{key})\s*[\.\)\]\:\-]?", answer, re.I):
            return key
    ans_norm = norm(answer).lower()
    hits = [k for k, v in options.items() if norm(v) and norm(v).lower() in ans_norm]
    return hits[0] if len(hits) == 1 else None


def load_model(model_id: str):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for evaluation.")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True, padding_side="left", trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    kwargs = dict(low_cpu_mem_usage=True, trust_remote_code=False)
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, attn_implementation="sdpa", **kwargs)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, attn_implementation="sdpa", **kwargs)
    model = model.to("cuda:0").eval()
    return tokenizer, model


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=MODEL_REGISTRY, default="gemma2_2b")
    ap.add_argument("--test_csv", default="./data/test.csv")
    ap.add_argument("--output_dir", default="./outputs/zero_shot")
    ap.add_argument("--max_prompt_length", type=int, default=2048)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    df = prepare_df(args.test_csv)
    if args.max_samples:
        df = df.iloc[:args.max_samples].copy()

    model_id = MODEL_REGISTRY[args.model]
    tokenizer, model = load_model(model_id)
    out_dir = Path(args.output_dir) / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = df.to_dict("records")
    records = []
    for start in range(0, len(rows), args.batch_size):
        batch = rows[start:start + args.batch_size]
        texts = [tokenizer.apply_chat_template(build_messages(r), tokenize=False, add_generation_prompt=True) for r in batch]
        enc = tokenizer(texts, add_special_tokens=False, return_tensors="pt", padding=True, truncation=True, max_length=args.max_prompt_length)
        enc = {k: v.to("cuda:0") for k, v in enc.items()}
        generated = model.generate(
            **enc, max_new_tokens=args.max_new_tokens, do_sample=False,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
        )
        prompt_width = enc["input_ids"].shape[1]
        for i, row in enumerate(batch):
            completion = tokenizer.decode(generated[i, prompt_width:], skip_special_tokens=True)
            options = {k: row[f"option_{k}"] for k in "abcd"}
            pred = parse_letter(completion, row["language"], options)
            records.append({
                "item_id": str(row["item_id"]), "language": row["language"],
                "gold_letter": row["gold_letter"], "pred_letter": pred,
                "correct": pred == row["gold_letter"], "completion": completion,
            })
        del enc, generated
        torch.cuda.empty_cache()

    pred = pd.DataFrame(records)
    lang = pred.groupby("language").agg(n=("correct", "size"), accuracy=("correct", "mean")).reset_index()
    summary = {
        "model_key": args.model, "model_id": model_id, "n": int(len(pred)),
        "micro_accuracy": float(pred.correct.mean()),
        "macro_language_accuracy": float(lang.accuracy.mean()),
        "prompt": SYSTEM_PROMPT,
    }
    pred.to_csv(out_dir / "predictions.csv", index=False, encoding="utf-8-sig")
    lang.to_csv(out_dir / "language_accuracy.csv", index=False, encoding="utf-8-sig")
    (out_dir / "run_config.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(lang.to_string(index=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
