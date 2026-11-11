#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Anonymous reference implementation for FinToT.

This release supports the model configurations reported in the paper:
  - Gemma-2-2B-IT
  - Qwen2.5-3B-Instruct
  - Qwen2.5-7B-Instruct

Experiment modes:
  - zero_shot       : base instruction-tuned model + mechanistic evaluation
  - grpo_partial    : existing SFT adapter continued with GRPO partial rewards
  - lr_disentangle  : existing SFT adapter + inference-time LR intervention
  - fintot          : existing SFT adapter + LR intervention + FinToT GRPO rewards


"""

import os
from pathlib import Path
from types import SimpleNamespace

MODEL_REGISTRY = {
    "gemma2_2b": {
        "model_id": "google/gemma-2-2b-it",
        "adapter_hint": "gemma-2-2b-it",
        "layers": "14-19",
    },
    "qwen25_3b": {
        "model_id": "Qwen/Qwen2.5-3B-Instruct",
        "adapter_hint": "qwen2.5-3b-instruct",
        "layers": "14-19",
    },
    "qwen25_7b": {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "adapter_hint": "qwen2.5-7b-instruct",
        "layers": "14-19",
    },
}

MODEL_KEY = os.environ.get("FINTOT_MODEL", "gemma2_2b").strip().lower()
if MODEL_KEY not in MODEL_REGISTRY:
    raise ValueError(
        f"Unknown FINTOT_MODEL={MODEL_KEY!r}. Choose one of {sorted(MODEL_REGISTRY)}"
    )
MODEL_CFG = MODEL_REGISTRY[MODEL_KEY]

EXPERIMENT = os.environ.get("FINTOT_EXPERIMENT", "fintot").strip().lower()
if EXPERIMENT not in {"zero_shot", "grpo_partial", "lr_disentangle", "fintot"}:
    raise ValueError("FINTOT_EXPERIMENT must be zero_shot, grpo_partial, lr_disentangle, or fintot")

TRAIN_CSV = os.environ.get("FINTOT_TRAIN_CSV", "./data/train.csv")
TEST_CSV = os.environ.get("FINTOT_TEST_CSV", "./data/test.csv")
OUTPUT_ROOT = os.environ.get("FINTOT_OUTPUT_ROOT", "./outputs")
SFT_ADAPTER_ROOT = os.environ.get("FINTOT_ADAPTER_ROOT", "./adapters")
SFT_ADAPTER_HINT = os.environ.get("FINTOT_ADAPTER_HINT", MODEL_CFG["adapter_hint"])
EXPECTED_SFT_BASE_MODEL = MODEL_CFG["model_id"]

LAYERS = os.environ.get("FINTOT_LAYERS", MODEL_CFG["layers"])
LAMBDA_SUPPRESS = float(os.environ.get("FINTOT_LAMBDA", "0.5"))
SUBSPACE_RANK = int(os.environ.get("FINTOT_SUBSPACE_RANK", "5"))
CALIBRATION_GROUPS = int(os.environ.get("FINTOT_CALIBRATION_GROUPS", "256"))
REBUILD_SUBSPACE = os.environ.get("FINTOT_REBUILD_SUBSPACE", "0") == "1"

SEMANTIC_THRESHOLD = float(os.environ.get("FINTOT_SEMANTIC_THRESHOLD", "0.80"))
ANSWER_EMERGENCE_PROB = float(os.environ.get("FINTOT_ANSWER_EMERGENCE_PROB", "0.50"))
LATE_LANGUAGE_LAYERS = int(os.environ.get("FINTOT_LATE_LANGUAGE_LAYERS", "4"))

MAX_PROMPT_LENGTH = int(os.environ.get("FINTOT_MAX_PROMPT_LENGTH", "2048"))
MAX_NEW_TOKENS = int(os.environ.get("FINTOT_MAX_NEW_TOKENS", "256"))
EVAL_TEMPERATURE = float(os.environ.get("FINTOT_EVAL_TEMPERATURE", "0.0"))
PREVIEW_PER_LANGUAGE = int(os.environ.get("FINTOT_PREVIEW_PER_LANGUAGE", "10"))
MAX_TEST_SAMPLES = None

LEARNING_RATE = float(os.environ.get("FINTOT_LEARNING_RATE", "1e-5"))
EPOCHS = float(os.environ.get("FINTOT_EPOCHS", "3"))
TRAIN_BATCH_SIZE = int(os.environ.get("FINTOT_TRAIN_BATCH_SIZE", "2"))
GRAD_ACCUM = int(os.environ.get("FINTOT_GRAD_ACCUM", "8"))
NUM_GENERATIONS = int(os.environ.get("FINTOT_NUM_GENERATIONS", "2"))
MAX_COMPLETION_LENGTH = int(os.environ.get("FINTOT_MAX_COMPLETION_LENGTH", "256"))
LOGGING_STEPS = int(os.environ.get("FINTOT_LOGGING_STEPS", "10"))

LORA_R = int(os.environ.get("FINTOT_LORA_R", "8"))
LORA_ALPHA = int(os.environ.get("FINTOT_LORA_ALPHA", "32"))
LORA_DROPOUT = float(os.environ.get("FINTOT_LORA_DROPOUT", "0.10"))

W_FORMAT = float(os.environ.get("FINTOT_W_FORMAT", "0.05"))
W_ACCURACY = float(os.environ.get("FINTOT_W_ACCURACY", "0.35"))
W_PARTIAL = float(os.environ.get("FINTOT_W_PARTIAL", "0.10"))
W_REASON = float(os.environ.get("FINTOT_W_REASON", "0.20"))
W_LANGUAGE = float(os.environ.get("FINTOT_W_LANGUAGE", "0.10"))
W_PRESERVE = float(os.environ.get("FINTOT_W_PRESERVE", "0.20"))
PARTIAL_REWARD_STYLE = os.environ.get("FINTOT_PARTIAL_REWARD_STYLE", "legacy")
SEED = int(os.environ.get("FINTOT_SEED", "42"))

MECH_PROBE_GROUPS = int(os.environ.get("FINTOT_MECH_PROBE_GROUPS", "96"))
CAUSAL_EVAL_SAMPLES = int(os.environ.get("FINTOT_CAUSAL_EVAL_SAMPLES", "240"))
RUN_LAMBDA_SWEEP = os.environ.get("FINTOT_RUN_LAMBDA_SWEEP", "1") == "1"
LAMBDA_SWEEP_VALUES = [0.25, 0.5, 0.7, 1.0]

print("FinToT anonymous configuration loaded")
print("Model      :", MODEL_KEY, "->", MODEL_CFG["model_id"])
print("Experiment :", EXPERIMENT)
print("Train CSV  :", TRAIN_CSV)
print("Test CSV   :", TEST_CSV)
print("Layers     :", LAYERS)
print("Lambda     :", LAMBDA_SUPPRESS)

# ================================================================
# FINToT IMPLEMENTATION
# ================================================================




import os

# Must be set before torch initializes CUDA.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ======================================================================================
# 1. IMPORTS
# ======================================================================================

import gc
import inspect
import json
import math
import random
import re
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
)
from peft import LoraConfig, get_peft_model, PeftConfig, PeftModel


try:
    import Levenshtein

    def normalized_string_similarity(a: str, b: str) -> float:
        return float(Levenshtein.ratio(a, b))

    STRING_SIM_BACKEND = "python-Levenshtein"

except ImportError:
    from difflib import SequenceMatcher

    def normalized_string_similarity(a: str, b: str) -> float:
        return float(SequenceMatcher(None, a, b).ratio())

    STRING_SIM_BACKEND = "difflib.SequenceMatcher"

try:
    from sentence_transformers import SentenceTransformer, util
except ImportError as e:
    raise ImportError(
        "sentence-transformers is required. Install with: pip install sentence-transformers"
    ) from e


# ----------------------------------------------------------------------
# DEVICE / DISTRIBUTED RUNTIME
# ----------------------------------------------------------------------
LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
GLOBAL_RANK = int(os.environ.get("RANK", "0"))
WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))

if torch.cuda.is_available():
    torch.cuda.set_device(LOCAL_RANK)
    DEVICE = torch.device(f"cuda:{LOCAL_RANK}")
else:
    DEVICE = torch.device("cpu")

IS_MAIN_PROCESS = GLOBAL_RANK == 0

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


def main_print(*args, **kwargs):
    if IS_MAIN_PROCESS:
        print(*args, **kwargs)


def detect_attention_backend() -> str:
    """Prefer FlashAttention-2 on A100; fall back to SDPA."""
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except Exception:
        return "sdpa"


ATTENTION_IMPL = detect_attention_backend()


def liger_available() -> bool:
    try:
        import liger_kernel  # noqa: F401
        return True
    except Exception:
        return False


LIGER_AVAILABLE = liger_available()


# ======================================================================================
# 2. CONSTANTS
# ======================================================================================

MODEL_ID = MODEL_CFG["model_id"]

LANGS = ["english", "hindi", "bengali", "marathi", "gujarati", "tamil"]
LANG2ID = {lang: i for i, lang in enumerate(LANGS)}

OPTION_LABEL_MAP = {
    "english":  {"a": "a", "b": "b", "c": "c", "d": "d"},
    "hindi":    {"a": "क", "b": "ख", "c": "ग", "d": "घ"},
    "bengali":  {"a": "ক", "b": "খ", "c": "গ", "d": "ঘ"},
    "marathi":  {"a": "क", "b": "ख", "c": "ग", "d": "घ"},
    "gujarati": {"a": "ક", "b": "ખ", "c": "ગ", "d": "ઘ"},
    "tamil":    {"a": "௧", "b": "௨", "c": "௩", "d": "௪"},
}

NULL_LIKE = {
    "", "idk", "i don't know", "i dont know", "unknown",
    "null", "none", "nan", "na", "n/a", "nil", "missing", "no answer"
}

TAG_RE = re.compile(
    r"<think>(.*?)</think>\s*<answer>(.*?)</answer>",
    flags=re.DOTALL | re.IGNORECASE,
)

ANSWER_ONLY_RE = re.compile(
    r"<answer>(.*?)</answer>",
    flags=re.DOTALL | re.IGNORECASE,
)

SYSTEM_INSTRUCTION = (
    "You are solving a multilingual multiple-choice financial reasoning problem. "
    "Read the question and all four options carefully. "
    "First reason about the problem, then provide exactly one final option. "
    "Use the following output format exactly: "
    "<think> reasoning process </think><answer> option-label. option-text </answer>. "
    "Use the language of the question for the reasoning and final answer."
)

# ======================================================================================
# 3. ARGUMENTS
# ======================================================================================


# ======================================================================================
# 4. BASIC HELPERS
# ======================================================================================

def normalize_string(value) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return re.sub(r"\s+", " ", str(value).strip())


def normalize_lower(value) -> str:
    return normalize_string(value).lower()


def normalize_difficulty(value) -> str:
    x = normalize_lower(value)
    aliases = {
        "easy": "easy", "e": "easy", "simple": "easy",
        "moderate": "moderate", "medium": "moderate", "med": "moderate",
        "m": "moderate", "moderately difficult": "moderate",
        "moderately-difficult": "moderate",
        "hard": "hard", "h": "hard", "difficult": "hard", "challenging": "hard",
    }
    return aliases.get(x, "unknown")


def canonical_gold(language: str, value) -> Optional[str]:
    language = normalize_lower(language)
    raw = normalize_string(value)
    low = raw.lower()

    if low in NULL_LIKE:
        return None

    if low in {"a", "b", "c", "d"}:
        return low

    reverse = {
        str(native).strip(): canonical
        for canonical, native in OPTION_LABEL_MAP.get(language, {}).items()
    }
    if raw in reverse:
        return reverse[raw]

    cleaned = re.sub(r"[\s\.\,\:\;\-\(\)\[\]\{\}]+", "", raw)

    for canonical, native in OPTION_LABEL_MAP.get(language, {}).items():
        native_clean = re.sub(
            r"[\s\.\,\:\;\-\(\)\[\]\{\}]+", "", str(native)
        )
        if cleaned == native_clean:
            return canonical

    if cleaned.lower() in {"a", "b", "c", "d"}:
        return cleaned.lower()

    return None


def parse_layer_spec(spec: str, n_layers: int) -> List[int]:
    layers = []
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            a, b = int(a), int(b)
            if a > b:
                a, b = b, a
            layers.extend(range(a, b + 1))
        else:
            layers.append(int(part))

    layers = sorted(set(layers))
    bad = [x for x in layers if x < 0 or x >= n_layers]
    if bad:
        raise ValueError(
            f"Invalid layer(s) {bad}. Model has {n_layers} decoder layers indexed 0..{n_layers-1}."
        )
    return layers


def experiment_tag(args) -> str:
    if args.experiment in {"lr_disentangle", "fintot"}:
        safe_layers = args.layers.replace(",", "_")
        lam = str(args.lambda_suppress).replace(".", "p")
        return f"{args.experiment}_L{safe_layers}_lambda{lam}"
    return args.experiment


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)



def _normalized_model_id(x: str) -> str:
    return str(x or "").strip().lower().rstrip("/")


def resolve_sft_adapter_dir(
    root: str,
    expected_base_model: str,
    hint: str = "",
) -> Path:
    """
    Find a local PEFT adapter by reading adapter_config.json.

    Search order:
      root/hint/final_adapter
      root/hint
      every child/final_adapter
      every child

    The adapter is accepted only when base_model_name_or_path matches the
    expected base model. This makes a folder name such as `phi-5` harmless
    when the actual adapter config points to microsoft/phi-4.
    """
    root_path = Path(root).expanduser().resolve()

    if not root_path.exists():
        raise FileNotFoundError(
            f"SFT_ADAPTER_ROOT does not exist: {root_path}"
        )

    candidates = []

    if hint:
        hinted = root_path / hint
        candidates.extend([
            hinted / "final_adapter",
            hinted,
        ])

    for child in sorted(root_path.iterdir()):
        if child.is_dir():
            candidates.extend([
                child / "final_adapter",
                child,
            ])

    seen = set()
    discovered = []

    expected = _normalized_model_id(expected_base_model)

    for candidate in candidates:
        try:
            candidate = candidate.resolve()
        except Exception:
            continue

        if candidate in seen:
            continue
        seen.add(candidate)

        cfg_path = candidate / "adapter_config.json"
        if not cfg_path.exists():
            continue

        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception as exc:
            discovered.append(
                f"{candidate} -> unreadable adapter_config.json ({exc})"
            )
            continue

        base = _normalized_model_id(
            cfg.get("base_model_name_or_path", "")
        )

        discovered.append(
            f"{candidate} -> {base or '<missing base_model_name_or_path>'}"
        )

        # Exact match first; basename fallback handles local cached paths.
        if (
            base == expected
            or Path(base).name.lower() == Path(expected).name.lower()
        ):
            print(
                f"[adapter] using SFT adapter: {candidate}\n"
                f"[adapter] base model verified: "
                f"{cfg.get('base_model_name_or_path')}"
            )
            return candidate

    raise RuntimeError(
        "Could not find an SFT adapter whose adapter_config.json matches "
        f"{expected_base_model!r} under {root_path}.\n"
        "Discovered adapters:\n  - "
        + "\n  - ".join(discovered or ["none"])
    )


def load_adapter_tokenizer(adapter_dir: Path, base_model_id: str, **kwargs):
    """
    Prefer tokenizer files saved with the SFT adapter; otherwise use base model.
    """
    try:
        return AutoTokenizer.from_pretrained(
            str(adapter_dir),
            **kwargs,
        )
    except Exception:
        return AutoTokenizer.from_pretrained(
            base_model_id,
            **kwargs,
        )

# ======================================================================================
# 5. DATA LOADING
# ======================================================================================

def validate_and_prepare_df(path: str, require_gold: bool = True) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path.resolve()}")

    df = pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8-sig",
    )

    required = {
        "question", "option_a", "option_b", "option_c", "option_d", "language"
    }
    if require_gold:
        required.add("final_answer")

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing required columns: {sorted(missing)}")

    df["language"] = df["language"].apply(normalize_lower)
    df = df[df["language"].isin(LANGS)].copy()

    if "difficulty_level" in df.columns:
        df["difficulty_level"] = df["difficulty_level"].apply(normalize_difficulty)
    elif "question_level" in df.columns:
        df["difficulty_level"] = df["question_level"].apply(normalize_difficulty)
    else:
        df["difficulty_level"] = "unknown"

    if "gold_letter" not in df.columns:
        df["gold_letter"] = [
            canonical_gold(lang, ans)
            for lang, ans in zip(df["language"], df["final_answer"])
        ]
    else:
        # Revalidate rather than blindly trusting CSV text.
        repaired = []
        for lang, gold, final_answer in zip(
            df["language"], df["gold_letter"], df["final_answer"]
        ):
            g = canonical_gold(lang, gold)
            if g is None:
                g = canonical_gold(lang, final_answer)
            repaired.append(g)
        df["gold_letter"] = repaired

    if require_gold:
        before = len(df)
        df = df[df["gold_letter"].isin(["a", "b", "c", "d"])].copy()
        removed = before - len(df)
        if removed:
            print(f"[data] removed {removed} rows with invalid/null gold labels from {path.name}")

    if "item_id" not in df.columns:
        if "input_id" in df.columns:
            df["item_id"] = df["input_id"].astype(str)
        elif "source_row_index" in df.columns:
            df["item_id"] = df["source_row_index"].astype(str)
        else:
            df["item_id"] = np.arange(len(df)).astype(str)

    if "input_id" not in df.columns:
        df["input_id"] = df["item_id"].astype(str)

    if "comparison_key" not in df.columns:
        df["comparison_key"] = (
            df["input_id"].astype(str) + "::" + df["language"].astype(str)
        )

    df["item_id"] = df["item_id"].astype(str)
    df["input_id"] = df["input_id"].astype(str)
    df["comparison_key"] = df["comparison_key"].astype(str)

    return df.reset_index(drop=True)


def check_train_test_group_leakage(train_df: pd.DataFrame, test_df: pd.DataFrame):
    train_ids = set(train_df["item_id"].astype(str))
    test_ids = set(test_df["item_id"].astype(str))
    overlap = train_ids & test_ids
    if overlap:
        raise ValueError(
            f"Train/test leakage detected: {len(overlap)} item_id groups occur in both splits. "
            f"First few: {sorted(list(overlap))[:10]}"
        )
    print("[data] train/test item_id leakage check PASSED.")


# ======================================================================================
# 6. MULTI-MODEL PROMPTS
# ======================================================================================

def native_label(language: str, canonical: str) -> str:
    return OPTION_LABEL_MAP.get(language, OPTION_LABEL_MAP["english"]).get(
        canonical, canonical
    )


def build_user_text(row: dict) -> str:
    language = normalize_lower(row.get("language", "english"))
    labels = OPTION_LABEL_MAP.get(language, OPTION_LABEL_MAP["english"])

    # For cross-model consistency, the system instruction is prepended to the user turn.
    return (
        f"{SYSTEM_INSTRUCTION}\n\n"
        f"Question: {row.get('question', '')}\n"
        f"Options:\n"
        f"{labels['a']}. {row.get('option_a', '')}\n"
        f"{labels['b']}. {row.get('option_b', '')}\n"
        f"{labels['c']}. {row.get('option_c', '')}\n"
        f"{labels['d']}. {row.get('option_d', '')}\n"
    )


def build_messages(row: dict) -> List[dict]:
    return [{"role": "user", "content": build_user_text(row)}]


def gold_target_text(row: dict) -> str:
    language = normalize_lower(row["language"])
    gold = canonical_gold(language, row.get("gold_letter", row.get("final_answer")))
    if gold is None:
        gold = canonical_gold(language, row.get("final_answer"))
    native = native_label(language, gold)
    option_text = normalize_string(row.get(f"option_{gold}", ""))
    return f"<think> None </think><answer>{native}. {option_text}</answer>"


def format_train_row(row: dict) -> dict:
    return {
        "prompt": build_messages(row),
        "solution": gold_target_text(row),
        "language": normalize_lower(row["language"]),
        "gold_letter": row["gold_letter"],
        "item_id": str(row["item_id"]),
        "input_id": str(row["input_id"]),
        "comparison_key": str(row["comparison_key"]),
        "difficulty_level": row.get("difficulty_level", "unknown"),
        "option_a": row.get("option_a", ""),
        "option_b": row.get("option_b", ""),
        "option_c": row.get("option_c", ""),
        "option_d": row.get("option_d", ""),
    }


# ======================================================================================
# 7. MODEL HELPERS
# ======================================================================================

def load_model():
    """Load the selected model, optionally with the matching SFT PEFT adapter."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for this release script.")

    use_sft_adapter = EXPERIMENT in {"grpo_partial", "lr_disentangle", "fintot"}
    adapter_dir = None
    if use_sft_adapter:
        adapter_dir = resolve_sft_adapter_dir(
            SFT_ADAPTER_ROOT,
            EXPECTED_SFT_BASE_MODEL,
            SFT_ADAPTER_HINT,
        )
        tokenizer = load_adapter_tokenizer(
            adapter_dir,
            MODEL_ID,
            use_fast=True,
            padding_side="left",
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_ID,
            use_fast=True,
            padding_side="left",
            trust_remote_code=False,
        )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    common_kwargs = dict(low_cpu_mem_usage=True, trust_remote_code=False)

    try:
        try:
            base = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                dtype=dtype,
                attn_implementation=ATTENTION_IMPL,
                **common_kwargs,
            )
        except TypeError:
            base = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                torch_dtype=dtype,
                attn_implementation=ATTENTION_IMPL,
                **common_kwargs,
            )
    except Exception as exc:
        main_print(f"[model] {ATTENTION_IMPL!r} failed ({exc}); falling back to SDPA")
        try:
            base = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                dtype=dtype,
                attn_implementation="sdpa",
                **common_kwargs,
            )
        except TypeError:
            base = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                torch_dtype=dtype,
                attn_implementation="sdpa",
                **common_kwargs,
            )

    base = base.to(DEVICE)
    model = base

    if adapter_dir is not None:
        trainable_adapter = EXPERIMENT in {"grpo_partial", "fintot"}
        model = PeftModel.from_pretrained(
            base,
            str(adapter_dir),
            is_trainable=trainable_adapter,
        )
        model.set_adapter("default")
        main_print(
            f"[adapter] loaded {adapter_dir} | trainable={trainable_adapter}"
        )

    model.config.use_cache = False
    if EXPERIMENT in {"grpo_partial", "fintot"}:
        model.train()
    else:
        model.eval()

    main_print(
        f"[model] loaded {MODEL_ID} on {DEVICE} | dtype={dtype} | "
        f"attention={ATTENTION_IMPL} | world_size={WORLD_SIZE}"
    )
    return tokenizer, model

def get_base_causal_lm(model):
    if hasattr(model, "get_base_model"):
        base = model.get_base_model()
        # Some PEFT versions return a wrapper containing `.model`.
        if hasattr(base, "model") and hasattr(base.model, "layers"):
            return base
    return model


def get_decoder_layers(model):
    base = get_base_causal_lm(model)

    # Common causal-LM layout -> .model.layers
    if hasattr(base, "model") and hasattr(base.model, "layers"):
        return base.model.layers

    # Conservative fallback.
    for module in base.modules():
        if isinstance(module, torch.nn.ModuleList) and len(module) > 0:
            first = module[0]
            if hasattr(first, "self_attn"):
                return module

    raise RuntimeError("Could not locate decoder layers for the selected model.")


def get_final_norm(model):
    base = get_base_causal_lm(model)
    if hasattr(base, "model") and hasattr(base.model, "norm"):
        return base.model.norm
    raise RuntimeError("Could not locate the final normalization layer for the selected model.")


def get_lm_head(model):
    base = get_base_causal_lm(model)
    if hasattr(base, "lm_head"):
        return base.lm_head
    raise RuntimeError("Could not locate the LM head for the selected model.")


# ======================================================================================
# 8. TEXT / ANSWER PARSING
# ======================================================================================

def completion_content(completion) -> str:
    """
    TRL conversational completions can be:
      [{"role":"assistant","content":"..."}]
    or a plain string in some versions.
    """
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        return str(completion.get("content", ""))
    if isinstance(completion, list) and completion:
        first = completion[0]
        if isinstance(first, dict):
            return str(first.get("content", ""))
        return str(first)
    return str(completion)


def extract_answer_text(text: str) -> str:
    m = ANSWER_ONLY_RE.search(text or "")
    if m:
        return m.group(1).strip()

    # If think exists but answer tags are missing, use text after </think>.
    if "</think>" in (text or "").lower():
        pos = (text or "").lower().rfind("</think>")
        return (text or "")[pos + len("</think>"):].strip()

    return normalize_string(text)


def parse_predicted_letter(
    output_text: str,
    language: str,
    options: Optional[Dict[str, str]] = None,
) -> Optional[str]:
    language = normalize_lower(language)
    answer = extract_answer_text(output_text)
    low = answer.lower()

    mapping = OPTION_LABEL_MAP.get(language, OPTION_LABEL_MAP["english"])
    reverse = {native: canonical for canonical, native in mapping.items()}

    # 1. Native or canonical label near beginning.
    for canonical in ["a", "b", "c", "d"]:
        native = re.escape(mapping[canonical])
        patterns = [
            rf"^\s*[\(\[]?\s*{native}\s*[\.\)\]\:\-]?",
            rf"^\s*[\(\[]?\s*{canonical}\s*[\.\)\]\:\-]?",
        ]
        for pattern in patterns:
            if re.search(pattern, answer, flags=re.IGNORECASE):
                return canonical

    # 2. Explicit answer cues.
    for canonical in ["a", "b", "c", "d"]:
        native = re.escape(mapping[canonical])
        if re.search(
            rf"(?:answer|option|correct|उत्तर|উত্তর|જવાબ|விடை)[^\n]{{0,25}}(?:{native}|{canonical})\b",
            answer,
            flags=re.IGNORECASE,
        ):
            return canonical

    # 3. Unique option text match.
    if options:
        norm_answer = normalize_lower(answer)
        matches = []
        for canonical, text in options.items():
            norm_opt = normalize_lower(text)
            if norm_opt and (norm_opt == norm_answer or norm_opt in norm_answer):
                matches.append(canonical)
        if len(matches) == 1:
            return matches[0]

    return None


def option_dict_from_row(row: dict) -> Dict[str, str]:
    return {
        "a": row.get("option_a", ""),
        "b": row.get("option_b", ""),
        "c": row.get("option_c", ""),
        "d": row.get("option_d", ""),
    }


# ======================================================================================
# 9. CHAT TEMPLATE HELPERS
# ======================================================================================

def prompt_to_text(tokenizer, prompt) -> str:
    return tokenizer.apply_chat_template(
        prompt,
        tokenize=False,
        add_generation_prompt=True,
    )


def tokenize_text_no_double_specials(tokenizer, text: str, **kwargs):
    return tokenizer(
        text,
        add_special_tokens=False,
        **kwargs,
    )


# ======================================================================================
# 10. BUILD CONTENT-CONTROLLED LANGUAGE SUBSPACE
# ======================================================================================

@torch.inference_mode()
def pooled_prompt_acts_all_layers(
    model,
    tokenizer,
    rows: List[dict],
    max_prompt_length: int,
) -> Dict[int, torch.Tensor]:
    """
    Returns {layer_idx: (B, hidden)} mean-pooled prompt activations on CPU.
    Hidden state index layer+1 corresponds to output of decoder layer `layer`.
    """
    prompt_texts = [
        prompt_to_text(tokenizer, build_messages(row))
        for row in rows
    ]

    enc = tokenize_text_no_double_specials(
        tokenizer,
        prompt_texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_prompt_length,
    )
    enc = {k: v.to(DEVICE) for k, v in enc.items()}

    out = model(
        **enc,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )

    mask = enc["attention_mask"].unsqueeze(-1).to(torch.float32)
    results = {}

    for layer in range(len(out.hidden_states) - 1):
        h = out.hidden_states[layer + 1].float()
        pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        results[layer] = pooled.detach().cpu()

    del out, enc
    torch.cuda.empty_cache()
    return results


def build_language_subspace(
    model,
    tokenizer,
    train_df: pd.DataFrame,
    cache_path: Path,
    max_prompt_length: int,
    calibration_groups: int,
    rank: int,
    seed: int,
):
    """
    Content-controlled language subspace:

      For each parallel semantic group q:
         c_q^l = mean_language h_{q,language}^l
         d_{q,language}^l = h_{q,language}^l - c_q^l

      SVD over d residuals gives a language-associated basis B_l.

    This avoids treating English itself as the universal semantic anchor.
    """
    if cache_path.exists():
        print(f"[subspace] loading cached language subspace: {cache_path}")
        return torch.load(cache_path, map_location="cpu", weights_only=False)

    rng = np.random.default_rng(seed)

    grouped = []
    for item_id, group in train_df.groupby("item_id", sort=False):
        group = group.drop_duplicates("language")
        if group["language"].nunique() >= 2:
            grouped.append((str(item_id), group.copy()))

    if not grouped:
        raise RuntimeError(
            "No multilingual item_id groups were found in the training CSV. "
            "A content-controlled language subspace cannot be built."
        )

    rng.shuffle(grouped)
    grouped = grouped[: min(calibration_groups, len(grouped))]

    print(
        f"[subspace] using {len(grouped)} multilingual calibration groups "
        f"(rank={rank})"
    )

    n_layers = len(get_decoder_layers(model))
    per_layer_groups = {l: [] for l in range(n_layers)}

    model.eval()
    model.config.use_cache = False

    for item_id, group in tqdm(grouped, desc="Building language subspace"):
        rows = group.to_dict("records")

        # One parallel group at a time; max 6 rows.
        acts = pooled_prompt_acts_all_layers(
            model=model,
            tokenizer=tokenizer,
            rows=rows,
            max_prompt_length=max_prompt_length,
        )

        langs_here = [normalize_lower(row["language"]) for row in rows]

        for layer in range(n_layers):
            per_layer_groups[layer].append(
                {
                    "item_id": item_id,
                    "languages": langs_here,
                    "acts": acts[layer],  # (B,d)
                }
            )

    subspace = {
        "model_id": MODEL_ID,
        "languages": LANGS,
        "rank": rank,
        "n_layers": n_layers,
        "mu": {},
        "basis": {},
        "anchors": {},
        "singular_values": {},
        "topr_explained_variance": {},
        "calibration_item_ids": [item_id for item_id, _ in grouped],
        "construction": "content_controlled_parallel_item_residual_svd_v2",
    }

    for layer in range(n_layers):
        all_acts = []
        all_residuals = []
        lang_residuals = {lang: [] for lang in LANGS}

        for record in per_layer_groups[layer]:
            H = record["acts"].float()
            centroid = H.mean(dim=0, keepdim=True)
            residual = H - centroid

            all_acts.append(H)
            all_residuals.append(residual)

            for i, lang in enumerate(record["languages"]):
                if lang in lang_residuals:
                    lang_residuals[lang].append(residual[i])

        A = torch.cat(all_acts, dim=0)
        D = torch.cat(all_residuals, dim=0)

        mu = A.mean(dim=0)

        max_rank = min(rank, D.shape[0] - 1, D.shape[1])
        if max_rank < 1:
            raise RuntimeError(f"Insufficient residual matrix rank at layer {layer}.")

        # CPU SVD. D is small because only calibration groups are used.
        _, S, Vh = torch.linalg.svd(D, full_matrices=False)
        B = Vh[:max_rank].T.contiguous()  # (hidden, rank)

        explained = float(
            (S[:max_rank].pow(2).sum() / S.pow(2).sum().clamp_min(1e-12)).item()
        )

        anchors = {}
        for lang in LANGS:
            if lang_residuals[lang]:
                a = torch.stack(lang_residuals[lang]).mean(dim=0)
                a = F.normalize(a, dim=0)
            else:
                a = torch.zeros_like(mu)
            anchors[lang] = a

        subspace["mu"][layer] = mu.half()
        subspace["basis"][layer] = B.half()
        subspace["singular_values"][layer] = S.half().cpu()
        subspace["topr_explained_variance"][layer] = explained
        subspace["anchors"][layer] = {
            lang: vec.half() for lang, vec in anchors.items()
        }

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    # Atomic save avoids partially-written caches when multiple jobs share storage.
    tmp_path = cache_path.with_suffix(
        cache_path.suffix + f".tmp.rank{GLOBAL_RANK}"
    )
    torch.save(subspace, tmp_path)
    os.replace(tmp_path, cache_path)

    main_print(f"[subspace] saved -> {cache_path}")
    return subspace


# ======================================================================================
# 11. LANGUAGE SUBSPACE OPERATIONS
# ======================================================================================

def projected_language_component(
    h: torch.Tensor,
    mu: torch.Tensor,
    basis: torch.Tensor,
) -> torch.Tensor:
    """
    h: (..., hidden)
    basis: (hidden, rank), orthonormal columns from SVD
    """
    x = h - mu
    return (x @ basis) @ basis.transpose(0, 1)


def language_energy(
    h: torch.Tensor,
    mu: torch.Tensor,
    basis: torch.Tensor,
) -> torch.Tensor:
    x = h - mu
    coeff = x @ basis
    inside = coeff.pow(2).sum(dim=-1)
    total = x.pow(2).sum(dim=-1).clamp_min(1e-8)
    return inside / total


def target_language_margin(
    h: torch.Tensor,
    mu: torch.Tensor,
    anchors: Dict[str, torch.Tensor],
    target_lang: str,
) -> float:
    x = F.normalize((h - mu).float(), dim=-1)

    sims = {}
    for lang, anchor in anchors.items():
        a = F.normalize(anchor.float(), dim=-1)
        sims[lang] = F.cosine_similarity(x, a, dim=-1).item()

    target = sims.get(target_lang, -1.0)
    others = [v for lang, v in sims.items() if lang != target_lang]

    if not others:
        return target

    return target - max(others)



# ======================================================================================
# 11B. SUBSPACE VALIDATION / REPRESENTATION PROBES
# ======================================================================================

def make_matched_random_bases(
    subspace,
    layers: List[int],
    seed: int,
) -> Dict[int, torch.Tensor]:
    """Matched-rank orthonormal random control for causal/selectivity tests."""
    result = {}
    for layer in layers:
        B = subspace["basis"][layer].float().cpu()
        g = torch.Generator(device="cpu")
        g.manual_seed(int(seed) + 104729 * int(layer + 1))
        R = torch.randn(
            B.shape[0],
            B.shape[1],
            generator=g,
            dtype=torch.float32,
        )
        Q, _ = torch.linalg.qr(R, mode="reduced")
        result[layer] = Q.contiguous()
    return result


def suppress_vectors(
    X: torch.Tensor,
    mu: torch.Tensor,
    basis: torch.Tensor,
    strength: float,
) -> torch.Tensor:
    Xf = X.float()
    muf = mu.float().view(1, -1)
    Bf = basis.float()
    centered = Xf - muf
    return Xf - float(strength) * ((centered @ Bf) @ Bf.T)


def _dual_ridge_language_probe(
    X: torch.Tensor,
    labels: List[str],
    item_ids: List[str],
    seed: int,
    alpha: float = 1.0,
) -> float:
    """
    Closed-form ridge probe in the dual space.

    Split is by semantic item_id, not by row, so parallel translations of the
    same question cannot leak from train to test.
    """
    unique_items = sorted(set(map(str, item_ids)))
    if len(unique_items) < 8:
        return float("nan")

    rng = np.random.default_rng(seed)
    perm = np.array(unique_items, dtype=object)
    rng.shuffle(perm)

    n_train_items = max(4, int(round(0.70 * len(perm))))
    train_items = set(map(str, perm[:n_train_items]))
    test_items = set(map(str, perm[n_train_items:]))

    tr = torch.tensor(
        [str(x) in train_items for x in item_ids],
        dtype=torch.bool,
    )
    te = torch.tensor(
        [str(x) in test_items for x in item_ids],
        dtype=torch.bool,
    )

    if tr.sum() < len(LANGS) or te.sum() < len(LANGS):
        return float("nan")

    lang2idx = {lang: i for i, lang in enumerate(LANGS)}
    y = torch.tensor(
        [lang2idx.get(normalize_lower(l), -1) for l in labels],
        dtype=torch.long,
    )

    valid = y >= 0
    tr = tr & valid
    te = te & valid

    Xtr = X[tr].float()
    Xte = X[te].float()
    ytr = y[tr]
    yte = y[te]

    mean = Xtr.mean(dim=0, keepdim=True)
    Xtr = Xtr - mean
    Xte = Xte - mean

    # Scale each sample to avoid probe domination by activation norm.
    Xtr = F.normalize(Xtr, dim=-1)
    Xte = F.normalize(Xte, dim=-1)

    Y = F.one_hot(
        ytr,
        num_classes=len(LANGS),
    ).float()

    # Dual ridge: W = X^T (X X^T + alpha I)^-1 Y
    K = Xtr @ Xtr.T
    eye = torch.eye(K.shape[0], dtype=K.dtype)
    coef = torch.linalg.solve(
        K + float(alpha) * eye,
        Y,
    )
    W = Xtr.T @ coef
    pred = torch.argmax(Xte @ W, dim=-1)

    return float((pred == yte).float().mean().item())


def _mean_parallel_pairwise_cosine(
    X: torch.Tensor,
    item_ids: List[str],
) -> float:
    vals = []
    ids = np.asarray(list(map(str, item_ids)), dtype=object)

    for item in sorted(set(ids.tolist())):
        idx = np.where(ids == item)[0]
        if len(idx) < 2:
            continue

        H = F.normalize(X[idx].float(), dim=-1)
        S = H @ H.T
        tri = torch.triu_indices(
            len(idx),
            len(idx),
            offset=1,
        )
        if tri.shape[1] > 0:
            vals.extend(
                S[tri[0], tri[1]].detach().cpu().tolist()
            )

    return float(np.mean(vals)) if vals else float("nan")


def _english_anchor_retrieval_accuracy(
    X: torch.Tensor,
    labels: List[str],
    item_ids: List[str],
) -> float:
    """
    Cross-lingual semantic retrieval:
    each non-English representation retrieves the matching English parallel
    question among all English candidates.
    """
    ids = list(map(str, item_ids))
    langs = [normalize_lower(x) for x in labels]

    english_indices = [
        i for i, lang in enumerate(langs)
        if lang == "english"
    ]
    if len(english_indices) < 4:
        return float("nan")

    # One English anchor per item.
    anchor_by_item = {}
    for i in english_indices:
        anchor_by_item.setdefault(ids[i], i)

    anchor_items = sorted(anchor_by_item.keys())
    anchor_indices = [anchor_by_item[x] for x in anchor_items]

    A = F.normalize(
        X[anchor_indices].float(),
        dim=-1,
    )

    queries = []
    gold_idx = []
    item_to_anchor = {
        item: j for j, item in enumerate(anchor_items)
    }

    for i, (lang, item) in enumerate(zip(langs, ids)):
        if lang == "english":
            continue
        if item not in item_to_anchor:
            continue
        queries.append(i)
        gold_idx.append(item_to_anchor[item])

    if not queries:
        return float("nan")

    Q = F.normalize(
        X[queries].float(),
        dim=-1,
    )

    sims = Q @ A.T
    pred = torch.argmax(sims, dim=-1)
    gold = torch.tensor(gold_idx, dtype=torch.long)

    return float((pred.cpu() == gold).float().mean().item())


@torch.inference_mode()
def collect_probe_activations(
    model,
    tokenizer,
    train_df: pd.DataFrame,
    subspace,
    max_prompt_length: int,
    probe_groups: int,
    seed: int,
):
    """
    Collect prompt activations on semantic groups NOT used to fit the subspace.
    """
    excluded = set(
        map(str, subspace.get("calibration_item_ids", []))
    )

    grouped = []
    for item_id, group in train_df.groupby("item_id", sort=False):
        if str(item_id) in excluded:
            continue
        group = group.drop_duplicates("language")
        if group["language"].nunique() >= 2:
            grouped.append((str(item_id), group.copy()))

    if not grouped:
        raise RuntimeError(
            "No held-out multilingual groups remain for subspace probing."
        )

    rng = np.random.default_rng(seed + 17)
    rng.shuffle(grouped)
    grouped = grouped[: min(int(probe_groups), len(grouped))]

    n_layers = len(get_decoder_layers(model))
    layer_vectors = {l: [] for l in range(n_layers)}
    labels = []
    item_ids = []

    model.eval()
    old_cache = bool(getattr(model.config, "use_cache", False))
    model.config.use_cache = False

    for item_id, group in tqdm(
        grouped,
        desc="Held-out language-subspace probes",
        disable=not IS_MAIN_PROCESS,
    ):
        rows = group.to_dict("records")
        acts = pooled_prompt_acts_all_layers(
            model=model,
            tokenizer=tokenizer,
            rows=rows,
            max_prompt_length=max_prompt_length,
        )

        local_langs = [
            normalize_lower(r["language"])
            for r in rows
        ]

        labels.extend(local_langs)
        item_ids.extend([str(item_id)] * len(rows))

        for layer in range(n_layers):
            layer_vectors[layer].append(
                acts[layer].half()
            )

    model.config.use_cache = old_cache

    packed = {
        layer: torch.cat(v, dim=0)
        for layer, v in layer_vectors.items()
    }

    return packed, labels, item_ids, [
        item_id for item_id, _ in grouped
    ]


def run_subspace_probe_suite(
    model,
    tokenizer,
    train_df,
    subspace,
    reasoning_layers,
    args,
    output_dir: Path,
    prefix: str,
):
   
    if not IS_MAIN_PROCESS:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)

    packed, labels, item_ids, probe_item_ids = collect_probe_activations(
        model=model,
        tokenizer=tokenizer,
        train_df=train_df,
        subspace=subspace,
        max_prompt_length=args.max_prompt_length,
        probe_groups=args.mech_probe_groups,
        seed=args.seed,
    )

    n_layers = len(packed)
    random_bases = make_matched_random_bases(
        subspace=subspace,
        layers=list(range(n_layers)),
        seed=args.seed + 991,
    )

    rows = []

    for layer in range(n_layers):
        X = packed[layer].float()
        mu = subspace["mu"][layer].float()
        B = subspace["basis"][layer].float()
        Br = random_bases[layer].float()

        X_true = suppress_vectors(
            X, mu, B, args.lambda_suppress
        )
        X_random = suppress_vectors(
            X, mu, Br, args.lambda_suppress
        )

        e_orig = float(
            language_energy(
                X, mu.view(1, -1), B
            ).mean().item()
        )
        e_true = float(
            language_energy(
                X_true, mu.view(1, -1), B
            ).mean().item()
        )
        e_random = float(
            language_energy(
                X_random, mu.view(1, -1), B
            ).mean().item()
        )

        record = {
            "prefix": prefix,
            "layer": layer,
            "is_intervention_layer": layer in reasoning_layers,
            "subspace_rank": int(B.shape[1]),
            "topr_explained_variance": float(
                subspace.get(
                    "topr_explained_variance", {}
                ).get(layer, float("nan"))
            ),
            "language_probe_original": _dual_ridge_language_probe(
                X, labels, item_ids, args.seed + layer
            ),
            "language_probe_true_suppression": _dual_ridge_language_probe(
                X_true, labels, item_ids, args.seed + layer
            ),
            "language_probe_random_suppression": _dual_ridge_language_probe(
                X_random, labels, item_ids, args.seed + layer
            ),
            "parallel_cosine_original": _mean_parallel_pairwise_cosine(
                X, item_ids
            ),
            "parallel_cosine_true_suppression": _mean_parallel_pairwise_cosine(
                X_true, item_ids
            ),
            "parallel_cosine_random_suppression": _mean_parallel_pairwise_cosine(
                X_random, item_ids
            ),
            "semantic_retrieval_original": _english_anchor_retrieval_accuracy(
                X, labels, item_ids
            ),
            "semantic_retrieval_true_suppression": _english_anchor_retrieval_accuracy(
                X_true, labels, item_ids
            ),
            "semantic_retrieval_random_suppression": _english_anchor_retrieval_accuracy(
                X_random, labels, item_ids
            ),
            "language_energy_original": e_orig,
            "language_energy_true_suppression": e_true,
            "language_energy_random_suppression": e_random,
        }

        record["language_probe_drop_true"] = (
            record["language_probe_original"]
            - record["language_probe_true_suppression"]
        )
        record["language_probe_drop_random"] = (
            record["language_probe_original"]
            - record["language_probe_random_suppression"]
        )
        record["semantic_retrieval_delta_true"] = (
            record["semantic_retrieval_true_suppression"]
            - record["semantic_retrieval_original"]
        )
        record["semantic_retrieval_delta_random"] = (
            record["semantic_retrieval_random_suppression"]
            - record["semantic_retrieval_original"]
        )

        rows.append(record)

    probe_df = pd.DataFrame(rows)
    probe_path = output_dir / f"{prefix}_language_subspace_probes.csv"
    probe_df.to_csv(
        probe_path,
        index=False,
        encoding="utf-8-sig",
    )

    spectrum_rows = []
    for layer in range(n_layers):
        S = subspace.get(
            "singular_values", {}
        ).get(layer, torch.tensor([]))
        spectrum_rows.append(
            {
                "prefix": prefix,
                "layer": layer,
                "topr_explained_variance": float(
                    subspace.get(
                        "topr_explained_variance", {}
                    ).get(layer, float("nan"))
                ),
                "singular_values": ";".join(
                    f"{float(x):.6g}"
                    for x in S[: min(20, len(S))].float().tolist()
                ),
            }
        )

    spectrum_df = pd.DataFrame(spectrum_rows)
    spectrum_df.to_csv(
        output_dir / f"{prefix}_subspace_spectrum.csv",
        index=False,
        encoding="utf-8-sig",
    )

    selected = probe_df[
        probe_df["is_intervention_layer"]
    ]

    summary = {
        "prefix": prefix,
        "probe_groups": len(probe_item_ids),
        "mean_language_probe_drop_true": float(
            selected["language_probe_drop_true"].mean()
        ),
        "mean_language_probe_drop_random": float(
            selected["language_probe_drop_random"].mean()
        ),
        "mean_semantic_retrieval_delta_true": float(
            selected["semantic_retrieval_delta_true"].mean()
        ),
        "mean_semantic_retrieval_delta_random": float(
            selected["semantic_retrieval_delta_random"].mean()
        ),
        "mean_parallel_cosine_gain_true": float(
            (
                selected["parallel_cosine_true_suppression"]
                - selected["parallel_cosine_original"]
            ).mean()
        ),
        "mean_energy_drop_true": float(
            (
                selected["language_energy_original"]
                - selected["language_energy_true_suppression"]
            ).mean()
        ),
    }

    pd.DataFrame([summary]).to_csv(
        output_dir / f"{prefix}_subspace_probe_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    main_print(
        f"[probe:{prefix}] true language-probe drop="
        f"{summary['mean_language_probe_drop_true']:.4f} | "
        f"random drop={summary['mean_language_probe_drop_random']:.4f} | "
        f"semantic retrieval Δ="
        f"{summary['mean_semantic_retrieval_delta_true']:+.4f}"
    )

    return {
        "layerwise": probe_df,
        "spectrum": spectrum_df,
        "summary": summary,
        "random_bases": random_bases,
    }


# ======================================================================================
# 12. FIXED LR-SUPPRESSION HOOKS
# ======================================================================================

class LanguageSuppressionController:
   

    def __init__(
        self,
        model,
        subspace,
        layers: List[int],
        strength: float,
        basis_override: Optional[Dict[int, torch.Tensor]] = None,
        name: str = "language",
    ):
        self.model = model
        self.subspace = subspace
        self.layers = list(layers)
        self.strength = float(strength)
        self.basis_override = basis_override or {}
        self.name = str(name)
        self.handles = []
        self._runtime_cache = {}
        self.calls = 0

    def _runtime_tensors(self, layer_idx: int, h: torch.Tensor):
        key = (layer_idx, h.device.type, h.device.index, h.dtype)
        if key in self._runtime_cache:
            return self._runtime_cache[key]

        mu = self.subspace["mu"][layer_idx].float().to(
            device=h.device,
            dtype=h.dtype,
        )

        base_B = self.basis_override.get(
            layer_idx,
            self.subspace["basis"][layer_idx],
        )

        B = base_B.float().to(
            device=h.device,
            dtype=h.dtype,
        )

        self._runtime_cache[key] = (mu, B)
        return mu, B

    def install(self):
        if self.handles:
            return

        decoder_layers = get_decoder_layers(self.model)

        for layer_idx in self.layers:
            module = decoder_layers[layer_idx]
            strength = self.strength

            def make_hook(idx, strength_local):
                def hook(module, inputs, output):
                    h = output[0] if isinstance(output, tuple) else output
                    if not torch.is_floating_point(h):
                        raise RuntimeError(
                            f"Layer {idx} activation is not floating point: {h.dtype}"
                        )

                    mu_local, B_local = self._runtime_tensors(idx, h)
                    x = h - mu_local
                    coeff = torch.matmul(x, B_local)
                    lang_comp = torch.matmul(coeff, B_local.transpose(0, 1))
                    h_new = h - strength_local * lang_comp
                    self.calls += 1

                    if isinstance(output, tuple):
                        return (h_new,) + output[1:]
                    return h_new

                return hook

            self.handles.append(
                module.register_forward_hook(
                    make_hook(layer_idx, strength)
                )
            )

        main_print(
            f"[LR] installed {self.name} suppression hooks on layers "
            f"{self.layers} with lambda={self.strength}"
        )

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles = []
        self._runtime_cache = {}

    @contextmanager
    def disabled(self):
        was_installed = bool(self.handles)
        if was_installed:
            self.remove()
        try:
            yield
        finally:
            if was_installed:
                self.install()


# ======================================================================================
# 13. LORA
# ======================================================================================

def apply_lora(model, args):
    target_modules = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]

    available = {
        name.rsplit(".", 1)[-1]
        for name, _ in model.named_modules()
    }
    target_modules = [m for m in target_modules if m in available]

    if not target_modules:
        raise RuntimeError("No expected LoRA target modules were found for the selected model.")

    config = LoraConfig(
        task_type="CAUSAL_LM",
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )

    model = get_peft_model(model, config)
    model.config.use_cache = False
    model.print_trainable_parameters()

    main_print("[LoRA] target modules:", target_modules)
    return model


# ======================================================================================
# 14. OPTION-LOGIT LENS
# ======================================================================================

def build_label_token_cache(tokenizer) -> Dict[str, Dict[str, int]]:
    cache = {}

    for lang in LANGS:
        cache[lang] = {}
        for canonical, label in OPTION_LABEL_MAP[lang].items():
            # The leading-space version better approximates answer-generation context.
            ids = tokenizer.encode(" " + label, add_special_tokens=False)
            if not ids:
                ids = tokenizer.encode(label, add_special_tokens=False)
            if not ids:
                raise RuntimeError(
                    f"Could not tokenize option label {label!r} for language {lang}."
                )
            cache[lang][canonical] = int(ids[0])

    return cache


def option_distribution_from_hidden(
    model,
    hidden_state: torch.Tensor,
    language: str,
    label_token_cache,
) -> Tuple[Dict[str, float], str, int]:
    """
    Approximate logit-lens distribution over the four option labels.
    Uses only the FIRST token of each native option label.
    """
    language = normalize_lower(language)

    final_norm = get_final_norm(model)
    lm_head = get_lm_head(model)

    h = hidden_state.unsqueeze(0)
    h_norm = final_norm(h)

    ids = [
        label_token_cache[language][canonical]
        for canonical in ["a", "b", "c", "d"]
    ]
    ids_tensor = torch.tensor(ids, device=h_norm.device, dtype=torch.long)

    weight = lm_head.weight.index_select(0, ids_tensor)
    logits = torch.matmul(
        h_norm.float(),
        weight.float().transpose(0, 1),
    ).squeeze(0)

    # Support LM heads with or without bias.
    if getattr(lm_head, "bias", None) is not None:
        logits = logits + lm_head.bias.index_select(0, ids_tensor).float()

    probs = torch.softmax(logits, dim=-1)

    canonical_order = ["a", "b", "c", "d"]
    prob_dict = {
        canonical_order[i]: float(probs[i].item())
        for i in range(4)
    }

    pred = canonical_order[int(torch.argmax(probs).item())]

    return prob_dict, pred, ids[int(torch.argmax(probs).item())]


# ======================================================================================
# 15. REWARD FUNCTIONS
# ======================================================================================

def make_reward_functions(
    model,
    tokenizer,
    subspace,
    reasoning_layers,
    args,
):
    """
    For grpo_partial:
        format + accuracy + partial

    For fintot:
        format + accuracy + partial
        + latent answer-formation reward
        + target-language re-grounding reward
        + late-semantic-preservation reward
    """
    sentence_model = SentenceTransformer(
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        device="cpu",
    )

    label_cache = build_label_token_cache(tokenizer)
    n_layers = len(get_decoder_layers(model))
    late_layers = list(
        range(max(0, n_layers - args.late_language_layers), n_layers)
    )

    def _sequence_arg(value, i, default=None):
        if value is None:
            return default
        if isinstance(value, (list, tuple)):
            return value[i] if i < len(value) else default
        return value

    def format_reward(completions, **kwargs):
        results = []
        pattern = re.compile(
            r"^\s*<think>.*?</think>\s*<answer>.*?</answer>\s*$",
            flags=re.DOTALL | re.IGNORECASE,
        )
        for completion in completions:
            text = completion_content(completion)
            results.append(args.w_format * (1.0 if pattern.match(text) else 0.0))
        return results

    def accuracy_reward(completions, language=None, gold_letter=None, **kwargs):
        rewards = []

        option_a = kwargs.get("option_a")
        option_b = kwargs.get("option_b")
        option_c = kwargs.get("option_c")
        option_d = kwargs.get("option_d")

        for i, completion in enumerate(completions):
            lang = normalize_lower(_sequence_arg(language, i, "english"))
            gold = normalize_lower(_sequence_arg(gold_letter, i, ""))

            options = {
                "a": _sequence_arg(option_a, i, ""),
                "b": _sequence_arg(option_b, i, ""),
                "c": _sequence_arg(option_c, i, ""),
                "d": _sequence_arg(option_d, i, ""),
            }
            pred = parse_predicted_letter(
                completion_content(completion),
                language=lang,
                options=options,
            )
            rewards.append(args.w_accuracy * (1.0 if pred == gold else 0.0))

        return rewards

    def partial_reward(completions, solution=None, **kwargs):
        pred_answers = [
            extract_answer_text(completion_content(c))
            for c in completions
        ]

        gold_answers = []
        for i in range(len(completions)):
            sol = _sequence_arg(solution, i, "")
            gold_answers.append(extract_answer_text(sol))

        pred_emb = sentence_model.encode(
            pred_answers,
            convert_to_tensor=True,
            show_progress_bar=False,
        )
        gold_emb = sentence_model.encode(
            gold_answers,
            convert_to_tensor=True,
            show_progress_bar=False,
        )

        cosines = F.cosine_similarity(pred_emb, gold_emb, dim=-1).cpu().tolist()

        rewards = []
        for pred_text, gold_text, cos in zip(pred_answers, gold_answers, cosines):
            lev = normalized_string_similarity(
                normalize_lower(pred_text),
                normalize_lower(gold_text),
            )
            sim = float((float(cos) + float(lev)) / 2.0)
            sim = max(0.0, min(1.0, sim))

            if args.partial_reward_style == "legacy":
                value = 1.0 if sim >= 0.999999 else sim - 1.0
            else:
                value = sim

            rewards.append(args.w_partial * value)

        return rewards

    def _char_masks_for_full_text(full_text, prompt_char_len, offsets):
        think_positions = []
        answer_positions = []

        completion_text = full_text[prompt_char_len:]
        m = TAG_RE.search(completion_text)

        if m:
            think_span = (
                prompt_char_len + m.start(1),
                prompt_char_len + m.end(1),
            )
            answer_span = (
                prompt_char_len + m.start(2),
                prompt_char_len + m.end(2),
            )

            for idx, (start, end) in enumerate(offsets):
                if end <= start:
                    continue
                if start >= think_span[0] and end <= think_span[1]:
                    think_positions.append(idx)
                if start >= answer_span[0] and end <= answer_span[1]:
                    answer_positions.append(idx)

        if not answer_positions:
            # Fallback: generated tokens.
            generated_positions = [
                i for i, (start, end) in enumerate(offsets)
                if end > start and start >= prompt_char_len
            ]
            if generated_positions:
                answer_positions = [generated_positions[-1]]
                think_positions = generated_positions[:-1] or generated_positions

        decision_pos = (
            max(0, answer_positions[0] - 1)
            if answer_positions
            else max(0, len(offsets) - 1)
        )

        return think_positions, answer_positions, decision_pos

    @torch.no_grad()
    def latent_fintot_reward(
        completions,
        prompts=None,
        language=None,
        gold_letter=None,
        **kwargs,
    ):
        rewards = []

        for i, completion in enumerate(completions):
            prompt = _sequence_arg(prompts, i, None)
            if prompt is None:
                prompt = _sequence_arg(kwargs.get("prompt"), i, None)
            if prompt is None:
                rewards.append(0.0)
                continue

            lang = normalize_lower(_sequence_arg(language, i, "english"))
            gold = normalize_lower(_sequence_arg(gold_letter, i, ""))

            prompt_text = prompt_to_text(tokenizer, prompt)
            completion_text = completion_content(completion)
            full_text = prompt_text + completion_text

            enc = tokenize_text_no_double_specials(
                tokenizer,
                full_text,
                return_tensors="pt",
                truncation=True,
                max_length=args.max_prompt_length + args.max_completion_length,
                return_offsets_mapping=True,
            )
            offsets = enc.pop("offset_mapping")[0].tolist()
            enc = {k: v.to(DEVICE) for k, v in enc.items()}

            out = model(
                **enc,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )

            think_positions, answer_positions, decision_pos = _char_masks_for_full_text(
                full_text,
                len(prompt_text),
                offsets,
            )

            # -------------------------
            # R_reason
            # -------------------------
            reason_probs = []
            for layer in reasoning_layers:
                h = out.hidden_states[layer + 1][0, decision_pos]
                p, _, _ = option_distribution_from_hidden(
                    model, h, lang, label_cache
                )
                reason_probs.append(float(p.get(gold, 0.0)))

            r_reason = max(reason_probs) if reason_probs else 0.0

            # -------------------------
            # R_language
            # -------------------------
            language_margins = []
            positions_for_language = answer_positions or think_positions or [decision_pos]

            for layer in late_layers:
                h = out.hidden_states[layer + 1][0, positions_for_language].float().mean(0)

                mu = subspace["mu"][layer].float().to(h.device)
                anchors = {
                    lg: vec.float().to(h.device)
                    for lg, vec in subspace["anchors"][layer].items()
                }

                margin = target_language_margin(
                    h=h,
                    mu=mu,
                    anchors=anchors,
                    target_lang=lang,
                )
                language_margins.append(margin)

            mean_margin = (
                float(np.mean(language_margins))
                if language_margins else -1.0
            )
            # Smooth 0..1 reward; positive margin should approach 1.
            r_language = float(torch.sigmoid(torch.tensor(5.0 * mean_margin)).item())

            # -------------------------
            # R_preserve (S4 prevention)
            # -------------------------
            # Compare strongest reasoning-region gold probability against the final
            # layer probability at the SAME answer-decision token position.
            h_final = out.hidden_states[-1][0, decision_pos]
            p_final_dict, _, _ = option_distribution_from_hidden(
                model, h_final, lang, label_cache
            )
            p_final_gold = float(p_final_dict.get(gold, 0.0))

            if r_reason >= 0.25:
                semantic_drop = max(0.0, r_reason - p_final_gold)
                r_preserve = max(0.0, 1.0 - semantic_drop)
            else:
                # Do not reward "preservation" if the model never formed a plausible answer.
                r_preserve = 0.0

            total = (
                args.w_reason * r_reason
                + args.w_language * r_language
                + args.w_preserve * r_preserve
            )
            rewards.append(float(total))

            del out, enc
            torch.cuda.empty_cache()

        return rewards

    funcs = [format_reward, accuracy_reward, partial_reward]

    if args.experiment == "fintot":
        funcs.append(latent_fintot_reward)

    return funcs


# ======================================================================================
# 16. GRPO TRAINING
# ======================================================================================

def make_grpo_config(args, output_dir: Path):
    from trl import GRPOConfig

    requested = {
        "output_dir": str(output_dir / "adapter"),
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.epochs,
        "per_device_train_batch_size": args.train_batch_size,
        "gradient_accumulation_steps": args.grad_accum,
        "max_completion_length": args.max_completion_length,
        "num_generations": args.num_generations,
        "max_prompt_length": args.max_prompt_length,
        "logging_steps": args.logging_steps,
        "save_strategy": "epoch",
        "report_to": "none",
        "remove_unused_columns": False,
        "bf16": bool(torch.cuda.is_bf16_supported()),
        "fp16": not bool(torch.cuda.is_bf16_supported()),
        "gradient_checkpointing": True,
    }

    signature = inspect.signature(GRPOConfig.__init__)
    accepted = set(signature.parameters.keys())

    usable = {
        key: value
        for key, value in requested.items()
        if key in accepted
    }

    dropped = sorted(set(requested) - set(usable))
    if dropped:
        print(f"[TRL compatibility] GRPOConfig ignored unsupported args: {dropped}")

    return GRPOConfig(**usable)


def train_with_grpo(
    model,
    tokenizer,
    train_df,
    subspace,
    reasoning_layers,
    args,
    output_dir,
):
    from trl import GRPOTrainer

    if args.train_batch_size * args.grad_accum < args.num_generations:
        raise ValueError(
            "Effective train batch is smaller than num_generations. "
            "Increase --train_batch_size or --grad_accum."
        )

    train_rows = [
        format_train_row(row)
        for row in train_df.to_dict("records")
    ]
    train_dataset = Dataset.from_list(train_rows)

    reward_funcs = make_reward_functions(
        model=model,
        tokenizer=tokenizer,
        subspace=subspace,
        reasoning_layers=reasoning_layers,
        args=args,
    )

    training_args = make_grpo_config(args, output_dir)

    trainer_signature = inspect.signature(GRPOTrainer.__init__)
    kwargs = {
        "model": model,
        "reward_funcs": reward_funcs,
        "args": training_args,
        "train_dataset": train_dataset,
    }

    if "processing_class" in trainer_signature.parameters:
        kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_signature.parameters:
        kwargs["tokenizer"] = tokenizer

    trainer = GRPOTrainer(**kwargs)

    print(f"[train] starting {args.experiment} on {len(train_dataset)} examples")
    trainer.train()

    adapter_dir = output_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    print(f"[train] adapter saved -> {adapter_dir}")
    return trainer.model


# ======================================================================================
# 17. GENERATION
# ======================================================================================

class TokenSuffixStoppingCriteria(StoppingCriteria):
    """Stop when every sequence ends with the tokenized closing answer tag."""
    def __init__(self, suffix_ids: List[int]):
        super().__init__()
        self.suffix_ids = list(map(int, suffix_ids))

    def __call__(self, input_ids, scores, **kwargs):
        if not self.suffix_ids:
            return False
        n = len(self.suffix_ids)
        if input_ids.shape[1] < n:
            return False
        suffix = torch.tensor(
            self.suffix_ids,
            device=input_ids.device,
            dtype=input_ids.dtype,
        )
        matches = (input_ids[:, -n:] == suffix.view(1, -1)).all(dim=1)
        return bool(matches.all().item())


@torch.inference_mode()
def generate_one(
    model,
    tokenizer,
    row: dict,
    max_prompt_length: int,
    max_new_tokens: int,
    temperature: float,
):
    prompt = build_messages(row)
    prompt_text = prompt_to_text(tokenizer, prompt)

    enc = tokenize_text_no_double_specials(
        tokenizer,
        prompt_text,
        return_tensors="pt",
        truncation=True,
        max_length=max_prompt_length,
    )
    enc = {k: v.to(DEVICE) for k, v in enc.items()}

    close_ids = tokenizer.encode(
        "</answer>",
        add_special_tokens=False,
    )

    generation_kwargs = {
        "max_new_tokens": max_new_tokens,
        "pad_token_id": tokenizer.pad_token_id,
        "do_sample": temperature > 0.0,
        "use_cache": True,
        "stopping_criteria": StoppingCriteriaList(
            [TokenSuffixStoppingCriteria(close_ids)]
        ),
    }

    if temperature > 0.0:
        generation_kwargs.update(
            temperature=temperature,
            top_p=0.95,
        )

    generated = model.generate(
        **enc,
        **generation_kwargs,
    )

    new_ids = generated[:, enc["input_ids"].shape[1]:]
    completion = tokenizer.decode(
        new_ids[0],
        skip_special_tokens=True,
    )

    del enc, generated, new_ids

    return prompt, prompt_text, completion



# ======================================================================================
# 17B. CAUSAL INTERVENTION CONTROLS
# ======================================================================================

def _balanced_sample_rows(
    df: pd.DataFrame,
    total_samples: int,
    seed: int,
) -> List[dict]:
    rng = np.random.default_rng(seed)
    per_lang = max(1, int(math.ceil(total_samples / max(1, len(LANGS)))))
    pieces = []

    for lang in LANGS:
        part = df[df["language"] == lang]
        if len(part) == 0:
            continue
        n = min(per_lang, len(part))
        idx = rng.choice(
            len(part),
            size=n,
            replace=False,
        )
        pieces.append(part.iloc[idx])

    if not pieces:
        return []

    sampled = pd.concat(
        pieces,
        ignore_index=True,
    )

    if len(sampled) > total_samples:
        sampled = sampled.sample(
            n=total_samples,
            random_state=seed,
        )

    return sampled.to_dict("records")


@torch.inference_mode()
def run_causal_intervention_generation(
    model,
    tokenizer,
    test_df,
    subspace,
    reasoning_layers,
    args,
    output_dir: Path,
    prefix: str,
    sample_count: int,
    include_lambda_sweep: bool = False,
):
    """
    Causal validation on a fixed balanced held-out subset.

    Controls:
      - no intervention
      - learned language-subspace suppression
      - matched-rank random-subspace suppression
      - language-subspace amplification (opposite-sign intervention)

    If the learned subspace is causal and language-specific, its suppression
    should alter reasoning accuracy more systematically than the random control,
    while amplification should often move behavior in the opposite direction.
    """
    if not IS_MAIN_PROCESS:
        return None

    rows = _balanced_sample_rows(
        test_df,
        total_samples=int(sample_count),
        seed=args.seed + 313,
    )

    if not rows:
        return None

    random_bases = make_matched_random_bases(
        subspace=subspace,
        layers=reasoning_layers,
        seed=args.seed + 2718,
    )

    conditions = [
        ("none", None),
        (
            f"language_suppress_lambda{args.lambda_suppress}",
            dict(
                basis_override=None,
                strength=args.lambda_suppress,
                name="language",
            ),
        ),
        (
            f"random_suppress_lambda{args.lambda_suppress}",
            dict(
                basis_override=random_bases,
                strength=args.lambda_suppress,
                name="random",
            ),
        ),
        (
            f"language_amplify_lambda{args.lambda_suppress}",
            dict(
                basis_override=None,
                strength=-args.lambda_suppress,
                name="language_amplify",
            ),
        ),
    ]

    if include_lambda_sweep:
        for lam in args.lambda_sweep_values:
            lam = float(lam)
            if math.isclose(lam, 0.0):
                continue
            name = f"language_suppress_lambda{lam}"
            if not any(c[0] == name for c in conditions):
                conditions.append(
                    (
                        name,
                        dict(
                            basis_override=None,
                            strength=lam,
                            name=f"language_lambda{lam}",
                        ),
                    )
                )

    records = []

    model.eval()
    old_cache = bool(getattr(model.config, "use_cache", False))
    model.config.use_cache = True

    for condition_name, spec in conditions:
        controller = None

        if spec is not None:
            controller = LanguageSuppressionController(
                model=model,
                subspace=subspace,
                layers=reasoning_layers,
                strength=spec["strength"],
                basis_override=spec["basis_override"],
                name=spec["name"],
            )
            controller.install()

        correct_count = 0

        for row in tqdm(
            rows,
            desc=f"Causal generation: {condition_name}",
        ):
            _, _, completion = generate_one(
                model=model,
                tokenizer=tokenizer,
                row=row,
                max_prompt_length=args.max_prompt_length,
                max_new_tokens=args.max_new_tokens,
                temperature=0.0,
            )

            lang = normalize_lower(row["language"])
            gold = normalize_lower(row["gold_letter"])
            pred = parse_predicted_letter(
                completion,
                language=lang,
                options=option_dict_from_row(row),
            )

            correct = bool(pred == gold)
            correct_count += int(correct)

            records.append(
                {
                    "prefix": prefix,
                    "condition": condition_name,
                    "item_id": str(row["item_id"]),
                    "input_id": str(row["input_id"]),
                    "language": lang,
                    "gold_letter": gold,
                    "pred_letter": pred,
                    "correct": correct,
                    "format_valid": bool(
                        TAG_RE.search(completion or "")
                    ),
                    "completion": completion,
                }
            )

        if controller is not None:
            controller.remove()

        main_print(
            f"[causal:{prefix}] {condition_name}: "
            f"accuracy={correct_count / max(1, len(rows)):.4f}"
        )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    model.config.use_cache = old_cache

    df = pd.DataFrame(records)
    df.to_csv(
        output_dir / f"{prefix}_causal_intervention_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )

    summary = (
        df.groupby(["condition", "language"], dropna=False)
        .agg(
            n=("correct", "size"),
            accuracy=("correct", "mean"),
            format_rate=("format_valid", "mean"),
        )
        .reset_index()
    )

    overall = (
        df.groupby(["condition"], dropna=False)
        .agg(
            n=("correct", "size"),
            accuracy=("correct", "mean"),
            format_rate=("format_valid", "mean"),
        )
        .reset_index()
    )
    overall.insert(1, "language", "ALL")

    summary = pd.concat(
        [overall, summary],
        ignore_index=True,
    )

    summary.to_csv(
        output_dir / f"{prefix}_causal_intervention_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    return {
        "predictions": df,
        "summary": summary,
    }


# ======================================================================================
# 18. FULL-SEQUENCE MECHANISTIC TRACE FOR ONE SAMPLE
# ======================================================================================

def _positions_from_offsets(
    completion_text: str,
    prompt_char_len: int,
    offsets: List[Tuple[int, int]],
):
    m = TAG_RE.search(completion_text)

    think_positions = []
    answer_positions = []

    if m:
        think_span = (
            prompt_char_len + m.start(1),
            prompt_char_len + m.end(1),
        )
        answer_span = (
            prompt_char_len + m.start(2),
            prompt_char_len + m.end(2),
        )

        for idx, (start, end) in enumerate(offsets):
            if end <= start:
                continue
            if start >= think_span[0] and end <= think_span[1]:
                think_positions.append(idx)
            if start >= answer_span[0] and end <= answer_span[1]:
                answer_positions.append(idx)

    generated_positions = [
        idx
        for idx, (start, end) in enumerate(offsets)
        if end > start and start >= prompt_char_len
    ]

    if not think_positions:
        think_positions = generated_positions[:-1] or generated_positions

    if not answer_positions and generated_positions:
        answer_positions = [generated_positions[-1]]

    if answer_positions:
        decision_pos = max(0, answer_positions[0] - 1)
    elif generated_positions:
        decision_pos = max(0, generated_positions[0] - 1)
    else:
        decision_pos = max(0, len(offsets) - 1)

    return think_positions, answer_positions, generated_positions, decision_pos


@torch.inference_mode()
def analyze_one_sample(
    model,
    tokenizer,
    subspace,
    row: dict,
    prompt_text: str,
    completion_text: str,
    label_cache,
    args,
):
    full_text = prompt_text + completion_text

    prompt_ids = tokenize_text_no_double_specials(
        tokenizer,
        prompt_text,
        truncation=True,
        max_length=args.max_prompt_length,
    )["input_ids"]
    prompt_token_count = len(prompt_ids)

    enc = tokenize_text_no_double_specials(
        tokenizer,
        full_text,
        return_tensors="pt",
        truncation=True,
        max_length=args.max_prompt_length + args.max_new_tokens,
        return_offsets_mapping=True,
    )

    offsets = enc.pop("offset_mapping")[0].tolist()
    enc = {k: v.to(DEVICE) for k, v in enc.items()}

    out = model(
        **enc,
        output_hidden_states=True,
        use_cache=False,
        return_dict=True,
    )

    think_positions, answer_positions, generated_positions, decision_pos = (
        _positions_from_offsets(
            completion_text=completion_text,
            prompt_char_len=len(prompt_text),
            offsets=offsets,
        )
    )

    # Guard against truncation.
    seq_len = out.hidden_states[0].shape[1]
    prompt_positions = list(range(min(prompt_token_count, seq_len)))
    think_positions = [p for p in think_positions if p < seq_len]
    answer_positions = [p for p in answer_positions if p < seq_len]
    decision_pos = min(decision_pos, seq_len - 1)

    language = normalize_lower(row["language"])
    gold = normalize_lower(row["gold_letter"])

    n_layers = len(out.hidden_states) - 1

    prompt_vectors = {}
    layer_rows = []

    for layer in range(n_layers):
        H = out.hidden_states[layer + 1][0].float()

        prompt_vec = (
            H[prompt_positions].mean(dim=0)
            if prompt_positions
            else H[0]
        )

        think_vec = (
            H[think_positions].mean(dim=0)
            if think_positions
            else H[decision_pos]
        )

        answer_vec = (
            H[answer_positions].mean(dim=0)
            if answer_positions
            else H[-1]
        )

        prompt_vectors[layer] = prompt_vec.detach().cpu().half()

        mu = subspace["mu"][layer].float().to(H.device)
        B = subspace["basis"][layer].float().to(H.device)
        anchors = {
            lang: vec.float().to(H.device)
            for lang, vec in subspace["anchors"][layer].items()
        }

        prompt_energy = float(
            language_energy(prompt_vec, mu, B).item()
        )
        think_energy = float(
            language_energy(think_vec, mu, B).item()
        )
        answer_energy = float(
            language_energy(answer_vec, mu, B).item()
        )

        lang_margin = float(
            target_language_margin(
                answer_vec,
                mu,
                anchors,
                target_lang=language,
            )
        )

        probs, layer_pred, _ = option_distribution_from_hidden(
            model=model,
            hidden_state=H[decision_pos],
            language=language,
            label_token_cache=label_cache,
        )

        ordered = sorted(
            probs.items(),
            key=lambda x: x[1],
            reverse=True,
        )
        rank = 1 + [x[0] for x in ordered].index(gold)

        layer_rows.append(
            {
                "layer": layer,
                "prompt_language_energy": prompt_energy,
                "think_language_energy": think_energy,
                "answer_language_energy": answer_energy,
                "target_language_margin": lang_margin,
                "gold_option_prob": float(probs[gold]),
                "gold_option_rank": int(rank),
                "gold_is_top1": bool(layer_pred == gold),
                "layer_pred_letter": layer_pred,
            }
        )

    del out, enc
    torch.cuda.empty_cache()

    return {
        "prompt_vectors": prompt_vectors,
        "layer_rows": layer_rows,
    }


# ======================================================================================
# 19. FAILURE CLASSIFICATION
# ======================================================================================

def first_stable_answer_emergence(
    layer_rows: List[dict],
    reasoning_end: int,
    prob_threshold: float,
) -> Optional[int]:
    """
    Preferred criterion:
      correct option is top-1 for two consecutive layers.

    Fallback:
      correct-option probability >= configured threshold.
    """
    rows = [
        row for row in layer_rows
        if row["layer"] <= reasoning_end
    ]

    for a, b in zip(rows[:-1], rows[1:]):
        if (
            a["gold_is_top1"]
            and b["gold_is_top1"]
            and b["layer"] == a["layer"] + 1
        ):
            return int(a["layer"])

    for row in rows:
        if row["gold_option_prob"] >= prob_threshold:
            return int(row["layer"])

    return None


def classify_failures(
    sample_record: dict,
    layer_rows: List[dict],
    reasoning_layers: List[int],
    late_layers: List[int],
    semantic_threshold: float,
    answer_emergence_prob: float,
):
    reasoning_end = max(reasoning_layers)

    # S1: no semantic alignment before/end of reasoning region.
    semantic_candidates = [
        row for row in layer_rows
        if row["layer"] <= reasoning_end
    ]

    first_alignment = None

    # Prefer 2-layer sustained alignment.
    for a, b in zip(semantic_candidates[:-1], semantic_candidates[1:]):
        if (
            a["semantic_similarity"] >= semantic_threshold
            and b["semantic_similarity"] >= semantic_threshold
            and b["layer"] == a["layer"] + 1
        ):
            first_alignment = int(a["layer"])
            break

    if first_alignment is None:
        for row in semantic_candidates:
            if row["semantic_similarity"] >= semantic_threshold:
                first_alignment = int(row["layer"])
                break

    s1 = first_alignment is None

    emergence = first_stable_answer_emergence(
        layer_rows=layer_rows,
        reasoning_end=reasoning_end,
        prob_threshold=answer_emergence_prob,
    )

    # S2: aligned, but answer never emerges.
    s2 = (not s1) and (emergence is None)

    # S3: answer emerged, but target-language margin does not dominate late.
    late_margins = [
        row["target_language_margin"]
        for row in layer_rows
        if row["layer"] in late_layers
    ]
    late_margin_mean = float(np.mean(late_margins)) if late_margins else float("nan")

    s3 = (
        emergence is not None
        and np.isfinite(late_margin_mean)
        and late_margin_mean <= 0.0
    )

    # S4: correct answer existed internally but was lost later.
    final_layer_row = layer_rows[-1]
    final_layer_wrong = not bool(final_layer_row["gold_is_top1"])
    generated_wrong = not bool(sample_record["correct"])

    s4_logit = emergence is not None and final_layer_wrong
    s4_generated = emergence is not None and generated_wrong
    s4 = bool(s4_logit or s4_generated)

    flags = []
    if s1:
        flags.append("S1")
    if s2:
        flags.append("S2")
    if s3:
        flags.append("S3")
    if s4:
        flags.append("S4")

    primary = "+".join(flags) if flags else "SUCCESS"

    return {
        "first_semantic_alignment_layer": first_alignment,
        "answer_emergence_layer": emergence,
        "late_target_language_margin": late_margin_mean,
        "S1_semantic_alignment_failure": bool(s1),
        "S2_reasoning_formation_failure": bool(s2),
        "S3_language_regrounding_failure": bool(s3),
        "S4_late_semantic_corruption": bool(s4),
        "S4_logit_corruption": bool(s4_logit),
        "S4_generated_corruption": bool(s4_generated),
        "primary_failure": primary,
    }


def layer_failure_state(
    row: dict,
    first_alignment: Optional[int],
    emergence: Optional[int],
    reasoning_layers: List[int],
    late_layers: List[int],
    semantic_threshold: float,
) -> str:
    layer = row["layer"]

    states = []

    if (
        first_alignment is None
        or (
            layer <= max(reasoning_layers)
            and row["semantic_similarity"] < semantic_threshold
        )
    ):
        states.append("S1")

    if layer in reasoning_layers:
        if emergence is None or layer < emergence:
            states.append("S2")

    if layer in late_layers and row["target_language_margin"] <= 0.0:
        states.append("S3")

    if (
        emergence is not None
        and layer > emergence
        and not row["gold_is_top1"]
    ):
        states.append("S4")

    return "+".join(states) if states else "OK"


# ======================================================================================
# 20. FULL TEST EVALUATION + LAYER-WISE FAILURES
# ======================================================================================

def evaluate_with_mechanistic_failures(
    model,
    tokenizer,
    subspace,
    test_df,
    reasoning_layers,
    args,
    output_dir,
):
    model.eval()
    model.config.use_cache = True

    n_layers = len(get_decoder_layers(model))
    late_layers = list(
        range(max(0, n_layers - args.late_language_layers), n_layers)
    )
    label_cache = build_label_token_cache(tokenizer)

    preview_counts = {lang: 0 for lang in LANGS}

    prediction_records = []
    sample_failure_records = []
    layer_failure_records = []

    # Group by semantic item so prompt representations from parallel languages
    # can be compared without storing the entire dataset's hidden states.
    groups = list(test_df.groupby("item_id", sort=False))

    if args.max_test_samples is not None:
        # A global sample limit for smoke-testing, preserving group integrity as much as possible.
        selected = []
        count = 0
        for item_id, group in groups:
            selected.append((item_id, group))
            count += len(group)
            if count >= args.max_test_samples:
                break
        groups = selected

    print(
        f"[eval] {len(groups)} semantic item groups; "
        f"{sum(len(g) for _, g in groups)} samples"
    )

    for item_id, group in tqdm(groups, desc="Test + layer diagnostics"):
        per_sample = []

        # --------------------------------------------------------------
        # First process all language versions in this semantic group.
        # --------------------------------------------------------------
        for row in group.to_dict("records"):
            language = normalize_lower(row["language"])
            gold = normalize_lower(row["gold_letter"])

            prompt, prompt_text, completion = generate_one(
                model=model,
                tokenizer=tokenizer,
                row=row,
                max_prompt_length=args.max_prompt_length,
                max_new_tokens=args.max_new_tokens,
                temperature=args.eval_temperature,
            )

            pred = parse_predicted_letter(
                completion,
                language=language,
                options=option_dict_from_row(row),
            )
            correct = bool(pred == gold)

            sample_record = {
                "experiment": args.experiment,
                "model_id": MODEL_ID,
                "item_id": str(row["item_id"]),
                "input_id": str(row["input_id"]),
                "comparison_key": str(row["comparison_key"]),
                "language": language,
                "difficulty_level": row.get("difficulty_level", "unknown"),
                "question_type": row.get("question_type", ""),
                "topic": row.get("topic", ""),
                "subject_domain": row.get("subject_domain", ""),
                "question": row.get("question", ""),
                "option_a": row.get("option_a", ""),
                "option_b": row.get("option_b", ""),
                "option_c": row.get("option_c", ""),
                "option_d": row.get("option_d", ""),
                "gold_letter": gold,
                "gold_native": row.get("final_answer", ""),
                "pred_letter": pred,
                "correct": correct,
                "completion": completion,
                "answer_text": extract_answer_text(completion),
                "layers_modified": ",".join(map(str, reasoning_layers)),
                "lambda_suppress": (
                    args.lambda_suppress
                    if args.experiment in {"lr_disentangle", "fintot"}
                    else 0.0
                ),
            }

            analysis = analyze_one_sample(
                model=model,
                tokenizer=tokenizer,
                subspace=subspace,
                row=row,
                prompt_text=prompt_text,
                completion_text=completion,
                label_cache=label_cache,
                args=args,
            )

            per_sample.append(
                {
                    "sample_record": sample_record,
                    "analysis": analysis,
                }
            )

            # Live first-10 preview PER language.
            if preview_counts[language] < args.preview_per_language:
                preview_counts[language] += 1
                print(
                    f"\n[{language.upper()} {preview_counts[language]}/{args.preview_per_language}] "
                    f"id={sample_record['input_id']} "
                    f"diff={sample_record['difficulty_level']} "
                    f"gold={gold} pred={pred} correct={correct}\n"
                    f"Q: {normalize_string(row.get('question'))[:220]}\n"
                    f"OUT: {normalize_string(completion)[:450]}\n"
                )

        # --------------------------------------------------------------
        # Compute semantic centroid for each layer from parallel versions.
        # --------------------------------------------------------------
        for layer in range(n_layers):
            vectors = torch.stack(
                [
                    record["analysis"]["prompt_vectors"][layer].float()
                    for record in per_sample
                ]
            )

            # IMPORTANT: use a leave-one-language-out semantic centroid.
            # Comparing a representation to a centroid that contains itself
            # artificially inflates cross-lingual similarity.
            for sample_idx, (record, vec) in enumerate(zip(per_sample, vectors)):
                if len(vectors) > 1:
                    other_mask = torch.ones(len(vectors), dtype=torch.bool)
                    other_mask[sample_idx] = False
                    semantic_reference = vectors[other_mask].mean(dim=0)
                else:
                    # Extremely rare fallback for an item with only one surviving language.
                    semantic_reference = vectors.mean(dim=0)

                sim = F.cosine_similarity(
                    F.normalize(vec, dim=0),
                    F.normalize(semantic_reference, dim=0),
                    dim=0,
                ).item()

                record["analysis"]["layer_rows"][layer][
                    "semantic_similarity"
                ] = float(sim)

        # --------------------------------------------------------------
        # Classify S1-S4 and emit layer rows.
        # --------------------------------------------------------------
        for record in per_sample:
            sample_record = record["sample_record"]
            layer_rows = record["analysis"]["layer_rows"]

            failures = classify_failures(
                sample_record=sample_record,
                layer_rows=layer_rows,
                reasoning_layers=reasoning_layers,
                late_layers=late_layers,
                semantic_threshold=args.semantic_threshold,
                answer_emergence_prob=args.answer_emergence_prob,
            )

            sample_summary = {
                **sample_record,
                **failures,
            }

            prediction_records.append(sample_record)
            sample_failure_records.append(sample_summary)

            for layer_row in layer_rows:
                layer = layer_row["layer"]

                if layer < min(reasoning_layers):
                    phase = "semantic_alignment"
                elif layer in reasoning_layers:
                    phase = "reasoning_disentanglement"
                else:
                    phase = (
                        "language_regrounding"
                        if layer in late_layers
                        else "post_reasoning"
                    )

                state = layer_failure_state(
                    row=layer_row,
                    first_alignment=failures["first_semantic_alignment_layer"],
                    emergence=failures["answer_emergence_layer"],
                    reasoning_layers=reasoning_layers,
                    late_layers=late_layers,
                    semantic_threshold=args.semantic_threshold,
                )

                layer_failure_records.append(
                    {
                        "experiment": args.experiment,
                        "item_id": sample_record["item_id"],
                        "input_id": sample_record["input_id"],
                        "comparison_key": sample_record["comparison_key"],
                        "language": sample_record["language"],
                        "difficulty_level": sample_record["difficulty_level"],
                        "layer": layer,
                        "phase": phase,
                        **layer_row,
                        "layer_failure_state": state,
                        "final_pred_letter": sample_record["pred_letter"],
                        "gold_letter": sample_record["gold_letter"],
                        "final_correct": sample_record["correct"],
                        "sample_S1": failures["S1_semantic_alignment_failure"],
                        "sample_S2": failures["S2_reasoning_formation_failure"],
                        "sample_S3": failures["S3_language_regrounding_failure"],
                        "sample_S4": failures["S4_late_semantic_corruption"],
                        "primary_failure": failures["primary_failure"],
                    }
                )

    predictions_df = pd.DataFrame(prediction_records)
    sample_fail_df = pd.DataFrame(sample_failure_records)
    layer_df = pd.DataFrame(layer_failure_records)

    # --------------------------------------------------------------
    # Save main tables.
    # --------------------------------------------------------------
    predictions_df.to_csv(
        output_dir / "predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    sample_fail_df.to_csv(
        output_dir / "sample_failure_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    layer_df.to_csv(
        output_dir / "layerwise_failures.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------------
    # Per-language failure files.
    # --------------------------------------------------------------
    for lang in LANGS:
        sample_fail_df[
            sample_fail_df["language"] == lang
        ].to_csv(
            output_dir / f"failures_{lang}.csv",
            index=False,
            encoding="utf-8-sig",
        )

    # --------------------------------------------------------------
    # Language-level sample failure summary.
    # --------------------------------------------------------------
    bool_cols = [
        "S1_semantic_alignment_failure",
        "S2_reasoning_formation_failure",
        "S3_language_regrounding_failure",
        "S4_late_semantic_corruption",
        "correct",
    ]

    language_summary = (
        sample_fail_df
        .groupby("language", dropna=False)
        .agg(
            n=("input_id", "count"),
            accuracy=("correct", "mean"),
            S1_rate=("S1_semantic_alignment_failure", "mean"),
            S2_rate=("S2_reasoning_formation_failure", "mean"),
            S3_rate=("S3_language_regrounding_failure", "mean"),
            S4_rate=("S4_late_semantic_corruption", "mean"),
            mean_alignment_layer=("first_semantic_alignment_layer", "mean"),
            mean_answer_emergence_layer=("answer_emergence_layer", "mean"),
            mean_late_language_margin=("late_target_language_margin", "mean"),
        )
        .reset_index()
    )

    language_summary.to_csv(
        output_dir / "language_failure_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    lang_diff_summary = (
        sample_fail_df
        .groupby(["language", "difficulty_level"], dropna=False)
        .agg(
            n=("input_id", "count"),
            accuracy=("correct", "mean"),
            S1_rate=("S1_semantic_alignment_failure", "mean"),
            S2_rate=("S2_reasoning_formation_failure", "mean"),
            S3_rate=("S3_language_regrounding_failure", "mean"),
            S4_rate=("S4_late_semantic_corruption", "mean"),
            mean_alignment_layer=("first_semantic_alignment_layer", "mean"),
            mean_answer_emergence_layer=("answer_emergence_layer", "mean"),
        )
        .reset_index()
    )

    lang_diff_summary.to_csv(
        output_dir / "language_difficulty_failure_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # --------------------------------------------------------------
    # Layer × language mechanistic summary.
    # --------------------------------------------------------------
    layer_language_summary = (
        layer_df
        .groupby(["language", "layer", "phase"], dropna=False)
        .agg(
            n=("input_id", "count"),
            semantic_similarity=("semantic_similarity", "mean"),
            prompt_language_energy=("prompt_language_energy", "mean"),
            think_language_energy=("think_language_energy", "mean"),
            answer_language_energy=("answer_language_energy", "mean"),
            target_language_margin=("target_language_margin", "mean"),
            gold_option_prob=("gold_option_prob", "mean"),
            gold_top1_rate=("gold_is_top1", "mean"),
        )
        .reset_index()
    )

    layer_language_summary.to_csv(
        output_dir / "layerwise_language_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Convenient language-energy-only table.
    energy_table = layer_language_summary[
        [
            "language",
            "layer",
            "phase",
            "prompt_language_energy",
            "think_language_energy",
            "answer_language_energy",
        ]
    ].copy()

    energy_table.to_csv(
        output_dir / "language_energy_by_layer_language.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Overall accuracy / difficulty.
    overall = pd.DataFrame(
        [
            {
                "experiment": args.experiment,
                "n": len(sample_fail_df),
                "accuracy": float(sample_fail_df["correct"].mean()),
                "S1_rate": float(
                    sample_fail_df["S1_semantic_alignment_failure"].mean()
                ),
                "S2_rate": float(
                    sample_fail_df["S2_reasoning_formation_failure"].mean()
                ),
                "S3_rate": float(
                    sample_fail_df["S3_language_regrounding_failure"].mean()
                ),
                "S4_rate": float(
                    sample_fail_df["S4_late_semantic_corruption"].mean()
                ),
            }
        ]
    )
    overall.to_csv(
        output_dir / "overall_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n=== OVERALL ===")
    print(overall.to_string(index=False))
    print("\n=== PER LANGUAGE ===")
    print(language_summary.round(4).to_string(index=False))
    print("\n=== LANGUAGE x DIFFICULTY ===")
    print(lang_diff_summary.round(4).to_string(index=False))

    return {
        "predictions": predictions_df,
        "sample_failures": sample_fail_df,
        "layerwise": layer_df,
        "language_summary": language_summary,
        "language_difficulty_summary": lang_diff_summary,
        "layer_language_summary": layer_language_summary,
    }



# ================================================================
# RUN THE SELECTED EXPERIMENT


args = SimpleNamespace(
    experiment=EXPERIMENT,
    train_csv=TRAIN_CSV,
    test_csv=TEST_CSV,
    output_root=OUTPUT_ROOT,

    layers=LAYERS,
    lambda_suppress=LAMBDA_SUPPRESS,
    subspace_rank=SUBSPACE_RANK,
    calibration_groups=CALIBRATION_GROUPS,
    rebuild_subspace=REBUILD_SUBSPACE,

    semantic_threshold=SEMANTIC_THRESHOLD,
    answer_emergence_prob=ANSWER_EMERGENCE_PROB,
    late_language_layers=LATE_LANGUAGE_LAYERS,

    max_prompt_length=MAX_PROMPT_LENGTH,
    max_new_tokens=MAX_NEW_TOKENS,
    eval_temperature=EVAL_TEMPERATURE,
    preview_per_language=PREVIEW_PER_LANGUAGE,
    max_test_samples=MAX_TEST_SAMPLES,

    learning_rate=LEARNING_RATE,
    epochs=EPOCHS,
    train_batch_size=TRAIN_BATCH_SIZE,
    grad_accum=GRAD_ACCUM,
    num_generations=NUM_GENERATIONS,
    max_completion_length=MAX_COMPLETION_LENGTH,
    logging_steps=LOGGING_STEPS,

    lora_r=LORA_R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,

    w_format=W_FORMAT,
    w_accuracy=W_ACCURACY,
    w_partial=W_PARTIAL,
    w_reason=W_REASON,
    w_language=W_LANGUAGE,
    w_preserve=W_PRESERVE,

    partial_reward_style=PARTIAL_REWARD_STYLE,

    mech_probe_groups=MECH_PROBE_GROUPS,
    causal_eval_samples=CAUSAL_EVAL_SAMPLES,
    run_lambda_sweep=RUN_LAMBDA_SWEEP,
    lambda_sweep_values=LAMBDA_SWEEP_VALUES,

    seed=SEED,
)

set_seed(args.seed)

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is required.")

print("=" * 100)
print("FinToT anonymous multi-model release")
print("=" * 100)
print("experiment        :", args.experiment)
print("model             :", MODEL_ID)
print("GPU               :", torch.cuda.get_device_name(0))
print("train_csv         :", Path(args.train_csv).resolve())
print("test_csv          :", Path(args.test_csv).resolve())
print("layers requested  :", args.layers)
print("lambda suppress   :", args.lambda_suppress)
print("string similarity :", STRING_SIM_BACKEND)
print("=" * 100)

# ------------------------------------------------------------------
# Load filtered train/test CSVs.
# ------------------------------------------------------------------
train_path = Path(args.train_csv)
test_path = Path(args.test_csv)

if not train_path.exists():
    raise FileNotFoundError(
        f"Training CSV not found:\\n{train_path.resolve()}\\n"
        "Set FINTOT_TRAIN_CSV or place the file at ./data/train.csv."
    )

if not test_path.exists():
    raise FileNotFoundError(
        f"Test CSV not found:\\n{test_path.resolve()}\\n"
        "Set FINTOT_TEST_CSV or place the file at ./data/test.csv."
    )

train_df = validate_and_prepare_df(
    str(train_path),
    require_gold=True,
)

test_df = validate_and_prepare_df(
    str(test_path),
    require_gold=True,
)

check_train_test_group_leakage(
    train_df,
    test_df,
)

print(
    f"[data] train={len(train_df):,} | test={len(test_df):,}\\n"
    f"[data] train languages={train_df['language'].value_counts().to_dict()}\\n"
    f"[data] test languages={test_df['language'].value_counts().to_dict()}"
)

# ------------------------------------------------------------------
# Load selected base model / SFT adapter.
# ------------------------------------------------------------------
tokenizer, model = load_model()

decoder_layers = get_decoder_layers(model)
n_layers = len(decoder_layers)

reasoning_layers = parse_layer_spec(
    args.layers,
    n_layers,
)

print(
    f"[model] decoder layers={n_layers}; "
    f"reasoning/suppression layers={reasoning_layers}"
)

# ------------------------------------------------------------------
# Output directory.
# ------------------------------------------------------------------
output_dir = (
    Path(args.output_root)
    / MODEL_KEY
    / experiment_tag(args)
)

output_dir.mkdir(
    parents=True,
    exist_ok=True,
)

print(
    "[output]",
    output_dir.resolve(),
)

# ------------------------------------------------------------------
# Build/load a common base-model language subspace.
# ------------------------------------------------------------------
subspace_cache = Path(args.output_root) / (
    f"{MODEL_KEY}_sft_content_controlled_v2_language_subspace_"
    f"rank{args.subspace_rank}_groups{args.calibration_groups}.pt"
)

if (
    args.rebuild_subspace
    and subspace_cache.exists()
):
    subspace_cache.unlink()

subspace = build_language_subspace(
    model=model,
    tokenizer=tokenizer,
    train_df=train_df,
    cache_path=subspace_cache,
    max_prompt_length=args.max_prompt_length,
    calibration_groups=args.calibration_groups,
    rank=args.subspace_rank,
    seed=args.seed,
)

# ------------------------------------------------------------------
# LR-only mechanistic validation BEFORE the main test intervention.
# These analyses use held-out training item groups for representation probes
# and a fixed balanced test subset for causal outcome controls.
# ------------------------------------------------------------------
if IS_MAIN_PROCESS and args.experiment in {"lr_disentangle", "fintot"}:
    run_subspace_probe_suite(
        model=model,
        tokenizer=tokenizer,
        train_df=train_df,
        subspace=subspace,
        reasoning_layers=reasoning_layers,
        args=args,
        output_dir=output_dir,
        prefix="lr_base",
    )

    run_causal_intervention_generation(
        model=model,
        tokenizer=tokenizer,
        test_df=test_df,
        subspace=subspace,
        reasoning_layers=reasoning_layers,
        args=args,
        output_dir=output_dir,
        prefix="lr_base",
        sample_count=args.causal_eval_samples,
        include_lambda_sweep=args.run_lambda_sweep,
    )

# ------------------------------------------------------------------
# GRPO continues the existing SFT PEFT adapter; no second LoRA adapter is added.
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# Install language suppression for LR disentangle / FinToT.
# ------------------------------------------------------------------
suppression = None

if args.experiment in {
    "lr_disentangle",
    "fintot",
}:
    suppression = LanguageSuppressionController(
        model=model,
        subspace=subspace,
        layers=reasoning_layers,
        strength=args.lambda_suppress,
    )
    suppression.install()

# ------------------------------------------------------------------
# GRPO training.
# ------------------------------------------------------------------
if args.experiment in {
    "grpo_partial",
    "fintot",
}:
    model = train_with_grpo(
        model=model,
        tokenizer=tokenizer,
        train_df=train_df,
        subspace=subspace,
        reasoning_layers=reasoning_layers,
        args=args,
        output_dir=output_dir,
    )

    # Trainer can wrap / replace the model object.
    # Reinstall the suppression controller on the trained model.
    if args.experiment == "fintot":
        try:
            suppression.remove()
        except Exception:
            pass

        suppression = LanguageSuppressionController(
            model=model,
            subspace=subspace,
            layers=reasoning_layers,
            strength=args.lambda_suppress,
        )
        suppression.install()

# ------------------------------------------------------------------
# Save experiment configuration.
# ------------------------------------------------------------------
resolved_late_layers = list(
    range(
        max(
            0,
            n_layers
            - args.late_language_layers,
        ),
        n_layers,
    )
)

config = vars(args).copy()

config.update(
    {
        "model_key": MODEL_KEY,

        "model_id":
            MODEL_ID,

        "n_decoder_layers":
            n_layers,

        "resolved_reasoning_layers":
            reasoning_layers,

        "late_language_layers_resolved":
            resolved_late_layers,

        "subspace_cache":
            str(
                subspace_cache.resolve()
            ),

        "string_similarity_backend":
            STRING_SIM_BACKEND,

        "failure_definitions": {
            "S1":
                "parallel prompts never cross the leave-one-language-out "
                "semantic similarity threshold before/end reasoning region",

            "S2":
                "semantic alignment occurs but gold option never stably "
                "emerges in the reasoning-region logit lens",

            "S3":
                "answer emerges but late-layer target-language anchor margin <= 0",

            "S4":
                "gold option emerges internally but is lost at final layer "
                "and/or the final generated answer",
        },
    }
)

with open(
    output_dir
    / "experiment_config.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(
        config,
        f,
        ensure_ascii=False,
        indent=2,
    )

# ------------------------------------------------------------------
# Test evaluation + layer-wise S1-S4 diagnostics.
# ------------------------------------------------------------------
results = evaluate_with_mechanistic_failures(
    model=model,
    tokenizer=tokenizer,
    subspace=subspace,
    test_df=test_df,
    reasoning_layers=reasoning_layers,
    args=args,
    output_dir=output_dir,
)



