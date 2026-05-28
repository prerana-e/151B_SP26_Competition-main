"""One-shot patch for submission_pipeline_full.ipynb.

Implements:
- Tier 1: A (stale-sample reset), B (multi-answer few-shot), C (stratified
  sampling), D (per-subpart fallback) — inference-time improvements.
- SFT data prep: MCQ corpus build, synthetic multi-[ANS] corpus build, and an
  SFT trainer cell that consumes all three corpora (math + MCQ + multi-[ANS]).
  All gated by `RUN_BUILD_EXT_MCQ`, `RUN_BUILD_MULTI_ANSWER`, `RUN_SFT` — set
  any/all to True when you want to retrain.

Idempotent: re-running this script overwrites the patched cells with the same
content and inserts each new cell at most once (keyed by cell id).
"""
import json
from pathlib import Path

NB_PATH = Path(__file__).parent / "submission_pipeline_full.ipynb"

# ──────────────────────────────────────────────────────────────────────────────
# Cell 4 (id=8f5e1b10): Config — Tier 1 switches + SFT data switches.
# ──────────────────────────────────────────────────────────────────────────────
CFG_SRC = '''import os, json, re, sys, gc, csv, math, random, time, glob, shutil, hashlib
from pathlib import Path
from typing import Optional, List, Tuple, Iterable
from collections import Counter, defaultdict

BASE_MODEL_ID    = "Qwen/Qwen3-4B-Thinking-2507"

PUBLIC_PATH      = "data/public.jsonl"
PRIVATE_PATH     = "data/private.jsonl"

RESULTS_DIR      = Path("results");  RESULTS_DIR.mkdir(exist_ok=True)
TRAIN_DIR        = Path("training"); TRAIN_DIR.mkdir(exist_ok=True)

EXT_SFT_PATH          = TRAIN_DIR / "external_sft.jsonl"          # single-answer math
EXT_SFT_MCQ_PATH      = TRAIN_DIR / "external_sft_mcq.jsonl"      # MCQ math (NEW)
MULTI_ANSWER_SFT_PATH = TRAIN_DIR / "multi_answer_sft.jsonl"      # synth multi-[ANS] (NEW)
SELF_SAMPLES_PATH     = TRAIN_DIR / "self_samples.jsonl"
DPO_PAIRS_PATH        = TRAIN_DIR / "dpo_pairs.jsonl"
GRPO_PROMPTS_PATH     = TRAIN_DIR / "grpo_prompts.jsonl"
SFT_LORA_DIR          = TRAIN_DIR / "sft_lora"
DPO_LORA_DIR          = TRAIN_DIR / "dpo_lora"
GRPO_LORA_DIR         = TRAIN_DIR / "grpo_lora"
MERGED_SFT_DIR = Path(f"/tmp/{os.environ[\'USER\']}/151b_training/merged_sft")
MERGED_SFT_DIR.parent.mkdir(parents=True, exist_ok=True)
MERGED_FINAL_DIR = Path(f"/tmp/{os.environ[\'USER\']}/151b_training/merged_final")
MERGED_FINAL_DIR.parent.mkdir(parents=True, exist_ok=True)
PRIVATE_SAMPLES_PATH = RESULTS_DIR / "private_samples.jsonl"
PRIVATE_HINTS_PATH   = RESULTS_DIR / "private_hint_samples.jsonl"
PRIVATE_SUBPART_PATH = RESULTS_DIR / "private_subpart_samples.jsonl"
SUBMISSION_CSV       = Path("submission.csv")

# ── Stage switches ──────────────────────────────────────────────────────────
RUN_BUILD_EXT_SFT       = False  # single-answer math corpus (existing)
RUN_BUILD_EXT_MCQ       = False  # MCQ corpus (NEW — flip to build MMLU math subsets)
RUN_BUILD_MULTI_ANSWER  = False  # synthetic multi-[ANS] corpus (NEW — flip to synthesize)
RUN_SFT                 = False  # train QLoRA on all available SFT corpora
RUN_MERGE_SFT           = True
RUN_SELF_SAMPLE         = False
RUN_BUILD_DPO           = False
RUN_DPO                 = False
RUN_GRPO                = False
RUN_MERGE_FINAL         = True
RUN_PRIVATE_INFER       = True
RUN_HINT_ROUND          = True
RUN_SUBPART_FALLBACK    = True   # option D — runs after voting+hint, multi-[ANS] only

# ── External SFT corpora ────────────────────────────────────────────────────
# Math single-answer / CoT — primary reasoning corpus.
EXT_DATASETS = [
    ("AI-MO/NuminaMath-CoT",        None,   50_000),
    ("meta-math/MetaMathQA",         None,   30_000),
    ("hendrycks/competition_math",   None,     None),
    ("openai/gsm8k",                 "main",   None),
]

# MCQ math — bridges the format gap for letter-style \\boxed{A} answers.
# MMLU is a benchmark; using subsets as SFT data is fine because it isn\'t the
# competition\'s evaluation set. Each row: question, choices[4], answer (idx).
EXT_DATASETS_MCQ = [
    ("cais/mmlu", "high_school_mathematics", None),
    ("cais/mmlu", "college_mathematics",     None),
    ("cais/mmlu", "elementary_mathematics",  None),
    ("cais/mmlu", "abstract_algebra",        None),
    ("cais/mmlu", "high_school_statistics",  None),
    ("cais/mmlu", "formal_logic",            None),
    ("cais/mmlu", "high_school_physics",     None),
]

# Synthetic multi-[ANS] — concatenates K∈[MIN,MAX] short single-answer problems
# into one multi-slot problem so the model learns to emit \\boxed{a, b, c}.
N_MULTI_ANSWER         = 8_000
MULTI_ANSWER_MIN_SLOTS = 2
MULTI_ANSWER_MAX_SLOTS = 5

# ── Hyperparameters ─────────────────────────────────────────────────────────
GPU_ID                = "0"
MAX_MODEL_LEN         = 8192
INFER_MAX_MODEL_LEN   = 16384
MAX_NEW_TOKENS_INFER  = 8192
MAX_NEW_TOKENS_SAMPLE = 4096

N_SAMPLES_PER_Q       = 16
N_SAMPLES_SELF        = 6
PROGRESSIVE_HINT_TAU  = 0.5
N_HINT_SAMPLES        = 8
FEW_SHOT_K_MCQ        = 2
FEW_SHOT_K_FREE       = 2

# Option C — stratified inference sampling (precision + exploration).
STRATIFIED_SAMPLING   = True
INFER_T_PRECISION     = 0.3
INFER_T_EXPLORATION   = 0.9
INFER_TEMPERATURE     = 0.7
INFER_TOP_P           = 0.95
INFER_TOP_K           = 20
INFER_CHUNK           = 64

# Option D — per-subpart fallback.
SUBPART_CONF_THRESHOLD = 0.3
N_SUBPART_SAMPLES      = 4
SUBPART_TEMP           = 0.5

SFT_EPOCHS            = 2          # 2 epochs over the combined corpus (bumped from 1)
SFT_LR                = 1e-4
SFT_BSZ               = 1
SFT_GRAD_ACCUM        = 8
SFT_SAVE_STEPS        = 200
SFT_LOG_STEPS         = 20

DPO_EPOCHS            = 1
DPO_LR                = 5e-6
DPO_BETA              = 0.1

GRPO_EPOCHS           = 1
GRPO_LR               = 5e-6
GRPO_NUM_GENERATIONS  = 4
GRPO_BETA             = 0.04

os.environ["CUDA_VISIBLE_DEVICES"]        = GPU_ID
os.environ["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
os.environ["VLLM_USE_DEEP_GEMM"]      = "0"
os.environ["VLLM_DEEP_GEMM_WARMUP"]   = "skip"
os.environ["TOKENIZERS_PARALLELISM"]      = "false"
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

random.seed(0)
print("Config OK.")
'''

# ──────────────────────────────────────────────────────────────────────────────
# Cell 8 (id=c8b0876d): Prompt construction — option B + slot helper for D.
# ──────────────────────────────────────────────────────────────────────────────
PROMPTS_SRC = '''SYSTEM_PROMPT_MATH = (
    "You are an expert mathematician. Reason carefully and step-by-step. "
    "At the very end, output ONLY the final answer wrapped in a single \\\\boxed{}. "
    "If the problem has multiple [ANS] placeholders, output ALL final answers in order, "
    "comma-separated, inside ONE \\\\boxed{}, e.g. \\\\boxed{3, 7, -1}. "
    "Do not add units inside \\\\boxed{} unless the question explicitly asks. "
    "Simplify radicals and fractions where possible."
)
SYSTEM_PROMPT_MCQ = (
    "You are an expert mathematician. Read the problem and the answer choices, reason step-by-step, "
    "then output ONLY the letter of the single best option inside \\\\boxed{}, e.g. \\\\boxed{C}."
)
SYSTEM_PROMPT_SLOT = (
    "You are an expert mathematician. The question has multiple answer slots labeled "
    "[ANS_1], [ANS_2], etc. Solve ONLY the slot you are asked about. Show your reasoning "
    "step-by-step, then output the value for that single slot inside one \\\\boxed{}."
)

def _short(item, lim_q=350, lim_a=80):
    if len(item["question"]) > lim_q: return False
    if item.get("options"):
        return len(str(item.get("answer", ""))) <= 3 and all(len(o) <= lim_a for o in item["options"])
    a = item.get("answer", [])
    if not isinstance(a, list): a = [a]
    return len(a) <= 2 and all(len(str(x)) <= 30 for x in a)

_mcq_short  = [d for d in public_data if d.get("options") and _short(d)]
_free_short = [d for d in public_data if not d.get("options") and _short(d)]
random.Random(7).shuffle(_mcq_short); random.Random(7).shuffle(_free_short)
FEW_SHOT_MCQ  = _mcq_short[:FEW_SHOT_K_MCQ]
FEW_SHOT_FREE = _free_short[:FEW_SHOT_K_FREE]

# Option B — hand-crafted multi-answer exemplars. Used for free-form questions
# that have >= 2 [ANS] placeholders, since the public-set few-shot pool filters
# those out (len(a) <= 2 in _short) and the model needs to see the comma-in-one-
# box format explicitly to keep Case-D dropouts in extract_free_answers low.
MULTI_ANSWER_EXEMPLARS = [
    {
        "question": (
            "A right triangle has legs of length 3 and 4. "
            "Its hypotenuse is [ANS] and its area is [ANS]."
        ),
        "answer": ["5", "6"],
    },
    {
        "question": (
            "Solve the system x + y = 7, x - y = 3. "
            "Then x = [ANS], y = [ANS], and x*y = [ANS]."
        ),
        "answer": ["5", "2", "10"],
    },
]

def _format_choices(options):
    labels = [chr(65 + i) for i in range(len(options))]
    return "\\n".join(f"{l}. {o.strip()}" for l, o in zip(labels, options))

def _exemplar_block(items, is_mcq):
    if not items: return ""
    out = ["Worked examples (study the format, then solve the new question):\\n"]
    for ex in items:
        q = ex["question"].strip()
        if is_mcq:
            out.append(f"Example question:\\n{q}\\n\\nOptions:\\n{_format_choices(ex[\'options\'])}\\n\\nFinal answer: \\\\boxed{{{str(ex[\'answer\']).strip().upper()}}}\\n")
        else:
            a = ex["answer"] if isinstance(ex["answer"], list) else [ex["answer"]]
            out.append(f"Example question:\\n{q}\\n\\nFinal answer: \\\\boxed{{{\', \'.join(str(x) for x in a)}}}\\n")
    out.append("Now solve the NEW question below. Show your reasoning then finish with \\\\boxed{...}.\\n")
    return "\\n".join(out)

def _n_ans(question_text: str) -> int:
    return question_text.count("[ANS]")

def build_messages(item, hints: Optional[List[str]] = None):
    is_mcq = bool(item.get("options"))
    system = SYSTEM_PROMPT_MCQ if is_mcq else SYSTEM_PROMPT_MATH
    if is_mcq:
        exemplars = FEW_SHOT_MCQ
    else:
        # Option B: pick multi-answer exemplars when the question has >= 2 [ANS] slots.
        exemplars = MULTI_ANSWER_EXEMPLARS if _n_ans(item.get("question", "")) >= 2 else FEW_SHOT_FREE
    block  = _exemplar_block(exemplars, is_mcq)
    parts  = [block] if block else []
    parts.append("New question:\\n" + item["question"].strip())
    if is_mcq:
        parts.append("\\nOptions:\\n" + _format_choices(item["options"]))
    if hints:
        parts.append(
            f"\\nHint: previous attempts produced the candidate answer(s) {{{\', \'.join(str(h) for h in hints)}}}. "
            "Reconsider from scratch and either confirm or correct. Final answer inside \\\\boxed{}."
        )
    return [
        {"role": "system", "content": system},
        {"role": "user",   "content": "\\n".join(parts)},
    ]

# Option D helper — re-label each [ANS] as [ANS_k] so subpart prompts can address slot k.
def _enumerate_ans_slots(question: str) -> str:
    parts = question.split("[ANS]")
    n = len(parts) - 1
    if n <= 0: return question
    out = parts[0]
    for k in range(1, n + 1):
        out += f"[ANS_{k}]" + parts[k]
    return out

def build_messages_for_slot(item, slot_idx: int):
    """Prompt the model to answer exactly one slot of a multi-[ANS] question."""
    q_enum = _enumerate_ans_slots(item["question"].strip())
    n = _n_ans(item.get("question", ""))
    user = (
        f"Question (it has {n} answer slots labeled [ANS_1] through [ANS_{n}]):\\n"
        f"{q_enum}\\n\\n"
        f"Solve only [ANS_{slot_idx}]. Output exactly one value inside a single \\\\boxed{{...}}."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT_SLOT},
        {"role": "user",   "content": user},
    ]

def chat_template_prompt(tok, messages):
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
'''

# ──────────────────────────────────────────────────────────────────────────────
# Cell 28 (id=2399d2c5): Private inference — option C stratified sampling.
# ──────────────────────────────────────────────────────────────────────────────
INFER_SRC = '''def _load_samples(path: Path):
    by_id = defaultdict(list)
    if not path.exists(): return by_id
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line); by_id[r["id"]].append(r)
            except Exception: pass
    return by_id

if RUN_PRIVATE_INFER:
    import os
    os.environ["VLLM_USE_DEEP_GEMM"]    = "0"
    os.environ["VLLM_DEEP_GEMM_WARMUP"] = "skip"
    os.environ["VLLM_USE_V1"] = "0"

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    model_path = str(MERGED_FINAL_DIR) if MERGED_FINAL_DIR.exists() else (
        str(MERGED_SFT_DIR) if MERGED_SFT_DIR.exists() else BASE_MODEL_ID)
    use_bnb = (model_path == BASE_MODEL_ID)
    print("Private inference with:", model_path)

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tok.pad_token = tok.eos_token

    by_id = _load_samples(PRIVATE_SAMPLES_PATH)

    # Build per-pass budgets. With stratified sampling we split into a precision
    # pass (low T) and an exploration pass (high T); legacy mode is a single pass.
    if STRATIFIED_SAMPLING:
        n_precision   = N_SAMPLES_PER_Q // 2
        n_exploration = N_SAMPLES_PER_Q - n_precision
        PASSES = [
            ("precision",   n_precision,   INFER_T_PRECISION),
            ("exploration", n_exploration, INFER_T_EXPLORATION),
        ]
    else:
        PASSES = [("legacy", N_SAMPLES_PER_Q, INFER_TEMPERATURE)]

    def _have_for_pass(item, pass_name):
        return sum(1 for r in by_id.get(item["id"], []) if r.get("pass") == pass_name)

    total_todo = 0
    plans = []
    for pass_name, n_target, temp in PASSES:
        todo = [(it, n_target - _have_for_pass(it, pass_name)) for it in private_data
                if _have_for_pass(it, pass_name) < n_target]
        plans.append((pass_name, temp, todo))
        total_todo += len(todo)
    print(f"Per-pass items needing samples: {[(p, len(t)) for p, _, t in plans]}  total={total_todo}")

    if total_todo:
        kw = dict(model=model_path, trust_remote_code=True,
                   gpu_memory_utilization=0.80, max_model_len=INFER_MAX_MODEL_LEN,
                   max_num_seqs=128,
                   max_num_batched_tokens=16384,
                   enable_prefix_caching=True,
                   enforce_eager=True)
        if use_bnb: kw.update(quantization="bitsandbytes", load_format="bitsandbytes")

        import gc, torch
        gc.collect(); torch.cuda.empty_cache()
        llm = LLM(**kw)

        for pass_name, temp, todo in plans:
            if not todo:
                print(f"  [{pass_name} T={temp}] up-to-date.")
                continue
            n_max = max(need for _, need in todo)
            sp = SamplingParams(n=n_max, max_tokens=MAX_NEW_TOKENS_INFER,
                                 temperature=temp, top_p=INFER_TOP_P, top_k=INFER_TOP_K)
            print(f"  [{pass_name} T={temp}] generating up to n={n_max} for {len(todo)} items")
            for start in range(0, len(todo), INFER_CHUNK):
                batch = todo[start:start+INFER_CHUNK]
                prompts = [chat_template_prompt(tok, build_messages(it)) for it, _ in batch]
                outs = llm.generate(prompts, sampling_params=sp)
                with open(PRIVATE_SAMPLES_PATH, "a") as f:
                    for (item, need), out in zip(batch, outs):
                        base_idx = len(by_id[item["id"]])
                        for j, c in enumerate(out.outputs[:need]):
                            rec = {"id": item["id"], "sample_idx": base_idx + j,
                                    "is_mcq": bool(item.get("options")),
                                    "pass": pass_name, "temperature": temp,
                                    "trace": c.text.strip()}
                            by_id[item["id"]].append(rec)
                            f.write(json.dumps(rec) + "\\n")
                print(f"    [{pass_name}] flushed {start+len(batch)}/{len(todo)}")

        del llm; gc.collect()
        try:
            import torch; torch.cuda.empty_cache()
        except Exception: pass
else:
    print("Skipping private inference.")
'''

# ──────────────────────────────────────────────────────────────────────────────
# Subpart fallback (option D) — inserted after cell 73a04b12.
# ──────────────────────────────────────────────────────────────────────────────
SUBPART_MD_SRC = '''## 14b. Per-subpart fallback for multi-[ANS] failures (option D)

For multi-[ANS] questions where the voted answer is empty, has the wrong number
of comma-separated parts, or the slot-wise confidence is below
`SUBPART_CONF_THRESHOLD`, ask the model each slot independently using
`build_messages_for_slot`. Vote per slot and reassemble. Resumable via
`results/private_subpart_samples.jsonl`.'''

SUBPART_CODE_SRC = '''def _needs_subpart(item):
    if item.get("options"): return False
    n = item.get("question", "").count("[ANS]")
    if n < 2: return False
    v = voted.get(item["id"], {})
    key = v.get("key") or ""
    if not key:
        return True
    parts = [p.strip() for p in key.split(",")]
    if len(parts) != n:
        return True
    if v.get("frac", 0.0) < SUBPART_CONF_THRESHOLD:
        return True
    return False

def _load_subpart_samples(path: Path):
    by_key = defaultdict(list)
    if not path.exists(): return by_key
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line); by_key[(r["id"], r["slot"])].append(r["trace"])
            except Exception: pass
    return by_key

if RUN_SUBPART_FALLBACK:
    failing = [it for it in private_data if _needs_subpart(it)]
    print(f"Subpart fallback candidates: {len(failing)}/{len(private_data)}")

    sub_by_key = _load_subpart_samples(PRIVATE_SUBPART_PATH)
    work = []
    for it in failing:
        n = it["question"].count("[ANS]")
        for k in range(1, n + 1):
            need = N_SUBPART_SAMPLES - len(sub_by_key[(it["id"], k)])
            if need > 0:
                work.append((it, k, n, need))
    print(f"Subpart prompts to run: {len(work)}")

    if work:
        import os
        os.environ["VLLM_USE_DEEP_GEMM"]    = "0"
        os.environ["VLLM_DEEP_GEMM_WARMUP"] = "skip"
        os.environ["VLLM_USE_V1"] = "0"
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        model_path = str(MERGED_FINAL_DIR) if MERGED_FINAL_DIR.exists() else (
            str(MERGED_SFT_DIR) if MERGED_SFT_DIR.exists() else BASE_MODEL_ID)
        use_bnb = (model_path == BASE_MODEL_ID)
        tok_sp = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tok_sp.pad_token = tok_sp.eos_token

        kw = dict(model=model_path, trust_remote_code=True,
                   gpu_memory_utilization=0.80, max_model_len=INFER_MAX_MODEL_LEN,
                   max_num_seqs=128, max_num_batched_tokens=16384,
                   enable_prefix_caching=True, enforce_eager=True)
        if use_bnb: kw.update(quantization="bitsandbytes", load_format="bitsandbytes")
        import gc, torch
        gc.collect(); torch.cuda.empty_cache()
        llm = LLM(**kw)
        n_max = max(w[3] for w in work)
        sp = SamplingParams(n=n_max, max_tokens=MAX_NEW_TOKENS_INFER,
                             temperature=SUBPART_TEMP, top_p=INFER_TOP_P, top_k=INFER_TOP_K)
        for start in range(0, len(work), INFER_CHUNK):
            batch = work[start:start+INFER_CHUNK]
            prompts = [chat_template_prompt(tok_sp, build_messages_for_slot(it, k))
                       for it, k, _, _ in batch]
            outs = llm.generate(prompts, sampling_params=sp)
            with open(PRIVATE_SUBPART_PATH, "a") as f:
                for (it, k, n, need), out in zip(batch, outs):
                    for c in out.outputs[:need]:
                        rec = {"id": it["id"], "slot": k, "n_slots": n,
                                "trace": c.text.strip()}
                        sub_by_key[(it["id"], k)].append(c.text.strip())
                        f.write(json.dumps(rec) + "\\n")
            print(f"  subpart flushed {start+len(batch)}/{len(work)}")
        del llm; gc.collect()
        try:
            import torch; torch.cuda.empty_cache()
        except Exception: pass

    # Vote per slot and reassemble. Replace voted entry only if every slot has at
    # least one extracted answer; prefer the subpart result over the old vote when
    # the old vote was empty, had the wrong number of parts, or had lower minimum
    # confidence.
    def _slot_answer(traces):
        c = Counter(); rep = {}
        for t in traces:
            post = _strip_thinking(t) or t
            boxes = _all_boxed_in_order(post) or _all_boxed_in_order(t)
            if not boxes: continue
            a = boxes[-1].strip()
            key = _vote_key(a)
            if not key: continue
            c[key] += 1
            if key not in rep or len(a) < len(rep[key]):
                rep[key] = a
        if not c: return None, 0.0
        top_k, top_n = c.most_common(1)[0]
        return rep[top_k].strip(), top_n / max(1, sum(c.values()))

    n_recovered = 0
    for it in [x for x in private_data if not x.get("options") and x["question"].count("[ANS]") >= 2]:
        n = it["question"].count("[ANS]")
        slot_answers, slot_fracs = [], []
        ok = True
        for k in range(1, n + 1):
            traces = sub_by_key.get((it["id"], k), [])
            if not traces: ok = False; break
            a, frac = _slot_answer(traces)
            if a is None: ok = False; break
            slot_answers.append(a); slot_fracs.append(frac)
        if not ok: continue
        new_key  = ", ".join(slot_answers)
        new_frac = min(slot_fracs)
        v        = voted.get(it["id"], {})
        old_key  = v.get("key", "") or ""
        old_frac = v.get("frac", 0.0)
        old_parts = [p.strip() for p in old_key.split(",")] if old_key else []
        replace = (not old_key) or (len(old_parts) != n) or (new_frac > old_frac)
        if replace:
            voted[it["id"]] = {"key": new_key, "frac": new_frac, "is_mcq": False}
            n_recovered += 1
    print(f"Subpart fallback updated {n_recovered} items.")
'''

# ──────────────────────────────────────────────────────────────────────────────
# MCQ SFT corpus build — inserted after cell 33020eb8.
# ──────────────────────────────────────────────────────────────────────────────
EXT_MCQ_MD_SRC = '''## 5b. Build the MCQ SFT corpus (checkpointed)

Downloads MMLU math/logic/physics subsets and converts each row to a
`{prompt, completion}` pair where the prompt is built with
`build_messages({question, options})` (MCQ system prompt + few-shot) and the
completion ends with `\\boxed{LETTER}`. This is what teaches the SFT model to
emit the letter-form answer required for the competition\'s MCQ slice. Skipped
if `training/external_sft_mcq.jsonl` already exists.'''

EXT_MCQ_CODE_SRC = '''def _mcq_row_to_pair(tok, hf_id, cfg, row):
    if hf_id == "cais/mmlu":
        q       = row.get("question")
        choices = row.get("choices")
        ans_idx = row.get("answer")
        if not q or not choices or ans_idx is None: return None
        try:
            ai = int(ans_idx)
        except Exception:
            return None
        if ai < 0 or ai >= len(choices): return None
        letter = chr(65 + ai)
        item = {"question": q.strip(), "options": [str(c).strip() for c in choices]}
        prompt = chat_template_prompt(tok, build_messages(item))
        sol = (
            "I will analyze each option carefully.\\n\\n"
            f"The correct answer is option {letter}: {item[\'options\'][ai]}.\\n\\n"
            f"\\\\boxed{{{letter}}}"
        )
        return prompt, sol
    return None

if RUN_BUILD_EXT_MCQ and not EXT_SFT_MCQ_PATH.exists():
    from datasets import load_dataset, concatenate_datasets
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    tok.pad_token = tok.eos_token

    n_written = 0
    tmp_path = EXT_SFT_MCQ_PATH.with_suffix(".jsonl.tmp")
    with open(tmp_path, "w") as f:
        for hf_id, cfg, cap in EXT_DATASETS_MCQ:
            print(f"→ {hf_id} (config={cfg}, cap={cap})")
            # MMLU has no \'train\' split; combine the non-test splits so we don\'t
            # train on the canonical test set. Falls back to whatever is available.
            ds = None
            for split in ("auxiliary_train", "dev+validation", "validation", "dev"):
                try:
                    ds = load_dataset(hf_id, cfg, split=split)
                    print(f"  using split={split}, rows={len(ds)}")
                    break
                except Exception:
                    continue
            if ds is None:
                try:
                    ds = load_dataset(hf_id, cfg, split="test")
                    print(f"  fallback to test split (no train available), rows={len(ds)}")
                except Exception as e:
                    print("  skipped:", e); continue
            kept = 0
            for i, row in enumerate(ds):
                if cap and i >= cap: break
                pair = _mcq_row_to_pair(tok, hf_id, cfg, row)
                if not pair: continue
                f.write(json.dumps({"prompt": pair[0], "completion": pair[1],
                                     "src": f"{hf_id}/{cfg}"}) + "\\n")
                kept += 1; n_written += 1
            print(f"  kept {kept}")
    tmp_path.replace(EXT_SFT_MCQ_PATH)
    print(f"Wrote {n_written} MCQ rows -> {EXT_SFT_MCQ_PATH}")
else:
    print("MCQ SFT corpus:", EXT_SFT_MCQ_PATH,
          "present" if EXT_SFT_MCQ_PATH.exists() else "SKIPPED")
'''

# ──────────────────────────────────────────────────────────────────────────────
# Multi-[ANS] synthesis — inserted after the MCQ build cell.
# ──────────────────────────────────────────────────────────────────────────────
MA_SYNTH_MD_SRC = '''## 5c. Synthesize multi-[ANS] SFT data (checkpointed)

Pulls short single-answer problems from GSM8K and NuminaMath-CoT, groups
K ∈ [`MULTI_ANSWER_MIN_SLOTS`, `MULTI_ANSWER_MAX_SLOTS`] of them into a single
multi-part question with `[ANS]` markers, and writes
`{prompt, completion}` rows where the completion ends with
`\\boxed{a, b, c}`. This directly attacks the multi-`[ANS]` format gap: the
public few-shot pool only contains single-answer items, so the SFT signal for
multi-slot output is otherwise zero. Skipped if
`training/multi_answer_sft.jsonl` already exists.'''

MA_SYNTH_CODE_SRC = '''def _extract_final_boxed(sol_text: str) -> Optional[str]:
    if not sol_text or "\\\\boxed{" not in sol_text: return None
    j = sol_text.rfind("\\\\boxed{")
    k = j + len("\\\\boxed{"); depth = 1
    while k < len(sol_text) and depth > 0:
        if   sol_text[k] == "{": depth += 1
        elif sol_text[k] == "}": depth -= 1
        k += 1
    if depth != 0: return None
    return sol_text[j + len("\\\\boxed{"):k - 1].strip()

if RUN_BUILD_MULTI_ANSWER and not MULTI_ANSWER_SFT_PATH.exists():
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    tok.pad_token = tok.eos_token

    # Build a pool of (question, final_answer) pairs from short single-answer items.
    pool = []
    sources = [
        ("openai/gsm8k",         "main", None),
        ("AI-MO/NuminaMath-CoT", None,   30_000),
    ]
    for hf_id, cfg, cap in sources:
        try:
            ds = load_dataset(hf_id, cfg, split="train", streaming=True)
        except Exception as e:
            print(f"  pool skip {hf_id}: {e}"); continue
        kept = 0
        for i, row in enumerate(ds):
            if cap and i >= cap: break
            if hf_id == "openai/gsm8k":
                q = row.get("question") or ""
                a = row.get("answer") or ""
                if "####" not in a: continue
                final = a.rpartition("####")[2].strip()
            else:  # NuminaMath-CoT
                q = row.get("problem") or ""
                final = _extract_final_boxed(row.get("solution") or "")
                if not final: continue
            q = q.strip()
            if not q or not final: continue
            if len(q) > 280 or len(final) > 24: continue   # keep parts short
            if "[ANS]" in q: continue                      # don\'t recurse
            pool.append((q, final))
            kept += 1
        print(f"  pool from {hf_id}: +{kept}, total={len(pool)}")

    rng = random.Random(0)
    rng.shuffle(pool)

    tmp_path = MULTI_ANSWER_SFT_PATH.with_suffix(".jsonl.tmp")
    n_written = 0
    LABELS = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)"]
    with open(tmp_path, "w") as f:
        i = 0
        while n_written < N_MULTI_ANSWER and i + MULTI_ANSWER_MAX_SLOTS <= len(pool):
            k = rng.randint(MULTI_ANSWER_MIN_SLOTS, MULTI_ANSWER_MAX_SLOTS)
            group = pool[i:i + k]; i += k
            labels = LABELS[:k]
            parts_q = []
            for lbl, (q, _) in zip(labels, group):
                parts_q.append(f"{lbl} {q} [ANS]")
            combined_q = (
                "Solve each independent part. Write the final answer for each "
                "where indicated by [ANS], then collect all answers in order.\\n\\n"
                + "\\n\\n".join(parts_q)
            )
            ans_list = [a for _, a in group]
            item = {"question": combined_q, "options": None}
            prompt = chat_template_prompt(tok, build_messages(item))
            sol_lines = [f"For {lbl}, the answer is {a}." for lbl, (_, a) in zip(labels, group)]
            completion = (
                "I will solve each part independently.\\n\\n"
                + "\\n".join(sol_lines)
                + f"\\n\\nFinal answer: \\\\boxed{{{\', \'.join(ans_list)}}}"
            )
            f.write(json.dumps({"prompt": prompt, "completion": completion,
                                 "src": "synthetic/multi-ans"}) + "\\n")
            n_written += 1
    tmp_path.replace(MULTI_ANSWER_SFT_PATH)
    print(f"Wrote {n_written} synthetic multi-[ANS] rows -> {MULTI_ANSWER_SFT_PATH}")
else:
    print("Multi-[ANS] SFT corpus:", MULTI_ANSWER_SFT_PATH,
          "present" if MULTI_ANSWER_SFT_PATH.exists() else "SKIPPED")
'''

# ──────────────────────────────────────────────────────────────────────────────
# Cell 14 (id=2ddfe045): SFT training — load all three corpora.
# ──────────────────────────────────────────────────────────────────────────────
SFT_SRC = '''def _latest_ckpt(dir_path: Path) -> Optional[str]:
    ckpts = sorted(glob.glob(str(dir_path / "checkpoint-*")),
                    key=lambda p: int(p.rsplit("-", 1)[-1]))
    return ckpts[-1] if ckpts else None

if RUN_SFT:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import LoraConfig, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig
    from datasets import load_dataset

    # Load every SFT corpus we have on disk: math single-answer + MCQ + synthetic
    # multi-[ANS]. Each corpus targets a different failure mode of the base model.
    data_files = []
    if EXT_SFT_PATH.exists():          data_files.append(str(EXT_SFT_PATH))
    if EXT_SFT_MCQ_PATH.exists():      data_files.append(str(EXT_SFT_MCQ_PATH))
    if MULTI_ANSWER_SFT_PATH.exists(): data_files.append(str(MULTI_ANSWER_SFT_PATH))
    assert data_files, "No SFT corpus on disk — run sections 5 / 5b / 5c first."
    print("SFT data files:", data_files)
    ds = load_dataset("json", data_files=data_files, split="train")
    ds = ds.shuffle(seed=0)
    print(f"SFT total rows: {len(ds)}")

    tok = AutoTokenizer.from_pretrained(BASE_MODEL_ID, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                              bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID, quantization_config=bnb, device_map="auto", trust_remote_code=True,
    )
    base = prepare_model_for_kbit_training(base)

    lora_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj",
                          "gate_proj","up_proj","down_proj"],
    )

    eos = tok.eos_token
    def fmt(ex): return {"text": ex["prompt"] + ex["completion"] + eos}
    ds = ds.map(fmt, remove_columns=[c for c in ds.column_names if c not in ("prompt","completion")])

    cfg = SFTConfig(
        output_dir=str(SFT_LORA_DIR),
        per_device_train_batch_size=SFT_BSZ,
        gradient_accumulation_steps=SFT_GRAD_ACCUM,
        num_train_epochs=SFT_EPOCHS,
        learning_rate=SFT_LR, lr_scheduler_type="cosine", warmup_ratio=0.03,
        bf16=True, logging_steps=SFT_LOG_STEPS,
        save_strategy="steps", save_steps=SFT_SAVE_STEPS, save_total_limit=3,
        max_length=2048, packing=True, report_to="none",
        gradient_checkpointing=True, optim="adamw_8bit",
        dataset_text_field="text",
    )
    trainer = SFTTrainer(model=base, args=cfg, train_dataset=ds,
                          peft_config=lora_cfg, processing_class=tok)
    resume = _latest_ckpt(SFT_LORA_DIR)
    print("Resuming from", resume) if resume else print("Starting SFT fresh")
    trainer.train(resume_from_checkpoint=resume)
    trainer.save_model(str(SFT_LORA_DIR))
    print("SFT-LoRA saved ->", SFT_LORA_DIR)
    del trainer, base; gc.collect(); torch.cuda.empty_cache()
else:
    print("Skipping SFT.")
'''


def _split_lines(src: str) -> list:
    return src.splitlines(keepends=True)


def main():
    nb = json.loads(NB_PATH.read_text())
    cells = nb["cells"]
    by_id = {c.get("id"): (i, c) for i, c in enumerate(cells)}

    # ── Update existing cells (idempotent rewrites) ─────────────────────────
    for cid, src in [
        ("8f5e1b10", CFG_SRC),
        ("c8b0876d", PROMPTS_SRC),
        ("2399d2c5", INFER_SRC),
        ("2ddfe045", SFT_SRC),
    ]:
        i, c = by_id[cid]
        c["source"] = _split_lines(src)
        c["outputs"] = []
        c["execution_count"] = None

    # ── Insert new cells idempotently ───────────────────────────────────────
    INSERTED_IDS = {
        "subpart-fallback-md", "subpart-fallback-code",
        "ext-mcq-md",          "ext-mcq-code",
        "multi-ans-md",        "multi-ans-code",
    }
    # Remove any prior insertions so re-running re-creates them at the right spot.
    cells[:] = [c for c in cells if c.get("id") not in INSERTED_IDS]

    def _insert_after(anchor_id, new_cells):
        idx = next(i for i, c in enumerate(cells) if c.get("id") == anchor_id)
        for off, nc in enumerate(new_cells):
            cells.insert(idx + 1 + off, nc)

    def _md(cell_id, src):
        return {"cell_type": "markdown", "id": cell_id, "metadata": {},
                 "source": _split_lines(src)}

    def _code(cell_id, src):
        return {"cell_type": "code", "id": cell_id, "metadata": {},
                 "execution_count": None, "outputs": [],
                 "source": _split_lines(src)}

    # MCQ + multi-answer cells go between the math-SFT builder (33020eb8) and
    # the SFT training cell (2ddfe045) so they\'re available when training runs.
    _insert_after("33020eb8", [
        _md  ("ext-mcq-md",   EXT_MCQ_MD_SRC),
        _code("ext-mcq-code", EXT_MCQ_CODE_SRC),
        _md  ("multi-ans-md",   MA_SYNTH_MD_SRC),
        _code("multi-ans-code", MA_SYNTH_CODE_SRC),
    ])

    # Subpart fallback stays after the voting+hint cell.
    _insert_after("73a04b12", [
        _md  ("subpart-fallback-md",   SUBPART_MD_SRC),
        _code("subpart-fallback-code", SUBPART_CODE_SRC),
    ])

    NB_PATH.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n")
    print(f"Patched {NB_PATH} — {len(cells)} cells total.")


if __name__ == "__main__":
    main()
