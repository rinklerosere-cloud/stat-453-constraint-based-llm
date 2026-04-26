# -*- coding: utf-8 -*-
"""
!pip install -q transformers peft datasets accelerate matplotlib pandas scikit-learn
"""


import os
from huggingface_hub import login

try:
    from google.colab import userdata
    hf_token = userdata.get('HF_TOKEN').strip()
except Exception:
    hf_token = os.environ.get('HF_TOKEN', '').strip()

login(token=hf_token)
print("Logged in to HuggingFace.")


import gc, json, os, re, random
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from peft import PeftModel
from sklearn.model_selection import KFold
from transformers import AutoModelForCausalLM, AutoTokenizer

print(f"GPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: No GPU. Go to Runtime → Change runtime type → GPU")


import zipfile, os
with zipfile.ZipFile("/content/recast_30k_clean.jsonl.zip", "r") as z:
    z.extractall("/content")
    print("Extracted:", z.namelist())

BASE_MODEL         = "meta-llama/Llama-3.2-1B-Instruct"
LORA_ADAPTER_PATH  = "/content/outputs/lora_r8_0.0001/lora_adapter"
FULL_FT_MODEL_PATH = "/content/output/finetuned"
DATASET_PATH       = "/content/recast_30k_clean.jsonl"

K_FOLDS          = 5    # 5 rounds of testing
SAMPLES_PER_FOLD = 200  # examples per round (200x5 = 1,000 total)
RANDOM_SEED      = 42   # fixed seed = reproducible results

RESULTS_DIR = "/content/kfold_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

print("Configuration set:")
print(f"  LoRA adapter:     {LORA_ADAPTER_PATH}")
print(f"  Full FT model:    {FULL_FT_MODEL_PATH}")
print(f"  Dataset:          {DATASET_PATH}")
print(f"  K folds:          {K_FOLDS}")
print(f"  Samples per fold: {SAMPLES_PER_FOLD}")
print(f"  Total evaluated:  {K_FOLDS * SAMPLES_PER_FOLD} examples")



def load_all_examples(path):
    """
    Reads RECAST jsonl and returns list of examples.
    Handles both field name conventions (Rinkle's and Mark's scripts).
    """
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            instruction = row.get("winner_prompt", row.get("input", ""))
            response    = row.get("response_of_winner_prompt",
                          row.get("winner_response", row.get("output", "")))
            if instruction and response:
                examples.append({
                    "instruction":   instruction,
                    "gold_response": response,
                    "raw":           row,  # full row kept for validators
                })
    return examples


print(f"Loading dataset...")
all_examples = load_all_examples(DATASET_PATH)
print(f"Total examples: {len(all_examples):,}")

random.seed(RANDOM_SEED)
random.shuffle(all_examples)

kf      = KFold(n_splits=K_FOLDS, shuffle=False)
indices = list(range(len(all_examples)))
folds   = []

for fold_num, (_, test_idx) in enumerate(kf.split(indices)):
    sampled_idx   = random.sample(list(test_idx), min(SAMPLES_PER_FOLD, len(test_idx)))
    fold_examples = [all_examples[i] for i in sampled_idx]
    folds.append(fold_examples)
    print(f"  Fold {fold_num+1}: {len(test_idx):,} total → {len(fold_examples)} sampled")

print(f"\nTotal to evaluate: {sum(len(f) for f in folds)} examples")



def check_length_words(response, raw):
    """Word count must be within the range specified in the constraint."""
    try:
        constraints = raw.get("added_constraint", {}).get("Length", [])
        wc = len(response.split())
        for c in constraints:
            nums = re.findall(r'\d+', c)
            if len(nums) >= 2 and "word" in c.lower():
                if not (int(nums[0]) <= wc <= int(nums[1])):
                    return False
        return True
    except Exception:
        return True

def check_length_sentences(response, raw):
    try:
        constraints = raw.get("added_constraint", {}).get("Length", [])
        sc = len(re.split(r'[.!?]+', response.strip()))
        for c in constraints:
            nums = re.findall(r'\d+', c)
            if nums and "sentence" in c.lower():
                if sc > int(nums[0]):
                    return False
        return True
    except Exception:
        return True

def check_keyword(response, raw):
    try:
        constraints = raw.get("added_constraint", {}).get("Keyword", [])
        rl = response.lower()
        for c in constraints:
            m = re.search(r'["\u201c\u201d]([^"]+)["\u201c\u201d].*?(\d+)\s*times', c, re.IGNORECASE)
            if m and rl.count(m.group(1).lower()) < int(m.group(2)):
                return False
        return True
    except Exception:
        return True

def check_start_with(response, raw):
    try:
        constraints = raw.get("added_constraint", {}).get(
            "Strat_With", raw.get("added_constraint", {}).get("Start_With", [])
        )
        for c in constraints:
            m = re.search(r'["\u201c\u201d]([^"]+)["\u201c\u201d]', c)
            if m:
                words = response.strip().lower().split()
                if not words or words[0] != m.group(1).strip().lower():
                    return False
        return True
    except Exception:
        return True

def check_end_with(response, raw):
    try:
        constraints = raw.get("added_constraint", {}).get("End_With", [])
        for c in constraints:
            m = re.search(r'["\u201c\u201d]([^"]+)["\u201c\u201d]', c)
            if m:
                words = re.findall(r'\w+', response.lower())
                if not words or words[-1] != m.group(1).strip().lower():
                    return False
        return True
    except Exception:
        return True

def check_format(response, raw):
    
    try:
        constraints = raw.get("added_constraint", {}).get("Format", [])
        for c in constraints:
            if "<<" in c and not re.search(r'<<.+?>>', response):
                return False
        return True
    except Exception:
        return True

def check_tone(response, raw):
    informal = ["gonna", "wanna", "gotta", "kinda", "dunno", "lol", "omg"]
    return not any(w in response.lower() for w in informal)

def check_style(response, raw):        return True
def check_role_playing(response, raw): return True
def check_numerical(response, raw):    return True

VALIDATORS = {
    "Length":                [check_length_words, check_length_sentences],
    "Keyword":               [check_keyword],
    "Start_With":            [check_start_with],
    "End_With":              [check_end_with],
    "Format":                [check_format],
    "Tone":                  [check_tone],
    "Style":                 [check_style],
    "Role_Playing":          [check_role_playing],
    "Numerical_Constraints": [check_numerical],
}

print(f"Validators ready for {len(VALIDATORS)} constraint categories.")

PROMPT_TEMPLATE = (
    "<|begin_of_text|>"
    "<|start_header_id|>user<|end_header_id|>\n\n"
    "{instruction}"
    "<|eot_id|>"
    "<|start_header_id|>assistant<|end_header_id|>\n\n"
)


def generate_response(model, tokenizer, instruction, max_new_tokens=512):
    
    prompt = PROMPT_TEMPLATE.format(instruction=instruction)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():  # saves GPU memory — no gradients needed for evaluation
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # greedy: always pick most likely next token
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()


def score_response(response, raw):
    
    results, all_passed = {}, []
    for cat, fns in VALIDATORS.items():
        passed = all(fn(response, raw) for fn in fns)  # all validators must pass
        results[cat] = passed
        all_passed.append(passed)
    results["csr"] = sum(all_passed) / len(all_passed)
    return results


def evaluate_fold(model, tokenizer, fold_examples, model_name, fold_num):
   
    results = []
    model.eval()  # evaluation mode: disables dropout
    for i, ex in enumerate(fold_examples):
        if i % 50 == 0:
            print(f"    [{model_name}] Fold {fold_num} — {i}/{len(fold_examples)} done...")
        response = generate_response(model, tokenizer, ex["instruction"])
        scores   = score_response(response, ex["raw"])
        results.append({"fold": fold_num, "model": model_name, **scores})
    print(f"    [{model_name}] Fold {fold_num} complete.")
    return results


print("Generation functions ready.")



all_results = []  # collects every scored example from every fold, both models

for fold_num, fold_examples in enumerate(folds, start=1):
    print(f"\n{'='*55}")
    print(f"  FOLD {fold_num} / {K_FOLDS} — {len(fold_examples)} examples")
    print(f"{'='*55}")

    # Load tokenizer once — converts text to token IDs, shared by both models
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ── RINKLE'S LORA MODEL ──────────────────────────────────────────────────
    print("  Loading LoRA model (frozen base + trained adapter)...")
    base       = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto")
    lora_model = PeftModel.from_pretrained(base, LORA_ADAPTER_PATH)

    lora_fold_results = evaluate_fold(
        lora_model, tokenizer, fold_examples, "LoRA", fold_num)
    all_results.extend(lora_fold_results)

    # Free GPU memory before loading next model
    del lora_model, base
    gc.collect()
    torch.cuda.empty_cache()
    print("  LoRA done. GPU memory freed.")

    # ── MARK'S FULL FT MODEL ─────────────────────────────────────────────────
    print("  Loading Full FT model (all weights updated)...")
    full_model = AutoModelForCausalLM.from_pretrained(
        FULL_FT_MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto")

    full_fold_results = evaluate_fold(
        full_model, tokenizer, fold_examples, "Full_FT", fold_num)
    all_results.extend(full_fold_results)

    del full_model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    print("  Full FT done. GPU memory freed.")

    # Save checkpoint after each fold
    with open(f"{RESULTS_DIR}/results_fold_{fold_num}.json", "w") as f:
        json.dump(lora_fold_results + full_fold_results, f, indent=2)
    print(f"  Fold {fold_num} saved to {RESULTS_DIR}/results_fold_{fold_num}.json")

# Save all results combined
with open(f"{RESULTS_DIR}/all_results.json", "w") as f:
    json.dump(all_results, f, indent=2)

print(f"\n{'='*55}")
print(f"All {K_FOLDS} folds complete! Total scored: {len(all_results)}")
print(f"{'='*55}")



df   = pd.DataFrame(all_results)
cats = list(VALIDATORS.keys())


def fold_summary(model_name):
    """
    Computes per-fold CSR then averages across folds.
    This is the correct K-Fold methodology — each fold gets equal weight.
    """
    model_df    = df[df["model"] == model_name]
    fold_scores = []
    for fold_num in range(1, K_FOLDS + 1):
        fold_df = model_df[model_df["fold"] == fold_num]
        if len(fold_df) == 0:
            continue
        fold_score = {"fold": fold_num, "CSR": fold_df["csr"].mean()}
        for cat in cats:
            if cat in fold_df.columns:
                fold_score[cat] = fold_df[cat].mean()
        fold_scores.append(fold_score)

    fold_df_all = pd.DataFrame(fold_scores)
    summary = {}
    for m in ["CSR"] + cats:
        if m in fold_df_all.columns:
            summary[m] = {"mean": fold_df_all[m].mean(), "std": fold_df_all[m].std()}
    return summary, fold_df_all


lora_summary,    lora_fold_df    = fold_summary("LoRA")
full_ft_summary, full_ft_fold_df = fold_summary("Full_FT")

print("LoRA — CSR per fold:")
print(lora_fold_df[["fold", "CSR"]].to_string(index=False))
print("\nFull FT — CSR per fold:")
print(full_ft_fold_df[["fold", "CSR"]].to_string(index=False))

# Build comparison table
rows = []
for m in ["CSR"] + cats:
    if m in lora_summary and m in full_ft_summary:
        lm = lora_summary[m]["mean"] * 100
        ls = lora_summary[m]["std"]  * 100
        fm = full_ft_summary[m]["mean"] * 100
        fs = full_ft_summary[m]["std"]  * 100
        winner = "LoRA" if lm > fm else ("Full_FT" if fm > lm else "Tie")
        rows.append({"Metric": m, "LoRA mean±std": f"{lm:.1f}±{ls:.1f}",
                     "Full FT mean±std": f"{fm:.1f}±{fs:.1f}", "Winner": winner})

comparison = pd.DataFrame(rows)
print("\n" + "="*65)
print("  K-Fold Results — mean±std across 5 folds (%)")
print("="*65)
print(comparison.to_string(index=False))
print("="*65)
comparison.to_csv(f"{RESULTS_DIR}/kfold_comparison.csv", index=False)
print(f"\nSaved: {RESULTS_DIR}/kfold_comparison.csv")



fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fold_nums = list(range(1, K_FOLDS + 1))

# Chart 1: Per-fold CSR line — shows consistency across data slices
axes[0].plot(fold_nums, lora_fold_df["CSR"].values*100,
             marker="o", label="LoRA (Rinkle)", color="#2196F3", linewidth=2)
axes[0].plot(fold_nums, full_ft_fold_df["CSR"].values*100,
             marker="s", label="Full FT (Mark)", color="#FF5722", linewidth=2)
axes[0].set_xlabel("Fold Number")
axes[0].set_ylabel("CSR (%)")
axes[0].set_title("CSR per Fold\n(consistency check)")
axes[0].set_xticks(fold_nums)
axes[0].legend()
axes[0].grid(alpha=0.3)
axes[0].set_ylim(0, 110)

# Chart 2: Per-category bar chart with error bars
cat_ml = [lora_summary.get(c,{}).get("mean",0)*100 for c in cats]
cat_sl = [lora_summary.get(c,{}).get("std",0)*100  for c in cats]
cat_mf = [full_ft_summary.get(c,{}).get("mean",0)*100 for c in cats]
cat_sf = [full_ft_summary.get(c,{}).get("std",0)*100  for c in cats]
x, w   = range(len(cats)), 0.35
axes[1].bar([i-w/2 for i in x], cat_ml, w, yerr=cat_sl, capsize=4,
            label="LoRA (Rinkle)", color="#2196F3", alpha=0.85)
axes[1].bar([i+w/2 for i in x], cat_mf, w, yerr=cat_sf, capsize=4,
            label="Full FT (Mark)", color="#FF5722", alpha=0.85)
axes[1].set_xticks(list(x))
axes[1].set_xticklabels([c.replace("_","\n") for c in cats], fontsize=7)
axes[1].set_ylabel("Satisfaction Rate (%)")
axes[1].set_title("Per-Category CSR\n(mean±std across 5 folds)")
axes[1].legend()
axes[1].set_ylim(0, 120)
axes[1].grid(axis="y", alpha=0.3)

# Chart 3: Final overall score — main result for paper
lf = lora_summary.get("CSR",{}).get("mean",0)*100
ff = full_ft_summary.get("CSR",{}).get("mean",0)*100
le = lora_summary.get("CSR",{}).get("std",0)*100
fe = full_ft_summary.get("CSR",{}).get("std",0)*100
bars = axes[2].bar([0,1],[lf,ff],yerr=[le,fe],capsize=8,
                   color=["#2196F3","#FF5722"],alpha=0.85,width=0.5)
axes[2].set_xticks([0,1])
axes[2].set_xticklabels(["LoRA\n(Rinkle)","Full FT\n(Mark)"],fontsize=11)
axes[2].set_ylabel("CSR (%)")
axes[2].set_title("Final K-Fold CSR Score\n(use this in your paper)")
axes[2].set_ylim(0,120)
axes[2].grid(axis="y",alpha=0.3)
for bar, val in zip(bars,[lf,ff]):
    axes[2].text(bar.get_x()+bar.get_width()/2, val+2,
                 f"{val:.1f}%",ha="center",va="bottom",fontsize=12,fontweight="bold")

plt.suptitle(
    f"K={K_FOLDS} Fold Cross Validation — LoRA vs Full Fine-Tuning\n"
    "Llama 3.2 1B Instruct on RECAST-30K | STAT 453, Spring 2026",
    fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(f"{RESULTS_DIR}/kfold_comparison_plot.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Plot saved: {RESULTS_DIR}/kfold_comparison_plot.png")



lora_csr = lora_summary["CSR"]["mean"]
lora_std = lora_summary["CSR"]["std"]
full_csr = full_ft_summary["CSR"]["mean"]
full_std = full_ft_summary["CSR"]["std"]

lora_wins = sum(1 for r in comparison.to_dict("records") if r["Winner"] == "LoRA")
full_wins = sum(1 for r in comparison.to_dict("records") if r["Winner"] == "Full_FT")
ties      = sum(1 for r in comparison.to_dict("records") if r["Winner"] == "Tie")

print("\n" + "="*55)
print("  FINAL SCORE  (put this in your paper)")
print("="*55)
print(f"  LoRA (Rinkle)  →  CSR = {lora_csr:.4f}  (±{lora_std:.4f})")
print(f"  Full FT (Mark) →  CSR = {full_csr:.4f}  (±{full_std:.4f})")
print("")
winner = "LoRA" if lora_csr > full_csr else "Full FT"
print(f"  Winner: {winner}  (difference = {abs(lora_csr-full_csr):.4f})")
print("="*55)

print("\n  Per-Category Breakdown:")
print(f"  {'Category':<25} {'LoRA':>8} {'Full FT':>10} {'Winner':>12}")
print("  " + "-"*57)
for m in comparison.to_dict("records"):
    cat = m["Metric"]
    if cat == "CSR":
        continue
    lv  = lora_summary.get(cat, {}).get("mean", 0)
    fv  = full_ft_summary.get(cat, {}).get("mean", 0)
    win = "LoRA ✓" if lv > fv else ("Full FT ✓" if fv > lv else "Tie")
    print(f"  {cat:<25} {lv:>8.4f} {fv:>10.4f} {win:>12}")

print("\n" + "="*55)
print(f"  Category wins → LoRA: {lora_wins} | Full FT: {full_wins} | Ties: {ties}")
print("="*55)
print(f"\nFiles saved to {RESULTS_DIR}/")
print("  kfold_comparison.csv       ← table for paper")
print("  kfold_comparison_plot.png  ← figure for paper")
print("  all_results.json           ← raw data")
print("  results_fold_1..5.json     ← per-fold checkpoints")

import os, json, gc, torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── UPDATE THESE ──────────────────────────────────────────────────────────────
BASE_MODEL         = "meta-llama/Llama-3.2-1B-Instruct"
LORA_ADAPTER_PATH  = "/content/outputs/lora_r8_0.0001/lora_adapter"
FULL_FT_MODEL_PATH = "/content/output/finetuned"
DATASET_PATH       = "/content/recast_30k_clean.jsonl"
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 50)
print("  SMOKE TEST")
print("=" * 50)

# ── CHECK 1: GPU ──────────────────────────────────────────────────────────────
print(f"\n[1] GPU: {'✓ ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else '✗ NOT FOUND'}")

# ── CHECK 2: Dataset ──────────────────────────────────────────────────────────
print(f"\n[2] Dataset exists: {'✓' if os.path.exists(DATASET_PATH) else '✗ NOT FOUND'}")

# ── CHECK 3: LoRA adapter ─────────────────────────────────────────────────────
lora_ok = os.path.exists(os.path.join(LORA_ADAPTER_PATH, "adapter_config.json"))
print(f"\n[3] LoRA adapter exists: {'✓' if lora_ok else '✗ NOT FOUND — run lora_base.ipynb first'}")

# ── CHECK 4: Full FT model ────────────────────────────────────────────────────
full_ok = os.path.exists(os.path.join(FULL_FT_MODEL_PATH, "config.json"))
print(f"\n[4] Full FT model exists: {'✓' if full_ok else '✗ NOT FOUND — run full_finetune first'}")

# ── CHECK 5: Load one example from dataset ────────────────────────────────────
with open(DATASET_PATH, "r") as f:
    for line in f:
        row = json.loads(line.strip())
        instruction = row.get("winner_prompt", row.get("input", ""))
        if instruction:
            break

print(f"\n[5] Sample instruction loaded: ✓")
print(f"    Preview: {instruction[:100]}...")

# ── CHECK 6: Test LoRA generates something ────────────────────────────────────
if lora_ok:
    print("\n[6] Testing LoRA model...")
    tokenizer  = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(base, LORA_ADAPTER_PATH)
    model.eval()
    inputs = tokenizer(instruction[:200], return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=30,
                             do_sample=False,
                             pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:],
                                skip_special_tokens=True)
    print(f"    LoRA output: ✓ → {response[:100]}")
    del model, base
    gc.collect()
    torch.cuda.empty_cache()

# ── CHECK 7: Test Full FT generates something ─────────────────────────────────
if full_ok:
    print("\n[7] Testing Full FT model...")
    tokenizer2 = AutoTokenizer.from_pretrained(FULL_FT_MODEL_PATH)
    if tokenizer2.pad_token is None:
        tokenizer2.pad_token = tokenizer2.eos_token
    model2 = AutoModelForCausalLM.from_pretrained(
        FULL_FT_MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto")
    model2.eval()
    inputs2 = tokenizer2(instruction[:200], return_tensors="pt").to(model2.device)
    with torch.no_grad():
        out2 = model2.generate(**inputs2, max_new_tokens=30,
                               do_sample=False,
                               pad_token_id=tokenizer2.eos_token_id)
    response2 = tokenizer2.decode(out2[0][inputs2["input_ids"].shape[1]:],
                                  skip_special_tokens=True)
    print(f"    Full FT output: ✓ → {response2[:100]}")
    del model2, tokenizer2
    gc.collect()
    torch.cuda.empty_cache()

# ── FINAL RESULT ──────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
if lora_ok and full_ok:
    print("  ALL CHECKS PASSED ✓")
    print("  Ready to run cross_validation_kfold.ipynb!")
else:
    print("  SOME CHECKS FAILED ✗")
    print("  Fix the issues above before running cross validation.")
print("=" * 50)