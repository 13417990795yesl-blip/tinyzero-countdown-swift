"""Evaluate an exported checkpoint on the independent Countdown test set."""

import argparse
from collections import Counter
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from countdown_plugin import evaluate_equation, extract_answer, validate_equation
from scripts.build_holdout import load_questions

MAX_NEW_TOKENS = 1024
BATCH_SIZE = 32


def classify(text, nums, target, *, split_eq=False):
    answer = extract_answer(text)
    if answer is None:
        return "no_answer_tag"
    expr = answer.split("=", 1)[0].strip() if split_eq else answer
    if not validate_equation(expr, nums):
        return "wrong_numbers"
    value = evaluate_equation(expr)
    if value is None:
        return "unevaluable"
    if abs(value - target) < 1e-5:
        return "correct"
    return "valid_wrong_value"


def generate(model_path, rows):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16).cuda().eval()
    prompts = [tokenizer.apply_chat_template(row["messages"], tokenize=False,
                                             add_generation_prompt=True) for row in rows]
    completions = []
    for i in range(0, len(prompts), BATCH_SIZE):
        batch = tokenizer(prompts[i:i+BATCH_SIZE], return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            tokens = model.generate(**batch, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
                                    pad_token_id=tokenizer.pad_token_id)
        generated = tokens[:, batch["input_ids"].shape[1]:]
        completions.extend(tokenizer.batch_decode(generated, skip_special_tokens=True))
        print(f"generated {len(completions)}/{len(rows)}", flush=True)
    return tokenizer, completions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    path = Path(f"{args.output_prefix}_completions.jsonl")
    summary_path = Path(f"{args.output_prefix}_summary.json")
    if path.exists() or summary_path.exists():
        parser.error("the output prefix already has results")
    rows = load_questions(args.dataset)
    if not rows:
        parser.error("the input dataset is empty")
    tokenizer, completions = generate(args.model, rows)
    strict = Counter()
    split = Counter()
    with path.open("x", encoding="utf-8") as output:
        for row, completion in zip(rows, completions, strict=True):
            nums, target = row["nums"], row["target"]
            cat_strict = classify(completion, nums, target)
            cat_split = classify(completion, nums, target, split_eq=True)
            strict[cat_strict] += 1
            split[cat_split] += 1
            output.write(json.dumps(dict(nums=nums, target=target, completion=completion,
                                         cat_strict=cat_strict, cat_split_eq=cat_split),
                                    ensure_ascii=False) + "\n")
    n = len(rows)
    lengths = [len(tokenizer(text)["input_ids"]) for text in completions]
    summary = {
        "n": n,
        "correct_strict": strict["correct"] / n,
        "correct_split_eq": split["correct"] / n,
        "categories_strict": {k: v/n for k, v in strict.items()},
        "categories_split_eq": {k: v/n for k, v in split.items()},
        "mean_completion_tokens": sum(lengths) / n,
        "truncated_at_1024": sum(length >= MAX_NEW_TOKENS for length in lengths) / n,
    }
    with summary_path.open("x", encoding="utf-8") as output:
        json.dump(summary, output, indent=2)
        output.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
