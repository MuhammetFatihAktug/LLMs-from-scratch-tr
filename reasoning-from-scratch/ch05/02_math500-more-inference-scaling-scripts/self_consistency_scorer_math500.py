# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt)
# Source for "Build a Reasoning Model (From Scratch)": https://mng.bz/lZ5B
# Code repository: https://github.com/rasbt/reasoning-from-scratch

import argparse
import json
from pathlib import Path
import time
import requests

import torch
from collections import Counter

from reasoning_from_scratch.ch02 import get_device
from reasoning_from_scratch.ch03 import (
    eta_progress_message,
    render_prompt,
    grade_answer,
    load_model_and_tokenizer,
    extract_final_candidate,
)
from reasoning_from_scratch.ch04 import (
    generate_text_stream_concat_flex,
    generate_text_top_p_stream_cache,
)
from reasoning_from_scratch.ch05 import (  # YENİ2
    heuristic_score,  # YENİ2
    avg_logprob_answer,  # YENİ2
)  # YENİ2


def self_consistency_vote(
    model,
    tokenizer,
    prompt,
    device,
    num_samples=10,
    temperature=0.8,
    top_p=0.9,
    max_new_tokens=2048,
    show_progress=True,
    show_long_answer=False,
    seed=None,
    early_stop=True,   # YENİ
):
    full_answers, short_answers = [], []
    counts = Counter()
    groups = {}
    majority_winners, final_answer = [], None

    for i in range(num_samples):
        if seed is not None:
            torch.manual_seed(seed + i + 1)

        answer = generate_text_stream_concat_flex(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            device=device,
            max_new_tokens=max_new_tokens,
            verbose=show_long_answer,
            generate_func=generate_text_top_p_stream_cache,
            temperature=temperature,
            top_p=top_p,
        )

        short = extract_final_candidate(answer, fallback="number_then_full")
        full_answers.append(answer)
        short_answers.append(short)
        counts[short] += 1
        groups.setdefault(short, []).append(i)

        if show_progress:
            print(f"[Sample {i+1}/{num_samples}] → {short!r}")

        #########################################################
        # YENİ
        # Bir yanıt zaten >= %50 çoğunluğa ulaştıysa erken dur
        if early_stop and counts[short] > num_samples / 2:
            majority_winners = [short]
            final_answer = short
            break
        #########################################################

    if final_answer is None:
        mc = counts.most_common()
        if mc:
            top_freq = mc[0][1]
            majority_winners = [s for s, f in mc if f == top_freq]
            final_answer = mc[0][0] if len(majority_winners) == 1 else None

    return {
        "full_answers": full_answers,
        "short_answers": short_answers,
        "counts": dict(counts),
        "groups": groups,
        "majority_winners": majority_winners,
        "final_answer": final_answer,
    }


def evaluate_math500_stream(
    model,
    tokenizer,
    device,
    math_data,
    out_path=None,
    max_new_tokens=2048,
    verbose=False,
    prompt_suffix="",    # YENİ
    temperature=1.0,     # YENİ
    top_p=1.0,           # YENİ
    seed=None,           # YENİ
    num_samples=10,      # YENİ
    early_stop=False,    # YENİ
    scoring="none",      # YENİ2
):

    if out_path is None:
        dev_name = str(device).replace(":", "-")
        out_path = Path(f"math500-{dev_name}.jsonl")

    num_examples = len(math_data)
    num_correct = 0
    start_time = time.time()

    with open(out_path, "w", encoding="utf-8") as f:
        for i, row in enumerate(math_data, start=1):
            prompt = render_prompt(row["problem"])

            ###################################################################
            # YENİ
            prompt += prompt_suffix
            results = self_consistency_vote(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                device=device,
                num_samples=num_samples,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                show_progress=False,
                show_long_answer=False,
                seed=seed,
                early_stop=early_stop,
            )

            # final_answer belirlenemediyse (berabere),
            # puanlama yoluyla çöz (puanlama yoksa ilk görülme sırasına göre)  # YENİ2
            if results["final_answer"] is None:  # YENİ2
                if scoring == "none":  # YENİ2
                    extracted = results["majority_winners"][0]  # YENİ2
                else:  # YENİ2
                    best = None  # YENİ2
                    best_score = float("-inf")  # YENİ2
                    for cand in results["majority_winners"]:  # YENİ2
                        scores = []  # YENİ2
                        for idx in results["groups"][cand]:  # YENİ2
                            candidate_full = results["full_answers"][idx]  # YENİ2
                            if scoring == "heuristic":  # YENİ2
                                score = heuristic_score(candidate_full, prompt=prompt)  # YENİ2
                            elif scoring == "logprob":  # YENİ2
                                score = avg_logprob_answer(  # YENİ2
                                    model=model,  # YENİ2
                                    tokenizer=tokenizer,  # YENİ2
                                    prompt=prompt,  # YENİ2
                                    answer=candidate_full,  # YENİ2
                                    device=device,  # YENİ2
                                )  # YENİ2
                            else:  # YENİ2
                                score = 0.0  # YENİ2
                            scores.append(float(score))  # YENİ2
                        cand_score = max(scores)  # YENİ2
                        if cand_score > best_score:  # YENİ2
                            best_score = cand_score  # YENİ2
                            best = cand  # YENİ2
                    extracted = best  # YENİ2
            else:  # YENİ2
                extracted = results["final_answer"]  # YENİ2

            # extracted = extract_final_candidate(
            #     gen_text
            # )

            # İsteğe bağlı olarak uzun yanıtı al
            if extracted is not None:
                for idx, s in enumerate(results["short_answers"]):
                    if s == extracted:
                        long_answer = results["full_answers"][idx]
                        break
            gen_text = long_answer
            ###################################################################

            is_correct = grade_answer(
                extracted, row["answer"]
            )
            num_correct += int(is_correct)

            record = {
                "index": i,
                "problem": row["problem"],
                "gtruth_answer": row["answer"],
                "generated_text": gen_text,
                "extracted": extracted,
                "correct": bool(is_correct),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

            progress_msg = eta_progress_message(
                processed=i,
                total=num_examples,
                start_time=start_time,
                show_eta=True,
                label="MATH-500",
            )
            print(progress_msg, end="\r", flush=True)
            if verbose:
                print(
                    f"\n\n{'='*50}\n{progress_msg}\n"
                    f"{'='*50}\nExtracted: {extracted}\n"
                    f"Expected:  {row['answer']}\n"
                    f"Correct so far: {num_correct}\n{'-'*50}"
                )

    seconds_elapsed = time.time() - start_time
    acc = num_correct / num_examples if num_examples else 0.0
    print(f"\nAccuracy: {acc*100:.1f}% ({num_correct}/{num_examples})")
    print(f"Total time: {seconds_elapsed/60:.1f} min")
    print(f"Logs written to: {out_path}")
    return num_correct, num_examples, acc


def get_data():
    local_path = Path("math500_test.json")
    url = (
        "https://raw.githubusercontent.com/rasbt/reasoning-from-scratch/"
        "main/ch03/01_main-chapter-code/math500_test.json"
    )

    if local_path.exists():
        with local_path.open("r", encoding="utf-8") as f:
            math_data = json.load(f)
    else:
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        math_data = r.json()

    return math_data


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: 'auto', or any torch device string like 'cpu', 'cuda', 'cuda:0', 'mps'.",
    )
    parser.add_argument(
        "--which_model",
        type=str,
        default="base",
        choices=["base", "reasoning", "instruct"],
        help="Model variant to load",
    )
    parser.add_argument(
        "--dataset_size",
        type=int,
        default=10,
        help="Number of MATH-500 examples to evaluate",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=2048,
        help="Max new tokens for generation",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Enable torch.compile for the model.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-sample correctness while evaluating.",
    )
    parser.add_argument(
        "--prompt_suffix",
        type=str,
        default="\n\nExplain step by step.",
        help="Adds a chain-of-thought prompt",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for self-consistency sampling",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Setting for temperature scaling",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=1.0,
        help="Threshold for top-p filtering (nucleus sampling)",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=3,
        help="Number of samples for self-consistency sampling",
    )
    parser.add_argument(
        "--early_stop",
        action="store_true",
        help="Enable early stopping when a strict majority is reached",
    )
    parser.add_argument(  # YENİ2
        "--scoring",  # YENİ2
        type=str,  # YENİ2
        default="none",  # YENİ2
        choices=["none", "heuristic", "logprob"],  # YENİ2
        help="Tie-break scoring for majority vote.",  # YENİ2
    )  # YENİ2
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.device == "auto":
        device = get_device()
    else:
        device = torch.device(args.device)

    which_model = args.which_model
    dataset_size = args.dataset_size
    max_new_tokens = args.max_new_tokens
    use_compile = args.compile

    print("Model:", which_model)
    print("Device:", device)
    dev_name = str(device).replace(":", "-")

    math_data = get_data()

    if args.which_model == "instruct":
        which_model = "reasoning"
    else:
        which_model = args.which_model

    model, tokenizer = load_model_and_tokenizer(
        which_model=which_model,
        device=device,
        use_compile=args.compile
    )
    if args.which_model == "instruct":
        tokenizer.add_thinking = False

    model.eval()
    torch.set_float32_matmul_precision("high")

    num_correct, num_examples, acc = evaluate_math500_stream(
        model=model,
        out_path=f"math500_{which_model}-{dev_name}-evaluate-script.jsonl",
        tokenizer=tokenizer,
        device=device,
        math_data=math_data[:dataset_size],
        max_new_tokens=max_new_tokens,
        verbose=args.verbose,
        prompt_suffix=args.prompt_suffix,  # YENİ
        temperature=args.temperature,      # YENİ
        top_p=args.top_p,                  # YENİ
        seed=args.seed,                    # YENİ
        num_samples=args.num_samples,      # YENİ
        early_stop=args.early_stop,        # YENİ2
        scoring=args.scoring               # YENİ2
    )
