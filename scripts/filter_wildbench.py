from datasets import load_dataset
from tqdm import tqdm
from typing import List, Dict, Tuple
from helm_instruct.annotators import InstructionDifficultyEstimator

import os
import sys
import random
import json
import time

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")
estimator = InstructionDifficultyEstimator()


def is_a_harder_than_b(a: str, b: str) -> bool:
    """
    Compare two instructions and determine if A is harder than B.
    """
    answer = estimator.rank(a, b)
    return answer[0]["most_difficult"] == "A"


def is_a_harder_than_b_batch(a: List[str], b: List[str]) -> List[bool]:
    """
    Compare two instructions and determine if A is harder than B.
    """
    answer = estimator.rank_batch(a, b)
    return [a["most_difficult"] == "A" for a in answer]


def elo_rating(Ra: float, Rb: float, K: float, result: int) -> Tuple[float, float]:
    """
    Calculate the new ELO rating.
    Ra: Current ELO rating of instruction A
    Rb: Current ELO rating of instruction B
    K: K-factor determining the weight of the match
    result: 1 if A wins, 0 if B wins
    """
    Ea = 1 / (1 + 10 ** ((Rb - Ra) / 400))
    Eb = 1 / (1 + 10 ** ((Ra - Rb) / 400))
    new_Ra = Ra + K * (result - Ea)
    new_Rb = Rb + K * ((1 - result) - Eb)
    return new_Ra, new_Rb


def update_elo_scores(
    prompts: List[Dict[str, str]],
    K: int = 30,
    comparisons_per_prompt: int = 5,
    batch_size: int = 64,
) -> List[Dict[str, str]]:
    for prompt in prompts:
        prompt["elo"] = 1000
    total_comparisons = len(prompts) * comparisons_per_prompt

    for _ in tqdm(range(total_comparisons // batch_size)):
        if batch_size > 1:
            prompts_batch = random.sample(prompts, batch_size * 2)
            a: List[Dict[str, str]] = prompts_batch[:batch_size]
            b: List[Dict[str, str]] = prompts_batch[batch_size:]
            results = is_a_harder_than_b_batch(
                [p["prompt"] for p in a], [p["prompt"] for p in b]
            )
            results = [1 if r else 0 for r in results]
            for i, result in enumerate(results):
                new_Ra, new_Rb = elo_rating(a[i]["elo"], b[i]["elo"], K, result)
                a[i]["elo"] = new_Ra
                b[i]["elo"] = new_Rb
        else:
            a, b = random.sample(prompts, 2)
            if is_a_harder_than_b(a["prompt"], b["prompt"]):
                result = 1
            else:
                result = 0

            new_Ra, new_Rb = elo_rating(a["elo"], b["elo"], K, result)
            a["elo"] = new_Ra
            b["elo"] = new_Rb

    return prompts


def get_prompts(num_per_category: int = 20) -> List[Dict[str, str]]:
    categories = {}
    for row in tqdm(
        load_dataset(
            "allenai/WildBench",
            split="test",
            cache_dir=CACHE_DIR,
        )
    ):
        if row["appropriate"] != "appropriate":
            continue
        conversation: List[Dict[str, str]] = row["conversation_input"]
        if len(conversation) > 1 or conversation[0]["role"] != "user":
            continue
        prompt = conversation[0]["content"]
        category = row["primary_tag"]
        categories[category] = categories.get(category, []) + [prompt]

    prompts: List[Dict[str, str]] = []
    print("Categories")
    for category, category_prompts in categories.items():
        samples_size = min(num_per_category, len(category_prompts))
        selection: List[str] = random.sample(category_prompts, samples_size)
        prompts.extend(
            [{"prompt": prompt, "category": category} for prompt in selection]
        )
        print(f" - {category}: {len(selection)}")

    return prompts


def main():
    prompts = get_prompts()
    print(f"Collected {len(prompts)} prompts")
    prompts = update_elo_scores(prompts)
    prompts = sorted(prompts, key=lambda x: x["elo"], reverse=True)

    # Save the prompts
    with open("prompts.json", "w") as f:
        json.dump(prompts, f, indent=2)


if __name__ == "__main__":
    main()
