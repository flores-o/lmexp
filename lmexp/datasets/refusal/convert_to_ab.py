"""
python -m lmexp.datasets.refusal.convert_to_ab
"""

import os
import json
import random

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
LENGHT_LIMIT_CHARS = 200


def get_ab_data():
    res = []
    with open(os.path.join(CURRENT_DIR, "refusal.json"), "r") as f:
        data = json.load(f)
    random.shuffle(data)
    for item in data:
        harmful = item["answer_harmful"].replace("\n", " ")
        harmless = item["answer_harmless"].replace("\n", " ")
        question = item["question"].replace("\n", " ")
        if len(harmful) > LENGHT_LIMIT_CHARS:
            sentences = harmful.split(".")
            harmful = ""
            while len(harmful) < LENGHT_LIMIT_CHARS:
                harmful += sentences.pop(0) + "."
        if len(harmless) > LENGHT_LIMIT_CHARS:
            sentences = harmless.split(".")
            harmless = ""
            while len(harmless) < LENGHT_LIMIT_CHARS:
                harmless += sentences.pop(0) + "."
        if random.random() > 0.5:
            prompt = f"{question}\n\nAnswer choices:\n(A) {harmful}\n(B) {harmless}"
            harmful_completion = "(A)"
            harmless_completion = "(B)"
        else:
            prompt = f"{question}\n\nAnswer choices:\n(A) {harmless}\n(B) {harmful}"
            harmful_completion = "(B)"
            harmless_completion = "(A)"
        res.append(
            {
                "prompt": prompt,
                "harmful_completion": harmful_completion,
                "harmless_completion": harmless_completion,
            }
        )
    return res


def save_ab_data():
    data = get_ab_data()
    with open(os.path.join(CURRENT_DIR, "refusal_ab.json"), "w") as f:
        json.dump(data, f, indent=4)


if __name__ == "__main__":
    save_ab_data()
