import io
import os
import re

from PIL import Image

dir_name = os.path.dirname(os.path.abspath(__file__))


def hrbench_doc_to_visual(doc):
    return [Image.open(io.BytesIO(doc["image"])).convert("RGB")]


def hrbench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"].strip()
    options = ["A", "B", "C", "D"]
    for option in options:
        question = f"{question}\n{option}. {doc[option]}"

    if "pre_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["pre_prompt"] != "":
        question = f"{lmms_eval_specific_kwargs['pre_prompt']}{question}"
    if "post_prompt" in lmms_eval_specific_kwargs and lmms_eval_specific_kwargs["post_prompt"] != "":
        question = f"{question}{lmms_eval_specific_kwargs['post_prompt']}"
    return question


def hrbench_process_results(doc, results):
    """
    Args:
        doc: a instance of the eval dataset
        results: [pred]
    Returns:
        a dictionary with key: metric name (in this case hrbench score), value: metric value
    """
    prediction = results[0]
    answer = doc["answer"]
    score = 0

    # More flexible regex patterns to handle various formats
    # Pattern 1: Matches "Answer: (X)" or "Answer: X" or "Answer is (X)" etc.
    # Pattern 2: Matches "(X)" or "X)" or "X." or "X" where X is A-Z
    # Pattern 3: Matches "option X" or "X option" where X is A-Z
    # Pattern 4: Matches "answer is X" or "X is the answer" where X is A-Z
    pred_patterns = [
        r"Answer:?\s*[\(]?([A-Z])[\)]?",  # Pattern 1
        r"[\(]?([A-Z])[\)\.]?",  # Pattern 2
        r"option\s*([A-Z])",  # Pattern 3
        r"([A-Z])\s*option",  # Pattern 3
        r"answer\s*is\s*([A-Z])",  # Pattern 4
        r"([A-Z])\s*is\s*the\s*answer",  # Pattern 4
    ]

    # Try each pattern on the prediction
    pred_letter = None
    for pattern in pred_patterns:
        match = re.search(pattern, prediction, re.IGNORECASE)
        if match:
            pred_letter = match.group(1).upper()
            break

    # Extract answer letter using the same patterns
    ans_letter = None
    for pattern in pred_patterns:
        match = re.search(pattern, answer, re.IGNORECASE)
        if match:
            ans_letter = match.group(1).upper()
            break

    if pred_letter and ans_letter:
        score = 1 if pred_letter == ans_letter else 0
    return {"accuracy": score}


def hrbench_aggregate_results(results):
    return sum(results) / len(results)
