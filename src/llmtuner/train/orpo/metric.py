from typing import Dict

import numpy as np

from ...extras.packages import is_jieba_available, is_nltk_available, is_rouge_available


if is_jieba_available():
    import jieba  # type: ignore

if is_nltk_available():
    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

if is_rouge_available():
    from rouge_chinese import Rouge


def compute_metrics(predict_results: list) -> Dict[str, float]:
    r"""
    Uses the model predictions to compute metrics.
    """
    score_dict = {"rouge-1": [], "rouge-2": [], "rouge-l": [], "bleu-4": []}

    for res in predict_results:
        pred, label = res["predict"], res["label"]
        hypothesis = list(jieba.cut(pred))
        reference = list(jieba.cut(label))

        if (
            len(" ".join(hypothesis).split()) == 0
            or len(" ".join(reference).split()) == 0
        ):
            result = {
                "rouge-1": {"f": 0.0},
                "rouge-2": {"f": 0.0},
                "rouge-l": {"f": 0.0},
            }
        else:
            rouge = Rouge()
            scores = rouge.get_scores(" ".join(hypothesis), " ".join(reference))
            result = scores[0]

        for k, v in result.items():
            score_dict[k].append(round(v["f"] * 100, 4))

        bleu_score = sentence_bleu(
            [list(label)],
            list(pred),
            smoothing_function=SmoothingFunction().method3,
        )
        score_dict["bleu-4"].append(round(bleu_score * 100, 4))

    return {k: float(np.mean(v)) for k, v in score_dict.items()}
