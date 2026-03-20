"""Evaluation metrics for OCR: CER, WER, and sequence accuracy."""

import editdistance


def character_error_rate(prediction: str, target: str) -> float:
    """Compute Character Error Rate (CER).

    CER = edit_distance(pred, target) / len(target)
    Returns 0.0 if target is empty and prediction is also empty, else 1.0.
    """
    if len(target) == 0:
        return 0.0 if len(prediction) == 0 else 1.0
    return editdistance.eval(prediction, target) / len(target)


def word_error_rate(prediction: str, target: str) -> float:
    """Compute Word Error Rate (WER).

    WER = edit_distance(pred_words, target_words) / len(target_words)
    """
    pred_words = prediction.split()
    target_words = target.split()
    if len(target_words) == 0:
        return 0.0 if len(pred_words) == 0 else 1.0
    return editdistance.eval(pred_words, target_words) / len(target_words)


def sequence_accuracy(prediction: str, target: str) -> float:
    """Exact-match sequence accuracy. Returns 1.0 if pred == target, else 0.0."""
    return 1.0 if prediction == target else 0.0


def batch_metrics(predictions: list[str], targets: list[str]) -> dict[str, float]:
    """Compute average metrics over a batch.

    Returns dict with keys: cer, wer, seq_acc
    """
    n = len(predictions)
    if n == 0:
        return {"cer": 0.0, "wer": 0.0, "seq_acc": 0.0}

    total_cer = sum(character_error_rate(p, t) for p, t in zip(predictions, targets))
    total_wer = sum(word_error_rate(p, t) for p, t in zip(predictions, targets))
    total_acc = sum(sequence_accuracy(p, t) for p, t in zip(predictions, targets))

    return {
        "cer": total_cer / n,
        "wer": total_wer / n,
        "seq_acc": total_acc / n,
    }
