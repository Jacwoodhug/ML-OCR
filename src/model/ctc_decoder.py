"""CTC decoders: greedy and beam search."""

import torch

from src.data.alphabet import BLANK_IDX, IDX_TO_CHAR


def greedy_decode(log_probs: torch.Tensor) -> list[str]:
    """Greedy CTC decoding (best-path).

    Args:
        log_probs: (T, B, C) log-probabilities from the model

    Returns:
        List of B decoded strings
    """
    # Argmax over classes at each timestep
    predictions = log_probs.argmax(dim=2)  # (T, B)
    predictions = predictions.permute(1, 0)  # (B, T)

    results = []
    for seq in predictions:
        decoded = []
        prev = BLANK_IDX
        for idx in seq.tolist():
            if idx != BLANK_IDX and idx != prev:
                if idx in IDX_TO_CHAR:
                    decoded.append(IDX_TO_CHAR[idx])
            prev = idx
        results.append("".join(decoded))

    return results


def beam_search_decode(
    log_probs: torch.Tensor,
    beam_width: int = 10,
) -> list[str]:
    """Simple beam search CTC decoding.

    Args:
        log_probs: (T, B, C) log-probabilities
        beam_width: number of beams to keep

    Returns:
        List of B decoded strings (best beam for each)
    """
    T, B, C = log_probs.shape
    results = []

    for b in range(B):
        # beams: list of (prefix_tuple, log_prob)
        beams = [((), 0.0)]

        for t in range(T):
            new_beams: dict[tuple, float] = {}

            for prefix, score in beams:
                for c in range(C):
                    lp = log_probs[t, b, c].item()
                    new_score = score + lp

                    if c == BLANK_IDX:
                        # Blank: keep prefix unchanged
                        key = prefix
                    elif len(prefix) > 0 and prefix[-1] == c:
                        # Repeat char without blank: same prefix
                        key = prefix
                    else:
                        # New character
                        key = prefix + (c,)

                    if key not in new_beams or new_beams[key] < new_score:
                        new_beams[key] = new_score

            # Prune to beam_width
            sorted_beams = sorted(new_beams.items(), key=lambda x: x[1], reverse=True)
            beams = sorted_beams[:beam_width]

        # Best beam
        best_prefix = beams[0][0] if beams else ()
        text = "".join(IDX_TO_CHAR.get(idx, "") for idx in best_prefix)
        results.append(text)

    return results
