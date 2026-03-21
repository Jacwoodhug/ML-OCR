"""Character set definition and encoding/decoding utilities for CTC-based OCR."""


# CTC blank is always index 0
BLANK_TOKEN = "<blank>"
BLANK_IDX = 0

# Full character set (order matters — indices are the class labels)
CHARS = (
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    ".,!?-:;'\"()@#$%&+=/_~[] {}\\|<>^`"
)

# Number of output classes = len(CHARS) + 1 (for CTC blank at index 0)
NUM_CLASSES = len(CHARS) + 1

# Mapping: char -> index (1-based, 0 is blank)
CHAR_TO_IDX = {ch: i + 1 for i, ch in enumerate(CHARS)}

# Mapping: index -> char
IDX_TO_CHAR = {i + 1: ch for i, ch in enumerate(CHARS)}
IDX_TO_CHAR[BLANK_IDX] = BLANK_TOKEN


def encode(text: str) -> list[int]:
    """Encode a text string to a list of integer indices.

    Characters not in the alphabet are silently skipped.
    """
    return [CHAR_TO_IDX[ch] for ch in text if ch in CHAR_TO_IDX]


def decode(indices: list[int], raw: bool = False) -> str:
    """Decode a list of integer indices back to a string.

    If raw=False (default), CTC blank tokens are stripped.
    If raw=True, blanks are included as '<blank>'.
    """
    chars = []
    for idx in indices:
        if idx == BLANK_IDX:
            if raw:
                chars.append(BLANK_TOKEN)
        elif idx in IDX_TO_CHAR:
            chars.append(IDX_TO_CHAR[idx])
    return "".join(chars)
