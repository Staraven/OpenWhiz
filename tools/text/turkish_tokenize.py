"""Reusable Turkish-correct tokenization, matching owTextTokenizer.hpp exactly.

Use this (not str.lower() + str.split()) wherever a vocabulary is extracted
from raw text that will later be matched against owTextTokenizer's runtime
output — otherwise words get mismatched or corrupted (see e.g. "Istanbul"
silently losing its leading letter under naive Python .lower(), or "de/da"
suffixes staying glued to the previous word via apostrophes).
"""

_UPPER_TO_LOWER = {ord('I'): 'ı', ord('İ'): 'i'}
for _u, _l in [('Ç', 'ç'), ('Ğ', 'ğ'), ('Ö', 'ö'), ('Ş', 'ş'), ('Ü', 'ü')]:
    _UPPER_TO_LOWER[ord(_u)] = _l
for _c in range(ord('A'), ord('Z') + 1):
    _UPPER_TO_LOWER.setdefault(_c, chr(_c + 0x20))

_WORD_CHARS = set("abcdefghijklmnopqrstuvwxyz0123456789çğıöşü")


def turkish_lower(text):
    return text.translate(_UPPER_TO_LOWER)


def tokenize(text):
    """Splits on anything that isn't a letter/digit, Turkish-correct lowercase."""
    lowered = turkish_lower(text)
    tokens = []
    current = []
    for ch in lowered:
        if ch in _WORD_CHARS:
            current.append(ch)
        elif current:
            tokens.append(''.join(current))
            current = []
    if current:
        tokens.append(''.join(current))
    return tokens


if __name__ == "__main__":
    for sample in ["İstanbul'da 2024!", "GÜNCEL Finans, İzmir ve İngilizce"]:
        print(sample, "->", tokenize(sample))
