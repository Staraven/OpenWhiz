#!/usr/bin/env python3
"""Prune fastText .vec files down to a project vocabulary, TR/EN/FR.

Not part of the C++ build — a one-off dev-time tool. Supports mixed-language
corpora (e.g. an app with Turkish and English content) by merging pruned
tables from up to three source files into one: a word is looked up in --tr
first, then --en (for words missing from --tr), then --fr (for words missing
from both). Vectors
from different languages are NOT in the same embedding space (trained
separately) — merging them doesn't make e.g. "kitap" and "book" comparable, it
just means every source language's words get *some* vector instead of being
silently dropped from average pooling.

Also strips common stopwords (ve, bir, the, to, le, la, ...) before matching,
since they add noise to embedAverage() without carrying topical signal.

Get the source vectors from https://fasttext.cc/docs/en/crawl-vectors.html
(cc.tr.300.vec, cc.en.300.vec, cc.fr.300.vec, ... — several GB each, not
shipped in this repo or anywhere in the library — see this tool's own README
for why: the library ships the pruning tool, never real vocabulary/vector
data, each project prunes its own).

Usage:
    python prune_fasttext_vectors.py --vocab vocabulary.txt --tr cc.tr.300.vec \\
        --en cc.en.300.vec --output pruned.vec

    # French only, no stopword filtering:
    python prune_fasttext_vectors.py --vocab vocabulary.txt --fr cc.fr.300.vec \\
        --output pruned.vec --no-stopwords

Output is in the same "<count> <dim>" header + "word f1 f2 ... fd" line format
that owEmbeddingLookup::loadFromFile expects.
"""

import argparse

DEFAULT_STOPWORDS = {
    # Turkish
    "ve", "bir", "bu", "da", "de", "en", "ile", "için", "ama", "çok",
    "daha", "gibi", "ki", "mi", "mı", "mu", "mü", "ne", "o", "şu", "ya",
    "yada", "veya", "ise", "her", "tüm", "diye", "olan", "olarak",
    # English
    "the", "to", "in", "my", "of", "a", "an", "and", "or", "is", "are",
    "on", "at", "for", "with", "by", "it", "this", "that", "you", "your",
    # French
    "le", "la", "les", "un", "une", "des", "et", "est", "à", "dans",
    "ce", "cette", "ces", "pour", "sur", "avec", "du", "au", "aux",
    "qui", "que", "ou", "mais", "se", "son", "sa", "ses", "il", "elle",
}


def load_words(path):
    if not path:
        return set()
    with open(path, "r", encoding="utf-8") as f:
        return {line.strip() for line in f if line.strip()}


def prune(source_path, vocabulary, existing):
    """Looks up `vocabulary` words in `source_path`, skipping ones already in `existing`."""
    found = {}
    dimension = None

    with open(source_path, "r", encoding="utf-8", errors="ignore") as source:
        first_line = source.readline()
        parts = first_line.split()
        if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
            dimension = int(parts[1])
        else:
            source.seek(0)

        for line in source:
            parts = line.rstrip("\n").split(" ")
            word = parts[0]
            if word not in vocabulary or word in existing or word in found:
                continue
            values = parts[1:]
            if dimension is None:
                dimension = len(values)
            elif len(values) != dimension:
                continue
            found[word] = values

    return found, dimension


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--vocab", required=True, help="Vocabulary file, one lowercased word per line.")
    parser.add_argument("--tr", help="Path to cc.tr.300.vec (or equivalent Turkish vectors). Looked up first.")
    parser.add_argument("--en", help="Path to cc.en.300.vec (or equivalent English vectors), looked up for words missing from --tr.")
    parser.add_argument("--fr", help="Path to cc.fr.300.vec (or equivalent French vectors), looked up for words missing from --tr and --en.")
    parser.add_argument("--output", required=True, help="Output path for the merged pruned table.")
    parser.add_argument("--stopwords", help="Extra stopwords file, one per line, added to the built-in list.")
    parser.add_argument("--no-stopwords", action="store_true", help="Disable stopword filtering entirely.")
    args = parser.parse_args()

    if not args.tr and not args.en and not args.fr:
        parser.error("at least one of --tr / --en / --fr is required")

    vocabulary = load_words(args.vocab)

    stopwords = set()
    if not args.no_stopwords:
        stopwords = set(DEFAULT_STOPWORDS) | load_words(args.stopwords)
        removed = vocabulary & stopwords
        vocabulary -= stopwords
        print(f"stopwords filtered out: {len(removed)}")

    print(f"vocabulary after stopword filtering: {len(vocabulary)} words")

    found = {}
    dimension = None

    if args.tr:
        tr_found, tr_dim = prune(args.tr, vocabulary, found)
        found.update(tr_found)
        dimension = dimension or tr_dim
        print(f"found in Turkish vectors: {len(tr_found)} words")

    if args.en:
        remaining = vocabulary - found.keys()
        en_found, en_dim = prune(args.en, remaining, found)
        found.update(en_found)
        dimension = dimension or en_dim
        print(f"found in English vectors (misses so far only): {len(en_found)} words")

    if args.fr:
        remaining = vocabulary - found.keys()
        fr_found, fr_dim = prune(args.fr, remaining, found)
        found.update(fr_found)
        dimension = dimension or fr_dim
        print(f"found in French vectors (misses so far only): {len(fr_found)} words")

    with open(args.output, "w", encoding="utf-8") as out:
        out.write(f"{len(found)} {dimension}\n")
        for word, values in found.items():
            out.write(word + " " + " ".join(values) + "\n")

    missing = vocabulary - found.keys()
    print(f"total found: {len(found)} words -> {args.output}")
    print(f"still missing (out-of-vocabulary in both): {len(missing)} words")
    if missing:
        sample = sorted(missing)[:20]
        print("sample missing words:", ", ".join(sample))


if __name__ == "__main__":
    main()
