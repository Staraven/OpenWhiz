# OpenWhiz/tools/text

Dev-time (not part of the C++ build) scripts that support `OpenWhiz/text/`'s
headers. Kept as a sibling of `include/`, not inside it, since these are Python
tooling, not headers to be `#include`d.

No language data (pretrained vectors, a project's vocabulary, labeled datasets)
lives here — these scripts *produce* a project's own small data files from
general-purpose inputs; the outputs themselves belong with the project that
generated them, kept in that project's own data directory.

## `prune_fasttext_vectors.py`

Prunes a full pretrained fastText vector file (several GB) down to just the words
a project actually needs, producing a small `.vec` table that
`owEmbeddingLookup::loadFromFile()` loads at runtime.

Needs:
1. Full pretrained vectors from https://fasttext.cc/docs/en/crawl-vectors.html for
   your language(s) — e.g. `cc.tr.300.vec`. Too large to ship here.
2. A vocabulary file — one lowercased word per line, extracted from your project's
   own text corpus (title/description/etc.) using the SAME tokenization rules as
   `owTextTokenizer.hpp` at runtime, or the results won't match up. `turkish_tokenize.py`
   (below) is one such tokenizer, for Turkish specifically.

```
python tools/text/prune_fasttext_vectors.py --vocab vocabulary.txt \
    --tr cc.tr.300.vec --en cc.en.300.vec --output pruned.vec
```

Supports mixed-language corpora via `--tr`/`--en`/`--fr` (any combination, in that
lookup order — each word tried against `--tr` first, then `--en` for what's still
missing, then `--fr`). Vectors from different languages are **not** in the same
embedding space (trained separately) — this doesn't make cross-language words
comparable, it just means a word missing from an earlier source still gets *some*
vector instead of being silently dropped from `embedAverage()`. It also strips
common stopwords (ve, bir, the, to, le, la, ...) before matching, since they add
noise to averaging without carrying topical signal. Run
`python tools/text/prune_fasttext_vectors.py --help` for all options
(single-language lookup, disabling stopword filtering, extra stopwords file).

## `turkish_tokenize.py`

A Python port of `owTextTokenizer.hpp`'s Turkish-correct lowercasing + word-boundary
logic (ASCII `I` -> dotless `ı`, `İ` -> ASCII `i` — a plain `.lower()` gets this
wrong and silently corrupts words like "İstanbul"). Used to build a vocabulary file
from a raw text corpus for `prune_fasttext_vectors.py`'s `--vocab` argument, so the
extracted vocabulary matches what the C++ tokenizer will actually produce at
runtime — building the vocabulary with a different tokenizer/lowercasing rule than
the one used at runtime causes silent lookup misses in `embedAverage()`.
