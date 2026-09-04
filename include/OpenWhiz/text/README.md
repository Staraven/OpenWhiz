# OpenWhiz/text

Text-processing extension for OpenWhiz — tokenization, stemming, embedding lookup,
and a sentiment classification preset built on OpenWhiz's existing layers/losses.
Lives inside OpenWhiz's own include tree (`OpenWhiz/text/...`) rather than a separate
sibling library, so it reads as a natural extension of OpenWhiz from the caller's
side and shares the `ow::` namespace (`ow::owTextTokenizer`, `ow::owSentimentPreset`,
...).

Not wired into `engine/CMakeLists.txt` — `libs/OpenWhiz/include` is already on every
GlistEngine project's include path, so `#include "OpenWhiz/text/..."` just works with
no extra CMake changes; projects that never `#include` these headers pay no cost.

Dev-time tooling (vocabulary pruning, etc. — not part of the C++ build) lives in the
sibling `libs/OpenWhiz/tools/text/` directory, kept out of `include/` on purpose.
Language data itself (pretrained vectors, a project's own vocabulary, labeled
datasets) is **not** shipped here — each project prunes/produces its own from
`tools/text/`'s scripts and keeps the result in its own data directory.

A runnable end-to-end sample (`examples/textClassificationExample/`, same layout/
build convention as OpenWhiz's other `examples/*`) shows the tokenize -> stem ->
embed -> classify pipeline in English, Turkish, and French, using a tiny hand-written
synthetic word-vector table (not real embeddings — see that example's own comments).

Another runnable sample (`examples/clusterLabelingExample/`) runs the tokenize ->
TF-IDF -> unsupervised-cluster -> bag-of-stems-label pipeline on a public,
category-labeled dataset (20 Newsgroups) and prints a purity metric against the known
labels — see that example's README for the measurement and what it covers.

## Headers

- `owLanguage.hpp` — `enum class owLanguage { English, Turkish, French }`, shared by
  `owTextTokenizer` and `owStemmer`. Default is `English` everywhere — callers that
  need Turkish/French tokenization or stemming must pass the language explicitly
  (don't rely on the default for a non-English corpus).
- `tokenizers/owTextTokenizer.hpp` — UTF-8 aware word-level tokenizer, constructed
  with an `owLanguage` (default `English`). Splits on anything that isn't a
  letter/digit; language controls which accented letters count as word characters
  and how they lowercase — English is plain ASCII, Turkish adds Ç Ğ İ Ö Ş Ü with
  the Turkish-specific ASCII `I` -> dotless `ı` / `İ` -> ASCII `i` mapping (generic
  ASCII case folding gets this wrong for Turkish), French adds À Â Ä Ç É È Ê Ë Î Ï
  Ô Ö Ù Û Ü Ÿ Œ Æ with standard case folding (no special-case pairs needed).
- `owStemmer.hpp` — language-dispatching facade over the per-language stemmers
  below; construct with an `owLanguage` and call `stem()`. English (no inflection
  worth stripping) is a no-op passthrough.
- `tokenizers/owTurkishStemmer.hpp` — rule-based Turkish suffix stripping (kök
  bulma), used directly by `owStemmer` for `owLanguage::Turkish`. No dictionary, no
  morphological analyzer, no external dependency. Heuristic longest-suffix-first
  stripping across a few passes — an approximation, not a correct morphological
  analyzer (see header comment for known limitations). Per-language stemmers/
  tokenizers live under `tokenizers/` as the module grows to more languages.
- `owEmbeddingLookup.hpp` — loads a pretrained fastText-format word vector table
  (word + N floats per line) into an in-memory lookup, plus a mean-pooling
  `embedAverage()` helper for turning a token list into one fixed-size vector.
  Expects the table already pruned to the project's real vocabulary — see
  `tools/text/prune_fasttext_vectors.py` — not the full multi-GB fastText release.
- `owSentimentPreset.hpp` — sentiment classification head over embedding vectors
  (e.g. from `owEmbeddingLookup::embedAverage()`). 2-class -> sigmoid output +
  `owBinaryCrossEntropyLoss`; 3+ class -> `owProbabilityLayer` (softmax) +
  `owCategoricalCrossEntropyLoss` — same pattern as OpenWhiz's own
  `classificationExample`. `owDataset` has no public API to inject a numeric matrix
  directly — only `loadFromCSV()` populates it — so `train()` here writes
  embeddings+labels to a temp CSV and loads them that way; this is how every OpenWhiz
  example actually feeds data in, not a workaround. `Options::trainRatio/valRatio/
  testRatio` expose `owDataset`'s split ratios (default 0.6/0.2/0.2) — on small
  datasets this can starve the training set (e.g. 8 samples -> only ~4 actually
  trained on), so tests/small experiments should widen `trainRatio`.
- `owWeightedBinaryCrossEntropyLoss.hpp` / `owWeightedCategoricalCrossEntropyLoss.hpp`
  — class-weighted variants of OpenWhiz's own BCE/categorical-cross-entropy losses,
  for imbalanced label distributions. Same eps-clamp as their unweighted counterparts.
- `owClusterLabeler.hpp` — unsupervised clustering over a set of embedding
  vectors (e.g. from `owEmbeddingLookup::embedAverage()`), plus a proposed
  human-readable label per cluster from its most frequent stemmed tokens
  (bag-of-stems frequency count). Reuses `owClusterLayer` directly (centroid
  distances, trained by pulling them toward the data with MSE-to-zero +
  `owADAMOptimizer` — same objective as `examples/clusterExample`); no
  `owDataset`/CSV round-trip needed since `owClusterLayer` takes a raw
  `owTensor` input. Contains no project-specific category names or
  vocabulary. Deliberately a stateless, one-shot static method rather
  than a constructed/held object like OpenWhiz's own layers - there's no
  trained model to keep around afterward (unlike `owSentimentPreset`,
  which does hold a trained network for repeated `predict()` calls).
- `owTfIdfVectorizer.hpp` — TF-IDF over already-tokenized (e.g. stemmed)
  documents: fits a vocabulary + smoothed IDF from a corpus, returns one dense
  L2-normalized TF-IDF vector per document. A bag-of-words alternative to
  `owEmbeddingLookup::embedAverage()` for when averaging pretrained vectors
  dilutes topic signal too much on short documents — output plugs into
  `owClusterLabeler::cluster()` the same way `embedAverage()` output does,
  since that function doesn't care what the vectors mean. `Options::
  maxVocabularySize` caps the vocabulary to the N highest-document-frequency
  terms (after `minDocFrequency`) — on a lexically rich corpus this can favor
  near-universal filler over the actually-discriminative terms sitting just
  below the cap, so also set `maxDocFrequencyRatio` (e.g. 0.5) to drop terms
  above that document-frequency ceiling first.

Stopword filtering before `embedAverage()` is project-specific (which filler words
hurt category signal depends on the corpus) and so is NOT shipped here — write a
small stopword list tuned against your own corpus and filter tokens with it before
calling `embedAverage()`.

## Choosing a training pipeline

`writeTrainingCSV()` (inside `owSentimentPreset::train()`) writes embedding floats
with `std::fixed` at high precision on purpose: `owDataset`'s numeric-column
detector does not recognize scientific notation (`1.23e-05`), and a single such
value silently turns an entire column into a text/category column instead of a
numeric one. Don't remove the `std::fixed`/`setprecision` formatting when touching
that function.
