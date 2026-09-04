#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <utility>

#include "OpenWhiz/text/owLanguage.hpp"
#include "OpenWhiz/text/owStemmer.hpp"
#include "OpenWhiz/text/owEmbeddingLookup.hpp"
#include "OpenWhiz/text/owSentimentPreset.hpp"
#include "OpenWhiz/text/tokenizers/owTextTokenizer.hpp"

// OpenWhiz/text pipeline demo: text in -> tokenize -> stem -> embed -> classify,
// run in English, Turkish, and French to show the SAME mechanism works across
// languages via owLanguage. The word vectors below are tiny, hand-written,
// synthetic (4 numbers per word, two made-up clusters: "animal" words near
// [1,0,0,0], "vehicle" words near [0,0,1,0]) - this demonstrates the pipeline
// mechanics, not real embedding quality. For real embeddings, prune your own
// vocabulary's vectors from a pretrained fastText release with
// tools/text/prune_fasttext_vectors.py - real vocabulary/vector data is
// deliberately never shipped in the library itself (see that script's and
// OpenWhiz/text/README.md's notes on why).
//
// English intentionally uses only exact vocabulary word forms in its sentences
// (owStemmer is a no-op for English), while Turkish and French sentences use
// inflected forms (plurals) that owStemmer actually reduces back to the words
// listed below - e.g. Turkish "kediler" -> "kedi", French "chats" -> "chat".

struct LangSample {
    ow::owLanguage language;
    std::string name;
    std::string vecFile;
    std::vector<std::pair<std::string, std::vector<float>>> words;
    std::vector<std::pair<std::string, int>> trainSentences; // text, label (0=animal, 1=vehicle)
    std::vector<std::string> testSentences;
};

void writeVecFile(const std::string& path, const std::vector<std::pair<std::string, std::vector<float>>>& words) {
    std::ofstream f(path);
    f << words.size() << " " << words.front().second.size() << "\n";
    for (auto& wv : words) {
        f << wv.first;
        for (float v : wv.second) f << " " << v;
        f << "\n";
    }
}

void runLanguageDemo(const LangSample& sample) {
    std::cout << "\n=== " << sample.name << " ===\n";
    writeVecFile(sample.vecFile, sample.words);

    ow::owTextTokenizer tokenizer(sample.language);
    ow::owStemmer stemmer(sample.language);
    ow::owEmbeddingLookup lookup;
    if (!lookup.loadFromFile(sample.vecFile)) {
        std::cout << "failed to load " << sample.vecFile << "\n";
        return;
    }

    auto embed = [&](const std::string& text) {
        std::vector<std::string> tokens = tokenizer.tokenize(text);
        std::vector<std::string> stemmed;
        for (auto& t : tokens) stemmed.push_back(stemmer.stem(t));
        return lookup.embedAverage(stemmed);
    };

    std::vector<std::vector<float>> trainEmb;
    std::vector<int> trainLabels;
    for (auto& ts : sample.trainSentences) {
        trainEmb.push_back(embed(ts.first));
        trainLabels.push_back(ts.second);
    }

    ow::owSentimentPreset::Options options;
    options.hiddenSizes = {4};
    options.maxEpochs = 100;
    // All 8 synthetic examples are used for training - this is a mechanism demo,
    // not a real held-out evaluation.
    options.trainRatio = 1.0f;
    options.valRatio = 0.0f;
    options.testRatio = 0.0f;
    // Fixed seed so the demo's printed predictions are the same every run -
    // owSentimentPreset::Options::useSeed/seed exists specifically for this kind
    // of reproducibility need, since owNeuralNetwork's weight initialization is
    // seed-sensitive on some architectures (an unlucky seed can start training
    // from a near-dead-gradient state).
    options.useSeed = true;
    options.seed = 42;

    ow::owSentimentPreset classifier;
    std::string tempCsv = sample.vecFile + ".train.csv";
    if (!classifier.train(trainEmb, trainLabels, 2, tempCsv, options)) {
        std::cout << "training failed\n";
        return;
    }

    for (auto& text : sample.testSentences) {
        auto scores = classifier.predict(embed(text));
        std::string predicted = scores[1] > scores[0] ? "vehicle" : "animal";
        std::cout << "\"" << text << "\" -> " << predicted
                   << " (P(animal)=" << scores[0] << " P(vehicle)=" << scores[1] << ")\n";
    }
}

int main() {
    std::cout << "=== OpenWhiz/text: tokenize -> stem -> embed -> classify (EN/TR/FR) ===\n";

    LangSample english{
        ow::owLanguage::English, "English", "examples/textClassificationExample/en.vec",
        {
            {"cat", {1.0f, 0.0f, 0.0f, 0.0f}}, {"dog", {0.9f, 0.1f, 0.0f, 0.0f}},
            {"bird", {0.8f, 0.2f, 0.0f, 0.0f}}, {"pet", {0.85f, 0.15f, 0.0f, 0.0f}},
            {"car", {0.0f, 0.0f, 1.0f, 0.0f}}, {"bus", {0.0f, 0.0f, 0.9f, 0.1f}},
            {"train", {0.0f, 0.0f, 0.8f, 0.2f}}, {"drive", {0.0f, 0.0f, 0.85f, 0.15f}},
        },
        {
            {"I have a cat", 0}, {"My dog is happy", 0}, {"The bird sings", 0}, {"I love my pet", 0},
            {"I drive my car", 1}, {"The bus is late", 1}, {"We took the train", 1}, {"I like to drive", 1},
        },
        {"My cat and dog are friends", "The train and bus were both late"}
    };

    LangSample turkish{
        ow::owLanguage::Turkish, "Turkish", "examples/textClassificationExample/tr.vec",
        {
            {"kedi", {1.0f, 0.0f, 0.0f, 0.0f}}, {"köpek", {0.9f, 0.1f, 0.0f, 0.0f}},
            {"kuş", {0.8f, 0.2f, 0.0f, 0.0f}}, {"hayvan", {0.85f, 0.15f, 0.0f, 0.0f}},
            {"araba", {0.0f, 0.0f, 1.0f, 0.0f}}, {"otobüs", {0.0f, 0.0f, 0.9f, 0.1f}},
            {"tren", {0.0f, 0.0f, 0.8f, 0.2f}}, {"sürmek", {0.0f, 0.0f, 0.85f, 0.15f}},
        },
        {
            {"Kediler çok tatlı", 0}, {"Köpekler mutlu", 0}, {"Kuşlar öter", 0}, {"Hayvanlar güzel", 0},
            {"Arabalar hızlı", 1}, {"Otobüsler geç kaldı", 1}, {"Trenler rahat", 1}, {"Sürmek güzel", 1},
        },
        {"Kediler ve köpekler arkadaş", "Otobüs ve tren geç kaldı"}
    };

    LangSample french{
        ow::owLanguage::French, "French", "examples/textClassificationExample/fr.vec",
        {
            {"chat", {1.0f, 0.0f, 0.0f, 0.0f}}, {"chien", {0.9f, 0.1f, 0.0f, 0.0f}},
            {"oiseau", {0.8f, 0.2f, 0.0f, 0.0f}}, {"animal", {0.85f, 0.15f, 0.0f, 0.0f}},
            {"voiture", {0.0f, 0.0f, 1.0f, 0.0f}}, {"bus", {0.0f, 0.0f, 0.9f, 0.1f}},
            {"train", {0.0f, 0.0f, 0.8f, 0.2f}}, {"conduire", {0.0f, 0.0f, 0.85f, 0.15f}},
        },
        {
            {"J'ai un chat", 0}, {"Mon chien est content", 0}, {"L'oiseau chante", 0}, {"Cet animal est mignon", 0},
            {"Je conduis ma voiture", 1}, {"Le bus est en retard", 1}, {"Nous avons pris le train", 1}, {"J'aime conduire", 1},
        },
        {"Mon chat et mon chien sont amis", "Le train et le bus étaient en retard"}
    };

    runLanguageDemo(english);
    runLanguageDemo(turkish);
    runLanguageDemo(french);

    return 0;
}
