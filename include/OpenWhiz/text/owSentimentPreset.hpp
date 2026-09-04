#pragma once

#include <cmath>
#include <fstream>
#include <iomanip>
#include <string>
#include <vector>

#include "OpenWhiz/openwhiz.hpp"
#include "OpenWhiz/text/owWeightedCategoricalCrossEntropyLoss.hpp"
#include "OpenWhiz/text/owWeightedBinaryCrossEntropyLoss.hpp"

namespace ow {

// Sentiment classification head built on OpenWhiz's existing pieces: 2-class
// uses owSigmoidActivation + owBinaryCrossEntropyLoss, 3+ classes use the
// softmax-style owProbabilityLayer + owCategoricalCrossEntropyLoss - same
// pattern as OpenWhiz's own classificationExample. Takes embedding vectors
// (e.g. from owEmbeddingLookup::embedAverage()) as input.
//
// owDataset has no public API to inject a numeric matrix directly, so
// train() writes embeddings+labels to a temp CSV and loads them through
// loadFromCSV() - the same data path OpenWhiz's own examples use.
class owSentimentPreset {
public:
    struct Options {
        std::vector<int> hiddenSizes = {32, 16};
        int maxEpochs = 300;
        bool autoNormalize = false;
        bool enablePrinting = false;
        // owDataset defaults to a 0.6/0.2/0.2 train/val/test split — fine for a real
        // corpus, but on small datasets it can starve the training set (e.g. 8 samples
        // -> only ~4 actually used for training). Adjust for your data size.
        float trainRatio = 0.6f;
        float valRatio = 0.2f;
        float testRatio = 0.2f;
        // owDataset::setRatios() shuffles row->split assignment by default.
        // Set this false for a stratified split where rows are pre-ordered
        // by the caller (first trainRatio fraction training, etc.).
        bool shuffleSplit = true;
        // Balanced class weighting for the numClasses>=3 (categorical) path only:
        // weight_c = totalExamples / (numClasses * count_c), the standard "balanced"
        // formula (matches sklearn's class_weight='balanced'). Computed from the full
        // label set passed to train() (not train-split-only, to stay independent of
        // shuffleSplit/row-ordering assumptions). Not oversampling - no rows are
        // duplicated, only the loss's per-example weight changes.
        bool useClassWeights = false;
        // Reproducible weight init, useful when diagnosing training issues
        // (e.g. sweeping seeds to check for a bad initial-weight state).
        // Leave false for normal use (time-based seed).
        bool useSeed = false;
        unsigned int seed = 0;
    };

    // embeddings.size() == labels.size(); labels are class indices in [0, numClasses).
    // numClasses == 2 -> binary sigmoid+BCE head. numClasses >= 3 -> softmax+categorical
    // cross-entropy head. tempCsvPath is overwritten with the training data (numeric
    // only, no PII beyond what's already in the embedding vectors).
    bool train(const std::vector<std::vector<float>>& embeddings,
               const std::vector<int>& labels,
               int numClasses,
               const std::string& tempCsvPath) {
        return train(embeddings, labels, numClasses, tempCsvPath, Options());
    }

    bool train(const std::vector<std::vector<float>>& embeddings,
               const std::vector<int>& labels,
               int numClasses,
               const std::string& tempCsvPath,
               const Options& options) {
        if (embeddings.empty() || embeddings.size() != labels.size() || numClasses < 2) {
            return false;
        }
        if (!writeTrainingCSV(tempCsvPath, embeddings, labels, numClasses)) {
            return false;
        }

        m_network = owNeuralNetwork();
        if (options.useSeed) m_network.setSeed(options.seed);
        m_network.getDataset()->setAutoNormalizeEnabled(options.autoNormalize);
        if (!m_network.loadData(tempCsvPath)) {
            return false;
        }
        m_network.getDataset()->setTargetVariableNum(numClasses == 2 ? 1 : numClasses);
        m_network.getDataset()->setRatios(options.trainRatio, options.valRatio, options.testRatio, options.shuffleSplit);

        if (numClasses == 2) {
            m_network.createNeuralNetwork(options.hiddenSizes, "ReLU", "Sigmoid", false);
            if (options.useClassWeights) {
                int countPos = 0, countNeg = 0;
                for (int lbl : labels) (lbl == 1 ? countPos : countNeg)++;
                float n = static_cast<float>(labels.size());
                float wPos = countPos > 0 ? n / (2.0f * countPos) : 1.0f;
                float wNeg = countNeg > 0 ? n / (2.0f * countNeg) : 1.0f;
                m_network.setLoss(std::make_shared<owWeightedBinaryCrossEntropyLoss>(wPos, wNeg));
            } else {
                m_network.setLoss(std::make_shared<owBinaryCrossEntropyLoss>());
            }
        } else {
            m_network.createNeuralNetwork(options.hiddenSizes, "ReLU", "Identity", false);
            m_network.addLayer(std::make_shared<owProbabilityLayer>());
            if (options.useClassWeights) {
                std::vector<int> counts(numClasses, 0);
                for (int lbl : labels) if (lbl >= 0 && lbl < numClasses) counts[lbl]++;
                std::vector<float> weights(numClasses, 1.0f);
                for (int c = 0; c < numClasses; ++c) {
                    weights[c] = counts[c] > 0
                        ? static_cast<float>(labels.size()) / (numClasses * static_cast<float>(counts[c]))
                        : 1.0f;
                }
                m_network.setLoss(std::make_shared<owWeightedCategoricalCrossEntropyLoss>(weights));
            } else {
                m_network.setLoss(std::make_shared<owCategoricalCrossEntropyLoss>());
            }
        }

        // owLBFGSOptimizer's line search can stall after one step on non-toy
        // data (its step size collapses and does not recover), so Adam is
        // used here instead.
        //
        // One forward+backward pass with no weight update, to record the
        // initial gradient norm - distinguishes a bad weight-init state
        // (near-zero gradient) from a stall caused by the optimizer/learning
        // rate despite a healthy initial gradient.
        {
            auto trainIn = m_network.getDataset()->getTrainInput();
            auto trainTarget = m_network.getDataset()->getTrainTarget();
            if (trainIn.size() > 0) {
                auto pred = m_network.forward(trainIn);
                m_network.backward(pred, trainTarget);
                owTensor<float, 1> grads(m_network.getTotalParameterCount());
                m_network.getGlobalGradients(grads);
                double sumSq = 0.0;
                for (size_t i = 0; i < grads.size(); ++i) sumSq += (double)grads.data()[i] * grads.data()[i];
                m_lastInitialGradNorm = static_cast<float>(std::sqrt(sumSq));
                m_network.reset();
            } else {
                m_lastInitialGradNorm = -1.0f;
            }
        }

        m_network.setOptimizer(std::make_shared<owADAMOptimizer>());
        m_network.setMaximumEpochNum(options.maxEpochs);
        m_network.setEnablePrinting(options.enablePrinting);
        m_network.train();
        m_numClasses = numClasses;
        m_lastFinishReason = m_network.getTrainingFinishReason();
        m_lastTrainLoss = m_network.getLastTrainError();
        return true;
    }

    // Diagnostics from the most recent train() call - see the debug probe above.
    float getLastInitialGradNorm() const { return m_lastInitialGradNorm; }
    std::string getLastFinishReason() const { return m_lastFinishReason; }
    float getLastTrainLoss() const { return m_lastTrainLoss; }

    // Returns per-class scores (2 values for binary [P(negative), P(positive)] derived
    // from the single sigmoid output, or numClasses softmax probabilities). Empty on
    // failure (e.g. train() wasn't called, or dimension mismatch).
    std::vector<float> predict(const std::vector<float>& embedding) {
        if (m_numClasses == 0 || embedding.empty()) {
            return {};
        }

        owTensor<float, 2> input(1, embedding.size());
        for (size_t i = 0; i < embedding.size(); ++i) {
            input(0, i) = embedding[i];
        }
        owTensor<float, 2> out = m_network.forward(input);

        if (m_numClasses == 2) {
            float positive = out(0, 0);
            return {1.0f - positive, positive};
        }

        std::vector<float> scores(m_numClasses);
        for (int c = 0; c < m_numClasses; ++c) {
            scores[c] = out(0, static_cast<size_t>(c));
        }
        return scores;
    }

private:
    bool writeTrainingCSV(const std::string& path,
                           const std::vector<std::vector<float>>& embeddings,
                           const std::vector<int>& labels,
                           int numClasses) const {
        std::ofstream file(path);
        if (!file.is_open()) {
            return false;
        }
        // owDataset's numeric-column detector doesn't recognize scientific
        // notation ('e'/'E') - one such value marks the whole column as
        // Text instead of numeric. Force fixed notation so embedding floats
        // never trigger it.
        file << std::fixed << std::setprecision(9);

        size_t dim = embeddings.front().size();
        for (size_t d = 0; d < dim; ++d) {
            file << "e" << d << ",";
        }
        if (numClasses == 2) {
            file << "label\n";
        } else {
            for (int c = 0; c < numClasses; ++c) {
                file << "t" << c << (c + 1 < numClasses ? "," : "\n");
            }
        }

        for (size_t i = 0; i < embeddings.size(); ++i) {
            if (embeddings[i].size() != dim) {
                return false;
            }
            for (float v : embeddings[i]) {
                file << v << ",";
            }
            if (numClasses == 2) {
                file << (labels[i] == 1 ? "1" : "0") << "\n";
            } else {
                for (int c = 0; c < numClasses; ++c) {
                    file << (labels[i] == c ? "1" : "0") << (c + 1 < numClasses ? "," : "\n");
                }
            }
        }
        return true;
    }

    owNeuralNetwork m_network;
    int m_numClasses = 0;
    float m_lastInitialGradNorm = 0.0f;
    std::string m_lastFinishReason;
    float m_lastTrainLoss = 0.0f;
};

} // namespace ow
