#pragma once

#include "OpenWhiz/openwhiz.hpp"

namespace ow {

// Class-weighted binary cross-entropy, same pattern as
// owWeightedCategoricalCrossEntropyLoss - lives in text/ (not the core
// layers/losses tree) since it's specific to class-imbalance handling for
// text classification. eps matches ow::owBinaryCrossEntropyLoss's own clamp
// (1e-12); no owProbabilityLayer is involved on the binary path (Sigmoid
// output feeds this loss directly), so there's no clamp-mismatch risk here
// like the softmax path had.
class owWeightedBinaryCrossEntropyLoss : public owLoss {
public:
    // weightPositive/weightNegative: per-class multiplier, e.g. balanced =
    // N / (2 * count_c).
    owWeightedBinaryCrossEntropyLoss(float weightPositive, float weightNegative)
        : m_weightPositive(weightPositive), m_weightNegative(weightNegative) {}

    float compute(const owTensor<float, 2>& prediction, const owTensor<float, 2>& target) override {
        float loss = 0.0f;
        size_t n = prediction.size();
        const float eps = 1e-12f;
        for (size_t i = 0; i < n; ++i) {
            float p = std::max(eps, std::min(1.0f - eps, prediction.data()[i]));
            float t = target.data()[i];
            float w = t > 0.5f ? m_weightPositive : m_weightNegative;
            loss -= w * (t * std::log(p) + (1.0f - t) * std::log(1.0f - p));
        }
        return loss / static_cast<float>(n);
    }

    owTensor<float, 2> gradient(const owTensor<float, 2>& prediction, const owTensor<float, 2>& target) override {
        owTensor<float, 2> grad(prediction.shape());
        size_t n = prediction.size();
        const float eps = 1e-12f;
        float factor = 1.0f / static_cast<float>(n);
        for (size_t i = 0; i < n; ++i) {
            float p = std::max(eps, std::min(1.0f - eps, prediction.data()[i]));
            float t = target.data()[i];
            float w = t > 0.5f ? m_weightPositive : m_weightNegative;
            grad.data()[i] = factor * w * (-(t / p) + (1.0f - t) / (1.0f - p));
        }
        return grad;
    }

    std::string getLossName() const override { return "Weighted Binary Cross-Entropy Loss"; }

    std::shared_ptr<owLoss> clone() const override {
        return std::make_shared<owWeightedBinaryCrossEntropyLoss>(m_weightPositive, m_weightNegative);
    }

private:
    float m_weightPositive;
    float m_weightNegative;
};

} // namespace ow
