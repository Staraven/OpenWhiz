/*
 * owBFGSOptimizer.hpp
 *
 *  Created on: Apr 27, 2026
 *      Author: Noyan Culum, AITIAL
 */

#pragma once
#include "owOptimizer.hpp"
#include <vector>
#include <iostream>
#include <chrono>
#include <cmath>
#include <algorithm>

namespace ow {

class owNeuralNetwork;
class owDataset;

/**
 * @class owBFGSOptimizer
 * @brief Broyden-Fletcher-Goldfarb-Shanno (BFGS) optimizer.
 */
class owBFGSOptimizer : public owOptimizer {
private:
    inline double dot(const std::vector<double>& a, const std::vector<double>& b) {
        double sum = 0;
        size_t n = a.size();
        for (size_t i = 0; i < n; ++i) sum += a[i] * b[i];
        return sum;
    }

public:
    explicit owBFGSOptimizer(float lr = 1.0f) {
        this->m_learningRate = lr;
    }

    void optimizeGlobal(owNeuralNetwork* nn, owDataset* ds) override {
        size_t nParams = nn->getTotalParameterCount();
        if (nParams == 0) return;

        auto trainIn = ds->getTrainInput();
        auto trainTarget = ds->getTrainTarget();

        owTensor<float, 1> x_f(nParams), g_f(nParams);
        nn->getGlobalParameters(x_f);

        std::vector<double> x(nParams), g(nParams), d(nParams), x_next(nParams);
        for(size_t i=0; i<nParams; ++i) x[i] = (double)x_f.data()[i];

        std::vector<double> invH(nParams * nParams, 0.0);
        auto resetHessian = [&]() {
            std::fill(invH.begin(), invH.end(), 0.0);
            for(size_t i=0; i<nParams; ++i) invH[i * nParams + i] = 1.0;
        };
        resetHessian();

        bool firstPass = true;
        auto compute_f_g = [&](const std::vector<double>& cur_x, std::vector<double>& cur_g) {
            for(size_t i=0; i<nParams; ++i) x_f.data()[i] = (float)cur_x[i];
            nn->setGlobalParameters(x_f);
            nn->reset();
            auto pred = nn->forward(trainIn);
            
            if (firstPass) {
                for (auto& layer : nn->getLayers()) layer->lockCache();
                firstPass = false;
            }

            const auto& activeTarget = nn->getActiveTarget(trainTarget);
            float loss = nn->calculateLoss(pred, activeTarget);
            nn->reset();
            nn->forward(trainIn);
            nn->backward(pred, activeTarget);
            nn->getGlobalGradients(g_f);
            for(size_t i=0; i<nParams; ++i) cur_g[i] = (double)g_f.data()[i];
            return (double)loss;
        };

        for (auto& layer : nn->getLayers()) layer->setTarget(&trainTarget);

        double f = compute_f_g(x, g);
        double bestLoss = f;
        int patience = 0;
        double lastStep = 1.0;
        auto startTime = std::chrono::high_resolution_clock::now();

        for (int k = 1; k <= nn->getMaximumEpochNum(); ++k) {
            nn->setTrainingEpochNum(k); // Ensure counter is updated at the start of epoch
            
            // Validation loss calculation
            float valLoss = 0.0f;
            auto valIn = ds->getValInput();
            if (valIn.size() > 0) {
                auto valTarget = ds->getValTarget();
                auto valPred = nn->forward(valIn);
                valLoss = nn->calculateLoss(valPred, valTarget);
                nn->setLastValError(valLoss);
            }

            // Print Status at the beginning of epoch
            if (nn->getPrintEpochInterval() > 0 && (k == 1 || k % nn->getPrintEpochInterval() == 0)) {
                auto now = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> currentElapsed = now - startTime;
                nn->printTrainingStatus(k, (float)f, valLoss, currentElapsed.count());
            }

            // 1. Direction Calculation: d = -H * g
            for (size_t i = 0; i < nParams; ++i) {
                double sum = 0;
                for (size_t j = 0; j < nParams; ++j) {
                    sum += invH[i * nParams + j] * g[j];
                }
                d[i] = -sum;
            }

            double g_dot_d = dot(g, d);
            if (g_dot_d >= 0) {
                resetHessian();
                for (size_t i = 0; i < nParams; ++i) d[i] = -g[i];
                g_dot_d = dot(g, d);
            }

            double step = (k == 1) ? 1.0 / std::sqrt(dot(g, g) + 1e-10) : lastStep;
            if (step > 1.0) step = 1.0;
            if (step < 1e-6) step = 1e-6;

            bool success = false;
            std::vector<double> g_next(nParams);
            double f_next = f;

            const double c1 = 1e-4;
            const double c2 = 0.01;

            for (int i = 0; i < 150; ++i) {
                for(size_t j=0; j<nParams; ++j) x_next[j] = x[j] + step * d[j];
                f_next = compute_f_g(x_next, g_next);

                if (f_next < f + c1 * step * g_dot_d + 1e-9) {
                    double g_next_dot_d = dot(g_next, d);
                    if (g_next_dot_d >= c2 * g_dot_d) {
                        success = true;
                        lastStep = step * 1.2; 
                        if (lastStep > 1.0) lastStep = 1.0;
                        break;
                    }
                }
                step *= 0.5; 
                if (step < 1e-40) break;
            }

            if (!success) {
                double eps = std::numeric_limits<double>::epsilon();
                bool perturbed = false;
                for (size_t i = 0; i < nParams; ++i) {
                    if (std::abs(g[i]) > 1e-35) {
                        x_next[i] = x[i] - (g[i] > 0 ? eps : -eps);
                        perturbed = true;
                    } else {
                        x_next[i] = x[i];
                    }
                }
                if (perturbed) {
                    f_next = compute_f_g(x_next, g_next);
                    if (f_next < f) {
                        success = true;
                        lastStep = 1e-6;
                    }
                }
            }

            if (!success) {
                resetHessian();
                lastStep = 1.0;
                if (patience > 10) {
                    nn->setTrainingFinishReason("Minimum Precision Limit");
                    // Print final status before exit
                    if (nn->getPrintEpochInterval() > 0 && k % nn->getPrintEpochInterval() != 0) {
                        auto now = std::chrono::high_resolution_clock::now();
                        std::chrono::duration<double> currentElapsed = now - startTime;
                        nn->printTrainingStatus(k, (float)f, valLoss, currentElapsed.count());
                    }
                    break; 
                }
                patience++;
                continue;
            }

            std::vector<double> sk(nParams), yk(nParams);
            for(size_t j=0; j<nParams; ++j) {
                sk[j] = x_next[j] - x[j];
                yk[j] = g_next[j] - g[j];
            }

            double ys = dot(yk, sk);
            
            if (k == 1 || g_dot_d >= 0) {
                double yy = dot(yk, yk);
                if (ys > 1e-30 && yy > 1e-30) {
                    double scale = ys / yy;
                    for (size_t i = 0; i < nParams; ++i) {
                        for (size_t j = 0; j < nParams; ++j) {
                            if (i == j) invH[i * nParams + j] = scale;
                            else invH[i * nParams + j] = 0.0;
                        }
                    }
                }
            }

            std::vector<double> Hs(nParams, 0.0);
            for(size_t i=0; i<nParams; ++i) {
                for(size_t j=0; j<nParams; ++j) Hs[i] += invH[i * nParams + j] * yk[j];
            }
            double sHs = dot(yk, Hs);

            if (ys > 1e-30 && sHs > 1e-30) {
                std::vector<double> bfgs_v(nParams);
                for (size_t i = 0; i < nParams; ++i) bfgs_v[i] = sk[i] / ys - Hs[i] / sHs;

                for (size_t i = 0; i < nParams; ++i) {
                    for (size_t j = 0; j < nParams; ++j) {
                        double update = (sk[i] * sk[j]) / ys 
                                      - (Hs[i] * Hs[j]) / sHs 
                                      + (bfgs_v[i] * bfgs_v[j]) * sHs;
                        invH[i * nParams + j] += update;
                    }
                }
            }

            x = x_next; g = g_next; f = f_next;
            nn->setLastTrainError((float)f);

            // --- MAPE Based Stopping ---
            if (nn->getMinimumPercentageError() > 0.0f) {
                float currentMape = 0.0f;
                auto pred = nn->forward(trainIn);
                const auto& activeTarget = nn->getActiveTarget(trainTarget);
                size_t n = pred.shape()[0], outDim = pred.shape()[1];
                for (size_t i = 0; i < n; ++i) {
                    for (size_t j = 0; j < outDim; ++j) {
                        float p = pred(i, j), t = activeTarget(i, j);
                        if (std::abs(t) > 1e-7f) currentMape += std::abs((p - t) / t);
                    }
                }
                currentMape = (currentMape / (n * outDim)) * 100.0f;
                if (currentMape <= nn->getMinimumPercentageError()) {
                    nn->setTrainingFinishReason("Minimum Error");
                    // Final print if MAPE reached
                    auto now = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> currentElapsed = now - startTime;
                    nn->printTrainingStatus(k, (float)f, valLoss, currentElapsed.count());
                    break;
                }
            }

            if (f < nn->getMinimumError()) {
                nn->setTrainingFinishReason("Minimum Error");
                // Final print if Error reached
                if (nn->getPrintEpochInterval() > 0 && k % nn->getPrintEpochInterval() != 0) {
                    auto now = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> currentElapsed = now - startTime;
                    nn->printTrainingStatus(k, (float)f, valLoss, currentElapsed.count());
                }
                break;
            }

            if (f < bestLoss - (double)nn->getLossStagnationTolerance()) {
                bestLoss = f; patience = 0;
            } else {
                patience++;
            }

            if (patience >= nn->getLossStagnationPatience()) {
                nn->setTrainingFinishReason("Loss Stagnation");
                // Final print if Stagnation reached
                if (nn->getPrintEpochInterval() > 0 && k % nn->getPrintEpochInterval() != 0) {
                    auto now = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> currentElapsed = now - startTime;
                    nn->printTrainingStatus(k, (float)f, valLoss, currentElapsed.count());
                }
                break;
            }
            
            nn->setTrainingFinishReason("Maximum Epoch Num");
            // Final print if Max Epoch reached
            if (k == nn->getMaximumEpochNum() && nn->getPrintEpochInterval() > 0 && k % nn->getPrintEpochInterval() != 0) {
                auto now = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> currentElapsed = now - startTime;
                nn->printTrainingStatus(k, (float)f, valLoss, currentElapsed.count());
            }
        }
        for(size_t i=0; i<nParams; ++i) x_f.data()[i] = (float)x[i];
        nn->setGlobalParameters(x_f);
    }

    void update(owTensor<float, 2>&, const owTensor<float, 2>&) override {}
    std::string getOptimizerName() const override { return "BFGS"; }
    std::shared_ptr<owOptimizer> clone() const override { return std::make_shared<owBFGSOptimizer>(m_learningRate); }
    bool supportsGlobalOptimization() const override { return true; }
};

} // namespace ow
