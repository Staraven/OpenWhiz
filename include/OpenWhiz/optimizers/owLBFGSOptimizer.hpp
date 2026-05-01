/*
 * owLBFGSOptimizer.hpp
 *
 *  Created on: Dec 16, 2025
 *      Author: Noyan Culum, AITIAL
 */


#pragma once
#include "owOptimizer.hpp"
#include <deque>
#include <vector>
#include <iostream>
#include <chrono>
#include <cmath>
#include <algorithm>

namespace ow {

class owNeuralNetwork;
class owDataset;

/**
 * @class owLBFGSOptimizer
 * @brief Limited-memory Broyden–Fletcher–Goldfarb–Shanno (L-BFGS) optimizer.
 */
 class owLBFGSOptimizer : public owOptimizer {
private:
    size_t m_m = 100;
    std::deque<std::vector<double>> s_list;
    std::deque<std::vector<double>> y_list;
    std::deque<double> rho_list;

    inline double dot(const std::vector<double>& a, const std::vector<double>& b) {
        double sum = 0;
        size_t n = a.size();
        for (size_t i = 0; i < n; ++i) sum += a[i] * b[i];
        return sum;
    }

public:
    explicit owLBFGSOptimizer(float lr = 1.0f, int m = 100) : m_m(m) {
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

            // 1. Direction Calculation
            if (s_list.empty()) {
                for(size_t i=0; i<nParams; ++i) d[i] = -g[i];
            } else {
                std::vector<double> q = g;
                std::vector<double> alphas(s_list.size());
                for (int i = (int)s_list.size() - 1; i >= 0; --i) {
                    alphas[i] = rho_list[i] * dot(s_list[i], q);
                    for(size_t j=0; j<nParams; ++j) q[j] -= alphas[i] * y_list[i][j];
                }

                double sy = dot(s_list.back(), y_list.back());
                double yy = dot(y_list.back(), y_list.back());
                double gamma = (std::abs(yy) > 1e-18) ? sy / yy : 1.0;

                for(size_t i=0; i<nParams; ++i) d[i] = q[i] * gamma;
                for (size_t i = 0; i < s_list.size(); ++i) {
                    double beta = rho_list[i] * dot(y_list[i], d);
                    for(size_t j=0; j<nParams; ++j) d[j] += s_list[i][j] * (alphas[i] - beta);
                }
                for(size_t i=0; i<nParams; ++i) d[i] *= -1.0;
            }

            double g_norm = 0;
            for(size_t i=0; i<nParams; ++i) g_norm += g[i]*g[i];
            g_norm = std::sqrt(g_norm);

            double g_dot_d = dot(g, d);
            if (g_dot_d > -1e-9 * g_norm) {
                s_list.clear(); y_list.clear(); rho_list.clear();
                for(size_t i=0; i<nParams; ++i) d[i] = -g[i];
                g_dot_d = dot(g, d);
            }

            double step = 1.0;
            if (g_norm > 1.0) step = std::min(1.0, 1.0 / g_norm);
            bool success = false;
            std::vector<double> g_next(nParams);
            double f_next = f;

            const double c1 = 1e-4;
            const double c2 = 0.9;

            for (int i = 0; i < 60; ++i) {
                for(size_t j=0; j<nParams; ++j) x_next[j] = x[j] + step * d[j];
                f_next = compute_f_g(x_next, g_next);

                if (f_next <= f + c1 * step * g_dot_d + 1e-12) {
                    double g_next_dot_d = dot(g_next, d);
                    if (std::abs(g_next_dot_d) <= c2 * std::abs(g_dot_d)) {
                        success = true;
                        break;
                    }
                    if (step < 1e-12) { success = true; break; }
                }
                step *= 0.6; 
                if (step < 1e-18) break;
            }

            if (!success && s_list.size() > 0) {
                s_list.clear(); y_list.clear(); rho_list.clear();
                continue;
            }

            if (!success) {
                nn->setTrainingFinishReason("Minimum Precision Limit");
                // Final print if failed
                if (nn->getPrintEpochInterval() > 0 && k % nn->getPrintEpochInterval() != 0) {
                    auto now = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> currentElapsed = now - startTime;
                    nn->printTrainingStatus(k, (float)f, valLoss, currentElapsed.count());
                }
                break;
            }

            std::vector<double> sk(nParams), yk(nParams);
            for(size_t j=0; j<nParams; ++j) {
                sk[j] = x_next[j] - x[j];
                yk[j] = g_next[j] - g[j];
            }
            double ys = dot(yk, sk);
            double yy = dot(yk, yk);
            
            if (ys > 1e-14 * yy && ys > 1e-18) {
                if (s_list.size() >= m_m) { s_list.pop_front(); y_list.pop_front(); rho_list.pop_front(); }
                s_list.push_back(sk); y_list.push_back(yk); rho_list.push_back(1.0 / ys);
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
                    auto now = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> currentElapsed = now - startTime;
                    nn->printTrainingStatus(k, (float)f, valLoss, currentElapsed.count());
                    break;
                }
            }

            bool canStop = (k > 100) && nn->isLossStagnationEnabled();

            if (f < bestLoss - (double)nn->getLossStagnationTolerance()) {
                bestLoss = f; patience = 0;
            } else {
                patience++;
            }

            if ((canStop && patience >= nn->getLossStagnationPatience()) || f < nn->getMinimumError()) {
                nn->setTrainingFinishReason(f < nn->getMinimumError() ? "Minimum Error" : "Loss Stagnation");
                if (nn->getPrintEpochInterval() > 0 && k % nn->getPrintEpochInterval() != 0) {
                    auto now = std::chrono::high_resolution_clock::now();
                    std::chrono::duration<double> currentElapsed = now - startTime;
                    nn->printTrainingStatus(k, (float)f, valLoss, currentElapsed.count());
                }
                break;
            }
            
            nn->setTrainingFinishReason("Maximum Epoch Num");

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
    std::string getOptimizerName() const override { return "L-BFGS"; }
    std::shared_ptr<owOptimizer> clone() const override { return std::make_shared<owLBFGSOptimizer>(m_learningRate); }
    bool supportsGlobalOptimization() const override { return true; }
};

} // namespace ow
