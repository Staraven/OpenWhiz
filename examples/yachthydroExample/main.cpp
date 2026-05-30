#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include "OpenWhiz/openwhiz.hpp"


int main() {
    std::cout << "=== OpenWhiz Yacht Hydrodynamics Approximation Example ===" << std::endl;

    const std::string csvFile = "examples/yachthydroExample/yacht_hydro.csv";

    ow::owNeuralNetwork nn;

    // 1. Setup Data
    // Increasing training ratio to 70% to give the model more samples to learn from
    nn.getDataset()->setRatios(0.7f, 0.15f, 0.15f);

    if (!nn.loadData(csvFile)) {
        std::cerr << "Failed to load data!" << std::endl;
        return -1;
    }
    nn.getDataset()->normalizeData();

    // 2. Build Network Architecture
    // LeakyReLU for processing negative inputs and sigmoid for positive result
    nn.createNeuralNetwork({128, 128}, "LeakyReLU", "Sigmoid", true);
    
    // Standard MSE loss
    nn.setLoss(std::make_shared<ow::owMeanSquaredErrorLoss>());

    // 3. Configure Training - BFGS Strategy
    nn.setOptimizer(std::make_shared<ow::owBFGSOptimizer>(0.001f));
    nn.setMaximumEpochNum(1000);
    nn.setLossStagnationPatience(100);
    nn.setMinimumError(0.00001f);
    nn.train();

    // 5. Final Evaluation
    std::cout << "\nFinal Performance on Test Set:" << std::endl;
    nn.printEvaluationReport(nn.evaluatePerformance());

    // Logging extra stats before manual prediction
    std::cout << "Total Epochs: " << nn.getTrainingEpochNum() << std::endl;
    std::cout << "Total Time: " << nn.getTrainingTime() << "s" << std::endl;
    std::cout << "Final Train Error: " << nn.getLastTrainError() << std::endl;
    std::cout << "Final Val Error: " << nn.getLastValError() << std::endl;
    std::cout << "First Sample Type: " << nn.getDataset()->getSampleTypeString(0) << std::endl;

    // 6. Manual Prediction Check (Using the new Smart Predict API!)
    ow::owTensor<float, 2> input(1, 6);
    input.setValues({-2.3f, 0.568f, 4.78f, 3.99f, 3.17f, 0.125f});
    
    // Smart predict handles both input normalization and output inverse normalization automatically!
    auto pred = nn.predict(input);
    
    std::cout << "Prediction for [-2.3, 0.568, 4.78, 3.99, 3.17, 0.125] = " << pred(0, 0) << " (Actual: 0.11)" << std::endl;

    return 0;
}
