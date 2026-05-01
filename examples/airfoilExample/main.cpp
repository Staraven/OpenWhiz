#include <iostream>
#include <fstream>
#include <cmath>
#include <vector>
#include "OpenWhiz/openwhiz.hpp"


int main() {
    std::cout << "=== OpenWhiz Airfoil Self Noise Approximation Example ===" << std::endl;

    const std::string csvFile = "examples/airfoilExample/airfoil_self_noise.csv";

    ow::owNeuralNetwork nn;

    // 1. Setup Data
    nn.getDataset()->setRatios(0.6f, 0.2f, 0.2f);

    if (!nn.loadData(csvFile)) {
        std::cerr << "Failed to load data!" << std::endl;
        return -1;
    }
    nn.getDataset()->normalizeData();

    // 2. Build Network Architecture
    // More capacity for complex non-linear airfoil physics
    nn.createNeuralNetwork({36, 36}, "ReLU", "Identity", true);
    
    // Mean Squared Error provides much faster convergence near the minimum for L-BFGS
    nn.setLoss(std::make_shared<ow::owMeanSquaredErrorLoss>());

    // 3. Configure Training
    nn.setOptimizer(std::make_shared<ow::owBFGSOptimizer>(1.0f));

    // 4. Train
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
    ow::owTensor<float, 2> input(1, 5);
    input.setValues({800.0f, 0.0f, 0.3048f, 71.3f, 0.00266337f});
    
    // Smart predict handles both input normalization and output inverse normalization automatically!
    auto pred = nn.predict(input);
    
    std::cout << "Prediction for [800, 0, 0.3048, 71.3, 0.00266337] = " << pred(0, 0) << " (Actual: 126.201)" << std::endl;

    return 0;
}
