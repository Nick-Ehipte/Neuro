package com.ehipte.neuralnetwork.weights;

import java.util.Random;

public class RandomWeightsInitializer implements WeightInitializer {

    @Override
    public double[] generateWeights(int inputs, int outputs) {
        double[] weights = new double[inputs];
        Random random = new Random();
        for (int i = 0; i < weights.length; i++) {
            weights[i] = 2 * random.nextDouble() - 1;
        }
        return weights;
    }
}
