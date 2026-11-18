package com.ehipte.neuralnetwork.weights;

import java.util.Random;

public class HeWeightsInitializer implements WeightInitializer {

    private final Random random = new Random();

    @Override
    public double[] generateWeights(int inputs, int outputs) {
        double[] weights = new double[inputs];
        double deviation = Math.sqrt(2.0 / inputs);
        for (int i = 0; i < weights.length; i++) {
            weights[i] = random.nextGaussian(0, deviation);
        }
        return weights;
    }
}
