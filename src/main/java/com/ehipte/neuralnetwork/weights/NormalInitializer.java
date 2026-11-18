package com.ehipte.neuralnetwork.weights;

import java.util.Random;

public class NormalInitializer implements WeightInitializer {

    private final Random random = new Random();

    @Override
    public double[] generateWeights(int inputs, int outputs) {
        double[] weights = new double[inputs];
        for (int i = 0; i < weights.length; i++) {
            weights[i] = random.nextGaussian(0, 1 / Math.sqrt(inputs));
        }
        return weights;
    }
}
