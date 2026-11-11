package com.ehipte.neuralnetwork.weights;

import java.util.Random;

public class XavierWeightsInitializer implements WeightInitializer {
    @Override
    public double[] generateWeights(int inputs, int outputs) {
        double[] weights = new double[inputs];
        Random random = new Random();
        double deviation = Math.sqrt(2.0 / (inputs + outputs));
        for (int i = 0; i < weights.length; i++) {
            weights[i] = random.nextGaussian(0, deviation);
        }
        return weights;
    }
}
