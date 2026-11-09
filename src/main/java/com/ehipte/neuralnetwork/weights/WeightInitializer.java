package com.ehipte.neuralnetwork.weights;

import lombok.extern.slf4j.Slf4j;

import java.util.Random;

@Slf4j
public class WeightInitializer {
    public double[] generateWeights(int neurons) {
        double[] weights = new double[neurons];
        Random random = new Random();
        double deviation = Math.sqrt(1.0 / neurons);
        for (int i = 0; i < weights.length; i++) {
            weights[i] = random.nextGaussian(0, deviation);
        }
        return weights;
    }
}
