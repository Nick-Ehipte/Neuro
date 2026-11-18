package com.ehipte.neuralnetwork.weights;

public interface WeightInitializer {

    public double[] generateWeights(int inputs, int outputs);
}
