package com.ehipte.neuralnetwork.activation;

public class SigmoidActivationFunction implements ActivationFunction {
    @Override
    public double activate(double input) {
        return 1 / (1 + Math.exp(-input));
    }
}
