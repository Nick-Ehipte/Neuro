package com.ehipte.neuralnetwork.activation;

public class SigmoidActivationFunction implements ActivationFunction {

    @Override
    public double activate(double inputSum) {
        return 1 / (1 + Math.exp(-inputSum));
    }

    @Override
    public double derivative(double output) {
        return output * (1 - output);
    }
}
