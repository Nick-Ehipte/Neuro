package com.ehipte.neuralnetwork.activation;

public class TanhActivationFunction implements ActivationFunction {

    @Override
    public double activate(double input) {
        return Math.tanh(input);
    }

    @Override
    public double derivative(double output) {
        return 1 - output * output;
    }
}
