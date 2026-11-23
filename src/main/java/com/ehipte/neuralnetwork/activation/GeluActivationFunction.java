package com.ehipte.neuralnetwork.activation;

public class GeluActivationFunction implements ActivationFunction {

    @Override
    public double activate(double input) {
        return 0.5 * input * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (input + 0.044715 * Math.pow(input, 3))));
    }

    //to do реализовать позже
    @Override
    public double derivative(double output) {
        return 0;
    }
}
