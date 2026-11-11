package com.ehipte.neuralnetwork.activation;

public class SoftmaxActivationFunction {

    public double[] activate(double[] input) {
        double[] outputs = new double[input.length];
        double max = 0.0;
        for (double v : input) {
            max = Math.max(v, max);
        }
        double sumExpInputs = 0.0;
        for (double v : input) {
            sumExpInputs += Math.exp(v - max);
        }
        for (int i = 0; i < input.length; i++) {
            outputs[i] = Math.exp(input[i]) / sumExpInputs;
        }
        return outputs;
    }
}
