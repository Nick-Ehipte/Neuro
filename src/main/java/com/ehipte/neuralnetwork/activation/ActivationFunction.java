package com.ehipte.neuralnetwork.activation;

public interface ActivationFunction {

    double activate(double input);

    double derivative(double output);
}
