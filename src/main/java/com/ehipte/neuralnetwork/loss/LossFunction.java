package com.ehipte.neuralnetwork.loss;

public interface LossFunction {

    double calculateLoss(double[] predicted, double[] target);

    double[] calculateErrors(double[] predicted, double[] target);
}
