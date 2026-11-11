package com.ehipte.neuralnetwork.loss;

public class CrossEntropyLossFunction implements LossFunction {
    private static final double EPSILON = 1e-15;

    @Override
    public double calculateLoss(double[] predicted, double[] target) {
        double loss = 0.0;
        for (int i = 0; i < predicted.length; i++) {
            loss += target[i] * Math.log(Math.max(predicted[i], EPSILON));
        }
        return -loss;
    }

    @Override
    public double[] calculateErrors(double[] predicted, double[] target) {
        double[] errors = new double[target.length];
        for (int i = 0; i < target.length; i++) {
            errors[i] = -(target[i] / Math.max(predicted[i], EPSILON));
        }
        return errors;
    }
}
