package com.ehipte.neuralnetwork.loss;

public class MeanSquaredErrorLossFunction implements LossFunction {

    @Override
    public double calculateLoss(double[] predicted, double[] target) {
        double sum = 0.0;
        for (int i = 0; i < predicted.length; i++) {
            sum += Math.pow(predicted[i] - target[i], 2);
        }
        return sum / predicted.length;
    }

    @Override
    public double[] calculateErrors(double[] predicted, double[] target) {
        double[] errors = new double[target.length];
        for (int i = 0; i < target.length; i++) {
            errors[i] = 2 * (predicted[i] - target[i]) / target.length;
        }
        return errors;
    }
}
