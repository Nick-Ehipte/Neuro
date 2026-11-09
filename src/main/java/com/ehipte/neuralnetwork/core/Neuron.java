package com.ehipte.neuralnetwork.core;

import com.ehipte.neuralnetwork.activation.ActivationFunction;
import com.ehipte.neuralnetwork.activation.SigmoidActivationFunction;
import com.ehipte.neuralnetwork.weights.WeightInitializer;
import lombok.extern.slf4j.Slf4j;

@Slf4j
public class Neuron {

    private WeightInitializer weightInitializer = new WeightInitializer();
    private ActivationFunction activationFunction = new SigmoidActivationFunction();
    private double[] weight;
    private double error;
    private double outputSignal;
    private double[] inputSignals;

    public Neuron(int inputSize) {
        this.weight = weightInitializer.generateWeights(inputSize);
    }

    public double calculateOutputSignal(double[] input) {
        inputSignals = input;
        double sum = 0.0;
        for (int i = 0; i < input.length; i++) {
            sum += weight[i] * input[i];
        }
        outputSignal = activationFunction.activate(sum);
        return outputSignal;
    }

    public void updateWeights(double learningRate) {
        for (int i = 0; i < weight.length; i++) {
            double weightDelta = learningRate * error * outputSignal * (1 - outputSignal) * inputSignals[i];
            weight[i] = weight[i] + weightDelta;
        }
    }

    public double[] getWeight() {
        return weight;
    }

    public void setError(double error) {
        this.error = error;
    }

    public double getOutputSignal() {
        return outputSignal;
    }
}
