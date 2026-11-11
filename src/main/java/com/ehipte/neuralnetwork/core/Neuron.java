package com.ehipte.neuralnetwork.core;

import com.ehipte.neuralnetwork.activation.ActivationFunction;
import com.ehipte.neuralnetwork.activation.SigmoidActivationFunction;
import com.ehipte.neuralnetwork.weights.HeWeightsInitializer;
import com.ehipte.neuralnetwork.weights.WeightInitializer;
import lombok.extern.slf4j.Slf4j;

@Slf4j
public class Neuron {

    private WeightInitializer weightInitializer = new HeWeightsInitializer();
    private ActivationFunction activationFunction = new SigmoidActivationFunction();
    private double[] weight;
    private double error;
    private double outputSignal;
    private double[] inputSignals;
    private double bias = 0.0;
    private double errorDelta;

    public Neuron(int inputSize, int outputSize) {
        this.weight = weightInitializer.generateWeights(inputSize, outputSize);
    }

    public double calculateOutputSignal(double[] input) {
        inputSignals = input;
        double sum = 0.0;
        for (int i = 0; i < input.length; i++) {
            sum += weight[i] * input[i];
        }
        sum = sum + bias;
        outputSignal = activationFunction.activate(sum);
        return outputSignal;
    }

    public void updateWeights(double learningRate) {
        for (int i = 0; i < weight.length; i++) {
            errorDelta = error * activationFunction.derivative(outputSignal);
            double weightDelta = learningRate * errorDelta * inputSignals[i];
            //веса отнимают согласно градиентного спуска
            weight[i] = weight[i] - weightDelta;
        }
    }

    public void updateBias(double learningRate) {
        bias = bias + (learningRate * errorDelta);
    }

    public double getWeightedSum() {
        double sum = 0.0;
        for (int i = 0; i < inputSignals.length; i++) {
            sum += weight[i] * inputSignals[i];
        }
        return sum;
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
