package com.ehipte.neuralnetwork.core;

public class NeuralLayer {

    private Neuron[] neurons;

    private boolean isFirstLayer = false;
    private boolean isLastLayer = false;

    public NeuralLayer(int neuronsCount, int inputSize) {
        this.neurons = new Neuron[neuronsCount];
        for (int i = 0; i < neuronsCount; i++) {
            neurons[i] = new Neuron(inputSize, neuronsCount);
        }
    }

    // по книжной архитектуре
    public double[] propagationSignal(double[] inputSignals) {
        if (isFirstLayer) {
            return inputSignals;
        }

        double[] output = new double[neurons.length];

        for (int i = 0; i < neurons.length; i++) {
            output[i] = neurons[i].calculateOutputSignal(inputSignals);
        }
        return output;
    }

    public double[] backpropagationError(double[] errors, NeuralLayer nextLayer) {
        if (isLastLayer) {
            for (int i = 0; i < neurons.length; i++) {
                neurons[i].setError(errors[i]);
            }
            return errors;
        } else if (isFirstLayer) {
            return new double[0];
        } else {
            if (!(errors.length == nextLayer.neurons.length) || !(nextLayer.neurons[0].getWeight().length == neurons.length)) {
                throw new IllegalStateException("Несовпадение числа ошибок и количества нейронов или количества весов и их нейронов");
            }
            double[] errorsOutput = new double[neurons.length];
            for (int i = 0; i < neurons.length; i++) {
                double error = 0.0;
                for (int j = 0; j < errors.length; j++) {
                    error += errors[j] * nextLayer.getNeurons()[j].getWeight()[i];
                }
                errorsOutput[i] = error;
                neurons[i].setError(error);
            }
            return errorsOutput;
        }
    }

    public void updateWeights(double learningRate) {
        for (Neuron neuron : neurons) {
            neuron.updateWeights(learningRate);
        }
    }

    public void updateBiases(double learningRate) {
        for (Neuron neuron : neurons) {
            neuron.updateBias(learningRate);
        }
    }

    public boolean isFirstLayer() {
        return isFirstLayer;
    }

    public void setFirstLayer(boolean firstLayer) {
        if (isLastLayer) {
            throw new IllegalStateException("Слой уже помечен как выходной");
        }
        isFirstLayer = firstLayer;
    }

    public boolean isLastLayer() {
        return isLastLayer;
    }

    public void setLastLayer(boolean lastLayer) {
        if (isFirstLayer) {
            throw new IllegalStateException("Слой уже помечен как входной");
        }
        isLastLayer = lastLayer;
    }

    public Neuron[] getNeurons() {
        return neurons;
    }
}
