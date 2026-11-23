package com.ehipte.neuralnetwork.core;

import java.util.ArrayList;
import java.util.List;


public class FeedForwardNetwork {

    private final List<LinearLayer> layers;
    private double learningRate;

    private FeedForwardNetwork(Builder builder) {
        this.learningRate = builder.learningRate;
        this.layers = builder.layers;
    }

    public static Builder builder() {
        return new Builder();
    }

    public void trainOnce(double[] inputSignals, double[] correctAnswer) {
        double[] signals = inputSignals;
        for (LinearLayer layer : layers) {
            signals = layer.propagationSignal(signals);
        }
        double[] errors = calculateErrors(correctAnswer, signals);
        for (int i = layers.size() - 1; i >= 0; i--) {
            LinearLayer currentLayer = layers.get(i);

            LinearLayer nextLayer = i < layers.size() - 1
                    ? layers.get(i + 1)
                    : null;

            errors = currentLayer.backpropagationError(errors, nextLayer);
        }

        for (LinearLayer layer : layers) {
            layer.updateWeights(learningRate);
            layer.updateBiases(learningRate);
        }
    }

    public void trainForEpochs(int epoch, double[][] inputSignals, double[][] correctAnswers) {
        for (int i = 0; i < epoch; i++) {
            for (int j = 0; j < inputSignals.length; j++) {
                trainOnce(inputSignals[j], correctAnswers[j]);
            }
        }
    }

    public double[] forward(double[] inputSignals) {
        double[] signals = inputSignals.clone();
        for (LinearLayer layer : layers) {
            signals = layer.propagationSignal(signals);
        }
        return signals;
    }

    private double[] calculateErrors(double[] correctAnswer, double[] outputSignals) {
        if (correctAnswer.length != outputSignals.length) {
            throw new IllegalStateException("Несовпадение размеров массива правильных ответов и массива выходных сигналов.");
        }
        double[] errors = new double[correctAnswer.length];
        for (int i = 0; i < correctAnswer.length; i++) {
            errors[i] = outputSignals[i] - correctAnswer[i];
        }
        return errors;
    }

    public static class Builder {

        private final List<LinearLayer> layers = new ArrayList<>();
        private double learningRate = 0.01;
        private int previousLayerSize = 0;


        public Builder inputLayer(int neuronCount) {
            if (neuronCount <= 0) {
                throw new IllegalArgumentException("Количество нейронов в слое должен быть больше 0");
            }
            if (!layers.isEmpty()) {
                throw new IllegalStateException("Входной слой должен быть первым");
            }

            LinearLayer layer = new LinearLayer(neuronCount, 0);
            layer.setFirstLayer(true);
            this.layers.add(layer);
            previousLayerSize = neuronCount;
            return this;
        }

        public Builder hiddenLayer(int neuronCount) {
            if (neuronCount <= 0) {
                throw new IllegalArgumentException("Количество нейронов в слое должен быть больше 0");
            }
            LinearLayer layer = new LinearLayer(neuronCount, previousLayerSize);
            this.layers.add(layer);
            previousLayerSize = neuronCount;
            return this;
        }

        public Builder outputLayer(int neuronCount) {
            if (neuronCount <= 0) {
                throw new IllegalArgumentException("Количество нейронов в слое должен быть больше 0");
            }
            if (layers.isEmpty()) {
                throw new IllegalStateException("Сначала необходимо создать входной слой");
            }

            LinearLayer layer = new LinearLayer(neuronCount, previousLayerSize);
            layer.setLastLayer(true);
            this.layers.add(layer);
            return this;
        }

        public Builder learningRate(double learningRate) {
            if (learningRate <= 0) {
                throw new IllegalArgumentException("Скорость обучения должен быть больше 0");
            }
            this.learningRate = learningRate;
            return this;
        }

        public FeedForwardNetwork build() {

            if (layers.isEmpty() || layers.size() == 1) {
                throw new IllegalArgumentException("Слоёв в нейросети должно быть больше 1");
            }

            if (!layers.getFirst().isFirstLayer()) {
                throw new IllegalStateException("Первый слой нейросети должен быть входным");
            }

            if (!layers.getLast().isLastLayer()) {
                throw new IllegalStateException("Последний слой нейросети должен быть выходным");
            }

            return new FeedForwardNetwork(this);
        }
    }
}
