package com.ehipte.neuralnetwork.core;

import lombok.extern.slf4j.Slf4j;

import java.util.ArrayList;
import java.util.List;

@Slf4j
public class NeuralNetwork {

    private List<NeuralLayer> layers = new ArrayList<>();
    private double learningRate;

    private NeuralNetwork(Builder builder) {
        this.learningRate = builder.learningRate;
        this.layers = builder.layers;
    }

    public static Builder builder() {
        return new Builder();
    }

    public void trainOnce(double[] inputSignals, double[] correctAnswer) {
        double[] signals = inputSignals;
        for (NeuralLayer layer : layers) {
            signals = layer.propagationSignal(signals);
        }

        double[] errors = calculateErrors(correctAnswer, signals);
        for (int i = layers.size() - 1; i >= 0; i--) {
            NeuralLayer currentLayer = layers.get(i);

            NeuralLayer nextLayer = i < layers.size() - 1
                    ? layers.get(i + 1)
                    : null;

            errors = currentLayer.backpropagationError(errors, nextLayer);
        }

        for (NeuralLayer layer : layers) {
            layer.updateWeights(learningRate);
        }
    }

    public void trainForEpochs(int epoch, double[][] inputSignals, double[][] correctAnswers) {
        for (int i = 0; i < epoch; i++) {
            for (int j = 0; j < inputSignals.length; j++) {
                trainOnce(inputSignals[j], correctAnswers[j]);
            }
        }
    }

    public int query(double[] inputSignals) {
        double[] signals = inputSignals;
        for (NeuralLayer layer : layers) {
            signals = layer.propagationSignal(signals);
        }
        int resultIndex = 0;
        for (int i = 0; i < signals.length; i++) {
            if (signals[i] > signals[resultIndex]) {
                resultIndex = i;
            }
        }
        return resultIndex;
    }

    private double[] calculateErrors(double[] correctAnswer, double[] outputSignals) {
//        log.info("Массив корректных значений: {}, массив выходных значений нейросети: {}", correctAnswer, outputSignals);
        if (correctAnswer.length != outputSignals.length) {
            throw new IllegalStateException("Несовпадение размеров массива правильных ответов и массива выходных сигналов.");
        }
        double[] errors = new double[correctAnswer.length];
        for (int i = 0; i < correctAnswer.length; i++) {
            errors[i] = correctAnswer[i] - outputSignals[i];
        }
        return errors;
    }

    public static class Builder {

        private final List<NeuralLayer> layers = new ArrayList<>();
        private double learningRate = 0.01;
        private int previousLayerSize = 0;


        public Builder inputLayer(int neuronCount) {
            if (neuronCount <= 0) {
                throw new IllegalArgumentException("Количество нейронов в слое должен быть больше 0");
            }
            if (!layers.isEmpty()) {
                throw new IllegalStateException("Входной слой должен быть первым");
            }

            NeuralLayer layer = new NeuralLayer(neuronCount, 0);
            layer.setFirstLayer(true);
            this.layers.add(layer);
            previousLayerSize = neuronCount;
            return this;
        }

        public Builder hiddenLayer(int neuronCount) {
            if (neuronCount <= 0) {
                throw new IllegalArgumentException("Количество нейронов в слое должен быть больше 0");
            }
            NeuralLayer layer = new NeuralLayer(neuronCount, previousLayerSize);
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

            NeuralLayer layer = new NeuralLayer(neuronCount, previousLayerSize);
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

        public NeuralNetwork build() {

            if (layers.isEmpty() || layers.size() == 1) {
                throw new IllegalArgumentException("Слоёв в нейросети должно быть больше 1");
            }

            if (!layers.getFirst().isFirstLayer()) {
                throw new IllegalStateException("Первый слой нейросети должен быть входным");
            }

            if (!layers.getLast().isLastLayer()) {
                throw new IllegalStateException("Последний слой нейросети должен быть выходным");
            }

            return new NeuralNetwork(this);
        }
    }
}
