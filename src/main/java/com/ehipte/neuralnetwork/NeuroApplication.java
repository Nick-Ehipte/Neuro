package com.ehipte.neuralnetwork;

import com.ehipte.neuralnetwork.core.NeuralNetwork;
import com.ehipte.neuralnetwork.training.TrainingDataService;

import java.util.List;

public class NeuroApplication {
    static void main() {
        int correct = 0;
        int fail = 0;
        TrainingDataService trainingDataService = new TrainingDataService();
        trainingDataService.initializeTrainingDataSetFromCsv("src/main/resources/data/mnist_train.csv");

        NeuralNetwork network = NeuralNetwork.builder()
                .learningRate(0.1)
                .inputLayer(784)
                .hiddenLayer(500)
                .outputLayer(10)
                .build();

        double[][] trainingData = trainingDataService.getTrainingData().toArray(double[][]::new);
        double[][] correctAnswers = trainingDataService.getCorrectAnswers().toArray(double[][]::new);


        network.trainForEpochs(5, trainingData, correctAnswers);
        trainingDataService.clearAllData();
        trainingDataService.initializeTrainingDataSetFromCsv("src/main/resources/data/mnist_test.csv");

        double[][] testData = trainingDataService.getTrainingData().toArray(double[][]::new);
        List<Integer> answers = trainingDataService.getCorrectIntAnswers();
        for (int i = 0; i < testData.length; i++) {
            int result = network.query(testData[i]);
            if (answers.get(i) == result) {
                correct++;
            } else {
                fail++;
            }
        }

        System.out.println(correct);
        System.out.println(fail);
        double accuracy = (double) correct / (correct + fail);
        System.out.println("Точность = " + accuracy);
    }
}

