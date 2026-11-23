package com.ehipte.neuralnetwork;

import com.ehipte.neuralnetwork.core.FeedForwardNetwork;
import com.ehipte.neuralnetwork.training.TrainingDataService;

import java.util.List;

public class NeuroApplication {
    static void main() {
        int correct = 0;
        int fail = 0;
        TrainingDataService trainingDataService = new TrainingDataService();
        trainingDataService.initializeTrainingDataSetFromCsv("src/main/resources/data/mnist_train.csv");

        FeedForwardNetwork network = FeedForwardNetwork.builder()
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
            double[] result = network.forward(testData[i]);
            int resultIndex = 0;
            for (int j = 0; j < result.length; j++) {
                if (result[i] > result[resultIndex]) {
                    resultIndex = j;
                }
            }
            if (answers.get(i) == result[resultIndex]) {
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

