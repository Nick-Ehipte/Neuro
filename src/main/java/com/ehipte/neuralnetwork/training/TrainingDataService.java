package com.ehipte.neuralnetwork.training;

import lombok.Getter;
import lombok.extern.slf4j.Slf4j;

import java.io.BufferedReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

@Slf4j
@Getter
public class TrainingDataService {

    private final List<double[]> trainingData = new ArrayList<>();
    private final List<double[]> correctAnswers = new ArrayList<>();
    private final List<Integer> correctIntAnswers = new ArrayList<>();

    public void initializeTrainingDataSetFromCsv(String path) {
        log.info("Старт получения тренировочных данных из csv файла: {}", path);
        Path paths = Paths.get(path);
        try (BufferedReader reader = Files.newBufferedReader(paths)) {
            String line;
            while ((line = reader.readLine()) != null) {
                String[] values = line.split(",");
                correctAnswers.add(generateCorrectAnswerInArrayFormat(Integer.parseInt(values[0])));
                correctIntAnswers.add(Integer.parseInt(values[0]));
                double[] image = new double[784];
                for (int i = 1; i < values.length; i++) {
                    image[i - 1] = (Double.parseDouble(values[i]) / 255.0 * 0.99) + 0.01;
                }
                trainingData.add(image);
            }
            log.info("Старт получения данных из csv файла: {}", path);
        } catch (Exception e) {
            log.error("Проблема при обработке тренировочных данных", e);
        }
    }

    public double[] generateCorrectAnswerInArrayFormat(int answer) {
        double[] correctAnswer = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        correctAnswer[answer] = 0.99;
        return correctAnswer;
    }

    public void clearAllData() {
        Iterator<double[]> trainingDataIterator = trainingData.iterator();
        while (trainingDataIterator.hasNext()) {
            trainingDataIterator.next();
            trainingDataIterator.remove();
        }
        Iterator<double[]> correctAnswersIterator = correctAnswers.iterator();
        while (correctAnswersIterator.hasNext()) {
            correctAnswersIterator.next();
            correctAnswersIterator.remove();
        }
        Iterator<Integer> correctIntAnswersIterator = correctIntAnswers.iterator();
        while (correctIntAnswersIterator.hasNext()) {
            correctIntAnswersIterator.next();
            correctIntAnswersIterator.remove();
        }
    }
}


