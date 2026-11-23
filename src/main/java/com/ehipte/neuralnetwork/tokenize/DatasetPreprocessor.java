package com.ehipte.neuralnetwork.tokenize;

import com.ehipte.neuralnetwork.tokenize.model.TrainingExample;

import java.util.ArrayList;
import java.util.List;

/**
 * Класс осуществляющий подготовку тренировочных данных из последовательности токенов.
 */
public class DatasetPreprocessor {


    /**
     * Преобразует последовательность токенов в тренировочные данные состоящие из входных и целевых данных.
     *
     * @param inputTokens входящая последовательность токенов
     * @param maxLength   максимальная длина последовательности токенов
     * @param stride      смещение в выборке данных для реализации скользящего окна
     * @return лист тренировочных данных
     */
    //добавить batchsize??
    public List<TrainingExample> getTrainingData(List<Integer> inputTokens, int maxLength, int stride) {
        List<TrainingExample> trainingData = new ArrayList<>();

        List<List<Integer>> inputsData = getTokensData(inputTokens, maxLength, stride);
        inputTokens.removeFirst();
        List<List<Integer>> targetData = getTokensData(inputTokens, maxLength, stride);

        for (int i = 0; i < inputsData.size(); i++) {
            trainingData.add(new TrainingExample(inputsData.get(i), targetData.get(i)));
        }
        return trainingData;
    }

    /**
     * Преобразует последовательность токенов в массив последовательностей заданной длины
     *
     * @param inputTokens входящая последовательность токенов
     * @param maxLength   максимальная длина последовательности токенов
     * @param stride       смещение в выборке данных для реализации скользящего окна
     * @return массив последовательностей заданной длины
     */
    private List<List<Integer>> getTokensData(List<Integer> inputTokens, int maxLength, int stride) {
        List<List<Integer>> tokensData = new ArrayList<>();
        for (int i = 0; i < inputTokens.size(); i += stride) {
            List<Integer> oneInput = new ArrayList<>();
            if (i + maxLength < inputTokens.size()) {
                for (int j = i; j < i + maxLength; j++) {
                    oneInput.add(inputTokens.get(j));
                }
                tokensData.add(oneInput);
            }
        }
        return tokensData;
    }
}
