package com.ehipte.neuralnetwork.attention;

import java.util.ArrayList;
import java.util.List;

/**
 * Класс реализующий многоцелевой механизм самовнимания. Количество целей определяется количеством "голов".
 */
public class MultiHeadAttention {

    private final List<SelfAttention> heads;

    public MultiHeadAttention(Builder builder) {
        this.heads = builder.heads;
    }

    /**
     * Процесс прямого распространения сигнала, использует одноцелевые механизмы самовнимания для получения матриц
     * контекстных векторов, которые соединяет в одну матрицу контекстных векторов.
     *
     * @param inputTokenEmbedding последовательность векторных представлений токенов
     * @return матрицу контекстных векторов
     */
    public double[][] forward(double[][] inputTokenEmbedding) {
        List<double[][]> vectorsList = new ArrayList<>();
        for (SelfAttention head : heads) {
            vectorsList.add(head.generateContextVectors(inputTokenEmbedding));
        }
        return concatenateVectors(vectorsList, inputTokenEmbedding.length, inputTokenEmbedding[0].length);
    }

    /**
     * Построчно соединяет матрицы контекстных векторов полученные от одноцелевых механизмов самовнимания(голов).
     * Использует системный метод System.arraycopy() для конкатенации.
     *
     * @param vectorsFromHeads        лист матриц контекстных векторов полученных от "голов".
     * @param numberOfInputEmbeddings размер входной последовательности токенов
     * @param outputVectorSize        размер выходного (контекстного) вектора
     * @return матрицу контекстных векторов
     */
    private double[][] concatenateVectors(List<double[][]> vectorsFromHeads, int numberOfInputEmbeddings, int outputVectorSize) {
        double[][] contextVectors = new double[numberOfInputEmbeddings][outputVectorSize];
        for (int i = 0; i < vectorsFromHeads.size(); i++) {
            double[][] vectorFromHead = vectorsFromHeads.get(i);
            for (int j = 0; j < vectorFromHead.length; j++) {
                System.arraycopy(vectorFromHead[j], 0, contextVectors[j], i * vectorFromHead[j].length, vectorFromHead[j].length);
            }
        }
        return contextVectors;
    }


    public static MultiHeadAttention.Builder builder() {
        return new MultiHeadAttention.Builder();
    }

    public static class Builder {

        private final List<SelfAttention> heads = new ArrayList<>();

        public MultiHeadAttention.Builder addHeads(int headsCount, int numberOfInputEmbeddings, int inputEmbeddingSize, int dropoutRate) {
            if (headsCount < 2) {
                throw new IllegalArgumentException("В многоцелевом механизме внимания должно быть минимум 2 цели.");
            }
            if (inputEmbeddingSize <= 0) {
                throw new IllegalArgumentException("Размер входного вложения должен быть больше 0");
            }
            if (inputEmbeddingSize % headsCount != 0) {
                throw new IllegalStateException("Невозможно нацело разделить входной вектор на количество целей.");
            }
            if (numberOfInputEmbeddings <= 0) {
                throw new IllegalArgumentException("Количество входных вложений должно быть больше 0");
            }
            if (dropoutRate <= 0) {
                throw new IllegalArgumentException("Коэффициент отсева должен быть больше 0");
            }
            for (int i = 0; i < headsCount; i++) {
                SelfAttention head = new SelfAttention(
                        numberOfInputEmbeddings,
                        inputEmbeddingSize,
                        inputEmbeddingSize / headsCount,
                        dropoutRate
                );
                this.heads.add(head);
            }
            return this;
        }

        public MultiHeadAttention build() {
            return new MultiHeadAttention(this);
        }
    }
}
