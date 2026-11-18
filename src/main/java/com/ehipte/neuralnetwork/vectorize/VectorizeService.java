package com.ehipte.neuralnetwork.vectorize;

import com.ehipte.neuralnetwork.weights.NormalInitializer;
import com.ehipte.neuralnetwork.weights.WeightInitializer;

import java.util.List;

/**
 * Класс отвечающий за создание векторных представлений токенов.
 */
public class VectorizeService {

    private final WeightInitializer weightInitializer = new NormalInitializer();
    /**
     * Веса вложений
     */
    private final double[][] tokenEmbeddings;

    /**
     * Позиционные веса
     */
    private final double[][] positionEmbeddings;

    /**
     * Размер вложения
     */
    private final int embeddingSize;

    /**
     * Длина последовательности
     */
    private final int tokenDataSize;

    public VectorizeService(int vocabularySize, int embeddingSize, int tokenDataSize) {
        this.embeddingSize = embeddingSize;
        this.tokenDataSize = tokenDataSize;
        this.tokenEmbeddings = new double[vocabularySize][embeddingSize];
        this.positionEmbeddings = new double[embeddingSize][tokenDataSize];
        initializeEmbeddingWeights(embeddingSize);
    }

    /**
     * Преобразует входную последовательность токенов в её векторное представление
     *
     * @param tokens входящая последовательность токенов
     * @return векторное представление последовательности токенов
     */
    public double[][] generateVectors(List<Integer> tokens) {
        double[][] vectors = new double[tokenDataSize][embeddingSize];
        for (int i = 0; i < tokens.size(); i++) {
            vectors[i] = calculateFinalVector(i, tokens.get(i), embeddingSize);
        }
        return vectors;
    }

    /**
     * Вычисляет векторное представление последовательности токенов учитывая их позиции
     *
     * @param position позиция токена
     * @param tokenId идентификатор токена
     * @param embeddingSize размер векторного представления
     * @return векторное представление последовательности токенов
     */
    private double[] calculateFinalVector(int position, int tokenId, int embeddingSize) {
        double[] finalVector = new double[embeddingSize];
        double[] tokenEmbedding = tokenEmbeddings[tokenId];
        double[] positionEmbedding = positionEmbeddings[position];
        for (int i = 0; i < tokenEmbedding.length; i++) {
            finalVector[i] = tokenEmbedding[i] + positionEmbedding[i];
        }
        return finalVector;
    }

    /**
     * Инициализация весов слоя вложения и позиционных весов
     *
     * @param embeddingSize размерность вложения
     */
    private void initializeEmbeddingWeights(int embeddingSize) {
        for (int i = 0; i < tokenEmbeddings.length; i++) {
            tokenEmbeddings[i] = weightInitializer.generateWeights(embeddingSize, 0);
        }
        for (int i = 0; i < positionEmbeddings.length; i++) {
            positionEmbeddings[i] = weightInitializer.generateWeights(embeddingSize, 0);
        }
    }
}
