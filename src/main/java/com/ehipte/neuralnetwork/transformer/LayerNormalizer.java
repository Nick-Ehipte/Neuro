package com.ehipte.neuralnetwork.transformer;

import java.util.Arrays;

/**
 * Класс выполняющий нормализацию входных векторов.
 */
public class LayerNormalizer {

    /**
     * Малое значение используемое для избегания ситуации деления на 0.
     */
    private static final double EPSILON = 1e-5;

    /**
     * Обучаемые коэффициенты масштабирования. Изначально все коэффициенты равны 1.
     */
    private final double[] scale;

    /**
     * Обучаемые коэффициенты смещения. Изначально все коэффициенты равны 0.
     */
    private final double[] shift;

    public LayerNormalizer(int inputVectorsCount) {
        this.scale = new double[inputVectorsCount];
        this.shift = new double[inputVectorsCount];
        initialize(inputVectorsCount);
    }

    /**
     * Нормализуем входную последовательность векторов.
     *
     * @param inputVectors входная последовательность векторов
     * @return нормализованная матрица векторов
     */
    public double[][] normalize(double[][] inputVectors) {
        double[][] normalizedVectors = new double[inputVectors.length][];
        for (int i = 0; i < inputVectors.length; i++) {
            double mean = calculateMean(inputVectors[i]);
            double dispersion = calculateDispersion(inputVectors[i], mean);
            normalizedVectors[i] = normalizeVector(inputVectors[i], mean, dispersion);
        }
        return normalizedVectors;
    }

    /**
     * Рассчитываем среднее значение вектора.
     *
     * @param inputVector входной вектор
     * @return среднее значение вектора
     */
    private double calculateMean(double[] inputVector) {
        double sum = 0.0;
        for (double v : inputVector) {
            sum += v;
        }
        return sum / inputVector.length;
    }

    /**
     * Рассчитываем дисперсию вектора.
     *
     * @param inputVector входной вектор
     * @param mean        среднее значение вектора
     * @return дисперсию вектора
     */
    private double calculateDispersion(double[] inputVector, double mean) {
        double sum = 0.0;
        for (double v : inputVector) {
            sum += Math.pow(v - mean, 2);
        }
        return sum / inputVector.length;
    }

    /**
     * Проводим нормализацию вектора, масштабирование и смещение его значений согласно коэффициентам.
     *
     * @param inputVector входной вектор
     * @param mean        среднее значение вектора
     * @param dispersion  дисперсия вектора
     * @return нормализованный вектор, значение которого масштабированы и смещены согласно коэффициентам.
     */
    private double[] normalizeVector(double[] inputVector, double mean, double dispersion) {
        for (int i = 0; i < inputVector.length; i++) {
            inputVector[i] = ((inputVector[i] - mean) / Math.sqrt(dispersion + EPSILON)) * scale[i] + shift[i];
        }
        return inputVector;
    }

    /**
     * Инициализируем обучающие коэффициенты масштабирования и смещения. Коэффициенты масштабирования заполняем единицами,
     * а смещения нулями.
     *
     * @param inputVectorsCount размер входной последовательности векторов
     */
    private void initialize(int inputVectorsCount) {
        Arrays.fill(scale, 1);
        Arrays.fill(shift, 0);
    }
}
