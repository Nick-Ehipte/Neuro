package com.ehipte.neuralnetwork.transformer;

import java.util.Random;

/**
 * Класс реализующий генерации маски отсева и её применение.
 */
public class Dropout {

    /**
     * Коэффициент отсева. Используется при создании маски отсева.
     */
    private final double dropoutRate;

    public Dropout(double dropoutRate) {
        this.dropoutRate = dropoutRate;
    }

    /**
     * Применяет маску отсева ко входной матрице, а так же увеличивает оставшиеся значения на величину:
     * scaleCoefficient = 1 / (1 - dropoutRate); Отсеянные позиции принимают значение "0".
     * dropoutRate - коэффициент отсева.
     *
     * @param inputMatrix входная матрица
     * @param dropoutMask маска отсева
     * @return возвращает матрицу весов внимания с отсеянными значениями, остальные значения весов увеличены на
     * коэффициент масштабирования
     */
    //можно не возвращать новый массив, изменять входящий
    public double[][] applyDropoutMaskAndScale(double[][] inputMatrix, double[][] dropoutMask) {
        double[][] weightsAfterDropoutAndScale = new double[inputMatrix.length][inputMatrix[0].length];
        double scaleCoefficient = 1 / (1 - dropoutRate);
        for (int i = 0; i < dropoutMask.length; i++) {
            for (int j = 0; j < dropoutMask[i].length; j++) {
                if (dropoutMask[i][j] == 0) {
                    weightsAfterDropoutAndScale[i][j] = 0;
                } else {
                    weightsAfterDropoutAndScale[i][j] = inputMatrix[i][j] * scaleCoefficient;
                }
            }
        }
        return weightsAfterDropoutAndScale;
    }

    /**
     * Генерирует маску отсева используемую для случайного отсева позиций в матрице
     * dropoutRate - коэффициент отсева.
     *
     * @param numberOfInputEmbeddings размер входной последовательности векторов
     * @return маску отсева, где 0 - отсеиваемый позиция, 1 - позиция которая останется
     */
    public double[][] generateDropoutMask(int numberOfInputEmbeddings) {
        double[][] dropoutMask = new double[numberOfInputEmbeddings][numberOfInputEmbeddings];
        Random random = new Random();
        for (int i = 0; i < dropoutMask.length; i++) {
            for (int j = 0; j < dropoutMask[i].length; j++) {
                double randomDouble = random.nextDouble();
                if (randomDouble < dropoutRate) {
                    dropoutMask[i][j] = 0;
                } else {
                    dropoutMask[i][j] = 1;
                }
            }
        }
        return dropoutMask;
    }
}
