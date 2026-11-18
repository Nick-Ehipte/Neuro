package com.ehipte.neuralnetwork.attention;

import com.ehipte.neuralnetwork.activation.SoftmaxActivationFunction;
import com.ehipte.neuralnetwork.weights.WeightInitializer;
import com.ehipte.neuralnetwork.weights.XavierWeightsInitializer;

import java.util.Random;

/**
 * Класс реализующий механизм самовнимания
 */
public class SelfAttention {

    private final SoftmaxActivationFunction softmaxFunction;
    private final WeightInitializer weightInitializer;

    /**
     * Веса векторов ключей
     */
    private double[][] keyWeights;

    /**
     * Веса векторов запросов
     */
    private double[][] queryWeights;

    /**
     * Веса векторов значений
     */
    private double[][] valueWeights;

    /**
     * Размер поступающей последовательности токенов
     */
    private final int numberOfInputEmbeddings;

    /**
     * Выходной размер контекстного вектора
     */
    private final int outputVectorSize;

    /**
     * Коэффициент отсева. Используется при создании маски отсева.
     */
    private final int dropoutRate;

    public SelfAttention(int numberOfInputEmbeddings, int inputEmbeddingSize, int outputVectorSize, int dropoutRate) {
        this.softmaxFunction = new SoftmaxActivationFunction();
        this.weightInitializer = new XavierWeightsInitializer();
        this.numberOfInputEmbeddings = numberOfInputEmbeddings;
        this.outputVectorSize = outputVectorSize;
        this.dropoutRate = dropoutRate;
        initializeWeightsMatrix(inputEmbeddingSize, outputVectorSize);
    }

    /**
     * Генерирует контекстные вектора для каждого из входящих векторов.
     *
     * @param inputTokenEmbedding последовательность векторов
     * @return матрицу контекстных векторов
     */
    public double[][] generateContextVectors(double[][] inputTokenEmbedding) {
        double[][] contextVectors;

        double[][] keysVectors = calculateProjectionVectors(inputTokenEmbedding, keyWeights, outputVectorSize);
        double[][] queryVectors = calculateProjectionVectors(inputTokenEmbedding, queryWeights, outputVectorSize);
        double[][] valueVectors = calculateProjectionVectors(inputTokenEmbedding, valueWeights, outputVectorSize);

        double[][] attentionScores = calculateAttentionScoreMatrix(queryVectors, keysVectors);
        attentionScores = scaleAttentionScores(attentionScores);
        double[][] normalizedAttentionWeights = applySoftmax(attentionScores);
        double[][] hideMask = generateHideMask(numberOfInputEmbeddings, outputVectorSize);
        double[][] normalizedWeightsAfterHideMask = applyHideMaskAndNormalizeWeights(normalizedAttentionWeights, hideMask);
        double[][] dropoutMask = generateDropoutMask(numberOfInputEmbeddings, outputVectorSize, dropoutRate);
        double[][] attentionWeightsAfterDropoutMask = applyDropoutMaskAndScaleWeights(normalizedWeightsAfterHideMask, dropoutMask, dropoutRate);
        contextVectors = calculateContextVectors(valueVectors, attentionWeightsAfterDropoutMask);

        return contextVectors;
    }

    /**
     * Рассчитывает проекционные (ключи, запросы, значения) вектора
     *
     * @param inputTokenEmbedding последовательность вложений
     * @param weights             весовая матрица проекции
     * @param outputVectorSize    размер выходного вектора
     * @return матрицу проекционных (ключей, запросов, значений) векторов
     */
    private double[][] calculateProjectionVectors(double[][] inputTokenEmbedding, double[][] weights, int outputVectorSize) {
        double[][] projectionVectors = new double[inputTokenEmbedding.length][outputVectorSize];

        for (int i = 0; i < inputTokenEmbedding.length; i++) {
            for (int j = 0; j < outputVectorSize; j++) {
                double sum = 0.0;
                for (int k = 0; k < inputTokenEmbedding[i].length; k++) {
                    sum += inputTokenEmbedding[i][k] * weights[k][j];
                }
                projectionVectors[i][j] = sum;
            }
        }
        return projectionVectors;
    }

    /**
     * Вычисляет матрицу контекстных векторов.
     *
     * @param valueVectors               матрица векторов значений
     * @param normalizedAttentionWeights матрица весов внимания после применения маски сокрытия, нормализации значений,
     *                                   применения маски отсева и масштабирования оставшихся значений.
     * @return матрицу контекстных векторов
     */
    private double[][] calculateContextVectors(double[][] valueVectors, double[][] normalizedAttentionWeights) {
        double[][] contextVectors = new double[normalizedAttentionWeights.length][outputVectorSize];

        for (int i = 0; i < normalizedAttentionWeights.length; i++) {
            for (int j = 0; j < outputVectorSize; j++) {
                double sum = 0.0;
                for (int k = 0; k < valueVectors.length; k++) {
                    sum += normalizedAttentionWeights[i][k] * valueVectors[k][j];
                }
                contextVectors[i][j] = sum;
            }
        }
        return contextVectors;
    }

    /**
     * Вычисляет матрицу показателей внимания
     *
     * @param queryVectors вектора для которого вычисляется показатель внимания
     * @param keysVectors  вектора относительно которого вычисляется показатель внимания
     * @return матрицу показателей внимания
     */
    private double[][] calculateAttentionScoreMatrix(double[][] queryVectors, double[][] keysVectors) {
        //точно ли размерность такая? может надо что-то другое?
        double[][] attentionScoreMatrix = new double[numberOfInputEmbeddings][outputVectorSize];
        for (int i = 0; i < queryVectors.length; i++) {
            //точно ли размерность такая? может надо что-то другое?
            double[] indexes = new double[outputVectorSize];
            for (int j = 0; j < keysVectors.length; j++) {
                indexes[j] = calculateAttentionScore(queryVectors[i], keysVectors[j]);
            }
            attentionScoreMatrix[i] = indexes;
        }
        return attentionScoreMatrix;
    }

    /**
     * Рассчитывает показатель внимания для одного вектора.
     *
     * @param queryVector вектор для которого вычисляется показатель внимания
     * @param keyVector   вектор относительно которого вычисляется показатель внимания
     * @return показатель внимания
     */
    private double calculateAttentionScore(double[] queryVector, double[] keyVector) {
        double attentionScore = 0.0;
        for (int i = 0; i < queryVector.length; i++) {
            attentionScore += queryVector[i] * keyVector[i];
        }
        return attentionScore;
    }

    /**
     * Масштабирует показатели внимания, для того чтобы значения не были слишком большими.
     * Каждый показатель внимания делится на квадратный корень размерности выходного вектора.
     *
     * @param attentionScores матрица показателей внимания.
     * @return матрицу масштабированных показателей внимания
     */
    //можно убрать этот метод и выполнить всё на уровне расчёта матрицы показателей внимания
    //метод сделан для наглядности
    private double[][] scaleAttentionScores(double[][] attentionScores) {
        //можно убрать создание нового массива
        double[][] scaledAttentionsScores = new double[attentionScores.length][attentionScores[0].length];
        for (int i = 0; i < attentionScores.length; i++) {
            for (int j = 0; j < attentionScores[i].length; j++) {
                scaledAttentionsScores[i][j] = attentionScores[i][j] / Math.sqrt(outputVectorSize);
            }
        }
        return scaledAttentionsScores;
    }


    /**
     * Нормализует весовые коэффициенты применяя функцию softmax.
     *
     * @param scaledAttentionScores матрица показателей внимания
     * @return возвращает матрицу весов внимания
     */
    private double[][] applySoftmax(double[][] scaledAttentionScores) {
        double[][] normalizedAttentionWeights = new double[scaledAttentionScores.length][scaledAttentionScores[0].length];
        for (int i = 0; i < scaledAttentionScores.length; i++) {
            normalizedAttentionWeights[i] = softmaxFunction.activate(scaledAttentionScores[i]);
        }
        return normalizedAttentionWeights;
    }

    /**
     * Применяет маску сокрытия к матрице весов внимания, а так же нормализует оставшиеся значения применяя функцию softmax.
     * Отсеянные позиции принимают значение 0.
     *
     * @param attentionWeights матрица весов внимания
     * @param hideMask         маска сокрытия
     * @return возвращает матрицу весов внимания, где значения выше диагонали матрицы отсеяны.
     */
    // можно не возвращать матрицу, а просто модифицировать входную матрицу весов,
    // но тогда надо, чтобы applySoftmax тоже не возвращал матрицу
    private double[][] applyHideMaskAndNormalizeWeights(double[][] attentionWeights, double[][] hideMask) {
        double[][] normalizedWeights = new double[attentionWeights.length][attentionWeights[0].length];
        for (int i = 0; i < attentionWeights.length; i++) {
            for (int j = 0; j < attentionWeights[i].length; j++) {
                if (hideMask[i][j] == Double.NEGATIVE_INFINITY) {
                    attentionWeights[i][j] = Double.NEGATIVE_INFINITY;
                }
            }
        }
        normalizedWeights = applySoftmax(attentionWeights);
        return normalizedWeights;
    }

    /**
     * Генерирует маску сокрытия используемую для сокрытия следующих за вектором весов для каждого
     * вектора в последовательности.
     *
     * @param numberOfInputEmbeddings размер входной последовательности векторов
     * @param outputVectorSize        размер выходного вектора
     * @return маску сокрытия, где позиции выше диагонали матрицы принимают значение NEGATIVE_INFINITY,
     * позиции ниже остаются без изменений.
     */
    private double[][] generateHideMask(int numberOfInputEmbeddings, int outputVectorSize) {
        double[][] hideMask = new double[numberOfInputEmbeddings][outputVectorSize];
        for (int i = 0; i < hideMask.length; i++) {
            for (int j = 0; j < hideMask[i].length; j++) {
                if (i >= j) {
                    hideMask[i][j] = 1;
                } else {
                    hideMask[i][j] = Double.NEGATIVE_INFINITY;
                }
            }
        }
        return hideMask;
    }

    /**
     * Применяет маску отсева к матрице весов внимания, а так же увеличивает оставшиеся значения на величину:
     * scaleCoefficient = 1 / (1 - dropoutRate); Отсеянные позиции принимают значение "0".
     *
     * @param attentionWeights матрица весов внимания
     * @param dropoutMask      маска отсева
     * @param dropoutRate      коэффициент отсева
     * @return возвращает матрицу весов внимания с отсеянными значениями, остальные значения весов увеличены на
     * коэффициент масштабирования
     */
    //можно не возвращать новый массив, изменять входящий
    private double[][] applyDropoutMaskAndScaleWeights(double[][] attentionWeights, double[][] dropoutMask, double dropoutRate) {
        double[][] weightsAfterDropoutAndScale = new double[attentionWeights.length][attentionWeights[0].length];
        double scaleCoefficient = 1 / (1 - dropoutRate);
        for (int i = 0; i < dropoutMask.length; i++) {
            for (int j = 0; j < dropoutMask[i].length; j++) {
                if (dropoutMask[i][j] == 0) {
                    weightsAfterDropoutAndScale[i][j] = 0;
                } else {
                    weightsAfterDropoutAndScale[i][j] = attentionWeights[i][j] * scaleCoefficient;
                }
            }
        }
        return weightsAfterDropoutAndScale;
    }

    /**
     * Генерирует маску отсева используемую для случайного отсева весов внимания
     *
     * @param numberOfInputEmbeddings размер входной последовательности векторов
     * @param outputVectorSize        размер выходного вектора
     * @param dropoutRate             коэффициент отсева
     * @return маску отсева, где 0 - отсеиваемый позиция, 1 - позиция которая останется
     */
    private double[][] generateDropoutMask(int numberOfInputEmbeddings, int outputVectorSize, double dropoutRate) {
        double[][] dropoutMask = new double[numberOfInputEmbeddings][outputVectorSize];
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

    /**
     * Инициализирует матрицы весов
     *
     * @param inputs  количество входных параметров (размер входного вектора)
     * @param outputs количество выходных параметров (размер выходного вектора)
     */
    private void initializeWeightsMatrix(int inputs, int outputs) {
        this.keyWeights = new double[inputs][outputs];
        this.queryWeights = new double[inputs][outputs];
        this.valueWeights = new double[inputs][outputs];
        for (int i = 0; i < keyWeights.length; i++) {
            keyWeights[i] = weightInitializer.generateWeights(inputs, outputs);
        }

        for (int i = 0; i < queryWeights.length; i++) {
            queryWeights[i] = weightInitializer.generateWeights(inputs, outputs);
        }

        for (int i = 0; i < valueWeights.length; i++) {
            valueWeights[i] = weightInitializer.generateWeights(inputs, outputs);
        }
    }
}
