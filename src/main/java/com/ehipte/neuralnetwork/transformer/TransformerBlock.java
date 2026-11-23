package com.ehipte.neuralnetwork.transformer;

import com.ehipte.neuralnetwork.attention.MultiHeadAttention;
import com.ehipte.neuralnetwork.core.FeedForwardNetwork;

/**
 * Основной элемент LLM.
 */
public class TransformerBlock {

    private final Dropout dropout;
    private final FeedForwardNetwork ffn;
    private final LayerNormalizer attentionLayerNormalizer;
    private final LayerNormalizer ffnLayerNormalizer;
    private final MultiHeadAttention multiHeadAttention;

    private TransformerBlock(
            Dropout dropout,
            FeedForwardNetwork ffn,
            LayerNormalizer attentionLayerNormalizer,
            LayerNormalizer ffnLayerNormalizer,
            MultiHeadAttention multiHeadAttention
    ) {
        this.dropout = dropout;
        this.ffn = ffn;
        this.attentionLayerNormalizer = attentionLayerNormalizer;
        this.ffnLayerNormalizer = ffnLayerNormalizer;
        this.multiHeadAttention = multiHeadAttention;
    }

    public static Builder builder() {
        return new Builder();
    }

    /**
     * Обеспечивает прохождение сигнала. Входные вектора проходят следующую цепочку:
     * Первый слой нормализации -> Многоцелевой механизм самовнимания -> Отсев -> Используется "короткий путь"
     * (к выходным значениям слоя поэлементно прибавляются входные значения блока) -> Второй слой нормализации ->
     * Полносвязная сеть прямого распространения (внутри используется GELU функция активации) -> Отсев -> Выход
     *
     * @param inputVectors входная матрица векторов
     * @return выходная матрица векторов
     */
    public double[][] forward(double[][] inputVectors) {
        double[][] outputVectors;
        double[][] dropoutMask = dropout.generateDropoutMask(inputVectors.length);
        outputVectors = attentionLayerNormalizer.normalize(inputVectors);
        outputVectors = multiHeadAttention.forward(outputVectors);
        outputVectors = dropout.applyDropoutMaskAndScale(outputVectors, dropoutMask);
        double[][] intermediateVectors = useShortcut(inputVectors, outputVectors);
        outputVectors = ffnLayerNormalizer.normalize(intermediateVectors);
        outputVectors = forwardInFFN(outputVectors);
        dropoutMask = dropout.generateDropoutMask(inputVectors.length);
        outputVectors = dropout.applyDropoutMaskAndScale(outputVectors, dropoutMask);
        outputVectors = useShortcut(intermediateVectors, outputVectors);
        return outputVectors;
    }

    /**
     * Для лучшей обучаемости используется "короткий путь". Выполняет поэлементное сложение
     * входных и выходных матриц векторов.
     *
     * @param inputVectors  матрица входных векторов
     * @param outputVectors матрица выходных векторов
     * @return возвращает матрицу векторов после поэлементного сложения.
     */
    private double[][] useShortcut(double[][] inputVectors, double[][] outputVectors) {
        double[][] calculatedOutput = new double[inputVectors.length][inputVectors[0].length];
        for (int i = 0; i < inputVectors.length; i++) {
            for (int j = 0; j < inputVectors[i].length; j++) {
                calculatedOutput[i][j] = inputVectors[i][j] + outputVectors[i][j];
            }
        }
        return calculatedOutput;
    }

    private double[][] forwardInFFN(double[][] inputVectors) {
        double[][] output = new double[inputVectors.length][];
        for (int i = 0; i < inputVectors.length; i++) {
            output[i] = ffn.forward(inputVectors[i]);
        }
        return output;
    }

    public static class Builder {

        private int modelDimension;
        private int headsCount;
        private double dropoutRate;
        private int ffnExpansionFactor;
        private int inputVectorsCount;

        public Builder modelDimension(int modelDimension) {
            this.modelDimension = modelDimension;
            return this;
        }

        public Builder headsCount(int headsCount) {
            this.headsCount = headsCount;
            return this;
        }

        public Builder dropoutRate(double dropoutRate) {
            this.dropoutRate = dropoutRate;
            return this;
        }

        public Builder ffnExpansionFactor(int ffnExpansionFactor) {
            this.ffnExpansionFactor = ffnExpansionFactor;
            return this;
        }

        public Builder inputVectorsCount(int inputVectorsCount) {
            this.inputVectorsCount = inputVectorsCount;
            return this;
        }


        public TransformerBlock build() {

            if (modelDimension <= 0) {
                throw new IllegalArgumentException("Размерность вектора должна быть больше 0");
            }

            if (inputVectorsCount <= 0) {
                throw new IllegalArgumentException("Последовательность входящих векторов должна быть больше 0");
            }

            if (headsCount < 2) {
                throw new IllegalArgumentException("В многоцелевом механизме внимания должно быть минимум 2 цели.");
            }

            if (ffnExpansionFactor < 2) {
                throw new IllegalArgumentException("Масштабирующий коэффициент не может быть меньше '2'");
            }

            if (modelDimension % headsCount != 0) {
                throw new IllegalStateException("Невозможно нацело разделить размер вектора на количество целей.");
            }

            Dropout dropout = new Dropout(dropoutRate);
            LayerNormalizer attentionLayerNormalizer = new LayerNormalizer(inputVectorsCount);
            LayerNormalizer ffnLayerNormalizer = new LayerNormalizer(inputVectorsCount);
            FeedForwardNetwork ffn = FeedForwardNetwork.builder()
                    .inputLayer(modelDimension)
                    .hiddenLayer(modelDimension * ffnExpansionFactor)
                    .outputLayer(modelDimension)
                    .learningRate(0)
                    .build();
            MultiHeadAttention multiHeadAttention = MultiHeadAttention.builder()
                    .inputVectorSize(modelDimension)
                    .dropoutRate(dropoutRate)
                    .headsCount(headsCount)
                    .inputVectorsCount(inputVectorsCount)
                    .build();

            return new TransformerBlock(
                    dropout,
                    ffn,
                    attentionLayerNormalizer,
                    ffnLayerNormalizer,
                    multiHeadAttention
            );
        }
    }
}
