package com.ehipte.neuralnetwork.core;


import com.ehipte.neuralnetwork.tokenize.DatasetPreprocessor;
import com.ehipte.neuralnetwork.tokenize.Tokenizer;
import com.ehipte.neuralnetwork.transformer.Dropout;
import com.ehipte.neuralnetwork.transformer.LayerNormalizer;
import com.ehipte.neuralnetwork.transformer.TransformerBlock;
import com.ehipte.neuralnetwork.vectorize.VectorizeService;

import java.util.ArrayList;
import java.util.List;

/**
 * Класс представляющий собой LLM.
 */
public class GptModel {

    private final Tokenizer tokenizer;
    private final DatasetPreprocessor datasetPreprocessor;
    private final VectorizeService vectorizeService;
    private final Dropout dropout;
    private final List<TransformerBlock> transformers;
    private final LayerNormalizer layerNormalizer;
    private final LinearLayer finalLayer;

    private GptModel(Tokenizer tokenizer,
                     DatasetPreprocessor datasetPreprocessor,
                     VectorizeService vectorizeService, Dropout dropout,
                     List<TransformerBlock> transformers,
                     LayerNormalizer layerNormalizer,
                     LinearLayer finalLayer) {
        this.tokenizer = tokenizer;
        this.datasetPreprocessor = datasetPreprocessor;
        this.vectorizeService = vectorizeService;
        this.dropout = dropout;
        this.transformers = transformers;
        this.layerNormalizer = layerNormalizer;
        this.finalLayer = finalLayer;
    }

    public static class Builder {

        private int transformersBlockCount;
        private int vocabularySize;
        private int modelDimension;
        private int inputVectorsCount;
        private double dropoutRate;
        private int headsCount;
        private int ffnExpansionFactor;
        private int stride;

        public Builder transformersBlockCount(int transformersBlockCount) {
            this.transformersBlockCount = transformersBlockCount;
            return this;
        }

        public Builder vocabularySize(int vocabularySize) {
            this.vocabularySize = vocabularySize;
            return this;
        }

        public Builder modelDimension(int modelDimension) {
            this.modelDimension = modelDimension;
            return this;
        }

        public Builder inputVectorsCount(int inputVectorsCount) {
            this.inputVectorsCount = inputVectorsCount;
            return this;
        }

        public Builder dropoutRate(int dropoutRate) {
            this.dropoutRate = dropoutRate;
            return this;
        }

        public Builder headsCount(int headsCount) {
            this.headsCount = headsCount;
            return this;
        }

        public Builder ffnExpansionFactor(int ffnExpansionFactor) {
            this.ffnExpansionFactor = ffnExpansionFactor;
            return this;
        }

        public Builder stride(int stride) {
            this.stride = stride;
            return this;
        }

        public GptModel build() {
            Tokenizer tokenizer = new Tokenizer();
            DatasetPreprocessor datasetPreprocessor = new DatasetPreprocessor();
            VectorizeService vectorizeService = new VectorizeService(vocabularySize, modelDimension, inputVectorsCount);
            Dropout dropout = new Dropout(dropoutRate);
            LayerNormalizer layerNormalizer = new LayerNormalizer(inputVectorsCount);
            //надо ли ему флаг, что он последний слой?
            LinearLayer finalLayer = new LinearLayer(vocabularySize, modelDimension);
            List<TransformerBlock> transformerBlocks = new ArrayList<>();
            for (int i = 0; i < transformersBlockCount; i++) {
                transformerBlocks.add(
                        TransformerBlock.builder()
                                .headsCount(headsCount)
                                .dropoutRate(dropoutRate)
                                .ffnExpansionFactor(ffnExpansionFactor)
                                .inputVectorsCount(inputVectorsCount)
                                .modelDimension(modelDimension)
                                .build()
                );
            }
            return new GptModel(
                    tokenizer,
                    datasetPreprocessor,
                    vectorizeService,
                    dropout,
                    transformerBlocks,
                    layerNormalizer,
                    finalLayer
            );
        }

    }
}
