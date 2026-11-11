package com.ehipte.neuralnetwork.weights;

import lombok.extern.slf4j.Slf4j;

import java.util.Random;

public interface WeightInitializer {

    public double[] generateWeights(int inputs, int outputs);


}
