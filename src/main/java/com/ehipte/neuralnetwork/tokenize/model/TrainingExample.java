package com.ehipte.neuralnetwork.tokenize.model;

import java.util.List;

public record TrainingExample(List<Integer> inputTokens, List<Integer> targetTokens) {
}
