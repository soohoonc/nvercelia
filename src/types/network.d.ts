interface NeuralNetworkConfig {
  inputSize: number;
  hiddenSize: number;
  outputSize: number;
  learningRate: number;
}

interface NetworkWeights {
  weightsHidden: Float32Array;
  weightsOutput: Float32Array;
  biasHidden: Float32Array;
  biasOutput: Float32Array;
}

interface TrainingMetrics {
  loss: number;
  epoch: number;
  accuracy?: number;
}

interface NetworkState {
  inputLayer: number[];
  hiddenLayer: number[];
  outputLayer: number[];
  weights: NetworkWeights;
} 