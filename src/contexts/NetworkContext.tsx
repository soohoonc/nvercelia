'use client';
import React, { createContext, useContext, useState, useCallback, useRef } from 'react';
import { 
  initializeWebGPU, 
  initializeNetworkWeights, 
  createNeuralNetworkPipeline, 
  createBuffers,
  trainNetwork 
} from '../utils/webgpu';

interface NetworkContextType {
  config: NeuralNetworkConfig;
  networkState: NetworkState | null;
  metrics: TrainingMetrics;
  isTraining: boolean;
  updateConfig: (newConfig: Partial<NeuralNetworkConfig>) => void;
  startTraining: () => Promise<void>;
  stopTraining: () => void;
}

const NetworkContext = createContext<NetworkContextType | undefined>(undefined);

export const NetworkProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [config, setConfig] = useState<NeuralNetworkConfig>({
    inputSize: 2,
    hiddenSize: 4,
    outputSize: 1,
    learningRate: 0.01
  });

  const [networkState, setNetworkState] = useState<NetworkState | null>(null);
  const [metrics, setMetrics] = useState<TrainingMetrics>({ loss: 0, epoch: 0 });
  const [isTraining, setIsTraining] = useState(false);
  
  // Refs to hold WebGPU resources
  const deviceRef = useRef<GPUDevice | null>(null);
  const buffersRef = useRef<ReturnType<typeof createBuffers> | null>(null);
  const pipelinesRef = useRef<ReturnType<typeof createNeuralNetworkPipeline> | null>(null);
  const animationFrameRef = useRef<number | null>(null);

  const initializeNetwork = useCallback(async () => {
    try {
      // Initialize WebGPU
      const { device } = await initializeWebGPU();
      deviceRef.current = device;

      // Initialize weights
      const weights = initializeNetworkWeights(config);

      // Create buffers
      const buffers = createBuffers(device, config, weights);
      buffersRef.current = buffers;

      // Remove the await since createNeuralNetworkPipeline already returns a Promise
      const pipelines = createNeuralNetworkPipeline(device, config);
      pipelinesRef.current = await pipelines;

      // Set initial network state
      setNetworkState({
        inputLayer: Array(config.inputSize).fill(0),
        hiddenLayer: Array(config.hiddenSize).fill(0),
        outputLayer: Array(config.outputSize).fill(0),
        weights
      });

    } catch (error) {
      console.error('Failed to initialize network:', error);
      setIsTraining(false);
    }
  }, [config]);

  // Improved training data generation with batching
  const generateTrainingData = useCallback(() => {
    // XOR truth table
    const xorData = [
      { input: [0, 0], target: [0] },
      { input: [0, 1], target: [1] },
      { input: [1, 0], target: [1] },
      { input: [1, 1], target: [0] }
    ];

    // Randomly select a batch of data
    const batchIndex = Math.floor(Math.random() * xorData.length);
    const sample = xorData[batchIndex];

    return {
      inputData: new Float32Array(sample.input),
      targetData: new Float32Array(sample.target)
    };
  }, []);

  const trainingLoop = useCallback(async () => {
    if (!deviceRef.current || !buffersRef.current || !pipelinesRef.current) {
      console.log('Missing required resources:', {
        hasDevice: !!deviceRef.current,
        hasBuffers: !!buffersRef.current,
        hasPipelines: !!pipelinesRef.current
      });
      return;
    }

    try {
      // Generate new training data for each iteration
      const { inputData, targetData } = generateTrainingData();

      console.log('Training iteration with:', {
        input: Array.from(inputData),
        target: Array.from(targetData)
      });

      const result = await trainNetwork(
        deviceRef.current,
        buffersRef.current,
        pipelinesRef.current,
        config,
        inputData,
        targetData
      );

      console.log('Training result:', result);

      // Update network state
      setNetworkState(prev => {
        if (!prev) return null;
        return {
          ...prev,
          inputLayer: Array.from(inputData),
          hiddenLayer: Array.from(result.hiddenLayer),
          outputLayer: Array.from(result.outputLayer),
          weights: prev.weights // Keep existing weights
        };
      });

      // Update metrics
      setMetrics(prev => ({
        ...prev,
        loss: result.loss,
        accuracy: result.accuracy,
        epoch: prev.epoch + 1
      }));

      // Schedule next iteration with a delay
      if (isTraining) {
        animationFrameRef.current = window.setTimeout(() => {
          requestAnimationFrame(trainingLoop);
        }, 100);
      }
    } catch (error) {
      console.error('Training error:', error);
      stopTraining();
    }
  }, [config, isTraining, generateTrainingData]);

  const startTraining = useCallback(async () => {
    console.log('startTraining');
    try {
      console.log('networkState', networkState);
      if (!networkState) {
        console.log('Initializing network...');
        await initializeNetwork();
      }
      console.log('Starting training...', {
        hasDevice: !!deviceRef.current,
        hasBuffers: !!buffersRef.current,
        hasPipelines: !!pipelinesRef.current
      });
      
      setIsTraining(true);
      requestAnimationFrame(() => {
        console.log('Training loop starting with isTraining:', isTraining);
        trainingLoop();
      });
    } catch (error) {
      console.error('Error starting training:', error);
      setIsTraining(false);
    }
  }, [initializeNetwork, networkState, trainingLoop, isTraining]);

  const stopTraining = useCallback(() => {
    setIsTraining(false);
    if (animationFrameRef.current) {
      clearTimeout(animationFrameRef.current);
      animationFrameRef.current = null;
    }
  }, []);

  const updateConfig = useCallback((newConfig: Partial<NeuralNetworkConfig>) => {
    setConfig((prev: NeuralNetworkConfig) => ({ ...prev, ...newConfig }));
    // Reset network state when config changes
    setNetworkState(null);
  }, []);

  // Initialize network state immediately
  React.useEffect(() => {
    initializeNetwork();
  }, [initializeNetwork]);

  // Cleanup on unmount
  React.useEffect(() => {
    return () => {
      stopTraining();
      if (buffersRef.current) {
        // Cleanup WebGPU resources
        Object.values(buffersRef.current).forEach(buffer => buffer.destroy());
      }
    };
  }, [stopTraining]);

  return (
    <NetworkContext.Provider
      value={{
        config,
        networkState,
        metrics,
        isTraining,
        updateConfig,
        startTraining,
        stopTraining
      }}
    >
      {children}
    </NetworkContext.Provider>
  );
};

export const useNetwork = () => {
  const context = useContext(NetworkContext);
  if (!context) {
    throw new Error('useNetwork must be used within a NetworkProvider');
  }
  return context;
}; 