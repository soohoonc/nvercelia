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
  maxEpochs: number;
  updateConfig: (newConfig: Partial<NeuralNetworkConfig>) => void;
  setMaxEpochs: (epochs: number) => void;
  startTraining: () => Promise<void>;
  stopTraining: () => void;
  sampleHistory: SampleHistory[];
  epochMetrics: EpochMetrics[];
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
  const [metrics, setMetrics] = useState<TrainingMetrics>({ 
    loss: 0, 
    epoch: 0, 
    accuracy: 0 
  });
  const [isTraining, setIsTraining] = useState(false);
  const [maxEpochs, setMaxEpochs] = useState<number>(10);
  const [sampleHistory, setSampleHistory] = useState<SampleHistory[]>([]);
  const [epochMetrics, setEpochMetrics] = useState<EpochMetrics[]>([]);
  
  // Refs to hold WebGPU resources
  const deviceRef = useRef<GPUDevice | null>(null);
  const buffersRef = useRef<ReturnType<typeof createBuffers> | null>(null);
  const pipelinesRef = useRef<ReturnType<typeof createNeuralNetworkPipeline> | null>(null);
  const animationFrameRef = useRef<number | null>(null);

  // Add a ref to track current training sample
  const currentSampleRef = useRef(0);

  // Add state to track total samples processed
  const samplesProcessedRef = useRef(0);
  const epochCompletedRef = useRef(0);

  // Replace state with refs for accumulation
  const epochLossSumRef = useRef(0);
  const epochAccuracySumRef = useRef(0);
  const samplesInEpochRef = useRef(0);

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
      pipelinesRef.current = pipelines;

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

  const generateTrainingData = useCallback(() => {
    // XOR truth table
    const xorData = [
      { input: [0, 0], target: [0] },
      { input: [0, 1], target: [1] },
      { input: [1, 0], target: [1] },
      { input: [1, 1], target: [0] }
    ];

    // Get current sample
    const sampleIndex = samplesProcessedRef.current % xorData.length;
    const sample = xorData[sampleIndex];
    
    console.log('Current training state:', {
      samplesProcessed: samplesProcessedRef.current,
      sampleIndex,
      epochsCompleted: epochCompletedRef.current,
      currentEpoch: Math.floor(samplesProcessedRef.current / 4),
      maxEpochs
    });
    
    // Increment counter
    samplesProcessedRef.current += 1;
    
    // Check if we completed an epoch
    if (sampleIndex === xorData.length - 1) {
      epochCompletedRef.current += 1;
      console.log(`Completed epoch ${epochCompletedRef.current}`);
    }

    return {
      inputData: new Float32Array(sample.input),
      targetData: new Float32Array(sample.target)
    };
  }, [maxEpochs]);

  let stopTraining = useCallback(() => {
    setIsTraining(false);
    if (animationFrameRef.current !== null) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
  }, []);


  const trainingLoop = useCallback(async () => {
    console.log('Training loop started', {
      hasDevice: !!deviceRef.current,
      hasBuffers: !!buffersRef.current,
      hasPipelines: !!pipelinesRef.current,
      isTraining
    });

    if (!deviceRef.current || !buffersRef.current || !pipelinesRef.current) {
      console.log('Missing required resources:', {
        hasDevice: !!deviceRef.current,
        hasBuffers: !!buffersRef.current,
        hasPipelines: !!pipelinesRef.current
      });
      return;
    }

    try {
      const currentEpoch = Math.floor(samplesProcessedRef.current / 4);
      console.log('Current epoch:', currentEpoch, 'max epochs:', maxEpochs);
      
      // Check if we've reached max epochs
      if (currentEpoch >= maxEpochs) {
        console.log('Reached max epochs, stopping training');
        stopTraining();
        return;
      }

      console.log('Generating training data...');
      const { inputData, targetData } = generateTrainingData();

      console.log('Running training iteration...');
      const result = await trainNetwork(
        deviceRef.current,
        buffersRef.current,
        pipelinesRef.current,
        config,
        inputData,
        targetData
      );
      console.log('Training iteration complete:', result);

      // Update network state
      setNetworkState(prev => ({
        ...prev!,
        inputLayer: Array.from(inputData),
        hiddenLayer: Array.from(result.hiddenLayer),
        outputLayer: Array.from(result.outputLayer),
        weights: prev!.weights
      }));

      // Update metrics
      setMetrics(prev => ({
        ...prev,
        loss: result.loss,
        accuracy: result.accuracy,
        epoch: Math.floor(samplesProcessedRef.current / 4)
      }));

      // Add to sample history
      setSampleHistory(prev => [...prev, {
        epoch: Math.floor(samplesProcessedRef.current / 4),
        input: Array.from(inputData),
        target: targetData[0],
        prediction: result.outputLayer[0],
        loss: result.loss
      }]);

      // Accumulate epoch metrics using refs
      epochLossSumRef.current += result.loss;
      epochAccuracySumRef.current += result.accuracy;
      samplesInEpochRef.current += 1;

      // If we completed an epoch (4 samples for XOR)
      if (samplesProcessedRef.current % 4 === 0) {
        const epochLoss = epochLossSumRef.current / 4;
        const epochAccuracy = epochAccuracySumRef.current / 4;
        
        setEpochMetrics(prev => [...prev, {
          epoch: Math.floor(samplesProcessedRef.current / 4),
          loss: epochLoss,
          accuracy: epochAccuracy 
        }]);

        // Update current metrics
        setMetrics({
          loss: epochLoss,
          epoch: Math.floor(samplesProcessedRef.current / 4),
          accuracy: epochAccuracy
        });
        
        // Reset accumulators for next epoch
        epochLossSumRef.current = 0;
        epochAccuracySumRef.current = 0;
        samplesInEpochRef.current = 0;

        console.log(`Epoch ${Math.floor(samplesProcessedRef.current / 4)} completed:`, {
          loss: epochLoss,
          accuracy: epochAccuracy
        });
      }

      // Schedule next iteration immediately if still training
      if (isTraining) {
        // Use requestAnimationFrame directly instead of setTimeout
        animationFrameRef.current = requestAnimationFrame(trainingLoop);
      }
    } catch (error) {
      console.error('Training error:', error);
      stopTraining();
    }
  }, [config, generateTrainingData, maxEpochs, stopTraining]);

  const startTraining = useCallback(async () => {
    try {
      console.log('Starting training...');
      if (!networkState) {
        await initializeNetwork();
      }
      
      // Reset counters
      samplesProcessedRef.current = 0;
      epochCompletedRef.current = 0;
      setMetrics({ loss: 0, epoch: 0 });
      
      // Create a ref to track training state
      const isTrainingRef = { current: true };
      setIsTraining(true);
      
      console.log('Starting animation frame loop...');
      
      async function animate() {
        if (!isTrainingRef.current) return;
        
        await trainingLoop();
        animationFrameRef.current = requestAnimationFrame(animate);
      }
      
      animationFrameRef.current = requestAnimationFrame(animate);
      
      // Update the stopTraining function to use the ref
      const originalStopTraining = stopTraining;
      stopTraining = () => {
        isTrainingRef.current = false;
        originalStopTraining();
      };
      
      epochLossSumRef.current = 0;
      epochAccuracySumRef.current = 0;
      samplesInEpochRef.current = 0;
    } catch (error) {
      console.error('Error starting training:', error);
      setIsTraining(false);
    }
  }, [initializeNetwork, networkState, trainingLoop]);

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
        maxEpochs,
        updateConfig,
        setMaxEpochs,
        startTraining,
        stopTraining,
        sampleHistory,
        epochMetrics,
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