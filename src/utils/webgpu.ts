/// <reference types="@webgpu/types" />

export const initializeWebGPU = async () => {
  if (!navigator.gpu) {
    throw new Error("WebGPU not supported");
  }

  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) {
    throw new Error("Couldn't request WebGPU adapter");
  }

  const device = await adapter.requestDevice();
  return { adapter, device };
};

export const createNeuralNetworkPipeline = (
  device: GPUDevice,
  config: NeuralNetworkConfig
) => {
  const shaderCode = `
    @group(0) @binding(0) var<storage, read> inputData: array<f32>;
    @group(0) @binding(1) var<storage, read_write> weightsHidden: array<f32>;
    @group(0) @binding(2) var<storage, read_write> weightsOutput: array<f32>;
    @group(0) @binding(3) var<storage, read_write> biasHidden: array<f32>;
    @group(0) @binding(4) var<storage, read_write> biasOutput: array<f32>;
    @group(0) @binding(5) var<storage, read_write> hiddenLayer: array<f32>;
    @group(0) @binding(6) var<storage, read_write> outputLayer: array<f32>;
    @group(0) @binding(7) var<storage, read> targetOutput: array<f32>;
    @group(0) @binding(8) var<storage, read_write> hiddenGradients: array<f32>;
    @group(0) @binding(9) var<storage, read_write> outputGradients: array<f32>;
    @group(0) @binding(10) var<storage, read_write> loss: array<f32>;

    fn sigmoid(x: f32) -> f32 {
      return 1.0 / (1.0 + exp(-x));
    }

    fn sigmoid_derivative(x: f32) -> f32 {
      let s = sigmoid(x);
      return s * (1.0 - s);
    }

    @compute @workgroup_size(1)
    fn forward(@builtin(global_invocation_id) global_id: vec3<u32>) {
      let neuronId = global_id.x;
      
      // Hidden layer computation
      if (neuronId < ${config.hiddenSize}u) {
        var sum = biasHidden[neuronId];
        for (var i = 0u; i < ${config.inputSize}u; i = i + 1u) {
          sum = sum + inputData[i] * weightsHidden[i * ${config.hiddenSize}u + neuronId];
        }
        hiddenLayer[neuronId] = sigmoid(sum);
      }
      
      // Output layer computation
      if (neuronId < ${config.outputSize}u) {
        var sum = biasOutput[neuronId];
        for (var i = 0u; i < ${config.hiddenSize}u; i = i + 1u) {
          sum = sum + hiddenLayer[i] * weightsOutput[i * ${config.outputSize}u + neuronId];
        }
        outputLayer[neuronId] = sigmoid(sum);
      }
    }

    @compute @workgroup_size(1)
    fn backward(@builtin(global_invocation_id) global_id: vec3<u32>) {
      let neuronId = global_id.x;
      let learning_rate = ${config.learningRate}f;

      // Calculate output layer gradients
      if (neuronId < ${config.outputSize}u) {
        let predicted = outputLayer[neuronId];
        let target = targetOutput[neuronId];
        
        // Calculate error
        let error = predicted - target;
        
        // Store error in loss buffer for reading later
        if (neuronId == 0u) {
          loss[0] = error;  // Store raw error, not squared
        }
        
        // Calculate gradient
        outputGradients[neuronId] = 2.0 * error * sigmoid_derivative(predicted);

        // Update output weights and biases
        for (var i = 0u; i < ${config.hiddenSize}u; i = i + 1u) {
          let weightIndex = i * ${config.outputSize}u + neuronId;
          let delta = learning_rate * outputGradients[neuronId] * hiddenLayer[i];
          weightsOutput[weightIndex] = weightsOutput[weightIndex] - delta;
        }
        biasOutput[neuronId] = biasOutput[neuronId] - learning_rate * outputGradients[neuronId];
      }
    }
  `;

  const shaderModule = device.createShaderModule({
    code: shaderCode,
  });

  const forwardPipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module: shaderModule,
      entryPoint: 'forward',
    },
  });

  const backwardPipeline = device.createComputePipeline({
    layout: 'auto',
    compute: {
      module: shaderModule,
      entryPoint: 'backward',
    },
  });

  return { forwardPipeline, backwardPipeline };
};

export const initializeNetworkWeights = (config: NeuralNetworkConfig): NetworkWeights => {
  const weightsHidden = new Float32Array(config.inputSize * config.hiddenSize);
  const weightsOutput = new Float32Array(config.hiddenSize * config.outputSize);
  const biasHidden = new Float32Array(config.hiddenSize);
  const biasOutput = new Float32Array(config.outputSize);

  // Initialize weights with random values between -1 and 1
  for (let i = 0; i < weightsHidden.length; i++) {
    weightsHidden[i] = Math.random() * 2 - 1;
  }
  for (let i = 0; i < weightsOutput.length; i++) {
    weightsOutput[i] = Math.random() * 2 - 1;
  }
  for (let i = 0; i < biasHidden.length; i++) {
    biasHidden[i] = Math.random() * 2 - 1;
  }
  for (let i = 0; i < biasOutput.length; i++) {
    biasOutput[i] = Math.random() * 2 - 1;
  }

  return {
    weightsHidden,
    weightsOutput,
    biasHidden,
    biasOutput,
  };
};

export const createBuffers = (
  device: GPUDevice,
  config: NeuralNetworkConfig,
  weights: NetworkWeights
) => {
  const inputBuffer = device.createBuffer({
    size: config.inputSize * Float32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  const weightsHiddenBuffer = device.createBuffer({
    size: weights.weightsHidden.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });

  const weightsOutputBuffer = device.createBuffer({
    size: weights.weightsOutput.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });

  const biasHiddenBuffer = device.createBuffer({
    size: weights.biasHidden.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });

  const biasOutputBuffer = device.createBuffer({
    size: weights.biasOutput.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });

  const hiddenLayerBuffer = device.createBuffer({
    size: config.hiddenSize * Float32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  const outputLayerBuffer = device.createBuffer({
    size: config.outputSize * Float32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  const targetOutputBuffer = device.createBuffer({
    size: config.outputSize * Float32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
  });

  const hiddenGradientsBuffer = device.createBuffer({
    size: config.hiddenSize * Float32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.STORAGE,
  });

  const outputGradientsBuffer = device.createBuffer({
    size: config.outputSize * Float32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.STORAGE,
  });

  const lossBuffer = device.createBuffer({
    size: Float32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  // Add these lines to write initial weights to buffers
  device.queue.writeBuffer(weightsHiddenBuffer, 0, weights.weightsHidden);
  device.queue.writeBuffer(weightsOutputBuffer, 0, weights.weightsOutput);
  device.queue.writeBuffer(biasHiddenBuffer, 0, weights.biasHidden);
  device.queue.writeBuffer(biasOutputBuffer, 0, weights.biasOutput);

  return {
    inputBuffer,
    weightsHiddenBuffer,
    weightsOutputBuffer,
    biasHiddenBuffer,
    biasOutputBuffer,
    hiddenLayerBuffer,
    outputLayerBuffer,
    targetOutputBuffer,
    hiddenGradientsBuffer,
    outputGradientsBuffer,
    lossBuffer,
  };
};

export const readBuffer = async (
  device: GPUDevice,
  buffer: GPUBuffer,
  size: number
): Promise<Float32Array> => {
  const readBuffer = device.createBuffer({
    size: size * Float32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
  });

  const commandEncoder = device.createCommandEncoder();
  commandEncoder.copyBufferToBuffer(
    buffer,
    0,
    readBuffer,
    0,
    size * Float32Array.BYTES_PER_ELEMENT
  );
  device.queue.submit([commandEncoder.finish()]);

  await readBuffer.mapAsync(GPUMapMode.READ);
  const copyArrayBuffer = readBuffer.getMappedRange();
  const data = new Float32Array(copyArrayBuffer.slice(0));
  readBuffer.unmap();
  readBuffer.destroy();

  return data;
};

export const trainNetwork = async (
  device: GPUDevice,
  buffers: ReturnType<typeof createBuffers>,
  pipelines: ReturnType<typeof createNeuralNetworkPipeline>,
  config: NeuralNetworkConfig,
  inputData: Float32Array,
  targetData: Float32Array
) => {
  // Initialize loss buffer to zero
  const zeroLoss = new Float32Array([0]);
  device.queue.writeBuffer(buffers.lossBuffer, 0, zeroLoss);

  // Write input and target data to buffers
  device.queue.writeBuffer(buffers.inputBuffer, 0, inputData);
  device.queue.writeBuffer(buffers.targetOutputBuffer, 0, targetData);

  // Create command encoder
  const commandEncoder = device.createCommandEncoder();

  // Forward pass
  const forwardPass = commandEncoder.beginComputePass();
  forwardPass.setPipeline(pipelines.forwardPipeline);
  forwardPass.setBindGroup(0, device.createBindGroup({
    layout: pipelines.forwardPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: buffers.inputBuffer } },
      { binding: 1, resource: { buffer: buffers.weightsHiddenBuffer } },
      { binding: 2, resource: { buffer: buffers.weightsOutputBuffer } },
      { binding: 3, resource: { buffer: buffers.biasHiddenBuffer } },
      { binding: 4, resource: { buffer: buffers.biasOutputBuffer } },
      { binding: 5, resource: { buffer: buffers.hiddenLayerBuffer } },
      { binding: 6, resource: { buffer: buffers.outputLayerBuffer } },
      { binding: 7, resource: { buffer: buffers.targetOutputBuffer } },
      { binding: 8, resource: { buffer: buffers.hiddenGradientsBuffer } },
      { binding: 9, resource: { buffer: buffers.outputGradientsBuffer } },
      { binding: 10, resource: { buffer: buffers.lossBuffer } },
    ],
  }));
  forwardPass.dispatchWorkgroups(Math.max(config.hiddenSize, config.outputSize));
  forwardPass.end();

  // Submit forward pass and wait for completion
  device.queue.submit([commandEncoder.finish()]);
  await device.queue.onSubmittedWorkDone();

  // Read output after forward pass
  const outputBeforeBackward = await readBuffer(device, buffers.outputLayerBuffer, config.outputSize);
  console.log('Output before backward:', Array.from(outputBeforeBackward));

  // New command encoder for backward pass
  const backwardEncoder = device.createCommandEncoder();
  const backwardPass = backwardEncoder.beginComputePass();
  backwardPass.setPipeline(pipelines.backwardPipeline);
  backwardPass.setBindGroup(0, device.createBindGroup({
    layout: pipelines.backwardPipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: buffers.inputBuffer } },
      { binding: 1, resource: { buffer: buffers.weightsHiddenBuffer } },
      { binding: 2, resource: { buffer: buffers.weightsOutputBuffer } },
      { binding: 3, resource: { buffer: buffers.biasHiddenBuffer } },
      { binding: 4, resource: { buffer: buffers.biasOutputBuffer } },
      { binding: 5, resource: { buffer: buffers.hiddenLayerBuffer } },
      { binding: 6, resource: { buffer: buffers.outputLayerBuffer } },
      { binding: 7, resource: { buffer: buffers.targetOutputBuffer } },
      { binding: 8, resource: { buffer: buffers.hiddenGradientsBuffer } },
      { binding: 9, resource: { buffer: buffers.outputGradientsBuffer } },
      { binding: 10, resource: { buffer: buffers.lossBuffer } },
    ],
  }));
  backwardPass.dispatchWorkgroups(Math.max(config.hiddenSize, config.outputSize));
  backwardPass.end();

  // Submit backward pass and wait for completion
  device.queue.submit([backwardEncoder.finish()]);
  await device.queue.onSubmittedWorkDone();

  // Read results
  const loss = await readBuffer(device, buffers.lossBuffer, 1);
  const outputLayer = await readBuffer(device, buffers.outputLayerBuffer, config.outputSize);
  const hiddenLayer = await readBuffer(device, buffers.hiddenLayerBuffer, config.hiddenSize);

  // Calculate MSE loss
  const prediction = outputLayer[0];
  const target = targetData[0];
  const error = prediction - target;
  const mseLoss = error * error;

  // Calculate accuracy (threshold at 0.5 for binary classification)
  const binaryPrediction = prediction > 0.5 ? 1 : 0;
  const accuracy = binaryPrediction === target ? 1 : 0;

  console.log('Training iteration details:', {
    input: Array.from(inputData),
    target,
    prediction,
    binaryPrediction,
    loss: mseLoss,
    accuracy,
    rawOutput: Array.from(outputLayer),
    rawHidden: Array.from(hiddenLayer)
  });

  return {
    loss: mseLoss,
    accuracy,
    outputLayer,
    hiddenLayer,
    inputLayer: inputData,
  };
}; 