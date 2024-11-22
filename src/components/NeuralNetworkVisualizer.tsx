'use client';
import React, { useMemo } from 'react';
import dynamic from 'next/dynamic';
import { useNetwork } from '../contexts/NetworkContext';

// Dynamically import ForceGraph3D with SSR disabled
const ForceGraph3D = dynamic(
  () => import('react-force-graph').then(mod => mod.ForceGraph3D),
  { ssr: false }
);

const NeuralNetworkVisualizer: React.FC = () => {
  const { networkState, config } = useNetwork();

  console.log('Network State:', networkState);
  console.log('Config:', config);

  const graphData = useMemo(() => {
    if (!networkState) return { nodes: [], links: [] };

    const nodes = [];
    const links = [];

    // Add input layer nodes
    for (let i = 0; i < config.inputSize; i++) {
      nodes.push({
        id: `input-${i}`,
        name: `Input ${i}`,
        val: networkState.inputLayer[i],
        group: 'input',
        layer: 0,
        color: 'green'
      });
    }

    // Add hidden layer nodes
    for (let i = 0; i < config.hiddenSize; i++) {
      nodes.push({
        id: `hidden-${i}`,
        name: `Hidden ${i}`,
        val: networkState.hiddenLayer[i],
        group: 'hidden',
        layer: 1,
      });
    }

    // Add output layer nodes
    for (let i = 0; i < config.outputSize; i++) {
      nodes.push({
        id: `output-${i}`,
        name: `Output ${i}`,
        val: networkState.outputLayer[i],
        group: 'output',
        layer: 2,
        color: 'red'
      });
    }

    // Add connections between input and hidden layer
    for (let i = 0; i < config.inputSize; i++) {
      for (let j = 0; j < config.hiddenSize; j++) {
        const weight = networkState.weights.weightsHidden[i * config.hiddenSize + j];
        links.push({
          source: `input-${i}`,
          target: `hidden-${j}`,
          value: Math.abs(weight),
          weight: weight,
        });
      }
    }

    // Add connections between hidden and output layer
    for (let i = 0; i < config.hiddenSize; i++) {
      for (let j = 0; j < config.outputSize; j++) {
        const weight = networkState.weights.weightsOutput[i * config.outputSize + j];
        links.push({
          source: `hidden-${i}`,
          target: `output-${j}`,
          value: Math.abs(weight),
          weight: weight,
        });
      }
    }

    return { nodes, links };
  }, [networkState, config]);

  if (!networkState) {
    return (
      <div>
        <div>Initializing network...</div>
        <div>Debug info:</div>
        <pre>
          Config: {JSON.stringify(config, null, 2)}
        </pre>
      </div>
    );
  }

  return (
    <div className="w-[600px] h-[600px] border m-auto relative">
      <ForceGraph3D
        graphData={graphData}
        // nodeColor={node => {
        //   const val = node.val as number;
        //   if (node.group === 'input') return `rgb(0, ${Math.floor(val * 255)}, 0)`;
        //   if (node.group === 'hidden') return `rgb(${Math.floor(val * 255)}, 0, ${Math.floor(val * 255)})`;
        //   return `rgb(${Math.floor(val * 255)}, 0, 0)`;
        // }}
        nodeVal={node => 5 + Math.abs(node.val as number) * 10}
        nodeLabel={node => `${node.name}: ${(node.val as number).toFixed(3)}`}
        nodeResolution={8}
        nodeOpacity={0.9}
        linkWidth={link => Math.abs(link.weight as number) * 3}
        linkColor={link => {
          const weight = link.weight as number;
          const intensity = Math.min(Math.abs(weight) * 255, 255);
          return weight > 0 
            ? `rgba(${intensity},0,0,0.5)` 
            : `rgba(0,0,${intensity},0.5)`;
        }}
        linkOpacity={0.3}
        linkDirectionalParticles={3}
        linkDirectionalParticleSpeed={d => Math.abs(d.weight as number) * 0.01}
        linkDirectionalParticleWidth={2}
        linkDirectionalParticleColor="#000000"
        // backgroundColor="#ffffff"
        forceEngine="d3"
        d3AlphaDecay={0.02}
        d3VelocityDecay={0.3}
        cooldownTime={5000}
        width={600}         // Match container width
        height={600}        // Match container height
      />
    </div>
  );
};

export default NeuralNetworkVisualizer; 