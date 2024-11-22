'use client';
import { NetworkProvider } from '../contexts/NetworkContext';
import NeuralNetworkVisualizer from '../components/NeuralNetworkVisualizer';
import TrainingControls from '../components/TrainingControls';
import GPUSpecs from '../components/GPUSpecs';
import TrainingHistory from '../components/TrainingHistory';
import { Label } from '@/components/ui/label';

export default function Home() {
  return (
    <NetworkProvider>
      <main className="flex justify-center w-full px-4">
        <div className="flex flex-col w-full max-w-screen-2xl space-y-8 p-4">
          <h1 className="text-2xl font-bold text-center">Neural Network Visualizer</h1>
          <div className="flex flex-col md:flex-row gap-8 justify-center w-full max-w-[1200px] mx-auto">
            <div className="space-y-2 flex flex-col items-center w-full md:w-1/2">
              <Label className="text-lg font-bold">Network Visualizer</Label>
              <NeuralNetworkVisualizer />
            </div>
            <div className="space-y-2 flex flex-col items-center w-full md:w-1/2">
              <Label className="text-lg font-bold">Training Controls</Label>
              <div className="space-y-4 w-full">
                <TrainingControls />
                <GPUSpecs />
              </div>
            </div>
          </div>
          
          <div className="flex flex-col items-center w-full space-y-2 max-w-[1200px] mx-auto">
            <Label className="text-lg font-bold">Training History</Label>
            <TrainingHistory />
          </div>
        </div>
      </main>
    </NetworkProvider>
  );
}
