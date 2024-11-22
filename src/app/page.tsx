'use client';
import { NetworkProvider } from '../contexts/NetworkContext';
import NeuralNetworkVisualizer from '../components/NeuralNetworkVisualizer';
import TrainingControls from '../components/TrainingControls';
import GPUSpecs from '../components/GPUSpecs';
import { Label } from '@/components/ui/label';

export default function Home() {
  return (
    <NetworkProvider>
      <main className="flex justify-center w-full px-4">
        <div className="flex flex-col w-full max-w-screen-2xl space-y-4 p-4 items-start">
          <div className="flex flex-col md:flex-row gap-8 network-container justify-center w-full max-w-screen-2xl">
            <div className="space-y-2 flex flex-col items-start">
              <Label className="text-lg font-bold">Network Visualizer</Label>
              <NeuralNetworkVisualizer />
            </div>
            <div className="space-y-2 flex flex-col items-start">
              <Label className="text-lg font-bold">Training Controls</Label>
              <div className="space-y-4 w-full">
                <TrainingControls />
                <GPUSpecs />
              </div>
            </div>
          </div>
        </div>
      </main>
    </NetworkProvider>
  );
}
