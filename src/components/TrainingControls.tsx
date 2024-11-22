import React from 'react';
import { useNetwork } from '../contexts/NetworkContext';
import { Label } from './ui/label';
import { Input } from './ui/input';
import { Button } from './ui/button';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';

const TrainingControls: React.FC = () => {
  const { 
    config, 
    updateConfig, 
    startTraining, 
    stopTraining, 
    isTraining, 
    metrics,
    maxEpochs,
    setMaxEpochs
  } = useNetwork();

  return (
    <Card>
      <CardHeader>
        <CardTitle>Neural Network Training</CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        <div className="flex flex-col gap-2">
          <div className="flex flex-row items-center gap-2">
            <Label className="text-sm font-semibold w-48">Input Size:</Label>
            <Input
              type="number"
              value={config.inputSize}
              onChange={(e) => updateConfig({ inputSize: parseInt(e.target.value) })}
              min={1}
            />
          </div>
          <div className="flex flex-row items-center gap-2">
            <Label className="text-sm font-semibold w-48">Hidden Layer Size:</Label>
            <Input
              type="number"
              value={config.hiddenSize}
              onChange={(e) => updateConfig({ hiddenSize: parseInt(e.target.value) })}
              min={1}
            />
          </div>
          <div className="flex flex-row items-center gap-2">
            <Label className="text-sm font-semibold w-48">Output Size:</Label>
            <Input
              type="number"
              value={config.outputSize}
              onChange={(e) => updateConfig({ outputSize: parseInt(e.target.value) })}
              min={1}
            />
          </div>
          <div className="flex flex-row items-center gap-2">
            <Label className="text-sm font-semibold w-48">Learning Rate:</Label>
            <Input
              type="number"
              value={config.learningRate}
              onChange={(e) => updateConfig({ learningRate: parseFloat(e.target.value) })}
              step={0.001}
              min={0.001}
            />
          </div>
          <div className="flex flex-row items-center gap-2">
            <Label className="text-sm font-semibold w-48">Max Epochs:</Label>
            <Input
              type="number"
              value={maxEpochs}
              onChange={(e) => setMaxEpochs(parseInt(e.target.value))}
              min={1}
            />
          </div>
        </div>

        <div className="flex flex-row justify-between gap-2">
          <div className="flex flex-col gap-2">
            <div className="grid grid-cols-3 gap-8">
            <div className="flex flex-col items-start">
              <Label className="text-sm font-semibold">Loss:</Label>
              <span className="w-16">{metrics.loss.toFixed(4)}</span>
            </div>
            <div className="flex flex-col items-start">
              <Label className="text-sm font-semibold">Accuracy:</Label>
              <span className="w-16">
                {metrics.accuracy ? `${(metrics.accuracy * 100).toFixed(1)}%` : 'N/A'}
              </span>
            </div>
            <div className="flex flex-col items-start">
              <Label className="text-sm font-semibold">Epoch:</Label>
              <span className="w-16">{metrics.epoch} / {maxEpochs}</span>
            </div>
          </div>
          </div>

          <div className="flex flex-row justify-end">
            <Button
              onClick={isTraining ? stopTraining : startTraining}
              className={isTraining ? 'stop' : 'start'}
            >
              {isTraining ? 'Stop Training' : 'Start Training'}
            </Button>
          </div>
        </div>
       
      </CardContent>
    </Card>
  );
};

export default TrainingControls;
