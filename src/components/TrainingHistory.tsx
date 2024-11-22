'use client';
import React from 'react';
import {
  Table,
  TableBody,
  TableCaption,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { useNetwork } from '../contexts/NetworkContext';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';

const TrainingHistory: React.FC = () => {
  const { sampleHistory, epochMetrics } = useNetwork();

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4 w-full">
      <Card>
        <CardHeader>
          <CardTitle>Sample History (Last 100)</CardTitle>
        </CardHeader>
        <CardContent className="max-h-[600px] overflow-y-auto">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Epoch</TableHead>
                <TableHead>Input</TableHead>
                <TableHead>Target</TableHead>
                <TableHead>Prediction</TableHead>
                <TableHead className="text-right">Loss</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {sampleHistory.slice(-100).map((sample, idx) => (
                <TableRow key={idx}>
                  <TableCell>{sample.epoch}</TableCell>
                  <TableCell>[{sample.input.join(', ')}]</TableCell>
                  <TableCell>{sample.target}</TableCell>
                  <TableCell>{sample.prediction.toFixed(4)}</TableCell>
                  <TableCell className="text-right">{sample.loss.toFixed(4)}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Epoch Metrics</CardTitle>
        </CardHeader>
        <CardContent className="max-h-[600px] overflow-y-auto">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>Epoch</TableHead>
                <TableHead>Loss</TableHead>
                <TableHead className="text-right">Accuracy</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {epochMetrics.map((metric, idx) => (
                <TableRow key={idx}>
                  <TableCell>{metric.epoch}</TableCell>
                  <TableCell>{metric.loss.toFixed(4)}</TableCell>
                  <TableCell className="text-right">
                    {(metric.accuracy * 100).toFixed(1)}%
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </CardContent>
      </Card>
    </div>
  );
};

export default TrainingHistory;