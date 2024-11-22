'use client';
import React, { useEffect, useState } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';

interface GPUInfo {
  description: string;
  vendor: string;
  architecture: string;
  maxBufferSize: number;
  maxBindGroups: number;
  maxBindingsPerBindGroup: number;
  maxDynamicUniformBuffersPerPipelineLayout: number;
  maxDynamicStorageBuffersPerPipelineLayout: number;
  maxSampledTexturesPerShaderStage: number;
  maxSamplersPerShaderStage: number;
  maxStorageBuffersPerShaderStage: number;
  maxStorageTexturesPerShaderStage: number;
  maxUniformBuffersPerShaderStage: number;
  maxComputeWorkgroupSizeX: number;
  maxComputeWorkgroupSizeY: number;
  maxComputeWorkgroupSizeZ: number;
  maxComputeInvocationsPerWorkgroup: number;
}

const GPUSpecs: React.FC = () => {
  const [gpuInfo, setGpuInfo] = useState<GPUInfo | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const getGPUInfo = async () => {
      try {
        if (!navigator.gpu) {
          throw new Error("WebGPU is not supported in this browser");
        }

        const adapter = await navigator.gpu.requestAdapter();
        if (!adapter) {
          throw new Error("Couldn't request WebGPU adapter");
        }

        // Get adapter info
        const info = adapter.info;
        
        // Get adapter limits
        const limits = adapter.limits;

        setGpuInfo({
          description: info.description,
          vendor: info.vendor,
          architecture: info.architecture,
          maxBufferSize: limits.maxBufferSize,
          maxBindGroups: limits.maxBindGroups,
          maxBindingsPerBindGroup: limits.maxBindingsPerBindGroup,
          maxDynamicUniformBuffersPerPipelineLayout: limits.maxDynamicUniformBuffersPerPipelineLayout,
          maxDynamicStorageBuffersPerPipelineLayout: limits.maxDynamicStorageBuffersPerPipelineLayout,
          maxSampledTexturesPerShaderStage: limits.maxSampledTexturesPerShaderStage,
          maxSamplersPerShaderStage: limits.maxSamplersPerShaderStage,
          maxStorageBuffersPerShaderStage: limits.maxStorageBuffersPerShaderStage,
          maxStorageTexturesPerShaderStage: limits.maxStorageTexturesPerShaderStage,
          maxUniformBuffersPerShaderStage: limits.maxUniformBuffersPerShaderStage,
          maxComputeWorkgroupSizeX: limits.maxComputeWorkgroupSizeX,
          maxComputeWorkgroupSizeY: limits.maxComputeWorkgroupSizeY,
          maxComputeWorkgroupSizeZ: limits.maxComputeWorkgroupSizeZ,
          maxComputeInvocationsPerWorkgroup: limits.maxComputeInvocationsPerWorkgroup,
        });
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Unknown error occurred');
      }
    };

    getGPUInfo();
  }, []);

  if (error) {
    return (
      <Card className="bg-destructive/10">
        <CardHeader>
          <CardTitle>GPU Error</CardTitle>
        </CardHeader>
        <CardContent>{error}</CardContent>
      </Card>
    );
  }

  if (!gpuInfo) {
    return (
      <Card>
        <CardHeader>
          <CardTitle>Loading GPU Information...</CardTitle>
        </CardHeader>
      </Card>
    );
  }

  return (
    <Card>
      <CardHeader>
        <CardTitle>GPU Specifications</CardTitle>
      </CardHeader>
      <CardContent className="grid grid-cols-1 md:grid-cols-2 gap-2 text-sm">
        <div className="space-y-2">
          <div>
            <span className="font-bold">Description:</span> {gpuInfo.description || 'Unknown'}
          </div>
          <div>
            <span className="font-bold">Vendor:</span> {gpuInfo.vendor === 'nvidia' ? 'NVIDIA!!!' : `${gpuInfo.vendor} (could be NVIDIA!)` || 'Unknown'}
          </div>
          <div>
            <span className="font-bold">Architecture:</span> {gpuInfo.architecture || 'Unknown'}
          </div>
          <div>
            <span className="font-bold">Max Buffer Size:</span>{' '}
            {(gpuInfo.maxBufferSize / (1024 * 1024)).toFixed(2)} MB
          </div>
          <div>
            <span className="font-bold">Max Bind Groups:</span> {gpuInfo.maxBindGroups}
          </div>
          <div>
            <span className="font-bold">Max Bindings Per Group:</span>{' '}
            {gpuInfo.maxBindingsPerBindGroup}
          </div>
        </div>
        <div className="space-y-2">
          <div>
            <span className="font-bold">Max Compute Workgroup Size:</span>
            <div className="pl-4">
              <div>X: {gpuInfo.maxComputeWorkgroupSizeX}</div>
              <div>Y: {gpuInfo.maxComputeWorkgroupSizeY}</div>
              <div>Z: {gpuInfo.maxComputeWorkgroupSizeZ}</div>
            </div>
          </div>
          <div>
            <span className="font-bold">Max Compute Invocations:</span>{' '}
            {gpuInfo.maxComputeInvocationsPerWorkgroup}
          </div>
        </div>
      </CardContent>
    </Card>
  );
};

export default GPUSpecs; 